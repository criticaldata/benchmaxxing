"""The shared text-lane model dispatch (`experiments/_lane.py`).

Every text runner used to carry its own copy of this logic and its own hardcoded Gemini id. These
tests pin the contract the runners now depend on, and in particular that the cache key is unchanged
from the per-runner caches, so every committed Gemini cache still replays with no API calls.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import _lane  # noqa: E402


def test_key_name_and_key_follow_the_model_id(monkeypatch):
    assert _lane.key_name("gemini-2.5-flash-lite") == "GEMINI_API_KEY"
    assert _lane.key_name("deepseek-ai/deepseek-v4-flash-0731") == "DEEPSEEK_API_KEY"
    assert _lane.key_name("nvidia/nemotron-3-super-120b-a12b") == "NVIDIA_API_KEY"
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("NVIDIA_API_KEY", "nv")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds")
    assert _lane.key_for("gemini-2.5-flash-lite") == "g"
    assert _lane.key_for("nvidia/nemotron-3-super-120b-a12b") == "nv"
    assert _lane.key_for("deepseek-ai/deepseek-v4-flash-0731") == "ds"


def test_gemini_routes_to_the_google_sdk(monkeypatch):
    """Dispatch only: building a real GeminiBackend would construct an SDK client."""
    seen = {}
    monkeypatch.setattr(_lane.gateway, "GeminiBackend",
                        lambda model, api_key: seen.update(model=model, api_key=api_key) or "gem")
    assert _lane.backend_for("gemini-2.5-flash-lite", "g") == "gem"
    assert seen == {"model": "gemini-2.5-flash-lite", "api_key": "g"}


def test_everything_else_routes_to_the_openai_compatible_path_with_a_cap():
    class _Stub:
        pass

    nim = _lane.backend_for("nvidia/nemotron-3-super-120b-a12b", "nvapi-test", client=_Stub())
    assert isinstance(nim, _lane.gateway.LocalOpenAICompatibleBackend)
    assert nim.base_url == _lane.NIM_BASE_URL
    # A cap that lands mid-reasoning is returned in `content` and would then be scored.
    assert nim.default_decoding["max_tokens"] == _lane.MAX_TOKENS
    ds = _lane.backend_for("deepseek-ai/deepseek-v4-flash-0731", "sk", client=_Stub())
    assert ds.base_url == _lane.DEEPSEEK_BASE_URL


def test_cache_key_is_unchanged_from_the_per_runner_caches(tmp_path):
    """The committed Gemini caches must keep replaying: same sha256(model NUL prompt) key."""
    model, prompt = "gemini-2.5-flash-lite", "Question: x\n\nOptions:\nA. a\nB. b\n\n"
    expected = hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"k": expected, "model": model, "resp": "B"}) + "\n")
    cache = _lane.Cache(path, None, model)
    assert cache.complete(prompt) == "B"
    assert cache.calls == 0


def test_a_miss_without_a_key_names_the_variable_it_wants(tmp_path):
    cache = _lane.Cache(tmp_path / "c.jsonl", None, "nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(SystemExit) as exc:
        cache.complete("uncached")
    assert "NVIDIA_API_KEY" in str(exc.value)


def test_reasoning_only_completion_is_refused(tmp_path, monkeypatch):
    """content=None must fail loudly rather than cache a null the parsers would read."""
    class _Null:
        def complete(self, prompt, decoding=None):
            return None

    monkeypatch.setattr(_lane, "backend_for", lambda model, key, client=None: _Null())
    monkeypatch.setattr(_lane.gateway, "RetryBackend", lambda b, tries=5, backoff=3.0: b)
    cache = _lane.Cache(tmp_path / "c.jsonl", "nvapi-test", "nvidia/x")
    with pytest.raises(SystemExit) as exc:
        cache.complete("hello")
    assert "content=None" in str(exc.value)


def test_default_model_keeps_the_committed_paths_and_others_are_scoped(tmp_path):
    default_cache = str(tmp_path / "results" / "arm_cache.jsonl")
    out, cache = _lane.scoped(_lane.DEFAULT_MODEL, str(tmp_path / "results"), default_cache)
    assert out == tmp_path / "results" and cache == Path(default_cache)
    out2, cache2 = _lane.scoped("nvidia/nemotron-3-super-120b-a12b", str(tmp_path / "results"),
                                default_cache)
    assert out2 == tmp_path / "results" / "nvidia_nemotron-3-super-120b-a12b"
    assert cache2.name == "nvidia_nemotron-3-super-120b-a12b_arm_cache.jsonl"
    assert cache2.parent == Path(default_cache).parent


def test_declared_reads_a_committed_letter_and_refuses_prose():
    opts = ["Psoriatic arthritis", "Reactive arthritis", "Gout", "Septic arthritis"]
    assert _lane.declared("B", opts) == "B"
    assert _lane.declared("Some reasoning.\n\nB", opts) == "B"
    assert _lane.declared("The answer is B.", opts) == "B"
    assert _lane.declared("The correct answer is **B**.", opts) == "B"
    assert _lane.declared("Answer: B", opts) == "B"
    # Truncated reasoning that merely mentions an option is not a declaration.
    assert _lane.declared("Psoriatic arthritis is unlikely because the patient", opts) is None
    assert _lane.declared("", opts) is None
    assert _lane.declared("Z", opts) is None


def test_pacing_follows_the_documented_nim_ceiling(monkeypatch):
    """#416 measured the NVIDIA endpoint at about 40 RPM and found it punishes bursts."""
    import time as _time
    monkeypatch.setattr(_lane, "MIN_CALL_INTERVAL", 0)
    # Gemini has no such restriction, so it is not paced at all.
    assert _lane.interval_for("gemini-2.5-flash-lite") == 0.0
    # The documented ceiling is 40 RPM, but a free-tier key sustains far less, so the default
    # interval is the measured sustained rate and stays well inside the documented one.
    gap = _lane.interval_for("nvidia/nemotron-3-super-120b-a12b")
    assert gap == _lane.NIM_SUSTAINED_INTERVAL
    assert 60.0 / gap <= _lane.NIM_RPM

    monkeypatch.setattr(_lane, "MIN_CALL_INTERVAL", 0.2)
    assert _lane.interval_for("nvidia/x") == 0.2
    monkeypatch.setattr(_lane, "_last_call", [_time.monotonic()])
    t0 = _time.monotonic()
    _lane._pace("nvidia/x")
    assert _time.monotonic() - t0 >= 0.15

    monkeypatch.setattr(_lane, "MIN_CALL_INTERVAL", 0)
    t0 = _time.monotonic()
    _lane._pace("gemini-2.5-flash-lite")
    assert _time.monotonic() - t0 < 0.05


def test_rate_limit_detection_covers_the_vendor_shapes():
    class _Vendor429(Exception):
        pass
    _Vendor429.__name__ = "RateLimitError"
    assert _lane._is_rate_limited(_Vendor429("Error code: 429"))
    assert _lane._is_rate_limited(RuntimeError("Error code: 429 - Too Many Requests"))

    class _Coded(Exception):
        status_code = 429
    assert _lane._is_rate_limited(_Coded())
    assert not _lane._is_rate_limited(RuntimeError("Error code: 500 - server error"))


def test_a_429_waits_for_a_refill_instead_of_losing_the_run(tmp_path, monkeypatch):
    """RetryBackend's five quick attempts expire while the bucket is still empty."""
    calls = {"n": 0}

    class _Flaky:
        def complete(self, prompt, decoding=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Error code: 429 - {'status': 429}")
            return "B"

    monkeypatch.setattr(_lane, "backend_for", lambda model, key, client=None: _Flaky())
    monkeypatch.setattr(_lane.gateway, "RetryBackend", lambda b, tries=5, backoff=3.0: b)
    monkeypatch.setattr(_lane, "RATE_LIMIT_SLEEP", 0.01)
    monkeypatch.setattr(_lane, "MIN_CALL_INTERVAL", 0.001)
    cache = _lane.Cache(tmp_path / "c.jsonl", "nvapi-test", "nvidia/x")
    assert cache.complete("p") == "B"
    assert calls["n"] == 3 and cache.calls == 1


def test_a_non_rate_limit_error_still_fails_fast(tmp_path, monkeypatch):
    class _Broken:
        def complete(self, prompt, decoding=None):
            raise RuntimeError("Error code: 500 - server error")

    monkeypatch.setattr(_lane, "backend_for", lambda model, key, client=None: _Broken())
    monkeypatch.setattr(_lane.gateway, "RetryBackend", lambda b, tries=5, backoff=3.0: b)
    monkeypatch.setattr(_lane, "MIN_CALL_INTERVAL", 0.001)
    cache = _lane.Cache(tmp_path / "c.jsonl", "nvapi-test", "nvidia/x")
    with pytest.raises(RuntimeError, match="500"):
        cache.complete("p")
