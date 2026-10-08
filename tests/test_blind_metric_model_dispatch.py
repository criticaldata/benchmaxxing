"""Model dispatch for the text blind-metric lane (the --model flag, #416's shape).

The imaging lane got ``--model`` and per-model key/backend dispatch in #416; this pins the same
contract for the text lane, since every text runner previously hardcoded one Gemini model id.
"""
import pytest

from experiments.blind_metric import blind_metric as bm


def test_key_name_follows_the_model_id():
    assert bm._key_name("gemini-2.5-flash-lite") == "GEMINI_API_KEY"
    assert bm._key_name("deepseek-ai/deepseek-v4-flash-0731") == "DEEPSEEK_API_KEY"
    assert bm._key_name("nvidia/nemotron-3-super-120b-a12b") == "NVIDIA_API_KEY"
    assert bm._key_name("moonshotai/kimi-k3") == "NVIDIA_API_KEY"


def test_key_reads_the_right_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("NVIDIA_API_KEY", "n")
    assert bm._key("gemini-2.5-flash-lite") == "g"
    assert bm._key("deepseek-ai/deepseek-v4-flash-0731") == "d"
    assert bm._key("nvidia/nemotron-3-super-120b-a12b") == "n"


def test_key_does_not_hand_a_gemini_key_to_a_nim_model(monkeypatch):
    """The failure #416 fixed in the imaging lane: one env var served every model."""
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert bm._key("nvidia/nemotron-3-super-120b-a12b") is None


def test_backend_dispatch_and_the_nim_output_cap():
    stub = object()  # the gateway's injection hook: no SDK client, no network
    nim = bm._backend("nvidia/nemotron-3-super-120b-a12b", "nvapi-test", client=stub)
    assert isinstance(nim, bm.gateway.LocalOpenAICompatibleBackend)
    assert nim.base_url == bm.NIM_BASE_URL
    # #417: an uncapped completion runs to the model ceiling and is then mis-scored.
    assert nim.default_decoding["max_tokens"] == bm.NIM_MAX_TOKENS

    deepseek = bm._backend("deepseek-ai/deepseek-v4-flash-0731", "sk-test", client=stub)
    assert deepseek.base_url == "https://api.deepseek.com"


def test_gemini_ids_still_route_to_the_google_sdk(monkeypatch):
    """Dispatch only: constructing a real GeminiBackend would build an SDK client."""
    seen = {}

    def _fake(model, api_key):
        seen["model"], seen["api_key"] = model, api_key
        return "gemini-backend"

    monkeypatch.setattr(bm.gateway, "GeminiBackend", _fake)
    assert bm._backend("gemini-2.5-flash-lite", "g") == "gemini-backend"
    assert seen == {"model": "gemini-2.5-flash-lite", "api_key": "g"}


def test_cache_miss_names_the_key_the_model_needs(tmp_path):
    cache = bm._Cache(tmp_path / "c.jsonl", None, "nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(SystemExit) as exc:
        cache.complete("nvidia/nemotron-3-super-120b-a12b", "hello")
    assert "NVIDIA_API_KEY" in str(exc.value)


def test_cache_key_is_model_scoped(tmp_path):
    """Two models must not read each other's cached completions."""
    cache = bm._Cache(tmp_path / "c.jsonl", None, "gemini-2.5-flash-lite")
    cache.store["seed"] = "x"
    import hashlib
    k1 = hashlib.sha256("gemini-2.5-flash-lite\x00p".encode()).hexdigest()
    k2 = hashlib.sha256("nvidia/nemotron-3-super-120b-a12b\x00p".encode()).hexdigest()
    assert k1 != k2
    cache.store[k1] = "gemini answer"
    assert cache.complete("gemini-2.5-flash-lite", "p") == "gemini answer"
    with pytest.raises(SystemExit):
        cache.complete("nvidia/nemotron-3-super-120b-a12b", "p")


def test_reasoning_only_completion_is_refused(tmp_path, monkeypatch):
    """A model that returns content=None must fail loudly, not cache a null."""
    class _Null:
        def complete(self, prompt, image=None, decoding=None):
            return None

    monkeypatch.setattr(bm, "_backend", lambda model, key: _Null())
    cache = bm._Cache(tmp_path / "c.jsonl", "nvapi-test", "nvidia/x")
    with pytest.raises(SystemExit) as exc:
        cache.complete("nvidia/x", "hello")
    assert "content=None" in str(exc.value)


def test_declared_letter_is_the_terminal_bare_letter_only():
    """A completion counts as declared only when its last non-empty line is a bare option letter."""
    L = ["A", "B", "C", "D", "E"]
    assert bm._declared("B", L) == "B"
    assert bm._declared("Justification sentence.  \nB", L) == "B"
    assert bm._declared("Reasoning.\n\n**D**\n", L) == "D"
    assert bm._declared("Reasoning.\n(C)", L) == "C"
    # truncated reasoning that merely mentions options is NOT a declaration
    assert bm._declared("The rubric awards full marks to option A. We need to", L) is None
    assert bm._declared("D. Excessive somatostatin secretion:", L) is None
    assert bm._declared("", L) is None
    assert bm._declared(None, L) is None
    # a letter outside the option set is not a declaration
    assert bm._declared("E", ["A", "B", "C", "D"]) is None


def test_nim_cap_is_high_enough_not_to_truncate_reasoning():
    """A cap that lands mid-reasoning puts the chain of thought in content, where the parser scores it."""
    assert bm.NIM_MAX_TOKENS >= 8192
