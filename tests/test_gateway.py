"""Tests for benchmaxxing.gateway.

Everything here runs offline: MockBackend stands in for a real model, and GeminiBackend is
exercised with an injected fake client (never a real API call). The one path that needs the
google-genai library either asserts a clear ImportError when it is absent or skips when it is
present.
"""

from __future__ import annotations

import pytest

from benchmaxxing import gateway


# --------------------------------------------------------------------------- MockBackend


def test_mock_backend_echoes_by_default():
    be = gateway.MockBackend()
    assert be.complete("hello") == "hello"
    assert be.n_calls == 1


def test_mock_backend_applies_rule():
    be = gateway.MockBackend(rule=lambda prompt, image, decoding: prompt.upper())
    assert be.complete("hi") == "HI"


def test_mock_backend_rule_sees_image_and_decoding():
    seen = {}

    def rule(prompt, image, decoding):
        seen["image"] = image
        seen["decoding"] = decoding
        return "ok"

    be = gateway.MockBackend(rule=rule)
    be.complete("p", image=b"img", decoding={"temperature": 0.2})
    assert seen == {"image": b"img", "decoding": {"temperature": 0.2}}


def test_wrappers_are_backends():
    be = gateway.MockBackend()
    assert isinstance(gateway.with_retry(be), gateway.Backend)
    assert isinstance(gateway.cached(be), gateway.Backend)


# --------------------------------------------------------------------------- with_retry


def _flaky_rule(fail_times):
    """Return a rule that raises RuntimeError the first ``fail_times`` calls, then succeeds."""
    state = {"n": 0}

    def rule(prompt, image, decoding):
        state["n"] += 1
        if state["n"] <= fail_times:
            raise RuntimeError(f"transient failure #{state['n']}")
        return "recovered"

    return rule


def test_with_retry_recovers_after_transient_failures():
    be = gateway.MockBackend(rule=_flaky_rule(fail_times=2))
    wrapped = gateway.with_retry(be, tries=3)
    assert wrapped.complete("go") == "recovered"
    assert be.n_calls == 3  # failed twice, succeeded on the third attempt


def test_with_retry_exhausts_and_raises_retry_error():
    be = gateway.MockBackend(rule=_flaky_rule(fail_times=99))
    wrapped = gateway.with_retry(be, tries=3)
    with pytest.raises(gateway.RetryError) as exc:
        wrapped.complete("go")
    assert be.n_calls == 3
    assert "3 attempt(s)" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)  # original error chained


def test_with_retry_terminal_error_not_retried():
    def rule(prompt, image, decoding):
        raise KeyError("terminal")

    be = gateway.MockBackend(rule=rule)
    # Only ValueError is transient here, so a KeyError surfaces immediately without retry.
    wrapped = gateway.with_retry(be, tries=5, transient=(ValueError,))
    with pytest.raises(KeyError):
        wrapped.complete("go")
    assert be.n_calls == 1


def test_with_retry_backoff_uses_injected_sleep():
    slept = []
    be = gateway.MockBackend(rule=_flaky_rule(fail_times=2))
    wrapped = gateway.with_retry(be, tries=3, backoff=0.5, sleep=slept.append)
    assert wrapped.complete("go") == "recovered"
    assert slept == [0.5, 1.0]  # backoff * attempt for attempts 1 and 2


def test_with_retry_rejects_bad_tries():
    with pytest.raises(ValueError):
        gateway.with_retry(gateway.MockBackend(), tries=0)


# --------------------------------------------------------------------------- cached


def test_cached_serves_second_identical_call_from_cache():
    be = gateway.MockBackend()  # default echo, counts calls
    wrapped = gateway.cached(be)
    first = wrapped.complete("same")
    second = wrapped.complete("same")
    assert first == second == "same"
    assert be.n_calls == 1  # second call served from cache, backend hit only once


def test_cached_distinguishes_different_prompts():
    be = gateway.MockBackend()
    wrapped = gateway.cached(be)
    wrapped.complete("a")
    wrapped.complete("b")
    assert be.n_calls == 2


def test_cached_key_includes_image_and_decoding():
    be = gateway.MockBackend()
    wrapped = gateway.cached(be)
    wrapped.complete("p")
    wrapped.complete("p", image=b"img")
    wrapped.complete("p", image=b"img", decoding={"temperature": 0.1})
    wrapped.complete("p", image=b"img", decoding={"temperature": 0.1})  # cache hit
    assert be.n_calls == 3  # three distinct keys, the fourth call is cached
    assert len(wrapped.cache) == 3


def test_cached_decoding_key_order_insensitive():
    be = gateway.MockBackend()
    wrapped = gateway.cached(be)
    wrapped.complete("p", decoding={"a": 1, "b": 2})
    wrapped.complete("p", decoding={"b": 2, "a": 1})  # same content, different order
    assert be.n_calls == 1


# --------------------------------------------------------------------------- GeminiBackend


class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self):
        self.received = []

    def generate_content(self, model, contents, **kwargs):
        self.received.append((model, contents, kwargs))
        return _FakeResp(f"ok:{contents[0]}")


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


def test_gemini_backend_missing_library_raises_clear_error():
    try:
        import google.genai  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError) as exc:
            gateway.GeminiBackend()
        assert "models" in str(exc.value)
    else:
        pytest.skip("google-genai is installed; the missing-library path is not exercised")


def test_gemini_backend_injected_client_text_path():
    client = _FakeClient()
    be = gateway.GeminiBackend(model="gemini-2.5-flash", client=client)
    out = be.complete("hi", decoding={"temperature": 0.0})
    assert out == "ok:hi"
    model, contents, kwargs = client.models.received[0]
    assert model == "gemini-2.5-flash"
    assert contents == ["hi"]
    assert kwargs["config"]["temperature"] == 0.0


def test_gemini_backend_injected_client_multimodal_path():
    client = _FakeClient()
    be = gateway.GeminiBackend(client=client)
    be.complete("describe", image=b"imgbytes")
    _, contents, kwargs = client.models.received[0]
    assert contents == ["describe", b"imgbytes"]
    assert "config" not in kwargs  # no decoding overrides -> no config passed


def test_gemini_backend_merges_default_decoding():
    client = _FakeClient()
    be = gateway.GeminiBackend(client=client, default_decoding={"temperature": 0.7, "top_p": 0.9})
    be.complete("hi", decoding={"temperature": 0.1})
    _, _, kwargs = client.models.received[0]
    assert kwargs["config"] == {"temperature": 0.1, "top_p": 0.9}  # per-call overrides default
