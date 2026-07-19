"""Tests for the extra provider backends in benchmaxxing.gateway.

Everything here runs offline: each real provider backend is exercised with an injected fake
client (never a real API call). The paths that need a vendor SDK either assert a clear
ImportError when the library is absent or skip when it is present.
"""

from __future__ import annotations

import base64

import pytest

from benchmaxxing import gateway


# --------------------------------------------------------------------------- fake clients


class _FakeChatMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeChatMessage(content)


class _FakeChatResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self):
        self.received = []

    def create(self, model, messages, **kwargs):
        self.received.append((model, messages, kwargs))
        return _FakeChatResp(f"ok:{model}")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = _FakeChat()


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeMessagesResp:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self):
        self.received = []

    def create(self, model, max_tokens, messages, **kwargs):
        self.received.append((model, max_tokens, messages, kwargs))
        return _FakeMessagesResp("ok")


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeMessages()


# --------------------------------------------------------------------------- contract


@pytest.mark.parametrize(
    "cls",
    [
        gateway.OpenAIBackend,
        gateway.AnthropicBackend,
        gateway.LocalOpenAICompatibleBackend,
    ],
)
def test_new_backends_are_backend_subclasses_with_complete(cls):
    assert issubclass(cls, gateway.Backend)
    assert callable(cls.complete)


def test_local_backend_is_openai_subclass():
    assert issubclass(gateway.LocalOpenAICompatibleBackend, gateway.OpenAIBackend)


# --------------------------------------------------------------------------- OpenAIBackend


def test_openai_backend_missing_library_raises_clear_error():
    try:
        import openai  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError) as exc:
            gateway.OpenAIBackend()
        assert "openai" in str(exc.value)
    else:
        pytest.skip("openai is installed; the missing-library path is not exercised")


def test_openai_backend_injected_client_text_path():
    client = _FakeOpenAIClient()
    be = gateway.OpenAIBackend(model="gpt-4o", client=client)
    out = be.complete("hi", decoding={"temperature": 0.0})
    assert out == "ok:gpt-4o"
    model, messages, kwargs = client.chat.completions.received[0]
    assert model == "gpt-4o"
    assert messages == [{"role": "user", "content": "hi"}]
    assert kwargs["temperature"] == 0.0


def test_openai_backend_merges_default_decoding():
    client = _FakeOpenAIClient()
    be = gateway.OpenAIBackend(client=client, default_decoding={"temperature": 0.7, "top_p": 0.9})
    be.complete("hi", decoding={"temperature": 0.1})
    _, _, kwargs = client.chat.completions.received[0]
    assert kwargs == {"temperature": 0.1, "top_p": 0.9}  # per-call overrides default


def test_openai_backend_image_path_encodes_base64():
    client = _FakeOpenAIClient()
    be = gateway.OpenAIBackend(client=client)
    be.complete("describe", image=b"imgbytes")
    _, messages, _ = client.chat.completions.received[0]
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"imgbytes"


# --------------------------------------------------------------------------- AnthropicBackend


def test_anthropic_backend_missing_library_raises_clear_error():
    try:
        import anthropic  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError) as exc:
            gateway.AnthropicBackend()
        assert "anthropic" in str(exc.value)
    else:
        pytest.skip("anthropic is installed; the missing-library path is not exercised")


def test_anthropic_backend_injected_client_text_path():
    client = _FakeAnthropicClient()
    be = gateway.AnthropicBackend(model="claude-opus-4-8", client=client, max_tokens=512)
    out = be.complete("hi")
    assert out == "ok"
    model, max_tokens, messages, kwargs = client.messages.received[0]
    assert model == "claude-opus-4-8"
    assert max_tokens == 512
    assert messages == [{"role": "user", "content": "hi"}]
    assert kwargs == {}


def test_anthropic_backend_decoding_overrides_max_tokens():
    client = _FakeAnthropicClient()
    be = gateway.AnthropicBackend(client=client)
    be.complete("hi", decoding={"max_tokens": 99, "temperature": 0.2})
    _, max_tokens, _, kwargs = client.messages.received[0]
    assert max_tokens == 99  # decoding wins over the backend default
    assert kwargs == {"temperature": 0.2}  # max_tokens consumed, not forwarded as a kwarg


def test_anthropic_backend_image_path_encodes_base64():
    client = _FakeAnthropicClient()
    be = gateway.AnthropicBackend(client=client)
    be.complete("describe", image=b"imgbytes")
    _, _, messages, _ = client.messages.received[0]
    content = messages[0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "image/png"
    assert base64.b64decode(content[0]["source"]["data"]) == b"imgbytes"
    assert content[1] == {"type": "text", "text": "describe"}


# ------------------------------------------------------------ LocalOpenAICompatibleBackend


def test_local_backend_missing_library_raises_clear_error():
    try:
        import openai  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError) as exc:
            gateway.LocalOpenAICompatibleBackend(model="qwen2.5", base_url="http://localhost/v1")
        assert "openai" in str(exc.value)
    else:
        pytest.skip("openai is installed; the missing-library path is not exercised")


def test_local_backend_injected_client_uses_base_url_and_reuses_completion():
    client = _FakeOpenAIClient()
    be = gateway.LocalOpenAICompatibleBackend(
        model="qwen2.5",
        base_url="http://localhost:11434/v1",
        client=client,
    )
    assert isinstance(be, gateway.OpenAIBackend)
    assert isinstance(be, gateway.Backend)
    assert be.base_url == "http://localhost:11434/v1"
    out = be.complete("hi")
    assert out == "ok:qwen2.5"
    model, messages, _ = client.chat.completions.received[0]
    assert model == "qwen2.5"
    assert messages == [{"role": "user", "content": "hi"}]
