"""Unified model backend (the gateway).

Every model call in benchmaxxing goes through one small interface, ``Backend.complete``,
so the rest of the pipeline never talks to a vendor SDK directly. Gemini is the default
backend for now; the design is deliberately extensible to other APIs.

Extendability (the ExtendableBackend contract)
----------------------------------------------
Adding another provider is a single new ``Backend`` subclass and nothing else changes:

    class MyProviderBackend(Backend):
        def complete(self, prompt, image=None, decoding=None) -> str:
            ...

Because callers depend only on the ``complete(prompt, image, decoding) -> str`` shape,
the wrappers in this module (``with_retry``, ``cached``) and every downstream consumer keep
working with the new backend unchanged. Reuse the originals: each real provider is a thin
adapter over that provider's own SDK, guarded so the pure-Python core installs and tests
anywhere.
"""

from __future__ import annotations

import abc
import hashlib
import time
from typing import Callable

__all__ = [
    "Backend",
    "MockBackend",
    "GeminiBackend",
    "RetryBackend",
    "CachedBackend",
    "RetryError",
    "with_retry",
    "cached",
]


class Backend(abc.ABC):
    """The one interface the whole pipeline agrees on for talking to a model.

    A backend maps a text ``prompt`` (optionally with an ``image`` and per-call
    ``decoding`` overrides) to a single text completion. Keep implementations thin: a
    backend is an adapter over a provider SDK or a deterministic stand-in, not a place for
    business logic.

    This is also the extension point. A new provider is a new ``Backend`` subclass that
    implements ``complete``; the wrappers and downstream code in benchmaxxing depend only on
    this signature, so nothing else has to change (the ExtendableBackend contract).
    """

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        image: object | None = None,
        decoding: dict | None = None,
    ) -> str:
        """Return the model completion for ``prompt``.

        Parameters
        ----------
        prompt:
            The text prompt.
        image:
            Optional image payload for multimodal backends (PIL image, numpy array, or
            raw bytes). Text-only backends may ignore it.
        decoding:
            Optional per-call decoding overrides (temperature, max tokens, ...). The exact
            keys are provider specific.
        """
        raise NotImplementedError


class MockBackend(Backend):
    """Deterministic, offline backend for tests and dry runs.

    With no ``rule`` it echoes the prompt verbatim. With a ``rule`` it returns
    ``rule(prompt, image, decoding)``, so tests can script any behaviour (including a flaky
    rule that raises on the first calls). ``n_calls`` counts how many times ``complete`` ran,
    which is handy for proving a cache served a repeat call without hitting the backend.
    """

    def __init__(self, rule: Callable[[str, object | None, dict | None], str] | None = None):
        self.rule = rule
        self.n_calls = 0

    def complete(
        self,
        prompt: str,
        image: object | None = None,
        decoding: dict | None = None,
    ) -> str:
        self.n_calls += 1
        if self.rule is None:
            return prompt
        return self.rule(prompt, image, decoding)


class GeminiBackend(Backend):
    """Thin adapter over Google's google-genai SDK. The default backend for now.

    The vendor SDK is imported lazily (guarded) so the pure-Python core installs and tests
    without it. If the library is missing, construction raises a clear ``ImportError`` telling
    the user to install the ``models`` extra. A ``client`` can be injected for offline tests,
    which bypasses the import entirely.

    Extending to another API is just another ``Backend`` subclass (see the module docstring);
    this class stays a thin, provider-specific adapter.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        client: object | None = None,
        default_decoding: dict | None = None,
    ):
        self.model = model
        self.default_decoding = dict(default_decoding or {})
        if client is not None:
            # Injected client (used by tests): no SDK import required.
            self._client = client
            return
        try:
            from google import genai
        except ImportError as exc:  # guarded import: keep the core dependency-free
            raise ImportError(
                "GeminiBackend requires the 'google-genai' package, which is not installed. "
                "Install the models extra: pip install 'benchmaxxing[models]' "
                "(or: pip install google-genai)."
            ) from exc
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def complete(
        self,
        prompt: str,
        image: object | None = None,
        decoding: dict | None = None,
    ) -> str:
        # Multimodal: google-genai accepts a PIL image (or a types.Part) alongside the text
        # directly in the contents list, so a single code path covers text and image calls.
        contents = [prompt] if image is None else [prompt, image]
        config = {**self.default_decoding, **(decoding or {})}
        kwargs = {"config": config} if config else {}
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            **kwargs,
        )
        return response.text


class RetryError(RuntimeError):
    """Raised when a backend keeps failing on transient errors until retries run out.

    The original exception is chained (``raise ... from exc``) so the underlying cause and
    the retry context are both visible.
    """


class RetryBackend(Backend):
    """Wrap a backend so transient failures are retried.

    Exceptions whose type is in ``transient`` are retried up to ``tries`` attempts, sleeping
    ``backoff * attempt`` seconds between them. When the attempts run out, the last failure
    surfaces as a ``RetryError`` with context (backend name and attempt count) chained to the
    original. Terminal errors (types not in ``transient``) are not retried and propagate
    immediately with their own traceback.
    """

    def __init__(
        self,
        backend: Backend,
        tries: int = 3,
        backoff: float = 0.0,
        transient: tuple[type[BaseException], ...] = (Exception,),
        sleep: Callable[[float], None] | None = None,
    ):
        if tries < 1:
            raise ValueError("tries must be >= 1")
        self.backend = backend
        self.tries = tries
        self.backoff = backoff
        self.transient = tuple(transient)
        self._sleep = sleep if sleep is not None else time.sleep

    def complete(
        self,
        prompt: str,
        image: object | None = None,
        decoding: dict | None = None,
    ) -> str:
        for attempt in range(1, self.tries + 1):
            try:
                return self.backend.complete(prompt, image=image, decoding=decoding)
            except self.transient as exc:
                if attempt >= self.tries:
                    raise RetryError(
                        f"{type(self.backend).__name__}.complete failed after "
                        f"{self.tries} attempt(s)"
                    ) from exc
                if self.backoff:
                    self._sleep(self.backoff * attempt)
        # Unreachable: the loop either returns or raises. Present for type checkers.
        raise RetryError("retry loop exited without a result")  # pragma: no cover


def _image_key(image: object | None) -> object:
    """Stable, hashable cache key for an image payload (or ``None``)."""
    if image is None:
        return None
    if isinstance(image, (bytes, bytearray, memoryview)):
        return hashlib.sha1(bytes(image)).hexdigest()
    tobytes = getattr(image, "tobytes", None)
    if callable(tobytes):
        try:
            raw = tobytes()
        except Exception:
            # Not all objects with a tobytes attribute produce bytes; fall back to repr.
            return repr(image)
        meta = repr(
            (
                getattr(image, "size", None),
                getattr(image, "mode", None),
                getattr(image, "shape", None),
                str(getattr(image, "dtype", "")),
            )
        )
        return hashlib.sha1(raw + meta.encode("utf-8")).hexdigest()
    return repr(image)


def _decoding_key(decoding: dict | None) -> object:
    """Stable, hashable cache key for a decoding dict (or ``None``)."""
    if not decoding:
        return None
    try:
        return repr(tuple(sorted(decoding.items())))
    except TypeError:
        return repr(decoding)


class CachedBackend(Backend):
    """Wrap a backend with an in-memory cache.

    Identical calls (same prompt, image, and decoding) are served from a dict instead of
    hitting the wrapped backend again. The key is ``(prompt, image-hash, decoding-repr)``.
    The cache is a plain dict exposed as ``.cache`` so tests can inspect or reset it, and a
    shared dict can be injected to persist across wrappers.
    """

    def __init__(self, backend: Backend, cache: dict | None = None):
        self.backend = backend
        self.cache = {} if cache is None else cache

    def complete(
        self,
        prompt: str,
        image: object | None = None,
        decoding: dict | None = None,
    ) -> str:
        key = (prompt, _image_key(image), _decoding_key(decoding))
        if key in self.cache:
            return self.cache[key]
        result = self.backend.complete(prompt, image=image, decoding=decoding)
        self.cache[key] = result
        return result


def with_retry(
    backend: Backend,
    tries: int = 3,
    backoff: float = 0.0,
    transient: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] | None = None,
) -> RetryBackend:
    """Return ``backend`` wrapped so transient exceptions retry (see ``RetryBackend``)."""
    return RetryBackend(
        backend,
        tries=tries,
        backoff=backoff,
        transient=transient,
        sleep=sleep,
    )


def cached(backend: Backend, cache: dict | None = None) -> CachedBackend:
    """Return ``backend`` wrapped with an in-memory cache (see ``CachedBackend``)."""
    return CachedBackend(backend, cache=cache)


def _image_to_base64(image: object) -> str:
    """Encode an image payload as a base64 ASCII string.

    Raw bytes are encoded verbatim (assumed to already be an encoded image). A PIL image
    or numpy array is rendered to PNG first; that path needs pillow, which is guarded so the
    core stays importable without it.
    """
    import base64

    if isinstance(image, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(image)).decode("ascii")

    import io

    save = getattr(image, "save", None)
    if callable(save):  # PIL image
        buffer = io.BytesIO()
        save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    try:
        from PIL import Image
    except ImportError as exc:  # guarded: converting arrays needs pillow
        raise ImportError(
            "Encoding a non-bytes image requires the 'pillow' package, which is not installed. "
            "Pass raw image bytes instead, or install pillow (pip install pillow)."
        ) from exc
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenAIBackend(Backend):
    """Thin adapter over the OpenAI Python SDK (chat completions).

    The vendor SDK is imported lazily (guarded) so the pure-Python core installs and tests
    without it. If ``openai`` is missing, construction raises a clear ``ImportError`` pointing
    at the ``models`` extra. A ``client`` can be injected for offline tests, bypassing the
    import entirely.

    Multimodal: when an ``image`` is supplied it is base64-encoded into an ``image_url`` data
    URI alongside the text, following the OpenAI chat-completions content-list convention.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        client: object | None = None,
        default_decoding: dict | None = None,
    ):
        self.model = model
        self.default_decoding = dict(default_decoding or {})
        if client is not None:
            # Injected client (used by tests): no SDK import required.
            self._client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # guarded import: keep the core dependency-free
            raise ImportError(
                "OpenAIBackend requires the 'openai' package, which is not installed. "
                "Install the models extra: pip install 'benchmaxxing[models]' "
                "(or: pip install openai)."
            ) from exc
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def complete(
        self,
        prompt: str,
        image: object | None = None,
        decoding: dict | None = None,
    ) -> str:
        if image is None:
            content: object = prompt
        else:
            data_uri = f"data:image/png;base64,{_image_to_base64(image)}"
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        config = {**self.default_decoding, **(decoding or {})}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            **config,
        )
        return response.choices[0].message.content


class AnthropicBackend(Backend):
    """Thin adapter over the Anthropic Python SDK (messages API).

    The vendor SDK is imported lazily (guarded) so the pure-Python core installs and tests
    without it. If ``anthropic`` is missing, construction raises a clear ``ImportError``
    pointing at the ``models`` extra. A ``client`` can be injected for offline tests.

    ``max_tokens`` is required by the messages API, so a default is carried on the backend and
    can be overridden per call via ``decoding``. Multimodal: an ``image`` is base64-encoded
    into an Anthropic ``image`` content block before the text block.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
        client: object | None = None,
        max_tokens: int = 1024,
        default_decoding: dict | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.default_decoding = dict(default_decoding or {})
        if client is not None:
            # Injected client (used by tests): no SDK import required.
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:  # guarded import: keep the core dependency-free
            raise ImportError(
                "AnthropicBackend requires the 'anthropic' package, which is not installed. "
                "Install the models extra: pip install 'benchmaxxing[models]' "
                "(or: pip install anthropic)."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(
        self,
        prompt: str,
        image: object | None = None,
        decoding: dict | None = None,
    ) -> str:
        if image is None:
            content: object = prompt
        else:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _image_to_base64(image),
                    },
                },
                {"type": "text", "text": prompt},
            ]
        config = {**self.default_decoding, **(decoding or {})}
        max_tokens = config.pop("max_tokens", self.max_tokens)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
            **config,
        )
        return response.content[0].text


class LocalOpenAICompatibleBackend(OpenAIBackend):
    """OpenAI-compatible backend pointed at a local server (Ollama / vLLM).

    Open-weights models in the cross-lineage arm are served behind an OpenAI-compatible HTTP
    API; this backend reuses the ``openai`` client (and ``OpenAIBackend.complete``) against a
    custom ``base_url``. Local servers usually ignore the key, so a placeholder ``api_key`` is
    sent by default. A ``client`` can be injected for offline tests.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "not-needed",
        client: object | None = None,
        default_decoding: dict | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.default_decoding = dict(default_decoding or {})
        if client is not None:
            # Injected client (used by tests): no SDK import required.
            self._client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # guarded import: keep the core dependency-free
            raise ImportError(
                "LocalOpenAICompatibleBackend requires the 'openai' package, which is not "
                "installed. Install the models extra: pip install 'benchmaxxing[models]' "
                "(or: pip install openai)."
            ) from exc
        self._client = OpenAI(base_url=base_url, api_key=api_key)
