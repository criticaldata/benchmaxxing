"""Shared model dispatch for the text lanes.

Every text runner used to hardcode ``MODEL = "gemini-2.5-flash-lite"`` and build
``GeminiBackend`` directly, so a contributor assigned a second-vendor model had nothing to run.
This module is the one place that maps a model id to its key, its backend and its output cap, so a
runner only has to take ``--model`` and pass it through.

The cache key is ``sha256(model \\x00 prompt)``, unchanged from the per-runner caches it replaces, so
every committed Gemini cache still replays byte for byte with no new API calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

from benchmaxxing import gateway
from benchmaxxing.extract import declared_mcq_choice

DEFAULT_MODEL = "gemini-2.5-flash-lite"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# Reasoning models need headroom. A cap that lands mid-reasoning returns the truncated chain of
# thought in `content`, which the legacy parsers would then score as if it were an answer. Whatever
# a cap still truncates is recorded as undeclared by `declared()` and excluded rather than scored.
MAX_TOKENS = 8192

_lock = threading.Lock()


def key_name(model: str) -> str:
    """Name the environment variable a model's key comes from."""
    m = model.lower()
    if "gemini" in m:
        return "GEMINI_API_KEY"
    if "deepseek" in m:
        return "DEEPSEEK_API_KEY"
    return "NVIDIA_API_KEY"


def key_for(model: str):
    """Resolve the API key strictly from the model id, as the imaging lane does."""
    m = model.lower()
    if "gemini" in m:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if "deepseek" in m:
        return os.environ.get("DEEPSEEK_API_KEY")
    return os.environ.get("NVIDIA_API_KEY")


def backend_for(model: str, key, client=None):
    """Gemini through the Google SDK, everything else through the OpenAI-compatible path.

    ``client`` is the gateway's own injection hook, so dispatch is testable without constructing
    an SDK client.
    """
    if "gemini" in model.lower():
        return gateway.GeminiBackend(model=model, api_key=key)
    base_url = DEEPSEEK_BASE_URL if "deepseek" in model.lower() else NIM_BASE_URL
    return gateway.LocalOpenAICompatibleBackend(
        model=model, base_url=base_url, api_key=key, client=client,
        default_decoding={"max_tokens": MAX_TOKENS},
    )


def letters(n: int) -> list[str]:
    return [chr(65 + i) for i in range(n)]


_TERMINAL_LETTER = re.compile(r"^\s*\**\(?([A-E])\)?\**[.:]?\s*$")


def declared(text: str, options) -> str | None:
    """The option letter the model actually committed to, or None if it committed to nothing.

    Prefers the shared declaration detector added in #418, and falls back to a bare option letter
    on the final non-empty line, which is the form the text prompts ask for. A completion that ends
    mid-reasoning, or in prose that merely mentions options, is undeclared and must not be scored:
    the legacy parser will still find *some* letter in it.
    """
    if not text:
        return None
    options = list(options)
    letter_of = letters(len(options))
    # declared_mcq_choice returns the option TEXT, so map it back to its letter.
    choice, ok = declared_mcq_choice(text, options)
    if ok and choice in options:
        return letter_of[options.index(choice)]
    valid = set(letter_of)
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        m = _TERMINAL_LETTER.match(lines[-1])
        if m and m.group(1) in valid:
            return m.group(1)
    return None


def add_model_arg(ap, default: str = DEFAULT_MODEL):
    ap.add_argument("--model", default=default,
                    help="Model id. Gemini ids go through the Google SDK; anything else through "
                         "the OpenAI-compatible endpoint (NVIDIA NIM by default).")


def scoped(model: str, out: str, default_cache: str, cache: str | None = None):
    """Model-scoped output directory and cache path.

    The default model keeps the committed paths untouched so its results and cache stay exactly
    where the paper's numbers were computed; every other model gets its own subdirectory and its
    own cache file, which also keeps a thirteen-way fan-out off one shared, conflict-prone file.
    """
    slug = model.replace("/", "_")
    out_dir = Path(out) if model == DEFAULT_MODEL else Path(out) / slug
    if cache:
        cache_path = Path(cache)
    elif model == DEFAULT_MODEL:
        cache_path = Path(default_cache)
    else:
        p = Path(default_cache)
        cache_path = p.with_name(f"{slug}_{p.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, cache_path


class Cache:
    """Prompt cache keyed on (model, prompt); a fully cached run needs no API key."""

    def __init__(self, path, key, model):
        self.path, self.key, self.model, self.store, self.calls = Path(path), key, model, {}, 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.store[r["k"]] = r["resp"]

    def complete(self, prompt, model=None):
        model = model or self.model
        k = hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()
        with _lock:
            if k in self.store:
                return self.store[k]
        if not self.key:
            raise SystemExit(f"Cache miss and no {key_name(model)} set for {model} "
                             "(a fully cached run needs no key).")
        resp = gateway.RetryBackend(backend_for(model, self.key), tries=5, backoff=3.0).complete(
            prompt, decoding={"temperature": 0})
        if resp is None:
            raise SystemExit(f"{model} returned an empty completion (content=None). Reasoning-only "
                             "models are not usable here: the parsers read `content`.")
        with _lock:
            self.store[k] = resp
            self.calls += 1
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "model": model, "resp": resp}) + "\n")
        return resp
