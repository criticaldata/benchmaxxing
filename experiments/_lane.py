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
import time
from pathlib import Path

from benchmaxxing import gateway
from benchmaxxing.extract import declared_mcq_choice

DEFAULT_MODEL = "gemini-2.5-flash-lite"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# An open-weights model served on the machine that runs the experiment has no vendor endpoint, no
# key and no request ceiling. BENCHMAXXING_LOCAL_BASE_URL names that server, and setting it is
# enough to point the OpenAI-compatible backend at it, skip the key lookup and switch pacing off.
# Gemini and DeepSeek ids keep their vendor routing whatever it is set to, so one variable cannot
# silently redirect a committed comparator arm to a different model behind the same id.
LOCAL_BASE_URL = os.environ.get("BENCHMAXXING_LOCAL_BASE_URL", "").strip()
# Reasoning models need headroom. A cap that lands mid-reasoning returns the truncated chain of
# thought in `content`, which the legacy parsers would then score as if it were an answer. Whatever
# a cap still truncates is recorded as undeclared by `declared()` and excluded rather than scored.
MAX_TOKENS = 8192
# The NVIDIA endpoint allows about 40 requests per minute and penalises concurrent bursts, which
# #416 measured and documented when it introduced this vendor; it returns HTTP 429 with no
# Retry-After header, so the retry wrapper burns its five attempts against a closed door. Pace the
# run instead of racing it: NIM_RPM is that documented ceiling, and BENCHMAXXING_MIN_CALL_INTERVAL
# overrides the interval directly when a vendor needs something different. Pacing is measured
# across threads, so it holds whatever max_workers a runner uses, and it is off for Gemini, which
# has no such restriction.
NIM_RPM = 40
_NIM_INTERVAL = 60.0 / NIM_RPM * 1.05  # a 5% margin, since the window is not published exactly
# 40 RPM is the documented ceiling, but a free-tier key sustains far less: the bucket is small and
# refills slowly, so a long run settles nearer 3 calls a minute. Measured on this account, one call
# every 20s completes 9 attempts in 10, and a single call succeeds again after 60s of idle.
NIM_SUSTAINED_INTERVAL = 20.0
# A 429 outlives RetryBackend's five quick attempts, which is what killed whole arms mid-run: the
# backoff schedule expires while the bucket is still empty. Wait for a refill instead of failing.
RATE_LIMIT_SLEEP = 90.0
RATE_LIMIT_TRIES = 12
TRANSIENT_SLEEP = 15  # a dropped connection needs a pause, not the full rate-limit cooldown
MIN_CALL_INTERVAL = float(os.environ.get("BENCHMAXXING_MIN_CALL_INTERVAL", "0") or 0)


def is_local(model: str) -> bool:
    """True when this model is served locally rather than by a vendor endpoint."""
    m = model.lower()
    return bool(LOCAL_BASE_URL) and "gemini" not in m and "deepseek" not in m


def interval_for(model: str) -> float:
    """Seconds to leave between outgoing calls for a model's endpoint."""
    if MIN_CALL_INTERVAL > 0:
        return MIN_CALL_INTERVAL
    if is_local(model):
        return 0.0
    if "gemini" in model.lower():
        return 0.0
    return NIM_SUSTAINED_INTERVAL


def _is_rate_limited(exc: Exception) -> bool:
    """True for a 429 from any vendor, without importing the vendor SDKs."""
    if type(exc).__name__ in ("RateLimitError", "ResourceExhausted"):
        return True
    if getattr(exc, "status_code", None) == 429 or getattr(exc, "code", None) == 429:
        return True
    return "429" in str(exc) or "too many requests" in str(exc).lower()


def _is_transient(exc: Exception) -> bool:
    """True for a dropped or timed-out connection, which is worth retrying like a 429.

    A second-vendor endpoint under load holds the socket open and then drops it rather than
    answering, so a long arm sees ``APIConnectionError`` or ``ReadTimeout`` even when paced well
    inside the rate limit. Without this the retry wrapper gives up and the whole arm dies, losing
    the run but not the calls already cached; observed on three of thirteen ablation arms.
    """
    name = type(exc).__name__.lower()
    if "timeout" in name or "connect" in name:
        return True
    # The NVIDIA endpoint under load also answers 503 "Service temporarily overloaded", 502/504, and an
    # intermittent 404 for a model that /v1/models still lists and that answers 200 a minute later
    # (observed 7 Sept 2026 on nemotron-3-super, 466 calls into an arm). Retrying is bounded by
    # RATE_LIMIT_TRIES, so a model that has genuinely been withdrawn still fails, just not on the first 404.
    text = str(exc).lower()
    return (
        "internalserver" in name or "serviceunavailable" in name or "notfound" in name
        or any(code in text for code in ("error code: 502", "error code: 503", "error code: 504", "error code: 404"))
        or "temporarily overloaded" in text
    )


_lock = threading.Lock()
_pace_lock = threading.Lock()
_last_call = [0.0]


def _pace(model: str):
    """Block until this model's minimum interval has passed since the previous outgoing call."""
    gap = interval_for(model)
    if gap <= 0:
        return
    with _pace_lock:
        wait = gap - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def is_gemini(model: str) -> bool:
    """The one lineage whose key, backend and pacing differ from every second-vendor model."""
    return "gemini" in model.lower()


GEMINI_IDS = ("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro")


def rebind_models(namespace: dict, model: str) -> int:
    """Rebind every Gemini id in a runner's module constants to `model`, in place.

    The Gemini-only runners name their seats with module constants such as HOLDOUT, MODELS, TIERS
    or MEMBERS, as strings, lists of strings, lists of (name, id) pairs or dicts of ids. When a
    second model is requested, every one of those seats becomes that model, so a committee runner
    compares the requested model's committee against Gemini's rather than mixing lineages. Returns
    the number of ids rebound; zero means the runner had nothing to rebind, which is a bug.
    """
    def swap(v):
        if isinstance(v, str):
            return (model, 1) if v in GEMINI_IDS else (v, 0)
        if isinstance(v, tuple):
            items = [swap(x) for x in v]
            return tuple(x for x, _ in items), sum(n for _, n in items)
        if isinstance(v, list):
            items = [swap(x) for x in v]
            out = [x for x, _ in items]
            if all(isinstance(x, str) for x in out):
                # A list of tiers collapses to one entry per distinct model, so a runner that loops
                # over tiers does not run the same model twice.
                out = list(dict.fromkeys(out))
            return out, sum(n for _, n in items)
        if isinstance(v, dict):
            items = {k: swap(x) for k, x in v.items()}
            return {k: x for k, (x, _) in items.items()}, sum(n for _, n in items.values())
        return v, 0

    total = 0
    for name, value in list(namespace.items()):
        if name.isupper() and not name.startswith("_") and isinstance(value, (str, list, tuple, dict)):
            new, n = swap(value)
            if n:
                namespace[name] = new
                total += n
    return total


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
    if is_local(model):
        # A cache miss on a local endpoint must not exit for a key that no server checks.
        return "not-needed"
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
    if is_local(model):
        base_url = LOCAL_BASE_URL
    elif "deepseek" in model.lower():
        base_url = DEEPSEEK_BASE_URL
    else:
        base_url = NIM_BASE_URL
    # A locally served model has no rate limit but can legitimately take minutes on a long
    # completion, so it gets a generous timeout; a hosted endpoint keeps the 60 s fast-fail, where a
    # stall is the failure mode worth converting into a retryable error.
    return gateway.LocalOpenAICompatibleBackend(
        model=model, base_url=base_url, api_key=key, client=client,
        default_decoding={"max_tokens": MAX_TOKENS},
        timeout=600.0 if is_local(model) else 60.0,
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


def paced_complete(model: str, key, prompt: str, decoding=None, client=None):
    """One completion, paced to the model's rate and retried through a 429 or a transient fault.

    This is the single call every runner cache goes through. The inner ``RetryBackend`` covers the
    quick retries; this loop covers the slow ones: an empty rate bucket (wait RATE_LIMIT_SLEEP) or a
    dropped connection, 5xx or intermittent 404 (wait TRANSIENT_SLEEP). Anything else, and the last
    attempt of anything, is re-raised so a real fault still fails the run.
    """
    backend = gateway.RetryBackend(backend_for(model, key, client=client), tries=5, backoff=3.0)
    for attempt in range(RATE_LIMIT_TRIES):
        _pace(model)
        try:
            return backend.complete(prompt, decoding=decoding or {"temperature": 0})
        except Exception as exc:  # noqa: BLE001  (re-raised below unless it is a 429 or transient)
            root = exc
            while root.__cause__ is not None:
                root = root.__cause__
            limited = _is_rate_limited(root)
            if attempt == RATE_LIMIT_TRIES - 1 or not (limited or _is_transient(root)):
                raise
            time.sleep(RATE_LIMIT_SLEEP if limited else TRANSIENT_SLEEP)
    return None


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
        resp = paced_complete(model, self.key, prompt, decoding={"temperature": 0})
        if resp is None:
            raise SystemExit(f"{model} returned an empty completion (content=None). Reasoning-only "
                             "models are not usable here: the parsers read `content`.")
        with _lock:
            self.store[k] = resp
            self.calls += 1
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "model": model, "resp": resp}) + "\n")
        return resp
