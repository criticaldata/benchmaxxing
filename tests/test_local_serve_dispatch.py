"""Serving a text-lane model locally (``BENCHMAXXING_LOCAL_BASE_URL``).

An open-weights arm can be served on the machine that runs the experiment instead of behind a
vendor endpoint. These tests pin the three things that change when it is: the OpenAI-compatible
backend points at the local server, the key lookup stops mattering because no local server checks
one, and the pacing that exists only to respect a vendor request ceiling goes to zero. They also
pin what must not change, which is the routing of the Gemini and DeepSeek ids whose committed
caches the cross-lineage comparison depends on.
"""
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import _lane  # noqa: E402

LOCAL = "http://127.0.0.1:8010/v1"
OPEN_WEIGHTS = "openai/gpt-oss-120b"
GEMINI = "gemini-2.5-flash-lite"
NIM = "nvidia/nemotron-3-super-120b-a12b"
DEEPSEEK = "deepseek-ai/deepseek-v4-flash-0731"


class _Stub:
    """Stands in for the OpenAI client, so dispatch is testable without constructing one."""


@pytest.fixture
def served_locally(monkeypatch):
    monkeypatch.setattr(_lane, "LOCAL_BASE_URL", LOCAL)


def test_a_local_server_captures_open_weights_ids_and_no_committed_comparator(served_locally):
    """A local server captures this lane's open-weights id and never a vendor-hosted comparator.

    An earlier version of this test asserted the opposite for the nemotron id, on the reasoning
    that an open-weights comparator could also be served on the machine and routing it elsewhere
    would be surprising. That is superseded: the id reaches a vendor endpoint in this repo and its
    committed cache is what a cross-lineage comparison rests on, so a shell serving these weights
    locally must not answer a cache miss for it from the wrong model. HOSTED_PREFIXES is kept
    identical to the other text lane's copy of the module, so consolidating them is a no-op.
    """
    assert _lane.HOSTED_PREFIXES == ("nvidia/",)
    assert _lane.is_local(OPEN_WEIGHTS)
    assert not _lane.is_local(NIM)
    assert not _lane.is_local(GEMINI)
    assert not _lane.is_local(DEEPSEEK)
    assert _lane.interval_for(NIM) > 0


def test_unset_variable_leaves_every_model_on_its_vendor_endpoint(monkeypatch):
    monkeypatch.setattr(_lane, "LOCAL_BASE_URL", "")
    assert not _lane.is_local(OPEN_WEIGHTS)
    backend = _lane.backend_for(OPEN_WEIGHTS, "nvapi-test", client=_Stub())
    assert backend.base_url == _lane.NIM_BASE_URL


def test_a_local_model_routes_to_the_local_server_with_the_same_cap(served_locally):
    backend = _lane.backend_for(OPEN_WEIGHTS, _lane.key_for(OPEN_WEIGHTS), client=_Stub())
    assert isinstance(backend, _lane.gateway.LocalOpenAICompatibleBackend)
    assert backend.base_url == LOCAL
    # Reasoning headroom is a property of the model, not of who serves it.
    assert backend.default_decoding["max_tokens"] == _lane.MAX_TOKENS


def test_the_comparator_arms_are_unaffected_by_a_local_server(served_locally, monkeypatch):
    seen = {}
    monkeypatch.setattr(_lane.gateway, "GeminiBackend",
                        lambda model, api_key: seen.update(model=model) or "gem")
    assert _lane.backend_for(GEMINI, "g") == "gem"
    assert seen == {"model": GEMINI}
    assert _lane.backend_for(DEEPSEEK, "sk", client=_Stub()).base_url == _lane.DEEPSEEK_BASE_URL


def test_pacing_is_off_locally_and_still_on_for_the_vendor(served_locally):
    assert _lane.interval_for(OPEN_WEIGHTS) == 0.0
    assert _lane.interval_for(GEMINI) == 0.0


def test_the_vendor_interval_survives_when_no_local_server_is_set(monkeypatch):
    monkeypatch.setattr(_lane, "LOCAL_BASE_URL", "")
    assert _lane.interval_for(NIM) == _lane.NIM_SUSTAINED_INTERVAL


def test_an_explicit_interval_still_overrides_a_local_server(served_locally, monkeypatch):
    monkeypatch.setattr(_lane, "MIN_CALL_INTERVAL", 5.0)
    assert _lane.interval_for(OPEN_WEIGHTS) == 5.0


def test_a_miss_on_a_local_endpoint_does_not_exit_for_a_vendor_key(tmp_path, served_locally,
                                                                   monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert _lane.key_for(OPEN_WEIGHTS) == "not-needed"

    class _Backend:
        def complete(self, prompt, decoding=None):
            return "B"

    monkeypatch.setattr(_lane.gateway, "RetryBackend",
                        lambda backend, tries, backoff: _Backend())
    monkeypatch.setattr(_lane, "backend_for", lambda model, key, client=None: _Backend())
    cache = _lane.Cache(tmp_path / "c.jsonl", _lane.key_for(OPEN_WEIGHTS), OPEN_WEIGHTS)
    assert cache.complete("uncached") == "B"
    assert cache.calls == 1


def test_a_miss_without_a_local_server_still_names_the_vendor_variable(tmp_path, monkeypatch):
    monkeypatch.setattr(_lane, "LOCAL_BASE_URL", "")
    cache = _lane.Cache(tmp_path / "c.jsonl", None, OPEN_WEIGHTS)
    with pytest.raises(SystemExit) as exc:
        cache.complete("uncached")
    assert "NVIDIA_API_KEY" in str(exc.value)


# The blind-metric lane carries its own copy of the key and backend dispatch, so the same variable
# has to reach that copy too, on the same terms.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "blind_metric"))
import blind_metric  # noqa: E402


def test_the_blind_metric_lane_honours_the_same_variable(monkeypatch):
    monkeypatch.setattr(blind_metric, "LOCAL_BASE_URL", LOCAL)
    assert blind_metric._is_local(OPEN_WEIGHTS)
    assert not blind_metric._is_local(GEMINI)
    # This copy must not disagree with _lane on a vendor-hosted id (HOSTED_PREFIXES).
    assert not blind_metric._is_local(NIM)
    assert blind_metric._key(OPEN_WEIGHTS) == "not-needed"
    backend = blind_metric._backend(OPEN_WEIGHTS, blind_metric._key(OPEN_WEIGHTS), client=_Stub())
    assert backend.base_url == LOCAL
    # The reasoning cap is unchanged by who serves the model.
    assert backend.default_decoding["max_tokens"] == blind_metric.NIM_MAX_TOKENS


def test_the_blind_metric_lane_keeps_vendor_routing_without_the_variable(monkeypatch):
    monkeypatch.setattr(blind_metric, "LOCAL_BASE_URL", "")
    assert not blind_metric._is_local(OPEN_WEIGHTS)
    assert blind_metric._backend(OPEN_WEIGHTS, "nvapi-test",
                                 client=_Stub()).base_url == blind_metric.NIM_BASE_URL
    assert blind_metric._key_name(OPEN_WEIGHTS) == "NVIDIA_API_KEY"


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "imaging_chexpert"))
import imaging_blind_metric  # noqa: E402


def test_the_imaging_lane_copy_agrees_with_the_lane_on_hosted_prefixes(monkeypatch):
    """The third dispatch copy must not route a vendor-hosted id to a local server (#430)."""
    monkeypatch.setattr(imaging_blind_metric, "LOCAL_BASE_URL", LOCAL)
    assert imaging_blind_metric.HOSTED_PREFIXES == _lane.HOSTED_PREFIXES
    assert imaging_blind_metric._is_local(OPEN_WEIGHTS)
    assert not imaging_blind_metric._is_local(NIM)
    assert not imaging_blind_metric._is_local(GEMINI)
    monkeypatch.setattr(imaging_blind_metric, "LOCAL_BASE_URL", "")
    assert not imaging_blind_metric._is_local(OPEN_WEIGHTS)


# ---------------------------------------------------------------------------
# The ``local/`` prefix: serving a vendor-hosted id's open weights locally.
#
# An id under HOSTED_PREFIXES is never captured by a local server, so its committed caches cannot be
# answered by the wrong model. To serve the same weights locally a run names them
# ``local/<vendor id>``: the request carries the vendor id, while the cache key, output directory
# and result rows carry the prefixed id, so a hosted run and a local run never share a cache entry.
# The third dispatch copy (CheXpert imaging blind metric) is held to the same contract for prefixed
# ids; since #431 it also applies the HOSTED_PREFIXES exclusion, so all three copies agree.
# ---------------------------------------------------------------------------

LOCAL_ID = "local/nvidia/cosmos-reason2-8b"
SERVED_ID = "nvidia/cosmos-reason2-8b"


def _load_chexpert_copy():
    """Import the CheXpert copy under its own name.

    ``experiments/imaging/imaging_blind_metric.py`` has the same module name, so it is loaded from
    its path. The module calls ``dotenv.load_dotenv()`` at import time and no extra declares
    python-dotenv, so when the package is absent a no-op stands in for the duration of the import.
    """
    pytest.importorskip("PIL")
    experiments = Path(__file__).resolve().parents[1] / "experiments"
    path = experiments / "imaging_chexpert" / "imaging_blind_metric.py"
    spec = importlib.util.spec_from_file_location("chexpert_imaging_blind_metric", path)
    module = importlib.util.module_from_spec(spec)
    stand_in = "dotenv" not in sys.modules and importlib.util.find_spec("dotenv") is None
    if stand_in:
        sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)
    try:
        spec.loader.exec_module(module)
    finally:
        if stand_in:
            del sys.modules["dotenv"]
    return module


chexpert = _load_chexpert_copy()

_COPIES = {
    "_lane": (_lane, _lane.is_local),
    "blind_metric": (blind_metric, blind_metric._is_local),
    "chexpert": (chexpert, chexpert._is_local),
}

# (id, server configured, expected per copy).
_ROUTING_TABLE = [
    (LOCAL_ID, True, {"_lane": True, "blind_metric": True, "chexpert": True}),
    (LOCAL_ID, False, {"_lane": False, "blind_metric": False, "chexpert": False}),
    # The prefix wins over the vendor-name rules: these open weights must not go to DeepSeek's API.
    ("local/" + DEEPSEEK, True, {"_lane": True, "blind_metric": True, "chexpert": True}),
    (NIM, True, {"_lane": False, "blind_metric": False, "chexpert": False}),
    (GEMINI, True, {"_lane": False, "blind_metric": False, "chexpert": False}),
    (OPEN_WEIGHTS, True, {"_lane": True, "blind_metric": True, "chexpert": True}),
    (OPEN_WEIGHTS, False, {"_lane": False, "blind_metric": False, "chexpert": False}),
]


@pytest.mark.parametrize("model,server,expected", _ROUTING_TABLE)
def test_the_three_dispatch_copies_route_the_same_ids_the_same_way(monkeypatch, model, server,
                                                                    expected):
    got = {}
    for name, (module, is_local) in _COPIES.items():
        monkeypatch.setattr(module, "LOCAL_BASE_URL", LOCAL if server else "")
        got[name] = is_local(model)
    assert got == expected


@pytest.fixture
def recorded(monkeypatch):
    """Replace LocalOpenAICompatibleBackend; the returned list holds (model, base_url) per request.

    All three copies reach the same ``benchmaxxing.gateway`` module, so one patch covers them.
    """
    requests = []

    class _RecordingBackend:
        def __init__(self, model, base_url, api_key=None, client=None, **kwargs):
            self.model, self.base_url = model, base_url

        def complete(self, prompt, image=None, decoding=None):
            requests.append((self.model, self.base_url))
            return "yes" if image is not None else "B"

    monkeypatch.setattr(_lane.gateway, "LocalOpenAICompatibleBackend", _RecordingBackend)
    return requests


def _gray():
    from PIL import Image
    return Image.new("L", (8, 8), 128)


def test_a_prefixed_id_reaches_the_local_server_without_its_prefix(tmp_path, recorded, monkeypatch):
    for module in (_lane, blind_metric, chexpert):
        monkeypatch.setattr(module, "LOCAL_BASE_URL", LOCAL)

    lane_cache = _lane.Cache(tmp_path / "lane.jsonl", _lane.key_for(LOCAL_ID), LOCAL_ID)
    assert lane_cache.complete("p") == "B"
    bm_cache = blind_metric._Cache(tmp_path / "bm.jsonl", blind_metric._key(LOCAL_ID), LOCAL_ID)
    assert bm_cache.complete(LOCAL_ID, "p") == "B"
    cx_cache = chexpert._Cache(tmp_path / "cx.jsonl", chexpert._key(LOCAL_ID), LOCAL_ID)
    assert cx_cache.ask("p", _gray()) == "yes"

    assert recorded == [(SERVED_ID, LOCAL)] * 3


def test_a_prefixed_id_keeps_its_prefix_in_cache_keys_and_output_paths(tmp_path, recorded,
                                                                        monkeypatch):
    for module in (_lane, blind_metric, chexpert):
        monkeypatch.setattr(module, "LOCAL_BASE_URL", LOCAL)
    expected_key = hashlib.sha256(f"{LOCAL_ID}\x00p".encode()).hexdigest()

    _lane.Cache(tmp_path / "lane.jsonl", _lane.key_for(LOCAL_ID), LOCAL_ID).complete("p")
    row = json.loads((tmp_path / "lane.jsonl").read_text().splitlines()[0])
    assert row == {"k": expected_key, "model": LOCAL_ID, "resp": "B"}

    bm_cache = blind_metric._Cache(tmp_path / "bm.jsonl", blind_metric._key(LOCAL_ID), LOCAL_ID)
    bm_cache.complete(LOCAL_ID, "p")
    row = json.loads((tmp_path / "bm.jsonl").read_text().splitlines()[0])
    assert row["k"] == expected_key and row["model"] == LOCAL_ID

    chexpert._Cache(tmp_path / "cx.jsonl", chexpert._key(LOCAL_ID), LOCAL_ID).ask("p", _gray())
    row = json.loads((tmp_path / "cx.jsonl").read_text().splitlines()[0])
    assert row["k"].startswith(f"{LOCAL_ID}:")

    out_dir, cache_path = _lane.scoped(LOCAL_ID, str(tmp_path / "results"),
                                       str(tmp_path / "results" / "call_cache.jsonl"))
    assert out_dir.name == "local_nvidia_cosmos-reason2-8b"
    assert cache_path.name == "local_nvidia_cosmos-reason2-8b_call_cache.jsonl"


def test_a_prefixed_id_with_no_local_server_exits_and_never_reaches_the_vendor(tmp_path, recorded,
                                                                               monkeypatch):
    """Even with an NVIDIA key in the environment, a ``local/`` id never falls back to the vendor.

    Before the prefix existed, ``local/nvidia/...`` was an unrecognised id that resolved to the
    NVIDIA key and endpoint, so this request went to the hosted service under an invalid name.
    """
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    for module in (_lane, blind_metric, chexpert):
        monkeypatch.setattr(module, "LOCAL_BASE_URL", "")
    # A local/ id never resolves a vendor key, whatever the environment holds.
    assert _lane.key_for(LOCAL_ID) is None
    assert blind_metric._key(LOCAL_ID) is None
    assert chexpert._key(LOCAL_ID) is None

    with pytest.raises(SystemExit) as exc:
        _lane.Cache(tmp_path / "lane.jsonl", _lane.key_for(LOCAL_ID), LOCAL_ID).complete("p")
    assert "BENCHMAXXING_LOCAL_BASE_URL" in str(exc.value)
    bm_cache = blind_metric._Cache(tmp_path / "bm.jsonl", blind_metric._key(LOCAL_ID), LOCAL_ID)
    with pytest.raises(SystemExit) as exc:
        bm_cache.complete(LOCAL_ID, "p")
    assert "BENCHMAXXING_LOCAL_BASE_URL" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        chexpert._Cache(tmp_path / "cx.jsonl", chexpert._key(LOCAL_ID), LOCAL_ID).ask("p", _gray())
    assert "BENCHMAXXING_LOCAL_BASE_URL" in str(exc.value)
    # The backend builders refuse too, so no caller can reach the vendor by passing a key directly.
    with pytest.raises(SystemExit):
        _lane.backend_for(LOCAL_ID, "nvapi-test")
    with pytest.raises(SystemExit):
        blind_metric._backend(LOCAL_ID, "nvapi-test")
    assert recorded == []


def test_a_fully_cached_prefixed_run_needs_no_server(tmp_path, recorded, monkeypatch):
    """Replaying committed local answers from a fresh clone must not require the server."""
    monkeypatch.setattr(_lane, "LOCAL_BASE_URL", "")
    key = hashlib.sha256(f"{LOCAL_ID}\x00p".encode()).hexdigest()
    path = tmp_path / "lane.jsonl"
    path.write_text(json.dumps({"k": key, "model": LOCAL_ID, "resp": "C"}) + "\n")
    assert _lane.Cache(path, _lane.key_for(LOCAL_ID), LOCAL_ID).complete("p") == "C"
    assert recorded == []


def _load_deliberation_channel():
    """experiments/medqa/deliberation_channel.py calls the OpenAI client itself, not backend.complete."""
    path = Path(__file__).resolve().parents[1] / "experiments" / "medqa" / "deliberation_channel.py"
    spec = importlib.util.spec_from_file_location("medqa_deliberation_channel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model, sent", [(LOCAL_ID, SERVED_ID),
                                         ("Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct")])
def test_the_direct_client_call_in_deliberation_channel_sends_the_served_id(monkeypatch, model, sent):
    """The channel arm builds its own request; a prefixed id must reach the server without its prefix."""
    channel = _load_deliberation_channel()
    monkeypatch.setattr(_lane, "LOCAL_BASE_URL", LOCAL)
    sent_kwargs = []

    class _Completions:
        def create(self, **kwargs):
            sent_kwargs.append(kwargs)
            message = types.SimpleNamespace(content="B", reasoning_content=None, model_extra={})
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
                                         usage=types.SimpleNamespace(completion_tokens=1))

    backend = types.SimpleNamespace(_client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions())))
    monkeypatch.setattr(channel._lane, "backend_for", lambda m, key, client=None: backend)
    result = channel._call(model, "not-needed", "prompt", "none")
    assert result["content"] == "B"
    assert [k["model"] for k in sent_kwargs] == [sent]
