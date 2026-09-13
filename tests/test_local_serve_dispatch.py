"""Serving a text-lane model locally (``BENCHMAXXING_LOCAL_BASE_URL``).

An open-weights arm can be served on the machine that runs the experiment instead of behind a
vendor endpoint. These tests pin the three things that change when it is: the OpenAI-compatible
backend points at the local server, the key lookup stops mattering because no local server checks
one, and the pacing that exists only to respect a vendor request ceiling goes to zero. They also
pin what must not change, which is the routing of the Gemini and DeepSeek ids whose committed
caches the cross-lineage comparison depends on.
"""
import sys
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


def test_a_local_server_captures_the_openai_compatible_ids_and_nothing_else(served_locally):
    """Every id that would go to the OpenAI-compatible vendor endpoint is served locally instead.

    That includes the nemotron id, deliberately: an open-weights comparator can also be served on
    the machine, and routing it anywhere else while a local server is configured would be
    surprising. The two ids that reach a vendor through its own SDK path are the ones that must
    not move, since their committed caches are what the cross-lineage comparison rests on.
    """
    assert _lane.is_local(OPEN_WEIGHTS)
    assert _lane.is_local(NIM)
    assert not _lane.is_local(GEMINI)
    assert not _lane.is_local(DEEPSEEK)


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
