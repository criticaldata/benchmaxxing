"""The twelve Gemini-only MedQA runners take --model through the shared dispatch.

Each keeps its own cache class and key format, so the committed Gemini arm replays unchanged; when
another model is requested, every Gemini seat in the module's constants becomes that model, the key
comes from the shared lookup and the backend from the shared dispatch.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import _lane  # noqa: E402

RUNNERS = ["break_it", "clean_a", "push_c", "scale_c", "hierarchy_dominance", "hierarchy_temp",
           "majority_pressure", "orchestrator_failure", "seed_timing", "true_peer_control",
           "unanimity_break", "reproduce"]
MODEL = "openai/gpt-oss-120b"


def test_rebind_replaces_every_gemini_id_and_leaves_seat_names_alone():
    ns = {"HOLDOUT": "gemini-2.5-flash-lite", "SEAT": "holdout", "_PRIVATE": "gemini-2.5-flash",
          "MEMBERS": [("a", "gemini-2.5-flash"), ("b", "gemini-2.5-flash-lite")],
          "BY_NAME": {"x": "gemini-2.5-flash"}, "N": 3}
    assert _lane.rebind_models(ns, MODEL) == 4
    assert ns["HOLDOUT"] == MODEL and ns["SEAT"] == "holdout" and ns["_PRIVATE"] == "gemini-2.5-flash"
    assert ns["MEMBERS"] == [("a", MODEL), ("b", MODEL)] and ns["BY_NAME"] == {"x": MODEL}


def test_rebind_collapses_a_tier_list_to_distinct_models():
    ns = {"TIERS": ["gemini-2.5-flash", "gemini-2.5-flash-lite"]}
    assert _lane.rebind_models(ns, MODEL) == 2
    assert ns["TIERS"] == [MODEL]


def test_rebind_returns_zero_when_there_is_nothing_to_rebind():
    assert _lane.rebind_models({"HOLDOUT": "holdout", "N": 1}, MODEL) == 0


@pytest.mark.parametrize("runner", RUNNERS)
def test_each_runner_goes_through_the_shared_dispatch(runner):
    src = (ROOT / "experiments" / "medqa" / f"{runner}.py").read_text()
    assert "import _lane" in src
    assert "_lane.add_model_arg(ap)" in src
    assert "_lane.rebind_models(globals(), model)" in src
    assert "_lane.scoped(model, args.out," in src
    # No direct Gemini construction remains: a Gemini id reaches GeminiBackend via backend_for.
    assert "gateway.GeminiBackend(" not in src
    assert re.search(r"_lane\.backend_for\((self\.)?model, (self\.)?(api_)?key\)", src)


@pytest.mark.parametrize("runner", RUNNERS)
def test_each_runner_has_gemini_seats_to_rebind(runner):
    import importlib.util
    sys.path.insert(0, str(ROOT / "experiments" / "medqa"))
    spec = importlib.util.spec_from_file_location(f"r_{runner}", ROOT / "experiments" / "medqa" / f"{runner}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert _lane.rebind_models(vars(mod), MODEL) > 0
    for name, value in vars(mod).items():
        if name.isupper() and isinstance(value, (str, list, tuple, dict)):
            # exact ids only: roster metadata such as lineage="gemini" is not a model seat
            assert not any(g in repr(value) for g in _lane.GEMINI_IDS), f"{runner}.{name} still names a Gemini id"


def test_the_default_model_path_is_unchanged():
    """With the default model the runners call their own _key() and the committed paths."""
    for runner in RUNNERS:
        src = (ROOT / "experiments" / "medqa" / f"{runner}.py").read_text()
        assert re.search(r"if model != _lane\.DEFAULT_MODEL else _(get_)?key\(\)", src), runner
