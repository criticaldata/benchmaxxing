"""The referee, cascade, contamination, model-dependence, SUPPORT2 and MIMIC-CXR text runners take
--model through the shared dispatch, on the same terms as the MedQA port: own cache and key format
kept, every Gemini seat rebound when another model is requested, paths model-scoped.
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import _lane  # noqa: E402

RUNNERS = ["referee/referee_deployable", "referee/referee_judge", "referee/referee_threshold",
           "referee/referee_requery_design", "cascade/multi_round", "contamination/contamination_audit",
           "model_dependence/cascade_C_flash", "mimic_cxr_text/break_it_a", "mimic_cxr_text/break_it_d",
           "mimic_cxr_text/push_c", "mimic_cxr_text/referee_deployable", "mimic_cxr_text/referee_judge",
           "support2/support2_solo", "support2/support2_cascade", "support2/support2_cascade_strength",
           "support2/support2_referee", "support2/support2_referee_judge"]
MODEL = "openai/gpt-oss-120b"


@pytest.mark.parametrize("runner", RUNNERS)
def test_each_runner_goes_through_the_shared_dispatch(runner):
    src = (ROOT / "experiments" / f"{runner}.py").read_text()
    assert "import _lane" in src and "_lane.add_model_arg(ap)" in src
    assert "_lane.rebind_models(globals(), model)" in src
    assert "_lane.scoped(model, args.out," in src
    assert "GeminiBackend(" not in src
    # a --cache default is None so the scoped path is used; an explicit path is still honoured
    assert not re.search(r'add_argument\("--(board-|requery-|noise-log|)cache", default="', src)


def test_support2_common_uses_the_shared_dispatch():
    src = (ROOT / "experiments/support2/_common.py").read_text()
    assert "_lane.backend_for(model, self.key)" in src and "GeminiBackend(" not in src


def test_mimic_solo_records_uses_the_central_parser():
    """build_solo_records lost its parser when #102 centralised extraction; it must use the survivor."""
    src = (ROOT / "experiments/mimic_cxr_text/build_solo_records.py").read_text()
    assert "_parse_choice" not in src, "the name #102 removed is back"
    assert "from benchmaxxing.extract import parse_legacy_string" in src
    assert 'parse_legacy_string(cache[key], list(payload["options"]))' in src


def test_mimic_refusal_aware_takes_a_model():
    """The re-analysis was pinned to the Gemini tiers, so no second lineage could be scored."""
    src = (ROOT / "experiments/mimic_cxr_text/refusal_aware_reanalysis.py").read_text()
    assert 'ap.add_argument("--model"' in src
    assert "TIERS = [args.model]" in src
    assert "if args.model:" in src, "the default must keep both committed Gemini tiers"


def test_cross_dataset_pilot_uses_the_shared_dispatch():
    """The MedQA/MedMCQA cue pilot keeps its own --model, cache and skip path; only the backend and the
    key for a non-Gemini model come from the dispatch, so the default model still needs a Gemini key."""
    src = (ROOT / "experiments/cross_dataset/run_cross_dataset.py").read_text()
    assert "_lane.backend_for(model, api_key)" in src and "GeminiBackend(" not in src
    assert "key = _lane.key_for(args.model)" in src
    assert 'os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")' in src


@pytest.mark.parametrize("runner", RUNNERS)
def test_each_runner_has_gemini_seats_to_rebind(runner):
    spec = importlib.util.spec_from_file_location(f"r_{runner.replace('/', '_')}", ROOT / "experiments" / f"{runner}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert _lane.rebind_models(vars(mod), MODEL) > 0
    for name, value in vars(mod).items():
        if name.isupper() and isinstance(value, (str, list, tuple, dict)):
            # exact ids only: roster metadata such as lineage="gemini" is not a model seat
            assert not any(g in repr(value) for g in _lane.GEMINI_IDS), f"{runner}.{name} still names a Gemini id"


@pytest.mark.parametrize("runner", RUNNERS)
def test_the_default_model_path_is_unchanged(runner):
    src = (ROOT / "experiments" / f"{runner}.py").read_text()
    assert re.search(r"if model != _lane\.DEFAULT_MODEL else (_key|api_key)\(\)", src)
