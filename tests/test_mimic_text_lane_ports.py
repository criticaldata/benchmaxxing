"""The MIMIC-CXR text lane on a second lineage: the two runners it reuses from the MedQA lane must
keep their prompts out of the MedQA cache, and the records builder must import."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "experiments"))
    spec.loader.exec_module(mod)
    return mod


def test_build_solo_records_imports():
    """72803a3 removed reproduce._parse_choice; this module kept importing it and was broken since."""
    mod = _load("experiments/mimic_cxr_text/build_solo_records.py")
    assert hasattr(mod, "parse_legacy_string")


def test_reproduce_takes_an_explicit_cache_path(tmp_path, monkeypatch):
    """reproduce.py is reused on MIMIC-CXR, whose prompts carry report text; --cache must be able to
    steer that away from the MedQA lane's tracked call_cache.jsonl."""
    sys.path.insert(0, str(ROOT / "experiments"))
    import _lane
    out, cache = _lane.scoped("some/other-model", str(tmp_path / "out"),
                              "experiments/medqa/results/call_cache.jsonl", str(tmp_path / "mine.jsonl"))
    assert cache == tmp_path / "mine.jsonl"
    mod = _load("experiments/medqa/reproduce.py")
    src = (ROOT / "experiments/medqa/reproduce.py").read_text()
    assert '"--cache"' in src and "args.cache" in src


def test_refusal_aware_reanalysis_takes_model():
    src = (ROOT / "experiments/mimic_cxr_text/refusal_aware_reanalysis.py").read_text()
    assert '"--model"' in src and "rebind_models" in src
