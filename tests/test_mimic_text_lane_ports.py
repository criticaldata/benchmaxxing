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

def test_a_hosted_vendor_id_is_never_served_locally():
    """Belt and braces for the local-serve guard, in a file the sibling lineage branches do not have.

    `_lane.is_local` must stay False for a hosted vendor id even when BENCHMAXXING_LOCAL_BASE_URL is
    set, or a cache miss for this lineage would be answered by whatever model a local server happens
    to hold. tests/test_local_serve_dispatch.py asserts the same thing, but that file and _lane.py
    both differ on the other lineage branches, so a merge that resolves them in the other direction
    would drop the guard and its test together. This assertion is duplicated here on purpose.
    """
    sys.path.insert(0, str(ROOT / "experiments"))
    import _lane

    hosted = ("nvidia/nemotron-3-super-120b-a12b", "gemini-2.5-flash-lite", "deepseek-chat")
    original = _lane.LOCAL_BASE_URL
    try:
        _lane.LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
        for model in hosted:
            assert not _lane.is_local(model), f"{model} would be answered by a local server"
        assert _lane.is_local("openai/gpt-oss-120b")
    finally:
        _lane.LOCAL_BASE_URL = original
