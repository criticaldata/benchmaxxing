"""The imaging runners take --model and go through the shared dispatch.

Every runner under ``experiments/imaging/`` hardcoded ``MODEL = "gemini-2.5-flash"`` and built
``GeminiBackend`` itself, so the imaging lane could not be run on a second model at all. A served
vision-language model reaches the same OpenAI-compatible backend the text lanes use, and the image
rides in the chat content list, so the port is the one the text lanes already had: the flag, the
shared paced dispatch, and a model-scoped output directory and cache.

These tests pin the three things a mis-port breaks silently:
  - the live call must go through ``_lane.paced_complete``, never a bare ``RetryBackend``, or a 429
    or a dropped connection ends the whole arm (the text lanes learned this the expensive way);
  - the committed Gemini paths must not move, because the paper's imaging numbers were computed
    in ``experiments/imaging/results`` and nowhere else;
  - a second model must not be able to append to a committed Gemini cache, which is what a
    hardcoded ``--cache`` default caused on two text-lane runners.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import _lane  # noqa: E402

IMAGING = sorted(p for p in (ROOT / "experiments" / "imaging").glob("imaging_*.py")
                 if "_lane.paced_complete" in p.read_text() or "GeminiBackend" in p.read_text())


def test_the_lane_has_imaging_runners_to_check():
    # A collection bug that finds nothing would make every test below pass vacuously.
    assert len(IMAGING) >= 15


@pytest.mark.parametrize("path", IMAGING, ids=lambda p: p.name)
def test_each_imaging_runner_takes_a_model_flag(path):
    src = path.read_text()
    # A runner names its own seat: most call it MODEL, imaging_judge_referee calls it JUDGE.
    assert any(f"_lane.add_model_arg(ap, {seat})" in src for seat in ("MODEL", "JUDGE")), \
        f"{path.name} has no --model flag"


@pytest.mark.parametrize("path", IMAGING, ids=lambda p: p.name)
def test_each_imaging_runner_goes_through_the_shared_dispatch(path):
    src = path.read_text()
    assert "_lane.paced_complete(" in src, f"{path.name} does not call the paced dispatch"
    assert "GeminiBackend(" not in src, f"{path.name} still constructs GeminiBackend directly"
    assert "RetryBackend(" not in src, (
        f"{path.name} wraps its own RetryBackend; its five quick attempts expire before a rate "
        "bucket refills, which ends the arm"
    )


@pytest.mark.parametrize("path", IMAGING, ids=lambda p: p.name)
def test_each_imaging_runner_rebinds_every_gemini_seat(path):
    src = path.read_text()
    assert "_lane.rebind_models(globals(), model)" in src, f"{path.name} does not rebind its seats"
    assert any(f"default_model = {seat}" in src for seat in ("MODEL", "JUDGE")), \
        f"{path.name} does not keep its own committed id"


@pytest.mark.parametrize("path", IMAGING, ids=lambda p: p.name)
def test_no_imaging_runner_defaults_its_cache_to_a_committed_file(path):
    """A ``--cache`` default naming a tracked Gemini cache writes a second model's rows into it."""
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--cache"):
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                assert isinstance(kw.value, ast.Constant) and kw.value.value is None, (
                    f"{path.name}: --cache default must be None and derived from --out"
                )


def test_scoped_keeps_the_imaging_lane_committed_paths_for_its_own_default(tmp_path):
    """The imaging lane ran on gemini-2.5-flash, not the text lane's flash-lite default."""
    out, cache = _lane.scoped("gemini-2.5-flash", str(tmp_path), str(tmp_path / "img_cache.jsonl"),
                              default="gemini-2.5-flash")
    assert out == tmp_path
    assert cache == tmp_path / "img_cache.jsonl"


def test_scoped_gives_a_served_vision_model_its_own_directory_and_cache(tmp_path):
    out, cache = _lane.scoped("Qwen/Qwen2.5-VL-72B-Instruct", str(tmp_path),
                              str(tmp_path / "img_cache.jsonl"), default="gemini-2.5-flash")
    assert out == tmp_path / "Qwen_Qwen2.5-VL-72B-Instruct"
    assert cache == tmp_path / "Qwen_Qwen2.5-VL-72B-Instruct_img_cache.jsonl"


def test_paced_complete_passes_an_image_through_to_the_backend():
    """The imaging arms need the film in the same call the text lanes use for pacing and retry."""
    seen = {}

    class _Backend:
        def complete(self, prompt, image=None, decoding=None):
            seen.update(prompt=prompt, image=image, decoding=decoding)
            return "yes"

    real = _lane.backend_for
    _lane.backend_for = lambda *a, **k: _Backend()
    try:
        out = _lane.paced_complete("gemini-2.5-flash", "k", "does this film show it?",
                                   image="<film>", decoding={"temperature": 0})
    finally:
        _lane.backend_for = real
    assert out == "yes"
    assert seen["image"] == "<film>"
    assert seen["prompt"] == "does this film show it?"


@pytest.mark.parametrize("path", IMAGING, ids=lambda p: p.name)
def test_each_imaging_runner_imports_and_its_parser_builds(path, monkeypatch):
    """A string check cannot catch ``add_model_arg(ap, MODEL)`` in a runner that names its seat
    something else: imaging_judge_referee calls its seat JUDGE, and the flag raised NameError when
    the parser was built while every text assertion above still passed."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_imgrunner_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", [path.name, "--help"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0, f"{path.name} --help did not build its parser cleanly"
