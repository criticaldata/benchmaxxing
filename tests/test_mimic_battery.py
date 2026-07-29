"""Tests for the MIMIC-CXR battery runner: arm coverage, argv shape, and the plant guard.

Offline: no images, no key, no network. The runner only builds argv and shells out, so the argv
itself is what these pin.
"""

from __future__ import annotations

from pathlib import Path

from experiments.mimic_cxr_image.build_subset import DEFAULT_SIZES
from experiments.mimic_cxr_image.run_battery import (
    ARMS,
    CACHE_NAME,
    WHOLE_MANIFEST,
    Arm,
    build_command,
    clean_read_planters,
)

MANIFESTS = Path("/m")
RESULTS = Path("/r")
IMAGE_ROOT = Path("/img")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "mimic_cxr_image" / "results"


def _cmd(arm: Arm) -> list[str]:
    return build_command(arm, MANIFESTS, IMAGE_ROOT, RESULTS, python="py")


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def _arm(name: str) -> Arm:
    return next(a for a in ARMS if a.name == name)


def test_every_committed_summary_has_an_arm_that_regenerates_it():
    # The gap #343 opened on: results/ held summaries no committed code could reproduce. Read the
    # real committed files rather than a hand-copied list, so a new summary with no producing arm
    # fails here instead of becoming the next unreproducible number.
    produced = {(a.out, a.module) for a in ARMS}
    checked, orphans = 0, []
    for path in sorted(RESULTS_DIR.rglob("*_summary.json")):
        if path.name == "plant_direction_summary.json":
            continue  # written by plant_direction_check.py, an offline reanalysis of transcripts
        rel = path.parent.relative_to(RESULTS_DIR)
        out = "" if rel == Path() else str(rel)
        checked += 1
        if (out, path.name.removesuffix("_summary.json")) not in produced:
            orphans.append(str(path.relative_to(RESULTS_DIR)))
    assert checked, f"no committed summaries found under {RESULTS_DIR}"
    assert not orphans, f"committed summaries with no arm that regenerates them: {orphans}"


def test_arms_reference_manifests_build_subset_actually_writes():
    # Arm names and sizes come from build_subset.DEFAULT_SIZES, so a resize there cannot silently
    # leave the battery pointing at a manifest filename that no longer exists.
    expected = {f"{arm}_{n}.csv" for arm, n in DEFAULT_SIZES.items()}
    assert {a.manifest for a in ARMS} <= expected


def test_cache_is_pinned_inside_the_mimic_results_dir():
    # Every runner but imaging_solo defaults --cache to the NIH lane's committed cache; an
    # unpinned MIMIC run would append credentialed-derived reads into it.
    for arm in ARMS:
        assert _flag(_cmd(arm), "--cache") == str(RESULTS / CACHE_NAME)


def test_one_shared_cache_so_nested_arms_reuse_calls():
    assert len({_flag(_cmd(a), "--cache") for a in ARMS}) == 1


def test_n_is_pinned_above_every_arm_size():
    # Unpinned, --n defaults to 35/40 and an 834-image arm silently measures its first 35.
    assert WHOLE_MANIFEST > max(DEFAULT_SIZES.values()) * 100
    for arm in ARMS:
        if arm.takes_n:
            assert _flag(_cmd(arm), "--n") == str(WHOLE_MANIFEST)


def test_referee_gets_no_n_flag():
    # imaging_referee replays a transcript and defines no --n; passing one is an argparse error.
    assert "--n" not in _cmd(_arm("referee"))


def test_referee_consumes_the_referee_300_cascade_transcript():
    cmd = _cmd(_arm("referee"))
    assert _flag(cmd, "--cascade-jsonl") == str(RESULTS / "referee_300" / "imaging_cascade.jsonl")
    assert _flag(cmd, "--out") == str(RESULTS / "referee_300")


def test_dependent_arms_run_after_the_arms_they_read():
    order = [a.name for a in ARMS]
    assert order.index("cascade") < order.index("system_flag")
    assert order.index("referee_cascade") < order.index("referee")


def test_system_flag_stages_the_cascade_transcript_it_contrasts_against():
    # imaging_system_flag reads imaging_cascade.jsonl from its own --out. The cascade arm writes to
    # cascade_150/, so without the copy vs_peer_assertion_cascade drops out of the summary.
    arm = _arm("system_flag")
    assert arm.stage == ("cascade_150/imaging_cascade.jsonl", "imaging_cascade.jsonl")
    assert _flag(_cmd(arm), "--out") == str(RESULTS)


def test_two_cascade_runs_do_not_overwrite_each_other():
    # imaging_cascade writes a fixed imaging_cascade.jsonl for the watermark cue, so a shared --out
    # would leave the 150-study and 300-study arms silently sharing one transcript.
    outs = [_flag(_cmd(a), "--out") for a in ARMS if a.module == "imaging_cascade"]
    assert len(set(outs)) == len(outs)


def test_plant_guard_flags_the_pre_fix_pattern(tmp_path):
    (tmp_path / "imaging_cascade.py").write_text('        wrong = "no" if clean == "yes" else "yes"\n')
    assert clean_read_planters(tmp_path) == ["imaging_cascade"]


def test_plant_guard_passes_the_ground_truth_plant(tmp_path):
    for module in sorted({a.module for a in ARMS}):
        (tmp_path / f"{module}.py").write_text('        wrong = "no"  # plant against ground truth\n')
    assert clean_read_planters(tmp_path) == []


def test_shipped_imaging_cascade_plants_against_ground_truth():
    # The #333 fix, guarded in the tree this battery actually runs. imaging_system_flag and
    # imaging_strength_cascade carry the same fix on main (#338) but not on this branch, so the
    # runner refuses those two arms until it is rebased rather than regenerating pre-fix numbers.
    assert "imaging_cascade" not in clean_read_planters()
