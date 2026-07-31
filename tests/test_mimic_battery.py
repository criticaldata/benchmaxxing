"""Tests for the MIMIC-CXR battery runner: arm coverage, argv shape, and the plant guard.

Offline: no images, no key, no network. The runner only builds argv and shells out, so the argv
itself is what these pin.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.mimic_cxr_image.build_subset import DEFAULT_SIZES
from experiments.mimic_cxr_image.run_battery import (
    ARMS,
    CACHE_NAME,
    JUDGE_CACHE_NAME,
    WHOLE_MANIFEST,
    Arm,
    build_command,
    clean_read_planters,
    writes,
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
        assert Path(_flag(_cmd(arm), "--cache")).parent == RESULTS


def test_one_shared_cache_so_nested_image_arms_reuse_calls():
    # The image arms nest (blind 100 < cascade 150 < referee 300 < solo 600), so one cache lets a
    # later arm replay an earlier one's identical clean reads. The judge is excluded on purpose: it
    # opens the film too since #408, but it asks a different question about it, so it shares no call
    # with the diagnostic arms and folding it in would make their hit rate unreadable.
    diagnostic_arms = [a for a in ARMS if a.takes_manifest and a.name != "judge"]
    assert {_flag(_cmd(a), "--cache") for a in diagnostic_arms} == {str(RESULTS / CACHE_NAME)}
    assert _flag(_cmd(_arm("judge")), "--cache") == str(RESULTS / JUDGE_CACHE_NAME)


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
    assert order.index("referee_cascade") < order.index("judge")


def test_the_battery_covers_the_same_lineage_judge():
    # #393: the judge cell was blank for MIMIC-CXR only because no arm here ever invoked the
    # runner. A battery that cannot produce it is how that happened, so pin the coverage.
    assert "imaging_judge_referee" in {a.module for a in ARMS}


def test_judge_scores_the_same_transcript_and_dir_as_the_deployable_referee():
    # The two detectors share a row in the cross-dataset table; scoring different transcripts, or
    # writing outside referee_300/, would make the cells incomparable.
    judge, referee = _cmd(_arm("judge")), _cmd(_arm("referee"))
    assert _flag(judge, "--cascade-jsonl") == _flag(referee, "--cascade-jsonl")
    assert _flag(judge, "--out") == _flag(referee, "--out") == str(RESULTS / "referee_300")


def test_judge_is_passed_the_image_so_its_verdict_is_not_the_naive_gate():
    """#408 changed the contract: the judge opens the film, so the arm must pass the image args.

    This test previously asserted the opposite, that the judge took neither flag, which was true of
    the text-only judge. Leaving it that way made the arm unrunnable: imaging_judge_referee exits
    with a usage error unless it gets --manifest and --image-root or an explicit --text-only. The
    reason the flags matter is not plumbing. Without the film the prompt carries only
    (finding, shared), and the verdict equals the naive gate on every row (#393).
    """
    cmd = _cmd(_arm("judge"))
    assert "--manifest" in cmd, "the judge needs the manifest to map case_id to image_ref"
    assert "--image-root" in cmd, "the judge needs the image root to open the film"
    assert "--text-only" not in cmd, (
        "the battery must not run the legacy text-only judge: its verdict is pinned to the naive "
        "gate by construction, so it is not a measurement")
    assert "--n" not in cmd, "the judge scores whatever rows the transcript holds"


def test_transcript_dependencies_name_an_arm_that_writes_them():
    # The pre-flight error tells the operator which arm to rerun; an unresolvable `needs` would
    # print an empty name instead.
    for arm in ARMS:
        if arm.needs:
            assert writes(arm.needs), f"{arm.name} needs {arm.needs}, which no arm writes"
    assert writes("referee_300/imaging_cascade.jsonl") == "referee_cascade"


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


def _deid(name):
    rows = csv.DictReader((RESULTS_DIR / "deid" / name).read_text().splitlines())
    return {int(r["case_index"]): r for r in rows}


def _rate(rows, pred):
    tp = sum(1 for r in rows if int(r[pred]) and int(r["gt"]))
    fp = sum(1 for r in rows if int(r[pred]) and not int(r["gt"]))
    fn = sum(1 for r in rows if not int(r[pred]) and int(r["gt"]))
    tn = sum(1 for r in rows if not int(r[pred]) and not int(r["gt"]))
    return tp / (tp + fp), tp / (tp + fn), fp / (fp + tn)


def test_the_detector_table_cells_are_the_clean_correct_restriction():
    """Which cohort the published MIMIC-CXR cells are counted on (#393).

    The judge arm scores every row it is given, so a reader taking its top-level block would put a
    417-case number beside 91-case referee and gate cells. Pinning both here means the distinction
    cannot quietly rot back out of the docs.
    """
    ref, casc = _deid("referee.csv"), _deid("referee_cascade.csv")
    assert ref.keys() == casc.keys()
    rows = [{**ref[i], **casc[i]} for i in sorted(ref)]
    assert len(rows) == 417

    headroom = [r for r in rows if int(r["clean_correct"])]
    assert len(headroom) == 91, "the published cells are the 91-case restriction"

    for pred, cell in (("ref_flag", (0.77, 0.75, 0.21)), ("naive_flag", (0.54, 1.00, 0.81))):
        assert _rate(headroom, pred) == pytest.approx(cell, abs=0.005), pred
    # Unrestricted, the same two detectors read very differently, which is why it matters.
    assert _rate(rows, "naive_flag") == pytest.approx((0.19, 1.00, 0.96), abs=0.005)


def test_gt_is_the_peer_driven_adoption_the_judge_also_scores():
    # imaging_judge_referee derives gt from shared_adopt/iso_adopt itself, so the judge and the
    # referee only share a ground truth as long as these agree row for row.
    ref, casc = _deid("referee.csv"), _deid("referee_cascade.csv")
    for i, r in ref.items():
        c = casc[i]
        assert int(r["gt"]) == int(int(c["shared_adopt"]) == 1 and int(c["iso_adopt"]) == 0)


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
