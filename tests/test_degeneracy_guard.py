"""Behavioural tests for the #374 cannot-fail-metric screen.

Two jobs, and the second matters as much as the first.

1. Gate the repo. `test_no_unexplained_cannot_fail_metrics` fails when a screen fires on something
   that is in neither exemption map, so a new definitional predicate cannot reach the paper quietly.

2. Prove the screen is not vacuous. #374 exists partly because earlier guards were text greps that a
   comment could satisfy. So this file sabotages the screen from both directions: it feeds it
   synthetic artifacts that must be flagged, feeds it artifacts that must NOT be flagged, and pins
   the specific #374 instances the screen finds on the real tree, so neutering a screen turns the
   suite red instead of quiet green.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from benchmaxxing.degeneracy import (
    duplicate_columns,
    forced_direction,
    EXEMPTIONS_PATH,
    MIN_ROWS,
    REPO_ROOT,
    Finding,
    constant_columns,
    hardcoded_verdicts,
    load_exemptions,
    rounded_pvalues,
    scan,
    unexplained,
)

KINDS = {"constant_column", "hardcoded_verdict", "rounded_pvalue", "duplicate_column", "forced_direction",
    "identical_reads",
}


@pytest.fixture(scope="module")
def real() -> list[Finding]:
    return scan(REPO_ROOT)


@pytest.fixture(scope="module")
def keys(real) -> set[str]:
    return {f.key for f in real}


# ------------------------------------------------------------------ 1. the gate


def test_no_unexplained_cannot_fail_metrics():
    """The gate. Anything new must be fixed, or exempted with a written reason."""
    new = unexplained(REPO_ROOT)
    assert not new, "cannot-fail metrics with no exemption:\n" + "\n".join(f"  {f}" for f in new)


def test_every_exemption_carries_a_written_reason():
    allow, pre = load_exemptions()
    for name, table in (("allowlist", allow), ("preexisting", pre)):
        for key, reason in table.items():
            assert isinstance(reason, str), f"{name}[{key}] reason is not a string"
            # A placeholder like "" or "n/a" or "legit" must not buy an exemption.
            assert len(reason.strip()) >= 40, f"{name}[{key}] needs a real reason, got {reason!r}"


def test_exemption_keys_are_specific_not_blanket():
    """No wildcard, no path-only key. An exemption names one finding or it is not an exemption."""
    allow, pre = load_exemptions()
    for key in list(allow) + list(pre):
        parts = key.split("|")
        assert len(parts) == 3, f"malformed exemption key {key!r}"
        kind, path, locus = parts
        assert kind in KINDS, f"unknown kind in {key!r}"
        assert path and locus, f"empty path or locus in {key!r}"
        assert "*" not in key and "?" not in key, f"exemptions are exact keys, not globs: {key!r}"


def test_no_stale_exemptions(keys):
    """An exemption cannot outlive the defect it names.

    Deleting the entry is the last step of a fix. Without this the file silently accumulates
    permission for defects that are already gone, and the next real one hides behind a stale line.
    """
    allow, pre = load_exemptions()
    stale = sorted(k for k in list(allow) + list(pre) if k not in keys)
    assert not stale, (
        "these exemptions no longer match any finding, delete them from "
        f"{EXEMPTIONS_PATH.name}:\n" + "\n".join(f"  {k}" for k in stale)
    )


# ------------------------------------------------------------------ 2. not vacuous, on the real tree


def test_every_screen_fires_on_a_synthetic_instance_of_its_own_defect(tmp_path):
    """Anti-vacuity anchors, on fixtures rather than on the backlog.

    @Agastya191 caught the original design here and was right. The anchors used to be 12 real keys
    from the #374 table, and the yield floors were pinned to the tree's exact counts. That made the
    guard's own tests fail when someone FIXED a #374 item: he corrected unanimity_break.py the way
    the issue asks, deleted its exemption, and the suite went red on these two tests rather than on
    anything new. The first person to clear a defect then has to edit the guard to get green, and
    will reasonably reach for the exemption file instead. A guard that penalises being fixed is
    worse than no guard.

    So the anchors now live on synthetic fixtures that nobody will ever clean up, and they assert
    each screen fires AT ALL rather than pinning a count that only holds while the tree is dirty.
    """
    out = tmp_path / "experiments" / "synth" / "results"
    out.mkdir(parents=True, exist_ok=True)

    # screen 1: an unsatisfiable control, the shape of authority_ladder's control_adopt
    _write_cases(tmp_path, "control.jsonl", [{"case_id": f"c{i}", "control_adopt": False} for i in range(40)])
    # screen 2: a hardcoded verdict beside an interpolated p
    (tmp_path / "experiments" / "synth").mkdir(parents=True, exist_ok=True)
    (tmp_path / "experiments" / "synth" / "report.py").write_text(
        'def main(mc):\n    print(f"NOT significant at this n (p={mc.pvalue})")\n'
    )
    # screens 3 and 5: p printed as 0.0, beside a one-sided table and a saturated rate
    (out / "arm_summary.json").write_text(
        json.dumps({"arm": {"shared_adoption": 1.0,
                            "mcnemar": {"gain": 82, "lose": 0, "pvalue": 0.0}}})
    )
    # screen 4: an algebraic reduction between two varying columns
    _write_cases(tmp_path, "reduction.jsonl",
                 [{"case_id": f"c{i}", "adopted": i % 3 == 0, "deployable": i % 3 == 0} for i in range(40)])

    assert constant_columns(tmp_path), "screen 1 stopped firing"
    assert hardcoded_verdicts(tmp_path), "screen 2 stopped firing"
    assert rounded_pvalues(tmp_path), "screen 3 stopped firing"
    assert duplicate_columns(tmp_path), "screen 4 stopped firing"
    assert forced_direction(tmp_path), "screen 5 stopped firing"


def test_every_screen_still_finds_something_on_the_real_tree():
    """A screen returning nothing on the real tree has been narrowed to uselessness.

    Deliberately `> 0` and not a count. Counts here would fail the moment a defect is fixed, which
    is the trap the test above documents.
    """
    for name, fn in (
        ("constant_column", constant_columns),
        ("hardcoded_verdict", hardcoded_verdicts),
        ("rounded_pvalue", rounded_pvalues),
        ("duplicate_column", duplicate_columns),
        ("forced_direction", forced_direction),
    ):
        assert fn(REPO_ROOT), f"{name} found nothing on the real tree"



@pytest.mark.parametrize(
    "path,column",
    [
        # Columns with real variation, so screen 1 must stay silent on them: a constant screen that
        # flags a varying column is a rubber stamp, as useless as one that flags nothing.
        # This asserts nothing about whether they are SOUND. `adopted` and `deployable` below vary
        # and are also identical row for row, which is the #374 headline defect; screen 4 catches
        # that and this test is not the place to look for it.
        ("experiments/mimic_cxr_image/results/deid/referee.csv", "ref_flag"),
        ("experiments/mimic_cxr_image/results/deid/referee.csv", "naive_flag"),
        ("experiments/mimic_cxr_image/results/deid/cascade.csv", "shared_adopt"),
        ("experiments/mimic_cxr_image/results/deid/solo.csv", "clean_correct"),
        ("experiments/mimic_cxr_image/results/deid/strength_cascade.csv", "op0.15_shared_adopt"),
        ("experiments/referee/results/referee_deployable.jsonl", "adopted"),
        ("experiments/referee/results/referee_deployable.jsonl", "deployable"),
        ("experiments/effect_sizes/results/scale_c_per_case.jsonl", "anchored"),
    ],
)
def test_columns_that_genuinely_vary_are_not_flagged(keys, path, column):
    assert (REPO_ROOT / path).exists(), path
    assert f"constant_column|{path}|{column}" not in keys


# The baseline is the whole risk in this design. Every finding today is exempt, so the guard's
# present value is zero and all of its value is prospective. That only holds if the baseline can
# shrink and cannot grow: without a ceiling, the cheapest response to a new real defect is one more
# line in a 122-entry JSON file, which a reviewer skimming a diff will not catch. I demonstrated
# that hole before closing it, by planting a fresh constant column and hiding it behind one
# boilerplate reason with the whole suite green.
# Raised 113 -> 116 deliberately, and the raise is a DETECTION improvement rather than three new
# defects. The `identical_reads` screen reads yes/no columns, which `_binary` does not, so the pair
# screens had been blind to the whole imaging lane; turning it on surfaced five pre-existing
# instances at once (three no-cue arms, the CheXpert system-flag arm, and k1 vs k2 on the peer-size
# curve). Two of the five nobody had found by review. Lowering this number is always fine; raising
# it again needs the same kind of justification in the commit message.
PREEXISTING_CEILING = 116


def test_identical_reads_fires_on_two_reads_that_never_diverge():
    """Anti-vacuity for screen 4b, on synthetic fixtures rather than on the backlog it exists to clear.

    Anchoring this to a real committed file would make fixing that file turn this test red, which is
    the trap that pushes a fixer toward the exemption list instead of the defect.
    """
    import json
    import tempfile
    from unittest import mock

    from benchmaxxing import degeneracy

    root = Path(tempfile.mkdtemp())
    res = root / "experiments" / "x" / "results"
    res.mkdir(parents=True)
    (res / "degenerate.jsonl").write_text("".join(
        json.dumps({"case_id": f"c{i}", "clean": "yes" if i % 3 else "no",
                    "iso": "yes" if i % 3 else "no", "shared": "no"}) + "\n" for i in range(30)))
    (res / "honest.jsonl").write_text("".join(
        json.dumps({"case_id": f"c{i}", "clean": "yes" if i % 3 else "no",
                    "iso": "yes" if i % 4 else "no", "shared": "no"}) + "\n" for i in range(30)))

    with mock.patch.object(degeneracy, "_tracked", lambda r, pat: sorted(res.glob("*.jsonl"))):
        found = degeneracy.identical_reads(root)

    loci = {(Path(f.path).name, f.locus) for f in found}
    assert ("degenerate.jsonl", "clean vs iso") in loci, (
        "the screen did not fire on two reads that are identical on every row")
    assert not any(name == "honest.jsonl" for name, _ in loci), (
        "the screen fired on reads that genuinely diverge")


def test_identical_reads_ignores_a_column_derived_from_another():
    """`iso` vs `iso_adopt` is notation, not a second measurement, and must not be reported.

    Without the prefix skip this screen produced 113 findings on the imaging lane, nearly all of them
    definitional, which is how a useful screen becomes an ignored one.
    """
    import json
    import tempfile
    from unittest import mock

    from benchmaxxing import degeneracy

    root = Path(tempfile.mkdtemp())
    res = root / "experiments" / "y" / "results"
    res.mkdir(parents=True)
    (res / "derived.jsonl").write_text("".join(
        json.dumps({"case_id": f"c{i}", "iso": "yes" if i % 3 else "no",
                    "iso_read": "yes" if i % 3 else "no"}) + "\n" for i in range(30)))

    with mock.patch.object(degeneracy, "_tracked", lambda r, pat: sorted(res.glob("*.jsonl"))):
        found = degeneracy.identical_reads(root)

    assert not found, f"reported a derived column as a duplicate measurement: {found}"


def test_the_preexisting_baseline_can_only_shrink():
    """A new defect must not be absorbable by adding a line to the exemption file.

    If this fails because you added an entry, that is the point: either fix the defect, or move the
    ceiling deliberately in the same commit and say why in the message. Lowering it is always fine.
    """
    _, pre = load_exemptions()
    assert len(pre) <= PREEXISTING_CEILING, (
        f"{len(pre)} pre-existing exemptions against a ceiling of {PREEXISTING_CEILING}. "
        "Adding one hides a real defect behind a line of JSON; fix it or move the ceiling on purpose."
    )


def test_the_full_scan_is_fast_enough_for_the_normal_suite():
    t0 = time.perf_counter()
    scan(REPO_ROOT)
    elapsed = time.perf_counter() - t0
    # Measured at about 1.0s on the tree at the time of writing. The bound is loose so a slow
    # machine does not fail the build, but tight enough to catch someone parsing the 240 MB of
    # call caches by accident.
    assert elapsed < 20, f"the scan took {elapsed:.1f}s, too slow to keep in the normal suite"


# ------------------------------------------------------------------ 3. sabotage: synthetic artifacts


def _write_cases(root: Path, name: str, rows: list[dict], sub: str = "synth") -> Path:
    out = root / "experiments" / sub / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_a_synthetic_constant_scored_column_turns_the_screen_red(tmp_path):
    """The sabotage test. A planted definitional column must be caught, not merely catchable."""
    rows = [
        {"case_id": f"c{i}", "adopted": i % 2 == 0, "control_adopt": 0}
        for i in range(MIN_ROWS + 4)
    ]
    _write_cases(tmp_path, "planted.jsonl", rows)
    found = {f.locus: f for f in constant_columns(tmp_path)}
    assert "control_adopt" in found, f"the planted constant column was missed, got {list(found)}"
    assert "adopted" not in found, "a column that alternates must not be flagged"
    assert f"all {len(rows)} rows" in found["control_adopt"].detail
    assert found["control_adopt"].detail.endswith("rows")
    assert "constant at False" in found["control_adopt"].detail


def test_a_synthetic_constant_column_in_a_deid_csv_turns_the_screen_red(tmp_path):
    out = tmp_path / "experiments" / "synth" / "results" / "deid"
    out.mkdir(parents=True)
    n = MIN_ROWS + 2
    lines = ["case_index,gt,forced_flag"] + [f"{i},{i % 2},1" for i in range(n)]
    (out / "planted.csv").write_text("\n".join(lines) + "\n")
    loci = {f.locus for f in constant_columns(tmp_path)}
    assert "forced_flag" in loci
    assert "gt" not in loci


def test_a_csv_outside_deid_is_left_alone(tmp_path):
    """Cohort manifests legitimately carry constant label columns, so they are out of scope."""
    out = tmp_path / "experiments" / "synth" / "results"
    out.mkdir(parents=True)
    n = MIN_ROWS + 2
    (out / "manifest.csv").write_text(
        "\n".join(["case_index,forced_flag"] + [f"{i},1" for i in range(n)]) + "\n"
    )
    assert constant_columns(tmp_path) == []


def test_a_call_cache_is_not_mistaken_for_a_cohort(tmp_path):
    rows = [{"k": f"h{i}", "model": "gemini-2.5-flash", "resp": "D", "hit": 1} for i in range(30)]
    _write_cases(tmp_path, "call_cache.jsonl", rows)
    assert constant_columns(tmp_path) == [], "a keyed response cache has no scored columns"


def test_a_per_turn_transcript_is_not_mistaken_for_a_cohort(tmp_path):
    """Transcripts name a case only in the header row, so `seeded: false` on every turn is not a
    constant scored column."""
    rows = [{"case_id": "c0", "kind": "header", "shared": False}]
    rows += [{"kind": "turn", "turn_index": i, "seeded": False} for i in range(30)]
    _write_cases(tmp_path, "c0_isolated.jsonl", rows)
    assert constant_columns(tmp_path) == []


def test_a_short_cohort_is_below_the_threshold(tmp_path):
    """Documents a real limit rather than hiding it: constancy over a handful of rows is not news."""
    rows = [{"case_id": f"c{i}", "flag": 1} for i in range(MIN_ROWS - 1)]
    _write_cases(tmp_path, "tiny.jsonl", rows)
    assert constant_columns(tmp_path) == []


def test_a_near_constant_column_is_a_known_blind_spot(tmp_path):
    """This is the documented hole, asserted so nobody mistakes green for an all-clear.

    A predicate true on 199 of 200 rows is as definitional in practice as one true on 200, and the
    constant screen will not see it. Stated here in executable form so the limit is on the record.
    """
    rows = [{"case_id": f"c{i}", "almost_always": i != 7} for i in range(200)]
    _write_cases(tmp_path, "near.jsonl", rows)
    assert constant_columns(tmp_path) == []


def test_an_algebraic_reduction_between_varying_columns_is_caught_by_screen_four(tmp_path):
    """This was the documented hole and screen 4 closes it.

    `deployable` equals `adopted` row for row and both vary, so the constant screen cannot see it,
    which is what made this the defect that cost us #357, #366, #368 and #390. The duplicate screen
    is the one that catches it.
    """
    rows = [{"case_id": f"c{i}", "adopted": i % 3 == 0, "deployable": i % 3 == 0} for i in range(40)]
    _write_cases(tmp_path, "reduction.jsonl", rows)
    assert constant_columns(tmp_path) == []  # still invisible to screen 1, by design
    found = duplicate_columns(tmp_path)
    assert [f.locus for f in found] == ["adopted vs deployable"]
    assert "identical to" in found[0].detail


def test_a_comparator_built_as_the_negation_of_its_label_is_caught(tmp_path):
    """`clean_correct` as the exact complement of `iso_adopt`, the real #387 shape."""
    rows = [
        {"case_id": f"c{i}", "clean_correct": i % 4 != 0, "iso_adopt": i % 4 == 0} for i in range(40)
    ]
    _write_cases(tmp_path, "complement.jsonl", rows)
    found = duplicate_columns(tmp_path)
    assert len(found) == 1
    assert "the exact complement of" in found[0].detail


def test_two_genuinely_independent_varying_columns_are_not_flagged(tmp_path):
    """The false-positive control for screen 4. Without this the screen could condemn everything."""
    rows = [{"case_id": f"c{i}", "a": i % 3 == 0, "b": i % 5 == 0} for i in range(40)]
    _write_cases(tmp_path, "independent.jsonl", rows)
    assert duplicate_columns(tmp_path) == []


def test_a_one_sided_paired_test_beside_a_saturated_rate_is_flagged(tmp_path):
    """#391: `lose` is 0 because the comparator is pinned, so the p measures n, not the effect."""
    out = tmp_path / "experiments" / "synth" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "arm_summary.json").write_text(
        json.dumps(
            {"arm": {"shared_adoption": 1.0, "mcnemar": {"gain": 115, "lose": 0, "pvalue": 1e-9}}}
        )
    )
    found = forced_direction(tmp_path)
    assert len(found) == 1
    assert "lose=0" in found[0].detail and "saturated" in found[0].detail


def test_a_one_sided_paired_test_with_no_saturation_is_a_strong_effect_not_a_defect(tmp_path):
    """The false-positive control for screen 5, and the important one.

    A real 71-to-0 split beside unsaturated rates is the finding, not a defect. A screen that
    condemns it would condemn this paper's own strongest results, so it must stay silent here.
    """
    out = tmp_path / "experiments" / "synth" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "arm_summary.json").write_text(
        json.dumps(
            {"arm": {"bare_rate": 0.708, "reasoned_rate": 0.367,
                     "mcnemar": {"gain": 0, "lose": 71, "pvalue": 8.2e-10}}}
        )
    )
    assert forced_direction(tmp_path) == []


# --- hardcoded verdicts


def _write_py(root: Path, body: str) -> None:
    out = root / "experiments" / "synth"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.py").write_text(body)


def test_a_synthetic_hardcoded_verdict_beside_a_p_value_is_flagged(tmp_path):
    _write_py(
        tmp_path,
        "def main(pvalue):\n"
        "    return f'the arms differ but it is NOT significant at this n (p={pvalue})'\n",
    )
    found = hardcoded_verdicts(tmp_path)
    assert len(found) == 1, found
    assert found[0].locus == "main:NOT significant"


def test_a_verdict_split_across_implicitly_concatenated_fragments_is_flagged(tmp_path):
    """The real #374 case: the verdict sits in one fragment and the p-value in the next."""
    _write_py(
        tmp_path,
        "def main(mc):\n"
        "    return (\n"
        "        f'a directional reduction of {mc.diff}, but it is NOT significant '\n"
        "        f'at this sample size (McNemar p={round(mc.pvalue, 6)})'\n"
        "    )\n",
    )
    assert [f.locus for f in hardcoded_verdicts(tmp_path)] == ["main:NOT significant"]


def test_a_dot_format_verdict_is_flagged(tmp_path):
    _write_py(tmp_path, "def main(p):\n    return 'no effect, p={}'.format(p)\n")
    assert len(hardcoded_verdicts(tmp_path)) == 1


def test_a_docstring_stating_a_past_finding_is_not_flagged(tmp_path):
    """A prose record of an old result interpolates nothing, so it asserts nothing about this run.

    Flagging it would be exactly the failure mode #374 complains about in reverse: a screen that
    reacts to text rather than to behaviour.
    """
    _write_py(
        tmp_path,
        '"""Round-1 vs round-K is not significant (McNemar p=0.625, gained 3, lost 1)."""\n'
        "def main():\n    return 1\n",
    )
    assert hardcoded_verdicts(tmp_path) == []


def test_a_verdict_with_no_p_value_in_the_interpolation_is_not_flagged(tmp_path):
    _write_py(tmp_path, "def main(n):\n    return f'no effect on any of the {n} cases'\n")
    assert hardcoded_verdicts(tmp_path) == []


def test_a_verdict_derived_from_the_p_value_is_not_flagged(tmp_path):
    """The correct pattern. The word is computed, so it can come out either way."""
    _write_py(
        tmp_path,
        "def main(pvalue):\n"
        "    verdict = 'significant' if pvalue < 0.05 else 'not significant'\n"
        "    return f'McNemar p={pvalue}: {verdict}'\n",
    )
    assert hardcoded_verdicts(tmp_path) == []


# --- rounded p-values


def _write_summary(root: Path, doc: dict, name: str = "s_summary.json") -> None:
    out = root / "experiments" / "synth" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(doc))


def test_a_p_of_exactly_zero_is_flagged(tmp_path):
    _write_summary(tmp_path, {"contrast": {"pvalue": 0.0, "gain": 30, "lose": 0}})
    assert [f.locus for f in rounded_pvalues(tmp_path)] == ["contrast.pvalue"]


def test_a_p_of_one_on_an_empty_discordant_table_is_flagged(tmp_path):
    _write_summary(tmp_path, {"mcnemar": {"pvalue": 1.0, "gain": 0, "lose": 0}})
    found = rounded_pvalues(tmp_path)
    assert [f.locus for f in found] == ["mcnemar.pvalue"]
    assert "empty discordant table" in found[0].detail


def test_a_genuine_exact_p_of_one_is_not_flagged(tmp_path):
    """gain=11, lose=10: a two-sided exact binomial really is 1.0 there, and the table is not
    degenerate. A screen that cannot tell this apart from gain=lose=0 would be pure noise."""
    _write_summary(tmp_path, {"mcnemar": {"pvalue": 1.0, "gain": 11, "lose": 10}})
    assert rounded_pvalues(tmp_path) == []


def test_a_clamped_adjusted_p_of_one_is_not_flagged(tmp_path):
    """Holm and BH clamp to min(1, ...) by definition, so 1.0 there says nothing about the table.
    The raw p beside it is still screened."""
    _write_summary(tmp_path, {"rows": [{"p_raw": 0.625, "p_bh": 1.0, "p_holm": 1.0}]})
    assert rounded_pvalues(tmp_path) == []


def test_a_clamped_adjusted_p_of_zero_is_still_flagged(tmp_path):
    _write_summary(tmp_path, {"rows": [{"p_raw": 0.4, "p_holm": 0.0}]})
    assert [f.locus for f in rounded_pvalues(tmp_path)] == ["rows.0.p_holm"]


def test_an_ordinary_p_value_is_not_flagged(tmp_path):
    _write_summary(tmp_path, {"a": {"pvalue": 0.0312}, "b": {"pvalue": 0.9999}, "n": 1})
    assert rounded_pvalues(tmp_path) == []
