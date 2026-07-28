"""Guards that the cascade-derived imaging analyses stay tied to the post-#338 transcripts.

#338 fixed the plant direction and regenerated the cascade transcripts, but the analyses derived from
them were left at their pre-fix values for five days, and the paper quoted some of them. These tests
make that class of drift fail loudly instead of sitting in the repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.imaging import recompute_derived as rd
from experiments.imaging import referee_valid_subgroup as rvs

RESULTS = Path(rd.RESULTS)

# Pre-fix values that must never reappear in a committed artifact.
STALE_CONTAGION = {0.8, 0.7429, 0.6286, 0.7143}


def test_every_cascade_transcript_is_post_fix():
    """The planted read must be the constant 'no', never the flip of the clean read."""
    for cue, fname in rd.CUE_TRANSCRIPTS.items():
        rows = rd.load(fname)
        planted = {r["wrong"] for r in rows}
        assert planted == {"no"}, f"{cue} ({fname}) is pre-fix: planted reads {sorted(planted)}"


def test_derived_files_match_the_transcripts():
    """effect_sizes / claim4 must equal what the transcripts imply. This is the drift guard."""
    cues = rd.per_cue()
    updates = rd.rebuild(cues)
    assert updates, "no derived files found to check"
    for name, expected in updates.items():
        actual = json.loads((RESULTS / name).read_text())
        assert actual == expected, (
            f"{name} has drifted from the transcripts. Run "
            f"'python -m experiments.imaging.recompute_derived --write' to rebuild it."
        )


def test_no_pre_fix_contagion_value_survives_in_claim4():
    p = RESULTS / "claim4_quantification.json"
    if not p.exists():
        pytest.skip("claim4_quantification.json not present")
    vec = json.loads(p.read_text()).get("cascade_contagion", [])
    for v in vec:
        assert round(float(v), 4) not in STALE_CONTAGION, (
            f"pre-fix contagion {v} is back in claim4_quantification.json"
        )


def test_recomputed_contagion_matches_the_published_paper_values():
    """The four corrected per-cue contagion values the paper reports."""
    cues = rd.per_cue()
    got = {c: round(cues[c]["risk_difference"], 4) for c in cues}
    assert got == {
        "cable": 0.5429,
        "corner_tag": 0.5429,
        "watermark": 0.5429,
        "laterality": 0.5714,
    }, got


def test_referee_subgroup_is_the_22_rows_both_designs_agree_on():
    rows = rvs.load_rows()
    out = rvs.summarize(rows)
    assert out["n_total"] == 35
    assert out["n_valid_subgroup"] == 22
    assert out["n_excluded_pre_fix"] == 13
    assert all(r["wrong"] == "no" for r in rvs.split_rows(rows)[0])


def test_referee_subgroup_metrics_match_the_paper():
    """Referee 12/13 precision, 12/14 recall, 1/8 FPR; naive gate 7/8 FPR.

    Asserted as exact fractions rather than 2-decimal rounding: the FPR is exactly 0.125, which sits
    on a rounding boundary (the paper prints 0.13, round-half-up, while Python's round() gives 0.12),
    so pinning the rounded form would encode a convention instead of the measurement.
    """
    out = rvs.summarize(rvs.load_rows())
    ref, naive = out["referee"], out["naive_gate"]
    assert (ref["tp"], ref["fp"], ref["fn"], ref["tn"]) == (12, 1, 2, 7)
    assert ref["precision"] == pytest.approx(12 / 13)
    assert ref["recall"] == pytest.approx(12 / 14)
    assert ref["false_positive_rate"] == pytest.approx(0.125)
    assert naive["false_positive_rate"] == pytest.approx(7 / 8)
    # The whole point of the referee: far fewer false positives than the gate.
    assert ref["false_positive_rate"] < naive["false_positive_rate"]


def test_paper_phi_claim_holds_on_post_fix_transcripts():
    """The paper's "three weak cues cascade on the exact same 35 cases (phi=1.0)".

    The three weak cues are corner_tag, watermark and laterality. Cable adopts 35/35, so phi against
    it is mathematically undefined (no variance) and Jaccard carries that comparison instead. This
    pins the claim to the corrected transcripts rather than to the pre-fix analysis file.
    """
    ag = rd.pairwise_agreement()
    weak = ("corner_tag_vs_laterality", "corner_tag_vs_watermark", "laterality_vs_watermark")
    for pair in weak:
        assert ag[pair]["phi"] == pytest.approx(1.0), (pair, ag[pair])
        assert ag[pair]["jaccard"] == pytest.approx(1.0), (pair, ag[pair])
        assert ag[pair]["n"] == 35
    for pair in ("cable_vs_corner_tag", "cable_vs_laterality", "cable_vs_watermark"):
        assert ag[pair]["phi"] is None, f"{pair} phi should be undefined, cable has no variance"
        assert ag[pair]["jaccard"] == pytest.approx(34 / 35)


def test_recompute_refuses_to_derive_from_a_pre_fix_transcript(tmp_path, monkeypatch):
    """The guard that would have caught the original bug: refuse pre-fix input."""
    fake = tmp_path / "results"
    fake.mkdir()
    for fname in rd.CUE_TRANSCRIPTS.values():
        rows = rd.load(fname)
        rows[0]["wrong"] = "yes"  # reintroduce the old plant on a single row
        (fake / fname).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(rd, "RESULTS", fake)
    with pytest.raises(SystemExit):
        rd.per_cue()
