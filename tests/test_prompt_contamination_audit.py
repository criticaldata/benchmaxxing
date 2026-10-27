"""#420 exclusion audit: headlines recompute from committed rows with the contaminated cases dropped."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import prompt_contamination_audit as pca  # noqa: E402


def test_committed_audit_matches_a_fresh_build():
    built = pca.build()
    committed = json.loads((ROOT / "experiments/prompt_contamination_audit.json").read_text())
    assert committed == built


def test_gemini_multi_round_exclusion_does_not_change_the_shape():
    arm = pca.build()["arms"][0]
    assert arm["excluded"] == ["medqa-38"]
    assert arm["excluded_adopted_isolated"] == [[False] * 5]
    # Denominator 39 instead of 40; the case never adopted, so the curve stays flat-then-tiny.
    assert arm["excluded_view"]["isolated_adoption_by_round"][0] == 0.0
    assert arm["committed"]["isolated_adoption_by_round"][1] == 0.025


def test_seed_timing_exclusions_are_the_posted_cases():
    report = pca.build()
    gemini, oss20 = report["arms"][1], report["arms"][2]
    assert gemini["excluded"] == ["medqa-38", "medqa-65"]
    assert oss20["excluded"] == ["medqa-3", "medqa-44", "medqa-91", "medqa-111"]
    assert gemini["excluded_view"]["holdout_first"] == 0.1356
    assert oss20["excluded_view"]["holdout_first"] == 0.0259
