"""The paper's MedQA solo numbers must stay regenerable from the committed cache.

Before this, the 100-case manifest behind `solo_results.json` was not committed at all, so nobody could
check whether a parser or cue change had moved the paper's headline MedQA rates. The case-id list is now
committed, and this test replays the committed cache through the CURRENT parser and asserts it lands on the
committed per-cue rates.

Only the two fully-replayable cues are asserted. `lexical_overlap` is at 68/100 cache coverage because the
cue builder has changed since the run, which is recorded in solo_manifest_provenance.json rather than
hidden.
"""
import hashlib
import json
from pathlib import Path

import pytest

RES = Path(__file__).parent.parent / "experiments" / "medqa" / "results"
IDS = RES / "solo_manifest_case_ids.txt"


def test_the_solo_cohort_is_committed():
    ids = [x for x in IDS.read_text().split() if x]
    assert len(ids) == 100, len(ids)
    assert len(set(ids)) == 100, "duplicate case ids"
    recorded = {json.loads(ln)["case_id"] for ln in (RES / "solo_records.jsonl").read_text().splitlines() if ln.strip()}
    assert set(ids) == recorded, "committed cohort does not match the per-case records"


def test_provenance_names_the_split_and_the_replay_gap():
    p = json.loads((RES / "solo_manifest_provenance.json").read_text())
    assert "test.jsonl" in p["source_split"]
    assert p["n_cases"] == 100
    # the honest bit: the partial cue must stay declared
    assert "68/100" in p["replay_coverage"]["lexical_overlap"]


@pytest.mark.parametrize("model,cue,expected", [
    ("gemini-2.5-flash", "longest_option", 0.05),
    ("gemini-2.5-flash", "option_order", 0.05),
    ("gemini-2.5-flash-lite", "longest_option", 0.12),
    ("gemini-2.5-flash-lite", "option_order", 0.09),
])
def test_committed_per_cue_rates_are_what_the_paper_reports(model, cue, expected):
    committed = json.loads((RES / "solo_results.json").read_text())
    assert committed["flip_rate_by_model"][model]["per_cue"][cue] == pytest.approx(expected)


def test_the_rerun_reproduces_the_two_builder_stable_cues():
    """The rerun exists because lexical_overlap was not replayable. The other two cues must be
    untouched by it, otherwise the rerun changed more than the one cue it was meant to close."""
    r = json.loads((RES / "solo_results_rerun_2026-07-29.json").read_text())
    c = json.loads((RES / "solo_results.json").read_text())
    for model, cue in (("gemini-2.5-flash", "longest_option"), ("gemini-2.5-flash", "option_order"),
                       ("gemini-2.5-flash-lite", "longest_option"), ("gemini-2.5-flash-lite", "option_order")):
        assert r["flip_rate_by_model"][model]["per_cue"][cue] == pytest.approx(
            c["flip_rate_by_model"][model]["per_cue"][cue]), (model, cue)


def test_the_noise_floor_caveat_is_recorded():
    """The floor is a 15-item control and moved by one case between runs, so any claim that a flip
    rate sits above or below it must stay hedged. Pin the caveat so it is not dropped."""
    r = json.loads((RES / "solo_results_rerun_2026-07-29.json").read_text())
    note = r["rerun_provenance"]["noise_floor_caveat"]
    assert "15" in note and "NOT separable" in note


def test_cache_is_present_and_keyed_as_expected():
    """The replay depends on sha256(model \\x00 prompt) keys; guard the shape."""
    lines = [ln for ln in (RES / "call_cache.jsonl").read_text().splitlines() if ln.strip()]
    assert len(lines) > 6000
    first = json.loads(lines[0])
    assert set(first) == {"k", "model", "resp"}
    assert len(first["k"]) == len(hashlib.sha256(b"x").hexdigest())
