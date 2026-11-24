"""Paraphrase noise band (experiments/matched_baseline/paraphrase_band.py).

Offline. Pins each committed band to its committed per-case records, checks the paraphrase log and
the cache sizes, and covers the generator-reply parser.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "matched_baseline"))
import matched_baseline_reason as mb  # noqa: E402
import paraphrase_band as pb  # noqa: E402

LANES = [("experiments/blind_metric/results", 40), ("experiments/medmcqa/results", 40),
         ("experiments/support2/results", 31)]


def _rows(path):
    return [json.loads(l) for l in (ROOT / path).read_text().splitlines() if l.strip()]


def test_parser_reads_a_fenced_json_array():
    assert pb.parse_paraphrases('```json\n["a", "b", "c"]\n```', 2) == ["a", "b"]
    assert pb.parse_paraphrases("not json", 5) == []


def test_numbers_check():
    assert pb.numbers_kept("A 67-year-old with pH 7.46", "pH 7.46 in a man aged 67")
    assert not pb.numbers_kept("2 viruses", "two viruses")


def test_text_prompt_with_the_original_question_is_the_committed_control():
    class C:
        question, options, answer_index, report = "Which is it?", ("alpha", "beta"), 0, ""
    assert pb.text_reason_prompt(C(), C.question) == mb.text_prompts(C())[3]
    assert pb.imaging_reason_prompt(pb.imaging_question("effusion")) == mb.imaging_prompts("effusion")[1]


@pytest.mark.parametrize("lane,n", LANES)
def test_committed_band_replays_from_committed_records(lane, n):
    s = json.loads((ROOT / lane / "paraphrase_band_summary.json").read_text())
    recs = _rows(f"{lane}/paraphrase_band.jsonl")
    full = [r for r in recs if r["n_paraphrases"] == s["paraphrases_per_case"]]
    got = mb.summarise_repeats(full, s["paraphrases_per_case"])
    for k in ("n", "cases_with_any_disagreement", "mean_pairwise_disagreement", "flip_vs_baseline_per_read",
              "flip_vs_baseline_band", "decoy_per_read", "decoy_band"):
        assert got[k] == s[k], k
    assert s["n"] == n and len(recs) == n


@pytest.mark.parametrize("lane,n", LANES)
def test_read_zero_is_the_committed_control(lane, n):
    """Read 0 of every paraphrase record equals read 0 of the committed same-session band."""
    ours = {r["case_id"]: r for r in _rows(f"{lane}/paraphrase_band.jsonl")}
    theirs = {r["case_id"]: r for r in _rows(f"{lane}/matched_baseline_reason_repeats.jsonl")}
    assert set(ours) == set(theirs)
    assert all(ours[c]["answers"][0] == theirs[c]["answers"][0] for c in ours)


@pytest.mark.parametrize("lane,n", LANES)
def test_paraphrase_log_and_caches_match_the_summary(lane, n):
    s = json.loads((ROOT / lane / "paraphrase_band_summary.json").read_text())
    paras = _rows(f"{lane}/paraphrases.jsonl")
    assert len(paras) == s["paraphrases_total"] == 5 * n
    hold = len(_rows(f"{lane}/paraphrase_band_cache.jsonl"))
    gen = len(_rows(f"{lane}/paraphrase_band_generator_cache.jsonl"))
    assert s["calls_in_cache"] == {"holdout": hold, "generator": gen}
