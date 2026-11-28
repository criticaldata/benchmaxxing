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


# Second lineages, served locally: the same runner over the Gemini lane's committed paraphrases
# (--generator-cache, --committed-paraphrases-only), holdout = the lineage, read 0 replayed from the lineage's
# committed control cache. A lineage band case whose question has no committed paraphrase is left out and listed.
# nemotron FP8 joins this tuple when its run is committed.
LINEAGE_SLUGS = ("openai_gpt-oss-120b", "Qwen_Qwen2.5-VL-72B-Instruct")
LINEAGE_LANES = [(lane, slug) for lane, _ in LANES for slug in LINEAGE_SLUGS]


@pytest.mark.parametrize("lane,slug", LINEAGE_LANES)
def test_lineage_band_replays_from_its_records(lane, slug):
    d = f"{lane}/{slug}"
    s = json.loads((ROOT / d / "paraphrase_band_summary.json").read_text())
    recs = _rows(f"{d}/paraphrase_band.jsonl")
    full = [r for r in recs if r["n_paraphrases"] == s["paraphrases_per_case"]]
    got = mb.summarise_repeats(full, s["paraphrases_per_case"])
    for k in ("n", "cases_with_any_disagreement", "mean_pairwise_disagreement", "flip_vs_baseline_per_read",
              "flip_vs_baseline_band", "decoy_per_read", "decoy_band"):
        assert got[k] == s[k], k
    assert s["calls_in_cache"]["generator"] == 0 and s["new_api_calls_this_run"]["generator"] == 0


@pytest.mark.parametrize("lane,slug", LINEAGE_LANES)
def test_lineage_read_zero_is_its_committed_control_and_every_case_is_accounted_for(lane, slug):
    d = f"{lane}/{slug}"
    s = json.loads((ROOT / d / "paraphrase_band_summary.json").read_text())
    ours = {r["case_id"]: r for r in _rows(f"{d}/paraphrase_band.jsonl")}
    theirs = {r["case_id"]: r for r in _rows(f"{d}/matched_baseline_reason_repeats.jsonl")}
    left_out = set(s["cases_without_committed_paraphrases"])
    assert set(ours) | left_out == set(theirs) and not set(ours) & left_out
    assert all(ours[c]["answers"][0] == theirs[c]["answers"][0] for c in ours)


@pytest.mark.parametrize("lane,slug", LINEAGE_LANES)
def test_lineage_paraphrases_are_the_committed_gemini_ones(lane, slug):
    gemini = {(p["case_id"], p["i"]): p["paraphrase"] for p in _rows(f"{lane}/paraphrases.jsonl")}
    mine = _rows(f"{lane}/{slug}/paraphrases.jsonl")
    assert mine and all(gemini[(p["case_id"], p["i"])] == p["paraphrase"] for p in mine)
    s = json.loads((ROOT / lane / slug / "paraphrase_band_summary.json").read_text())
    hold = len(_rows(f"{lane}/{slug}/paraphrase_band_cache.jsonl"))
    assert s["calls_in_cache"]["holdout"] == hold == s["paraphrases_total"]


@pytest.mark.parametrize("slug", LINEAGE_SLUGS)
def test_support2_lineage_band_replays_keyless(slug, tmp_path):
    """SUPPORT2's manifest is committed, so this lane's lineage bands rerun here with no server and no key."""
    import os
    import shutil
    import subprocess
    lane = ROOT / "experiments/support2/results"
    for f in ("paraphrase_band_cache.jsonl", "matched_baseline_reason_repeats_summary.json"):
        shutil.copy(lane / slug / f, tmp_path / f)
    model = json.loads((lane / slug / "paraphrase_band_summary.json").read_text())["model"]
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "experiments/matched_baseline/paraphrase_band.py", "--lane", "text",
                        "--manifest", "experiments/support2/results/support2_manifest.csv", "--model", model,
                        "--cache", f"experiments/support2/results/{slug}_blind_metric_call_cache.jsonl",
                        "--out", str(tmp_path), "--generator-cache",
                        "experiments/support2/results/paraphrase_band_generator_cache.jsonl",
                        "--committed-paraphrases-only"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
    got = json.loads((tmp_path / "paraphrase_band_summary.json").read_text())
    want = json.loads((lane / slug / "paraphrase_band_summary.json").read_text())
    assert got["new_api_calls_this_run"] == {"holdout": 0, "generator": 0}
    assert {k: v for k, v in got.items() if k != "new_api_calls_this_run"} == \
        {k: v for k, v in want.items() if k != "new_api_calls_this_run"}
    assert (tmp_path / "paraphrase_band.jsonl").read_bytes() == (lane / slug / "paraphrase_band.jsonl").read_bytes()
