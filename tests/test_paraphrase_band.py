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
LINEAGE_SLUGS = ("openai_gpt-oss-120b", "nemotron-3-super-120b-a12b-fp8-local", "Qwen_Qwen2.5-VL-72B-Instruct")
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
    for f in ("paraphrase_band_cache.jsonl", "matched_baseline_reason_repeats_summary.json",
              "matched_baseline_reason_repeats.jsonl"):
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


# Bands without read 0. Read 0 is replayed from the committed control cache, which for a locally served
# lineage was filled in an earlier serving session than reads 1 to 5. ``without_read_0`` gives the band
# over the paraphrase reads alone, and the committed same-prompt repeats alone on the same cases.
FOLDERS = [lane for lane, _ in LANES] + [f"{lane}/{slug}" for lane, slug in LINEAGE_LANES]
BAND_FIELDS = ("n", "cases_with_any_disagreement", "mean_pairwise_disagreement", "flip_vs_baseline_per_read",
               "flip_vs_baseline_band", "decoy_per_read", "decoy_band")


def test_band_of_reads_keeps_only_the_chosen_reads():
    # baseline answer A, decoy B: c1 moves to the decoy on its last read, c3 starts on it
    recs = [mb._repeat_record(cid, answers[0], "A", "B", answers[1:])
            for cid, answers in (("c1", ["A", "A", "B"]), ("c2", ["A", "A", "A"]), ("c3", ["B", "A", "A"]))]
    every = pb.band_of_reads(recs, lambda r: range(3))
    want = mb.summarise_repeats(recs, 2)
    assert {k: every[k] for k in BAND_FIELDS} == {k: want[k] for k in BAND_FIELDS}
    assert every["cases_with_any_disagreement"]["n"] == 2 and every["cases_per_read"] == [3, 3, 3]

    keep = {"c1": [0, 1], "c2": [0, 1, 2], "c3": [0]}
    part = pb.band_of_reads(recs, lambda r: keep[r["case_id"]])
    # c1 loses the read that disagreed, c3 keeps one read and cannot disagree with itself
    assert part["n"] == 2 and part["cases_with_any_disagreement"]["n"] == 0
    assert part["cases_with_fewer_than_two_reads"] == ["c3"]
    assert part["reads"] == [0, 1, 2] and part["cases_per_read"] == [3, 2, 1]
    assert part["flip_vs_baseline_per_read"] == [0.3333, 0.0, 0.0]
    assert part["flip_vs_baseline_band"] == [0.0, 0.3333] and part["decoy_per_read"] == [0.3333, 0.0, 0.0]

    tail = pb.band_of_reads(recs, lambda r: [1, 2])
    assert tail["reads"] == [1, 2] and tail["cases_with_any_disagreement"]["n"] == 1
    assert tail["flip_vs_baseline_per_read"] == [0.0, 0.3333]


@pytest.mark.parametrize("folder", FOLDERS)
def test_band_of_reads_over_every_read_is_the_committed_band(folder):
    s = json.loads((ROOT / folder / "paraphrase_band_summary.json").read_text())
    got = pb.band_of_reads(_rows(f"{folder}/paraphrase_band.jsonl"), lambda r: range(len(r["answers"])))
    assert {k: got[k] for k in BAND_FIELDS} == {k: s[k] for k in BAND_FIELDS}


@pytest.mark.parametrize("folder", FOLDERS)
def test_bands_without_read_0_recount_from_the_committed_rows(folder):
    s = json.loads((ROOT / folder / "paraphrase_band_summary.json").read_text())
    recs = _rows(f"{folder}/paraphrase_band.jsonl")
    repeats = _rows(f"{folder}/matched_baseline_reason_repeats.jsonl")
    block = s["without_read_0"]
    assert block == pb.without_read_0(recs, s["paraphrases_per_case"], repeats)

    para = block["paraphrase_reads"]
    assert para["reads"] == [1, 2, 3, 4, 5] and para["n"] == s["n"]
    assert para["flip_vs_baseline_per_read"] == s["flip_vs_baseline_per_read"][1:]
    assert para["flip_vs_baseline_band"] == [min(s["flip_vs_baseline_per_read"][1:]), max(s["flip_vs_baseline_per_read"][1:])]
    assert para["decoy_band"] == [min(s["decoy_per_read"][1:]), max(s["decoy_per_read"][1:])]
    assert para["cases_with_any_disagreement"]["n"] == sum(len(set(r["answers"][1:])) > 1 for r in recs)

    # the same-prompt repeats are restricted to the cases of this band, so both sides have the same cases
    mine = {r["case_id"] for r in recs}
    same = [r for r in repeats if r["case_id"] in mine]
    rep = block["same_session_repeats"]
    assert rep["n"] == len(same) == s["n"] and rep["reads"] == [1, 2, 3, 4, 5]
    assert rep["cases_with_any_disagreement"]["n"] == sum(len(set(r["answers"][1:])) > 1 for r in same)
    if len(same) == len(repeats):
        committed = json.loads((ROOT / folder / "matched_baseline_reason_repeats_summary.json").read_text())
        assert rep["flip_vs_baseline_per_read"] == committed["flip_vs_baseline_per_read"][1:]
        assert rep["decoy_per_read"] == committed["decoy_per_read"][1:]


def test_any_disagreement_without_read_0_on_every_lane_and_lineage():
    """Cases with any disagreement over reads 1 to 5: paraphrase reads, same-prompt repeats, cases."""
    want = {
        "experiments/blind_metric/results": (12, 0, 40),
        "experiments/blind_metric/results/openai_gpt-oss-120b": (7, 3, 40),
        "experiments/blind_metric/results/nemotron-3-super-120b-a12b-fp8-local": (7, 5, 40),
        "experiments/blind_metric/results/Qwen_Qwen2.5-VL-72B-Instruct": (9, 0, 40),
        "experiments/medmcqa/results": (17, 0, 40),
        "experiments/medmcqa/results/openai_gpt-oss-120b": (8, 5, 40),
        "experiments/medmcqa/results/nemotron-3-super-120b-a12b-fp8-local": (11, 8, 40),
        "experiments/medmcqa/results/Qwen_Qwen2.5-VL-72B-Instruct": (12, 0, 40),
        "experiments/support2/results": (4, 0, 31),
        "experiments/support2/results/openai_gpt-oss-120b": (1, 1, 26),
        "experiments/support2/results/nemotron-3-super-120b-a12b-fp8-local": (2, 0, 26),
        "experiments/support2/results/Qwen_Qwen2.5-VL-72B-Instruct": (6, 0, 31),
    }
    assert set(want) == set(FOLDERS)
    for folder, (paraphrases, repeats, n) in want.items():
        block = json.loads((ROOT / folder / "paraphrase_band_summary.json").read_text())["without_read_0"]
        got = (block["paraphrase_reads"]["cases_with_any_disagreement"]["n"],
               block["same_session_repeats"]["cases_with_any_disagreement"]["n"], block["paraphrase_reads"]["n"])
        assert got == (paraphrases, repeats, n), folder
