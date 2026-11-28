"""referee_noise_logprobs.py: the committed rows replay from their caches with no server, and the helpers
behave on hand-built inputs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "referee"))

import referee_noise_logprobs as rn  # noqa: E402

MANIFEST = Path(os.environ.get("BENCHMAXXING_MEDQA_MANIFEST", str(Path.home() / "Documents/DATASET/medqa/manifest_test.csv")))
LINEAGES = {"openai/gpt-oss-120b": "openai_gpt-oss-120b", "openai/gpt-oss-20b": "openai_gpt-oss-20b",
            "nemotron-3-super-120b-a12b-fp8-local": "nemotron-3-super-120b-a12b-fp8-local",
            "Qwen/Qwen2.5-VL-72B-Instruct": "Qwen_Qwen2.5-VL-72B-Instruct"}
VOLATILE = {"new_api_calls_this_run", "calls_by_kind"}
RESULTS = ROOT / "experiments/referee/results"


def _summary(slug):
    return json.loads((RESULTS / slug / "referee_noise_logprobs_summary.json").read_text())


def _rows(slug):
    return [json.loads(line) for line in (RESULTS / slug / "referee_noise_logprobs.jsonl").read_text().splitlines() if line]


def test_letter_distribution_reads_the_last_letter_token():
    rec = {"letter_tokens": [
        {"token": "A", "logprob": 0.0, "top": [["A", 0.0]]},
        {"token": "B", "logprob": -0.1, "top": [["B", -0.1053605], [" C", -2.302585], ["x", -1.0]]}]}
    d = rn._letter_dist(rec, ["A", "B", "C"])
    assert d["A"] == 0.0 and d["B"] == pytest.approx(0.9, abs=1e-3) and d["C"] == pytest.approx(0.1, abs=1e-3)
    assert rn._letter_dist({"letter_tokens": []}, ["A"]) is None


def test_sign_test_is_two_sided_and_exact():
    assert rn._sign_test(0, 0) is None
    assert rn._sign_test(2, 0) == 0.5
    assert rn._sign_test(5, 5) == 1.0


# ---------------------------------------------------------------- draws: all cases, and the adopted ones
def test_per_case_exact_test_with_five_draws_a_side():
    assert [rn.planted_fisher_p(b, a, 5) for b, a in ((5, 0), (4, 0), (5, 1), (3, 0), (5, 3), (1, 0), (0, 0))] == [
        0.007937, 0.047619, 0.047619, 0.166667, 0.444444, 1.0, 1.0]


def test_paired_all_cases_counts_every_case_and_tests_the_ones_that_differ():
    t = rn.paired_all_cases([(5, 0), (1, 0), (0, 1), (0, 0)], 5)
    assert (t["gain"], t["lose"], t["ties"]) == (2, 1, 1)
    assert (t["board_planted_draws"], t["bare_planted_draws"], t["draws_per_side"]) == (6, 1, 20)
    assert (t["board_rate"], t["bare_rate"]) == (0.3, 0.05)
    assert t["sign_test_p"] == 1.0
    # sign flips of (1.0, 0.2, -0.2): six of the eight sums are at least as far from zero as the observed 1.0
    assert t["permutation_p"] == 0.75
    tied = rn.paired_all_cases([(0, 0), (2, 2)], 5)
    assert (tied["gain"], tied["lose"], tied["sign_test_p"], tied["permutation_p"]) == (0, 0, None, None)


def _row(case_id, board, bare, adopted, k=5):
    return {"case_id": case_id, "adopted_t0": adopted, "board_draws_planted": board, "bare_draws_planted": bare,
            "board_rate": board / k, "bare_rate": bare / k, "board_rate_above_bare": board > bare,
            "planted_fisher_p": rn.planted_fisher_p(board, bare, k), "prob_planted_bare": 0.0, "prob_planted_board": 0.0,
            "margin_bare": 1.0, "margin_board": 1.0,
            "t0_live_bare_matches_cache": True, "t0_live_board_matches_cache": True}


def test_summary_counts_an_empty_bare_draw_as_planted_in_the_worst_case():
    rows = [_row("a", 5, 0, True), _row("b", 4, 0, True), _row("c", 0, 0, False), _row("d", 1, 0, False)]
    s = rn.summarise(rows, 5, {"bare_draws": {"a": 3, "c": 1}, "board_draws": {}})
    assert s["adopted_t0_board_rate_above_bare"] == s["n_adopted_t0"] == 2
    assert s["adopted_t0_tests"]["fisher_below_alpha"] == 2
    # a: 5 against 0 + 3 empty no longer separates; b has no empty draw and still does
    assert s["adopted_t0_tests"]["fisher_below_alpha_empty_bare_as_planted"] == 1
    assert [c["bare_draws_empty"] for c in s["adopted_t0_tests"]["cases"]] == [3, 0]
    free = s["selection_free"]
    assert (free["all_cases"]["gain"], free["all_cases"]["lose"], free["all_cases"]["bare_planted_draws"]) == (3, 0, 0)
    worst = free["all_cases_empty_bare_as_planted"]
    assert (worst["gain"], worst["lose"], worst["bare_planted_draws"]) == (3, 1, 4)


# all 40 cases: more planted draws on the board, fewer, exact sign test p, paired permutation p
ALL_CASES = {"openai_gpt-oss-120b": (6, 1, 0.125, 0.078125), "openai_gpt-oss-20b": (7, 1, 0.070312, 0.054688),
             "nemotron-3-super-120b-a12b-fp8-local": (4, 0, 0.125, 0.125),
             "Qwen_Qwen2.5-VL-72B-Instruct": (11, 0, 0.000977, 0.000977)}
# temperature-0 adoptions: how many, with a higher planted share on the board, passing the per-case exact test,
# and passing it with every empty bare draw counted as planted
ADOPTED = {"openai_gpt-oss-120b": (3, 3, 1, 1), "openai_gpt-oss-20b": (3, 3, 1, 1),
           "nemotron-3-super-120b-a12b-fp8-local": (2, 2, 1, 0), "Qwen_Qwen2.5-VL-72B-Instruct": (10, 10, 6, 6)}


@pytest.mark.parametrize("slug", sorted(LINEAGES.values()))
def test_all_cases_comparison_is_recounted_from_the_rows(slug):
    rows, s = _rows(slug), _summary(slug)
    t = s["selection_free"]["all_cases"]
    gain = sum(1 for r in rows if r["board_draws_planted"] > r["bare_draws_planted"])
    lose = sum(1 for r in rows if r["board_draws_planted"] < r["bare_draws_planted"])
    assert (t["gain"], t["lose"], t["sign_test_p"], t["permutation_p"]) == (gain, lose, *ALL_CASES[slug][2:]) == ALL_CASES[slug]
    assert t["gain"] + t["lose"] + t["ties"] == len(rows) == 40 and t["draws_per_side"] == 200
    assert t["board_planted_draws"] == sum(r["board_draws_planted"] for r in rows)
    assert t["bare_planted_draws"] == sum(r["bare_draws_planted"] for r in rows)
    assert t == rn.paired_all_cases([(r["board_draws_planted"], r["bare_draws_planted"]) for r in rows], s["k"])


@pytest.mark.parametrize("slug", sorted(LINEAGES.values()))
def test_adopted_cases_all_sit_above_their_bare_rate_and_few_pass_an_exact_test(slug):
    """The selected-case check passes on every adopted case, which is what selecting them predicts; the
    per-case exact test on the same draws passes on far fewer."""
    rows, s = _rows(slug), _summary(slug)
    adopted = [r for r in rows if r["adopted_t0"]]
    tests = s["adopted_t0_tests"]
    assert (s["n_adopted_t0"], s["adopted_t0_board_rate_above_bare"], tests["fisher_below_alpha"],
            tests["fisher_below_alpha_empty_bare_as_planted"]) == ADOPTED[slug]
    assert len(adopted) == s["n_adopted_t0"] and all(r["board_rate_above_bare"] for r in adopted)
    assert tests["fisher_below_alpha"] == sum(1 for r in adopted if r["planted_fisher_p"] < tests["alpha"])
    for r in rows:
        assert r["board_rate_above_bare"] == (r["board_rate"] > r["bare_rate"])
        assert r["planted_fisher_p"] == rn.planted_fisher_p(r["board_draws_planted"], r["bare_draws_planted"], s["k"])


def test_empty_replies_are_listed_by_case_and_change_the_worst_case_only_where_they_are():
    empty = {slug: _summary(slug)["empty_replies"] for slug in LINEAGES.values()}
    assert all(set(e) == set(rn.EMPTY_KINDS) for e in empty.values())
    assert not any(empty["openai_gpt-oss-120b"].values()) and not any(empty["Qwen_Qwen2.5-VL-72B-Instruct"].values())
    # gpt-oss-20b: one empty bare draw among those replayed from the earlier arm, on one of its adopted cases
    assert {k: v for k, v in empty["openai_gpt-oss-20b"].items() if v} == {
        "bare_draws": {"medqa-31": 1}, "committee_t0_bare": {"medqa-34": 1}}
    assert empty["nemotron-3-super-120b-a12b-fp8-local"] == {
        "bare_draws": {"medqa-15": 1, "medqa-23": 1, "medqa-3": 1, "medqa-31": 3, "medqa-9": 1},
        "board_draws": {"medqa-23": 1}, "logprob_t0_bare": {}, "logprob_t0_board": {"medqa-9": 1},
        "committee_t0_bare": {"medqa-31": 1}, "committee_t0_board": {"medqa-9": 1}}
    for slug in LINEAGES.values():
        free = _summary(slug)["selection_free"]
        n_empty = sum(empty[slug]["bare_draws"].values())
        assert free["all_cases_empty_bare_as_planted"]["bare_planted_draws"] == free["all_cases"]["bare_planted_draws"] + n_empty
        if not n_empty:
            assert free["all_cases_empty_bare_as_planted"] == free["all_cases"]
    worst = _summary("nemotron-3-super-120b-a12b-fp8-local")["selection_free"]["all_cases_empty_bare_as_planted"]
    assert (worst["gain"], worst["lose"], worst["bare_planted_draws"], worst["sign_test_p"]) == (4, 2, 8, 0.6875)


# ---------------------------------------------------------------- replay of the committed runs
@pytest.mark.skipif(not MANIFEST.exists(), reason="the MedQA manifest is not on this machine")
@pytest.mark.parametrize("model", sorted(LINEAGES))
def test_committed_rows_replay_with_no_server(model):
    d = ROOT / "experiments/referee/results" / LINEAGES[model]
    files = {p: p.read_bytes() for p in (d / "referee_noise_logprobs_summary.json", d / "referee_noise_logprobs.jsonl")}
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    try:
        r = subprocess.run([sys.executable, "experiments/referee/referee_noise_logprobs.py", "--manifest", str(MANIFEST),
                            "--model", model], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
        after = json.loads((d / "referee_noise_logprobs_summary.json").read_text())
        before = json.loads(files[d / "referee_noise_logprobs_summary.json"])
        assert after["new_api_calls_this_run"] == 0
        assert {k: v for k, v in after.items() if k not in VOLATILE} == {k: v for k, v in before.items() if k not in VOLATILE}
        assert (d / "referee_noise_logprobs.jsonl").read_bytes() == files[d / "referee_noise_logprobs.jsonl"]
    finally:
        for p, b in files.items():
            p.write_bytes(b)
