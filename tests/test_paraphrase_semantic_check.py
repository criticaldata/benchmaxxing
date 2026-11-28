"""Semantic check of the committed paraphrases (experiments/matched_baseline/paraphrase_semantic_check.py).

Offline. Covers the keyless rule check, the judge prompt and its reply parser, and the bands with
failed paraphrases left out, on small hand-built inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "matched_baseline"))
import matched_baseline_reason as mb  # noqa: E402
import paraphrase_semantic_check as sc  # noqa: E402

EXCEPT = "All of the following are true regarding Class-II inlay gingival margins preparation except?"


def test_a_faithful_rewording_of_an_exception_passes():
    """The rule compares presence of a marker, not the word: these rewordings keep the exception."""
    for reworded in (
        "Regarding Class-II inlay gingival margins preparation, all of these are true, with one exception?",
        "Which statement is not true concerning Class-II inlay gingival margins preparation?",
        "All of these hold for Class-II inlay gingival margins preparation, save for one. Which?",
        "Apart from one, every statement below about Class-II inlay gingival margins preparation is true?",
        "Which of these is false about Class-II inlay gingival margins preparation?",
    ):
        got = sc.rule_check(EXCEPT, reworded)
        assert got["rule_check"] and got["rule_reason"] == "kept", reworded
    assert sc.rule_check(EXCEPT, EXCEPT)["markers_original"] == ["except"]


def test_a_dropped_or_added_negation_fails():
    dropped = sc.rule_check(EXCEPT, "Which of the following is true of Class-II inlay gingival margins preparation?")
    assert (dropped["rule_check"], dropped["rule_reason"], dropped["markers_paraphrase"]) == (False, "dropped", [])
    least = sc.rule_check("Which of the following is the least likely diagnosis?", "Which diagnosis is most likely?")
    assert (least["rule_check"], least["rule_reason"], least["markers_original"]) == (False, "dropped", ["least"])
    added = sc.rule_check("Which drug is indicated?", "Which drug is not indicated?")
    assert (added["rule_check"], added["rule_reason"], added["markers_paraphrase"]) == (False, "added", ["not"])
    contraction = sc.rule_check("Which finding isn't expected?", "Which finding is expected?")
    assert (contraction["rule_reason"], contraction["markers_original"]) == ("dropped", ["n't"])
    plain = sc.rule_check("Drug of choice for X is", "The preferred drug for X is")
    assert (plain["rule_check"], plain["rule_reason"]) == (True, "none")


def test_only_the_question_part_is_read():
    """Negations among the facts of a vignette are not what the rule compares; the judge reads those."""
    original = ("A 45-year-old man has no fever. His pH is 7.4 and he does not smoke. "
                "Which of the following is the most likely diagnosis?")
    assert sc.question_part(original) == "Which of the following is the most likely diagnosis?"
    assert sc.markers(original) == []
    assert sc.rule_check(original, "An afebrile man aged 45 has a pH of 7.4. What diagnosis is most likely?")["rule_check"]
    # a decimal point is not a sentence end
    assert sc.question_part("The pH is 7.4. Which value of 7.4 is not normal?") == "Which value of 7.4 is not normal?"
    # no question mark: the last sentence
    assert sc.question_part("Regarding X. All are true except") == "All are true except"
    record = "Patient record:\n- diabetes: no\n- dementia: no\n\nWill this patient survive to hospital discharge?"
    assert sc.markers(record) == [] and sc.question_part(record) == "Will this patient survive to hospital discharge?"
    assert sc.markers("Is it true that X is other than Y, or neither?") == ["other than", "neither"]
    # a question of fewer than five words is read with the sentence before it
    assert sc.question_part("He has no fever. All of these hold, save for one. Which?") == \
        "All of these hold, save for one. Which?"
    assert sc.question_part("One is false. Which one is the false one?") == "Which one is the false one?"


def test_judge_prompt_shows_both_versions_and_the_options_once():
    prompt = sc.judge_prompt("ORIGINAL TEXT?", "REWORDED TEXT?", ["alpha", "beta"])
    assert "ORIGINAL:\nORIGINAL TEXT?\n\nREWORDED:\nREWORDED TEXT?\n\nOPTIONS:\nA. alpha\nB. beta" in prompt
    assert "VERDICT: yes or no" in prompt and "REASON: one short sentence" in prompt
    assert prompt.count("alpha") == 1


def test_judge_reply_parser():
    assert sc.parse_judge("VERDICT: yes\nREASON: Same facts and question.") == ("yes", "Same facts and question.")
    assert sc.parse_judge("**VERDICT:** No\n**REASON:** The exception is dropped.") == ("no", "The exception is dropped.")
    # no verdict line is not a yes
    assert sc.parse_judge("They look equivalent to me.") == (None, "")
    assert sc.parse_judge("") == (None, "") and sc.parse_judge(None) == (None, "")


def _records():
    """Baseline A, decoy B, two paraphrases per case: c1 moves on read 2, c2 never, c3 on both paraphrases."""
    recs = []
    for cid, answers in (("c1", ["A", "A", "B"]), ("c2", ["A", "A", "A"]), ("c3", ["A", "B", "B"])):
        rec = mb._repeat_record(cid, answers[0], "A", "B", answers[1:])
        rec["n_paraphrases"] = 2
        recs.append(rec)
    return recs


def _checks(failed_rule=(), failed_judge=()):
    out = {}
    for cid in ("c1", "c2", "c3"):
        for i in (1, 2):
            rule, judge = (cid, i) not in failed_rule, (cid, i) not in failed_judge
            out[(cid, i)] = {"rule": rule, "judge": judge, "either": rule and judge}
    return out


def test_bands_leave_out_the_reads_of_failed_paraphrases():
    bands = sc.lineage_bands(_records(), _checks(failed_rule=[("c1", 2)], failed_judge=[("c3", 1)]), 2)
    assert bands["cases"] == 3 and bands["paraphrase_reads"] == 6
    assert bands["reads_left_out"] == {"rule": 1, "judge": 1, "either": 2}

    def any_dis(name, reads="reads_0_to_k"):
        return bands[name][reads]["cases_with_any_disagreement"]["n"], bands[name][reads]["n"]

    assert any_dis("all_paraphrases") == (2, 3)
    # c1 loses the read that disagreed; c3 still has read 2 against read 0
    assert any_dis("failed_left_out") == (1, 3)
    assert any_dis("rule_failed_left_out") == (1, 3) and any_dis("judge_failed_left_out") == (2, 3)
    # without read 0, c1 and c3 keep a single paraphrase read each and cannot disagree with themselves
    assert any_dis("failed_left_out", "reads_1_to_k") == (0, 1)
    assert bands["failed_left_out"]["reads_1_to_k"]["cases_with_fewer_than_two_reads"] == ["c1", "c3"]
    assert bands["failed_left_out"]["reads_0_to_k"]["cases_per_read"] == [3, 2, 2]
    clean = bands["cases_with_every_paraphrase_passing"]
    assert clean["n"] == 1 and clean["reads_0_to_k"]["cases_with_any_disagreement"]["n"] == 0


def test_with_nothing_failed_the_bands_are_the_committed_band():
    recs = _records()
    bands = sc.lineage_bands(recs, _checks(), 2)
    want = mb.summarise_repeats(recs, 2)
    for name in ("all_paraphrases", "failed_left_out", "cases_with_every_paraphrase_passing"):
        got = bands[name]["reads_0_to_k"]
        for k in ("n", "cases_with_any_disagreement", "mean_pairwise_disagreement", "flip_vs_baseline_band", "decoy_band"):
            assert got[k] == want[k], (name, k)
    assert bands["reads_left_out"] == {"rule": 0, "judge": 0, "either": 0}


def test_reverse_question_turns_the_first_reversible_marker_around():
    """The control: a paraphrase turned around by rule asks the opposite question."""
    want = {
        "Which statement among the following is not true concerning X?":
            ("Which statement among the following is true concerning X?", "not"),
        "Regarding X, all of the following statements are true, with one exception?":
            ("Regarding X, all of the following statements are true, with no exception?", "exception"),
        "All of the following are true regarding X except?": ("All of the following are true regarding X?", "except"),
        "Which of the following is NOT a feature of X?": ("Which of the following is a feature of X?", "not"),
        "Which of these is false about X?": ("Which of these is true about X?", "false"),
        "Which is the Least likely diagnosis?": ("Which is the Most likely diagnosis?", "least"),
        # only the question part is edited: the negated fact before it stays
        "A man has no fever. Which finding isn't expected?": ("A man has no fever. Which finding is expected?", "n't"),
    }
    for text, (reversed_text, marker) in want.items():
        got = sc.reverse_question(text)
        assert (got["text"], got["marker"]) == (reversed_text, marker), text
    # a marker with no entry in the table leaves the paraphrase out of the control
    for text in ("All of these hold, save for one. Which?", "Apart from one, every statement below is true?",
                 "Each of these, except one, is true. Which is it?", "Drug of choice for X is"):
        assert sc.reverse_question(text) is None, text


def test_the_rule_refuses_a_reversal_that_removes_the_marker_and_not_one_that_keeps_one():
    """What the rule can and cannot see, on known reversals; the judge is there for the second kind."""
    removed = sc.reverse_question("Which statement is not true concerning Class-II inlay margins?")["text"]
    assert sc.rule_check(EXCEPT, removed)["rule_reason"] == "dropped"
    still_marked = sc.reverse_question("All of these are true of Class-II inlay margins, with one exception?")["text"]
    assert "with no exception" in still_marked
    assert sc.rule_check(EXCEPT, still_marked)["rule_reason"] == "kept"


def test_control_summary_counts_refusals():
    checks = [{"rule_reason": "kept"}] * 4 + [{"rule_reason": "none"}] * 6
    control = [{"marker": "not", "rule_reason": "dropped", "judge_verdict": "no"},
               {"marker": "exception", "rule_reason": "kept", "judge_verdict": "no"},
               {"marker": "not", "rule_reason": "dropped", "judge_verdict": "yes"}]
    got = sc.control_summary(checks, control)
    assert (got["paraphrases_with_a_marker_kept"], got["reversed"], got["not_reversible_by_the_table"]) == (4, 3, 1)
    assert (got["judge_said_no"], got["judge_said_yes"], got["judge_gave_no_verdict"]) == (2, 1, 0)
    assert got["rule_refused"] == 2 and got["refused_by_judge_or_rule"] == 3
    assert got["reversed_by_marker"] == {"not": 2, "exception": 1}


# ----------------------------------------------------------------------------- the committed check
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

import pytest  # noqa: E402

LANES = {"experiments/blind_metric/results": 200, "experiments/medmcqa/results": 200,
         "experiments/support2/results": 155}
MANIFESTS = {
    "experiments/blind_metric/results": Path(os.environ.get(
        "BENCHMAXXING_MEDQA_MANIFEST", str(Path.home() / "Documents/DATASET/medqa/manifest_test.csv"))),
    "experiments/medmcqa/results": Path(os.environ.get(
        "BENCHMAXXING_MEDMCQA_MANIFEST", str(Path.home() / "Documents/DATASET/medmcqa_dev_500.csv"))),
    "experiments/support2/results": ROOT / "experiments/support2/results/support2_manifest.csv",
}
BAND_FIELDS = ("n", "cases_with_any_disagreement", "mean_pairwise_disagreement", "flip_vs_baseline_per_read",
               "flip_vs_baseline_band", "decoy_per_read", "decoy_band")


def _rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def _summary(lane):
    return json.loads((ROOT / lane / sc.SUMMARY_FILE).read_text())


@pytest.mark.parametrize("lane,n", LANES.items())
def test_every_committed_paraphrase_is_checked_once(lane, n):
    checks = _rows(f"{lane}/{sc.CHECK_FILE}")
    paras = _rows(f"{lane}/paraphrases.jsonl")
    assert len(checks) == len(paras) == n
    assert [(c["case_id"], c["i"]) for c in checks] == [(p["case_id"], p["i"]) for p in paras]
    # one judge call per paraphrase, all by the judge model, none empty
    cache = _rows(f"{lane}/{sc.JUDGE_CACHE}")
    assert len(cache) == len({c["k"] for c in cache}) == n and {c["model"] for c in cache} == {sc.JUDGE}
    assert all(c["resp"].strip() for c in cache)
    s = _summary(lane)
    assert s["paraphrases"] == n and s["calls_in_cache"]["judge"] == n and s["judge"] == "gemini-2.5-flash"
    # the committed summary is what a keyless replay writes
    assert s["new_api_calls_this_run"] == {"judge": 0, "judge_control": 0}


@pytest.mark.parametrize("lane", LANES)
def test_the_rule_columns_recount_from_the_committed_paraphrase_text(lane):
    """The paraphrase text is committed, so its side of the rule is recomputed here with no manifest."""
    text = {(p["case_id"], p["i"]): p["paraphrase"] for p in _rows(f"{lane}/paraphrases.jsonl")}
    for c in _rows(f"{lane}/{sc.CHECK_FILE}"):
        assert c["markers_paraphrase"] == sc.markers(text[(c["case_id"], c["i"])]), (c["case_id"], c["i"])
        same = bool(c["markers_original"]) == bool(c["markers_paraphrase"])
        assert c["rule_check"] == same
        assert c["rule_reason"] == (("kept" if c["markers_original"] else "none") if same
                                    else ("dropped" if c["markers_original"] else "added"))
        assert c["judge_verdict"] in ("yes", "no", None) and c["judge"] == sc.JUDGE


@pytest.mark.parametrize("lane", LANES)
def test_the_failures_file_lists_exactly_the_failing_paraphrases(lane):
    checks = _rows(f"{lane}/{sc.CHECK_FILE}")
    failures = _rows(f"{lane}/{sc.FAILURES_FILE}")
    want = [(c["case_id"], c["i"]) for c in checks if not (c["rule_check"] and c["judge_verdict"] == "yes")]
    assert [(f["case_id"], f["i"]) for f in failures] == want
    text = {(p["case_id"], p["i"]): p["paraphrase"] for p in _rows(f"{lane}/paraphrases.jsonl")}
    for f in failures:
        assert f["paraphrase"] == text[(f["case_id"], f["i"])]
        rule_failed, judge_failed = not f["rule_check"], f["judge_verdict"] != "yes"
        assert f["failed"] == " and ".join(n for n, bad in (("rule", rule_failed), ("judge", judge_failed)) if bad)
    s = _summary(lane)["failed"]
    assert s["either"] == len(failures)
    assert s["rule"] == sum(not c["rule_check"] for c in checks)
    assert s["judge"] == sum(c["judge_verdict"] != "yes" for c in checks)
    assert s["judge"] == s["judge_said_no"] + s["judge_gave_no_verdict"]
    assert s["either"] == s["rule"] + s["judge"] - s["both"]


@pytest.mark.parametrize("lane", LANES)
def test_the_summary_recounts_from_the_check_rows_and_the_band_rows(lane):
    s = _summary(lane)
    folders = sc.band_folders(ROOT / lane)
    assert len(folders) == 4  # the Gemini lane folder and the three locally served lineages
    assert s == sc.summarise(_rows(f"{lane}/{sc.CHECK_FILE}"), folders, s["new_api_calls_this_run"],
                             s["calls_in_cache"], _rows(f"{lane}/{sc.CONTROL_FILE}"))


@pytest.mark.parametrize("lane", LANES)
def test_the_band_with_all_paraphrases_is_the_committed_band(lane):
    s = _summary(lane)
    for folder in sc.band_folders(ROOT / lane):
        committed = json.loads((folder / "paraphrase_band_summary.json").read_text())
        mine = s["bands"][committed["model"]]
        whole = mine["all_paraphrases"]["reads_0_to_k"]
        assert {k: whole[k] for k in BAND_FIELDS} == {k: committed[k] for k in BAND_FIELDS}, folder
        assert mine["all_paraphrases"]["reads_1_to_k"] == committed["without_read_0"]["paraphrase_reads"]
        assert mine["cases"] == committed["n"] and mine["paraphrase_reads"] == committed["paraphrases_total"]
        # leaving reads out can only remove disagreement, never add it
        for name in ("failed_left_out", "rule_failed_left_out", "judge_failed_left_out"):
            for reads in ("reads_0_to_k", "reads_1_to_k"):
                assert mine[name][reads]["cases_with_any_disagreement"]["n"] <= \
                    mine["all_paraphrases"][reads]["cases_with_any_disagreement"]["n"]
        # an independent recount of the left-out band over reads 0 to 5
        failed = {(c["case_id"], c["i"]) for c in _rows(f"{lane}/{sc.CHECK_FILE}")
                  if not (c["rule_check"] and c["judge_verdict"] == "yes")}
        recs = [json.loads(line) for line in (folder / "paraphrase_band.jsonl").read_text().splitlines() if line.strip()]
        kept = [[r["answers"][0]] + [a for i, a in enumerate(r["answers"][1:], 1) if (r["case_id"], i) not in failed]
                for r in recs]
        assert mine["failed_left_out"]["reads_0_to_k"]["cases_with_any_disagreement"]["n"] == \
            sum(len(set(k)) > 1 for k in kept if len(k) >= 2)
        assert mine["reads_left_out"]["either"] == sum(len(r["answers"]) - len(k) for r, k in zip(recs, kept))


@pytest.mark.parametrize("lane", LANES)
def test_keyless_replay_reproduces_the_check(lane, tmp_path):
    """The whole check reruns from the committed judge cache with no key (needs the lane's manifest)."""
    manifest = MANIFESTS[lane]
    if not manifest.exists():
        pytest.skip(f"the manifest of {lane} is not on this machine")
    src = ROOT / lane
    for f in ("paraphrases.jsonl", sc.JUDGE_CACHE, sc.CONTROL_CACHE):
        if (src / f).exists():  # a lane with no reversible question has no control cache
            shutil.copy(src / f, tmp_path / f)
    for folder in sc.band_folders(src):
        dst = tmp_path if folder == src else tmp_path / folder.name
        dst.mkdir(exist_ok=True)
        for f in ("paraphrase_band.jsonl", "paraphrase_band_summary.json"):
            shutil.copy(folder / f, dst / f)
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    env["PYTHONPATH"] = str(ROOT)
    r = subprocess.run([sys.executable, "experiments/matched_baseline/paraphrase_semantic_check.py",
                        "--manifest", str(manifest), "--results", str(tmp_path)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    for f in (sc.CHECK_FILE, sc.FAILURES_FILE, sc.CONTROL_FILE, sc.SUMMARY_FILE):
        assert (tmp_path / f).read_bytes() == (src / f).read_bytes(), f


def test_what_failed_on_each_lane():
    """9 of 555 paraphrases fail the rule, 1 the judge, none both; SUPPORT2 has no failure."""
    want = {"experiments/blind_metric/results": (1, {"added": 1}, 0, 1),
            "experiments/medmcqa/results": (8, {"dropped": 6, "added": 2}, 1, 5),
            "experiments/support2/results": (0, {}, 0, 0)}
    for lane, (rule, by_reason, judge, cases) in want.items():
        s = _summary(lane)
        f = s["failed"]
        assert (f["rule"], f["rule_by_reason"], f["judge"], f["both"]) == (rule, by_reason, judge, 0), lane
        assert f["judge_gave_no_verdict"] == 0 and f["either"] == rule + judge
        assert s["cases_with_a_failed_paraphrase"] == cases


def test_leaving_the_failed_paraphrases_out_barely_moves_any_band():
    changed = {}
    for lane in LANES:
        for model, m in _summary(lane)["bands"].items():
            for reads in ("reads_0_to_k", "reads_1_to_k"):
                a, f = m["all_paraphrases"][reads], m["failed_left_out"][reads]
                before = (a["cases_with_any_disagreement"]["n"], a["n"])
                after = (f["cases_with_any_disagreement"]["n"], f["n"])
                if reads == "reads_0_to_k":
                    assert before == after, (lane, model)  # read 0 keeps every case in
                elif before[0] != after[0]:
                    changed[(lane, model)] = (before, after)
                for k in ("flip_vs_baseline_band", "decoy_band"):
                    assert max(abs(x - y) for x, y in zip(a[k], f[k])) < 0.025, (lane, model, reads, k)
                assert abs(a["mean_pairwise_disagreement"] - f["mean_pairwise_disagreement"]) < 0.01
    # over reads 1 to 5, one case of one lineage leaves the count: it keeps a single paraphrase read
    assert changed == {("experiments/medmcqa/results", "openai/gpt-oss-120b"): ((8, 40), (7, 39))}


def test_the_judge_control_is_the_reversal_of_every_reversible_kept_paraphrase():
    """The control rows recount from the committed paraphrase text with no manifest and no key."""
    for lane in LANES:
        text = {(p["case_id"], p["i"]): p["paraphrase"] for p in _rows(f"{lane}/paraphrases.jsonl")}
        kept = [(c["case_id"], c["i"]) for c in _rows(f"{lane}/{sc.CHECK_FILE}") if c["rule_reason"] == "kept"]
        want = {key: sc.reverse_question(text[key]) for key in kept}
        control = _rows(f"{lane}/{sc.CONTROL_FILE}")
        assert [(r["case_id"], r["i"]) for r in control] == [key for key in kept if want[key]]
        for r in control:
            turned = want[(r["case_id"], r["i"])]
            assert (r["reversed"], r["marker"], r["replacement"]) == (turned["text"], turned["marker"],
                                                                      turned["replacement"])
        cache = _rows(f"{lane}/{sc.CONTROL_CACHE}") if (ROOT / lane / sc.CONTROL_CACHE).exists() else []
        assert len(cache) == len({c["k"] for c in cache}) == len(control) == _summary(lane)["calls_in_cache"]["judge_control"]


def test_the_judge_refuses_37_of_38_reversed_questions():
    want = {"experiments/blind_metric/results": (10, 5, 5, 0, 5, 5),
            "experiments/medmcqa/results": (39, 33, 32, 1, 30, 32),
            "experiments/support2/results": (0, 0, 0, 0, 0, 0)}
    for lane, numbers in want.items():
        c = _summary(lane)["judge_control"]
        assert (c["paraphrases_with_a_marker_kept"], c["reversed"], c["judge_said_no"], c["judge_said_yes"],
                c["rule_refused"], c["refused_by_judge_or_rule"]) == numbers, lane
        assert c["judge_gave_no_verdict"] == 0
    # the one reversal the judge accepted keeps a marker ("with no exception"), so the rule cannot see it either
    accepted = [r for r in _rows(f"experiments/medmcqa/results/{sc.CONTROL_FILE}") if r["judge_verdict"] == "yes"]
    assert [(r["marker"], r["replacement"], r["rule_reason"]) for r in accepted] == [
        ("exception", "with no exception", "kept")]
