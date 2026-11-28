"""The self-hosted nemotron FP8 plausible-distractor number, and the caveat that goes with it.

0.1782 against 0.0594 (15 vs 3) depends on how the arm's empty replies are scored: it holds on the
80 rows with no empty reply and reverses when the 21 rows that carry one are all scored against it.
This file recomputes those figures from the committed rows, audit and cache, and checks that the
experiment README states them in the same paragraph as the number. No model call is made: the cache
is only read, and a prompt that is not in it fails the test.
"""
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.extract import declared_mcq_choice
from benchmaxxing.stats import mcnemar

ROOT = Path(__file__).resolve().parents[1]
MODEL = "nemotron-3-super-120b-a12b-fp8-local"
RES = ROOT / "experiments/medqa/results"
ROWS = RES / MODEL / "plausible_distractor.jsonl"
SUMMARY = RES / MODEL / "plausible_distractor_summary.json"
AUDIT = RES / MODEL / "plausible_distractor_declared_audit.json"
CACHE = RES / f"{MODEL}_plausible_distractor_cache.jsonl"
AUDIT_SCRIPT = ROOT / "experiments/medqa/declared_only_audit_plausible_distractor.py"
README = ROOT / "experiments/medqa/README.md"
MANIFEST = Path(os.environ.get("BENCHMAXXING_MEDQA_MANIFEST",
                               str(Path.home() / "Documents/DATASET/medqa/manifest_test.csv")))
N_ASKED = 120  # the size the arm was run and committed at


def _rows():
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]


def _paired(rows):
    """Second-choice adopters, least-likely adopters, and the two discordant counts."""
    second = sum(r["plausible_adopt"] for r in rows)
    least = sum(r["implausible_adopt"] for r in rows)
    gain = sum(1 for r in rows if r["plausible_adopt"] and not r["implausible_adopt"])
    lose = sum(1 for r in rows if r["implausible_adopt"] and not r["plausible_adopt"])
    return second, least, gain, lose


def _p(gain, lose):
    return round(mcnemar(gain, lose).pvalue, 6)


def _replay():
    """Every reply of the arm, attributed to its case and prompt from the committed cache.

    The prompts are rebuilt with the declared-only audit's own builders, and the two boards of a
    scored case use the wrong answers its committed row records, as that audit does.
    """
    spec = importlib.util.spec_from_file_location("declared_only_audit_plausible_distractor", AUDIT_SCRIPT)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    store = {}
    for line in CACHE.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            store[entry["k"]] = entry["resp"]
    committed = {r["case_id"]: r for r in _rows()}
    used, cases = set(), []

    def reply(prompt):
        k = hashlib.sha256(f"{MODEL}\x00{prompt}".encode()).hexdigest()
        used.add(k)
        return store[k]  # a KeyError is a cache miss: nothing is ever asked of a model

    for case in load_cases(MANIFEST)[:N_ASKED]:
        options = list(case.options)
        payload = {"question": case.question, "options": options}
        replies = {
            "bare": reply(audit._mcq_prompt(payload)),
            "second": reply(audit._rank_prompt(payload, "SECOND most likely")),
            "least": reply(audit._rank_prompt(payload, "LEAST likely")),
        }
        row = committed.get(case.case_id)
        if row:  # a scored case was also shown its two boards
            for name in ("plausible", "implausible"):
                board = audit._seed_for(case, row[f"{name}_wrong"])
                replies[name] = reply(audit._mcq_prompt(payload, board))
        cases.append({"case_id": case.case_id, "options": options, "replies": replies, "row": row})
    return store, used, cases


def test_the_headline_is_the_committed_rows():
    rows = _rows()
    second, least, gain, lose = _paired(rows)
    assert (len(rows), second, least, gain, lose) == (101, 18, 6, 15, 3)
    summary = json.loads(SUMMARY.read_text())
    assert summary["adoption"] == {"plausible": 0.1782, "implausible": 0.0594}
    assert summary["adoption"] == {"plausible": round(second / 101, 4), "implausible": round(least / 101, 4)}
    assert summary["plausible_vs_implausible"] == {"gain": 15, "lose": 3, "pvalue": _p(15, 3)}
    assert _p(15, 3) == 0.007538


def test_the_result_reverses_when_the_rows_with_an_empty_reply_are_scored_against_it():
    audit = json.loads(AUDIT.read_text())
    kept = audit["declared_only"]
    assert (audit["n_cases"], kept["n"], audit["n_undeclared"]) == (101, 80, 21)
    assert kept["adoption"] == {"plausible": 0.175, "implausible": 0.0375}
    assert kept["plausible_vs_implausible"] == {"gain": 12, "lose": 1, "pvalue": _p(12, 1)}
    assert _p(12, 1) == 0.003418
    # each of the 21 rows counted as: least-likely answer adopted, second choice not adopted
    second = round(kept["adoption"]["plausible"] * kept["n"])
    least = round(kept["adoption"]["implausible"] * kept["n"]) + audit["n_undeclared"]
    gain = kept["plausible_vs_implausible"]["gain"]
    lose = kept["plausible_vs_implausible"]["lose"] + audit["n_undeclared"]
    assert (second, least, gain, lose) == (14, 24, 12, 22)
    assert (round(second / 101, 4), round(least / 101, 4)) == (0.1386, 0.2376)
    assert lose > gain and _p(gain, lose) == 0.121449


@pytest.mark.skipif(not MANIFEST.exists(), reason="the MedQA manifest is not on this machine")
def test_the_empty_replies_recounted_from_the_cache():
    store, used, cases = _replay()
    assert len(store) == 562 and used == set(store)  # every cached reply is attributed to a prompt
    empty = Counter(name for c in cases for name, resp in c["replies"].items() if resp == "")
    assert empty == {"second": 21, "least": 19, "bare": 6, "plausible": 3, "implausible": 1}
    assert sum(empty.values()) == sum(1 for resp in store.values() if resp == "") == 50

    scored = [c for c in cases if c["row"]]
    unscored = [c for c in cases if not c["row"]]
    with_empty = [c for c in scored if "" in c["replies"].values()]
    assert (len(scored), len(unscored), len(with_empty)) == (101, 19, 21)
    assert sum(1 for c in with_empty for resp in c["replies"].values() if resp == "") == 27
    # the audit's undeclared rows are exactly the rows with an empty reply
    undeclared = [c for c in scored
                  if not all(declared_mcq_choice(resp, c["options"])[1] for resp in c["replies"].values())]
    assert {c["case_id"] for c in undeclared} == {c["case_id"] for c in with_empty}

    clean = [c["row"] for c in scored if c not in with_empty]
    assert (len(clean), *_paired(clean)) == (80, 14, 3, 12, 1)
    gain, lose = 12, 1 + len(with_empty)
    assert (gain, lose, _p(gain, lose)) == (12, 22, 0.121449)
    # 14 unscored cases lost a ranking reply too; counted against the result as well
    lost_ranking = [c for c in unscored if "" in (c["replies"]["second"], c["replies"]["least"])]
    assert len(lost_ranking) == 14 == sum(1 for c in unscored if "" in c["replies"].values())
    assert sum(1 for c in unscored for resp in c["replies"].values() if resp == "") == 23
    assert _p(gain, lose + len(lost_ranking)) == 0.000717


def test_the_readme_quotes_the_number_only_with_its_caveat():
    paragraphs = [" ".join(p.split()) for p in README.read_text().split("\n\n")]
    quoting = [p for p in paragraphs if "0.1782" in p and "0.0594" in p and not p.startswith("|")]
    assert quoting, "the README no longer quotes the plausible-distractor number"
    for paragraph in quoting:
        for figure in ("15 vs 3", "21 of the 101", "0.175", "0.0375", "12 vs 1,",
                       "0.1386", "0.2376", "12 vs 22", "P = 0.121"):
            assert figure in paragraph, f"{figure!r} is missing beside 0.1782 against 0.0594"
