"""deliberation_channel: a reply cut at the token limit is no answer, and a non-default limit is its own reading.

Each test runs the runner's main() keyless in a temporary tree that holds a two-case manifest, the
committed cascade cache the planted wrong option is read from, and a filled deliberation cache.
"""
import hashlib
import json
import sys

import pytest

from benchmaxxing.data import write_manifest
from benchmaxxing.schema import Case, Modality
from experiments.medqa import deliberation_channel as dc

MODEL = "local/test-model"
SLUG = "local_test-model"
OPTIONS = ("Alpha", "Beta", "Gamma", "Delta")


def _key(*parts):
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def _tree(tmp_path, monkeypatch, seeded_replies):
    """Two cases whose correct answer is A; the bare answer is A, so the planted wrong option is Beta."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BENCHMAXXING_LOCAL_BASE_URL", raising=False)
    results = tmp_path / "experiments" / "medqa" / "results"
    results.mkdir(parents=True)
    cases = [Case(f"c{i}", f"p{i}", Modality.TEXT, question=f"Question number {i} about a patient?",
                  options=OPTIONS, answer_index=0) for i in range(len(seeded_replies))]
    write_manifest(cases, tmp_path / "manifest.csv")
    cascade, store = [], []
    for case, (text, finish) in zip(cases, seeded_replies):
        payload = {"question": case.question, "options": list(OPTIONS)}
        bare_prompt = dc._mcq_prompt(payload)
        cascade.append({"k": _key(MODEL, bare_prompt), "model": MODEL, "resp": "A"})
        instruction = dc._instruction(MODEL, "open")
        for board, reply, fin in ((dc._seed(case, "Beta"), text, finish), ("", "Thinking.\nA", "stop")):
            prompt = dc._mcq_prompt(payload, board, instruction)
            store.append({"k": _key(MODEL, "open", prompt), "model": MODEL, "condition": "open",
                          "content": reply, "reasoning_content": None, "finish_reason": fin})
    (results / f"{SLUG}_contamination_cascade_cache.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in cascade))
    standard_cache = results / f"{SLUG}_deliberation_channel_cache.jsonl"
    standard_cache.write_text("".join(json.dumps(r) + "\n" for r in store))
    return results, standard_cache


def _run(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["deliberation_channel", "--manifest", "manifest.csv", "--model", MODEL,
                                      "--out", "experiments/medqa/results", "--conditions", "open", *extra])
    dc.main()


def test_a_cut_reply_with_a_declared_letter_is_no_answer(tmp_path, monkeypatch):
    results, _ = _tree(tmp_path, monkeypatch, [("The attending says B. So perhaps\nB", "length"),
                                               ("Reasoned.\nB", "stop")])
    _run(monkeypatch)
    summary = json.loads((results / SLUG / "deliberation_channel_summary.json").read_text())
    rows = [json.loads(line) for line in (results / SLUG / "deliberation_channel.jsonl").read_text().splitlines()]
    assert summary["new_api_calls_this_run"] == 0
    assert [r["open_declared"] for r in rows] == [None, "B"]
    assert [r["open_adopt"] for r in rows] == [0, 1]
    assert summary["declared_adoption_by_condition"]["open"] == 0.5
    assert summary["declared_adoption_complete_by_condition"]["open"] == {"declared_adopt": 1, "complete": 1, "rate": 1.0}
    assert summary["finish_reason_counts"]["open"] == {"length": 1, "stop": 1}
    assert summary["unseeded_finish_reason_counts"]["open"] == {"stop": 2}
    assert summary["max_tokens"] == 8192


def test_a_cut_unseeded_reply_is_not_correct(tmp_path, monkeypatch):
    results, cache = _tree(tmp_path, monkeypatch, [("Reasoned.\nB", "stop")])
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    rows[1]["finish_reason"] = "length"   # the unseeded reply "Thinking.\nA" now stopped at the limit
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _run(monkeypatch)
    summary = json.loads((results / SLUG / "deliberation_channel_summary.json").read_text())
    assert summary["unseeded_accuracy_by_condition"]["open"] == 0.0
    assert summary["unseeded_finish_reason_counts"]["open"] == {"length": 1}


def test_a_non_default_limit_never_reads_or_writes_the_standard_files(tmp_path, monkeypatch):
    results, standard_cache = _tree(tmp_path, monkeypatch, [("Reasoned.\nB", "stop")])
    before = standard_cache.read_bytes()
    with pytest.raises(SystemExit, match="Cache miss"):
        _run(monkeypatch, "--max-tokens", "16384")
    assert standard_cache.read_bytes() == before
    assert not (results / SLUG / "deliberation_channel_summary.json").exists()


def test_a_non_default_limit_has_its_own_cache_keys(tmp_path, monkeypatch):
    # The same prompt at another limit is another request: a row keyed like a standard-limit request
    # must not answer it, or any reader that matches cache keys across files (cross_lineage_report's
    # repeated-prompt count) takes the two readings for repeats of one request.
    results, standard_cache = _tree(tmp_path, monkeypatch, [("Reasoned.\nB", "stop")])
    (results / f"{SLUG}_deliberation_channel_max_tokens_16384_cache.jsonl").write_bytes(standard_cache.read_bytes())
    with pytest.raises(SystemExit, match="Cache miss"):
        _run(monkeypatch, "--max-tokens", "16384")


def test_a_non_default_reading_replays_from_its_own_keys(tmp_path, monkeypatch):
    results, standard_cache = _tree(tmp_path, monkeypatch, [("Reasoned.\nB", "stop")])
    payload = {"question": "Question number 0 about a patient?", "options": list(OPTIONS)}
    case = Case("c0", "p0", Modality.TEXT, question=payload["question"], options=OPTIONS, answer_index=0)
    instruction = dc._instruction(MODEL, "open")
    rows = [json.loads(line) for line in standard_cache.read_text().splitlines()]
    for r, board in zip(rows, (dc._seed(case, "Beta"), "")):
        r["k"] = _key(MODEL, "open", dc._mcq_prompt(payload, board, instruction), "max_tokens=16384")
    (results / f"{SLUG}_deliberation_channel_max_tokens_16384_cache.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    _run(monkeypatch, "--max-tokens", "16384")
    summary = json.loads((results / SLUG / "deliberation_channel_max_tokens_16384_summary.json").read_text())
    assert summary["new_api_calls_this_run"] == 0
    assert summary["max_tokens"] == 16384


def test_a_non_default_limit_is_refused_for_gemini(tmp_path, monkeypatch):
    _tree(tmp_path, monkeypatch, [("Reasoned.\nB", "stop")])
    monkeypatch.setattr(sys, "argv", ["deliberation_channel", "--manifest", "manifest.csv",
                                      "--model", "gemini-2.5-flash-lite", "--max-tokens", "16384"])
    with pytest.raises(SystemExit, match="not supported for Gemini"):
        dc.main()
