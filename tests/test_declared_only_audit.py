"""The declared-only rescore must actually separate declared from undeclared draws.

The audit's whole point is the ``fully_declared`` filter: a holdout draw counts only when every
completion is an explicit option declaration. No test in the tree exercised that filter, so forcing
it to ``True`` left the suite green while the finding disappeared (``n_declared`` 39 to 40 on the
committed Gemini arm). This test runs the deployable audit end to end on a synthetic manifest with a
stubbed cache: even-numbered cases answer with a bare letter on every draw, odd-numbered ones with
prose that names no option, and the audit must count exactly the even ones as declared.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from benchmaxxing.data import write_manifest
from benchmaxxing.schema import Case, Modality

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/referee/declared_only_audit_deployable.py"

N = 6
CASES = [
    Case(
        case_id=f"q{i}",
        patient_id=f"p{i}",
        modality=Modality.TEXT,
        question=f"Synthetic vignette {i}: most likely diagnosis?",
        options=(f"Correct option {i}", f"Second option {i}", f"Third option {i}"),
        answer_index=0,
        report="",
    )
    for i in range(N)
]
PROSE = "Several of these could fit; I would weigh the history before committing to any of them."


def _load():
    spec = importlib.util.spec_from_file_location("declared_only_audit_deployable", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubCache:
    """Answers from the case the prompt is about: a bare letter for even cases, prose for odd."""

    def __init__(self, path):
        self.path = path

    def complete(self, model, prompt):
        for i, case in enumerate(CASES):
            if case.options[0] in prompt:
                return "B" if i % 2 == 0 else PROSE
        raise AssertionError("prompt names no synthetic case")


@pytest.fixture
def audit_run(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "_CacheReader", _StubCache)
    manifest = write_manifest(CASES, tmp_path / "cases.csv")
    results = tmp_path / "results"
    results.mkdir()
    # The holdout stub never moves off its bare answer, so the replay has no adoptions and no
    # false positives; the committed summary the audit checks itself against says the same.
    (results / "referee_deployable_summary.json").write_text(json.dumps({
        "n_cases": N, "n_holdout_adopted_shortcut": 0, "n_false_positive_on_clean_control": 0,
    }))
    monkeypatch.setattr(sys, "argv", [
        "declared_only_audit_deployable.py", "--manifest", str(manifest), "--results", str(results),
        "--cache", str(tmp_path / "unused_cache.jsonl"), "--n", str(N),
    ])
    mod.main()
    return json.loads((results / "referee_deployable_declared_audit.json").read_text())


def test_declared_filter_counts_only_fully_declared_cases(audit_run):
    assert audit_run["n_cases"] == N
    assert audit_run["n_declared"] == N // 2, "the declared filter is not separating draws"
    assert audit_run["n_undeclared"] == N - N // 2


def test_undeclared_draws_are_counted_per_draw_not_per_case(audit_run):
    # Each odd case has three undeclared draws: bare, planted board, clean board.
    assert audit_run["undeclared_draws"] == 3 * (N - N // 2)


def test_declared_only_block_is_computed_on_the_declared_subset(audit_run):
    assert audit_run["declared_only"]["n_cases"] == N // 2
    assert audit_run["current"]["n_cases"] == N


class _MixedStub(_StubCache):
    """q0 declares on every draw; q1/q2/q3 each answer in prose on exactly one draw.

    q1 on the bare re-query, q2 on the planted board (peers assert a wrong option), q3 on the
    clean-control board (peers assert the correct option). Each of the three conjuncts in
    ``fully_declared`` is then load-bearing on its own.
    """

    def complete(self, model, prompt):
        board, _, _ = prompt.partition("Question:")
        for i, case in enumerate(CASES[:4]):
            if case.options[0] in prompt:
                if not board:
                    undeclared = i == 1
                elif case.options[0] in board:
                    undeclared = i == 3
                else:
                    undeclared = i == 2
                return PROSE if undeclared else "C"
        raise AssertionError("prompt names no synthetic case")


def test_one_undeclared_draw_is_enough_to_exclude_a_case(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "_CacheReader", _MixedStub)
    manifest = write_manifest(CASES[:4], tmp_path / "cases.csv")
    results = tmp_path / "results"
    results.mkdir()
    (results / "referee_deployable_summary.json").write_text(json.dumps({
        "n_cases": 4, "n_holdout_adopted_shortcut": 0, "n_false_positive_on_clean_control": 0,
    }))
    monkeypatch.setattr(sys, "argv", [
        "declared_only_audit_deployable.py", "--manifest", str(manifest), "--results", str(results),
        "--cache", str(tmp_path / "unused_cache.jsonl"), "--n", "4",
    ])
    mod.main()
    audit = json.loads((results / "referee_deployable_declared_audit.json").read_text())
    assert audit["n_declared"] == 1, "a case with one undeclared draw must not count as declared"
    assert audit["undeclared_draws"] == 3
