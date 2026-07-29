"""Regression suite for ``experiments/medqa/reproduce.py``'s answer parser (#345).

The parser reproduce.py uses is the function behind every MedQA number currently in the paper
(solo flip rates, cascade contagion, referee metrics, …). It has no other committed test. PR #342
showed that swapping it for the shared ``extract.parse_mcq_choice`` silently disagrees on ~5.5% of
real cached Gemini replies — caught only by an ad hoc cache diff. This suite is the standing
guard: synthetic cases covering each reply shape, plus a frozen sample of real ``call_cache.jsonl``
responses.

The suite must collect under pytest (``test_*`` names). A meta-test below asserts a minimum number
of collected cases so a rename/empty-file regression fails loudly rather than silently running
zero tests (the failure mode of #342's ``tests/test_parse_regression.py``).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "experiments" / "medqa" / "reproduce.py"
_FIXTURES = _REPO / "tests" / "fixtures" / "medqa_parse_choice_cache.json"

# Floor for the meta-test. Keep this below the real collected count so adding cases is fine, but
# high enough that deleting the suite or emptying the fixture file fails CI.
_MIN_SYNTHETIC_CASES = 8
_MIN_CACHE_FIXTURES = 15


def _load_reproduce():
    spec = importlib.util.spec_from_file_location("medqa_reproduce", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


reproduce = _load_reproduce()
# #102 centralized the parser: reproduce.py no longer defines its own _parse_choice, it imports
# parse_legacy_string from benchmaxxing.extract. Resolve whichever this checkout exposes so the
# fixture coverage keeps testing the function actually behind every MedQA number in the paper.
parse = getattr(reproduce, "_parse_choice", None) or reproduce.parse_legacy_string


# Full-text options, matching what reproduce.py passes from MedQA payloads (not bare A–E).
OPTS_5 = (
    "Reassure the mother and schedule routine follow-up",
    "Start empiric antibiotics for community-acquired pneumonia",
    "Advise the patient to stop masturbating",
    "Order CT angiography of the chest",
    "Refer for immediate surgical exploration",
)


def _load_cache_fixtures() -> list[dict]:
    payload = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert isinstance(cases, list) and cases, f"fixture file {_FIXTURES} has no cases"
    return cases


CACHE_FIXTURES = _load_cache_fixtures()


# --------------------------------------------------------------------------- synthetic shapes


SYNTHETIC = [
    # bare letter
    ("D", OPTS_5, OPTS_5[3], "bare_letter"),
    ("b", OPTS_5, OPTS_5[1], "bare_letter_lowercase"),
    # \boxed{X}
    (
        "Long reasoning about A and B.\n\nThe final answer is $\\boxed{C}$",
        OPTS_5,
        OPTS_5[2],
        "boxed_letter",
    ),
    # Gemini \boxed{\text{C. ...}} — letter inside text box (reproduce's boxed regex is letter-only;
    # when the box holds only a phrase with no letter, option-text match must still win).
    (
        "Therefore the diagnosis is clear.\n\nThe final answer is "
        "$\\boxed{\\text{Order CT angiography of the chest}}$",
        OPTS_5,
        OPTS_5[3],
        "boxed_text_phrase",
    ),
    # explicit declarations
    ("The answer is: B", OPTS_5, OPTS_5[1], "answer_is_colon"),
    ("The correct answer is **D**.", OPTS_5, OPTS_5[3], "correct_answer_bold"),
    ("After review, final answer is A.", OPTS_5, OPTS_5[0], "final_answer_is"),
    # multi-option CoT: names several options, concludes with a letter — must not take a rejected
    # bullet's letter / text over the declaration (the #342 failure mode on shared extract).
    (
        "Looking at the choices:\n"
        f"* **A. {OPTS_5[0]}** — possible but incomplete.\n"
        f"* **B. {OPTS_5[1]}** — not first-line here.\n"
        f"* **C. {OPTS_5[2]}** — distractor.\n"
        f"* **D. {OPTS_5[3]}** — best next step.\n"
        f"* **E. {OPTS_5[4]}** — too aggressive.\n\n"
        "The correct answer is **D**.",
        OPTS_5,
        OPTS_5[3],
        "multi_option_cot_bold_D",
    ),
    # option text named last, no letter declaration
    (
        f"Not {OPTS_5[0]}. Not {OPTS_5[1]}. The right move is {OPTS_5[3]}.",
        OPTS_5,
        OPTS_5[3],
        "option_text_last_mention",
    ),
    # empty / unparseable → returns raw text (legacy contract used by FlipRecord fields)
    ("", OPTS_5, "", "empty"),
]


@pytest.mark.parametrize(
    "text,options,expected,case_id",
    SYNTHETIC,
    ids=[row[3] for row in SYNTHETIC],
)
def test_synthetic_reply_shapes(text, options, expected, case_id):
    assert parse(text, list(options)) == expected


# --------------------------------------------------------------------------- real cache fixtures


@pytest.mark.parametrize(
    "fixture",
    CACHE_FIXTURES,
    ids=[f"{c['kind']}:{c['id']}" for c in CACHE_FIXTURES],
)
def test_real_cache_fixture(fixture):
    """Pin the parser against frozen real Gemini replies from call_cache.jsonl."""
    assert parse(fixture["resp"], list(fixture["options"])) == fixture["expected"]


def test_cache_fixtures_cover_required_kinds():
    kinds = {c["kind"] for c in CACHE_FIXTURES}
    required = {"bare_letter", "boxed_letter", "boxed_text", "declaration", "multi_option_cot"}
    missing = required - kinds
    assert not missing, f"fixture file missing kinds {missing}"


def test_cache_fixture_keys_resolve_in_committed_call_cache():
    """Each frozen case cites a real ``k`` still present in the committed MedQA call cache."""
    cache_path = _REPO / "experiments" / "medqa" / "results" / "call_cache.jsonl"
    assert cache_path.is_file(), "committed call_cache.jsonl missing"
    keys = set()
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.add(json.loads(line)["k"])
    missing = [c["cache_k"] for c in CACHE_FIXTURES if c["cache_k"] not in keys]
    assert not missing, f"{len(missing)} fixture cache_k values not in call_cache.jsonl"


# --------------------------------------------------------------------------- loud emptiness guard


def test_suite_collects_enough_cases():
    """Fail loudly if this file or the fixture set is gutted (the #342 silent-zero-tests bug)."""
    assert len(SYNTHETIC) >= _MIN_SYNTHETIC_CASES, (
        f"synthetic cases shrank to {len(SYNTHETIC)}; expected >= {_MIN_SYNTHETIC_CASES}"
    )
    assert len(CACHE_FIXTURES) >= _MIN_CACHE_FIXTURES, (
        f"cache fixtures shrank to {len(CACHE_FIXTURES)}; expected >= {_MIN_CACHE_FIXTURES}"
    )
    assert _FIXTURES.is_file(), f"missing fixture file {_FIXTURES}"
    assert _SCRIPT.is_file(), f"missing reproduce.py at {_SCRIPT}"
