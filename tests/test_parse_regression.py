"""Regression guard for the centralized MCQ parser (#102, #342).

This file previously contained no ``def test_`` at all, so pytest imported it and collected zero
cases while the same PR deleted ``tests/test_reproduce_parse_choice.py`` and its fixture. Net effect
was five real tests removed and none added, on the parser every MedQA and imaging number depends on.
The coverage below restores that guard and pins the specific defect the collapse hid: for full-text
options the parser used to fall through to a trailing-letter scan on multi-match, so a bare "A" in
the prose (nearly always the English article) became a confident vote for option A.
"""
import json
import re
from pathlib import Path

import pytest

from benchmaxxing.extract import Abstention, parse_legacy_string, parse_mcq_choice

FIXTURE = Path(__file__).parent / "fixtures" / "medqa_parse_choice_cache.json"
CLINICAL = ("Type II pneumocytes", "Alveolar macrophages", "Club cells", "Goblet cells")


def _pre_centralization_parse(text, options):
    """The ad-hoc ``_parse`` every experiment carried before #342, kept as the parity reference.

    All 39 duplicated copies under experiments/ were behaviorally identical, differing only by an
    ``m`` vs ``m2`` rename, so one copy is a faithful oracle for "did the refactor move a number".
    """
    if not text:
        return ""
    t = text.strip()
    letters = "".join(chr(ord("A") + i) for i in range(len(options)))
    m = re.findall(r"\\boxed\{\s*([A-Z])\s*\}", t)
    if not m:
        m = re.findall(r"(?:final answer|the answer|answer)\s*(?:is|:)?\s*\**\(?([A-Z])\)?\b", t, re.I)
    if m and m[-1].upper() in letters:
        return options[letters.index(m[-1].upper())]
    low = t.lower()
    hits = [(low.rfind(o.lower()), o) for o in options if o.lower() in low]
    hits = [(p, o) for p, o in hits if p >= 0]
    if hits:
        return max(hits)[1]
    m2 = re.search(r"\b([A-Z])\b\s*[.)]?\s*$", t.upper())
    if m2 and m2.group(1) in letters:
        return options[letters.index(m2.group(1))]
    if len(t) == 1 and t.upper() in letters:
        return options[letters.index(t.upper())]
    return ""


# The three rows that caught the fall-through. Each mentions two options and contains a bare
# capital letter that is not an answer. Before the fix, rows 1 and 2 returned option A.
MULTI_MATCH = [
    ("Both Type II pneumocytes and Club cells are plausible. A surfactant deficiency argues for "
     "the former.", "Club cells"),
    ("A comparison of Alveolar macrophages and Goblet cells is useful here. Ultimately the "
     "histology favours Club cells.", "Club cells"),
    ("Consider Club cells versus Goblet cells. The lining here is ciliated, which points to "
     "Club cells.", "Club cells"),
]


@pytest.mark.parametrize("text,expected", MULTI_MATCH)
def test_multi_match_never_invents_an_answer_from_a_stray_letter(text, expected):
    assert parse_legacy_string(text, CLINICAL) == expected


@pytest.mark.parametrize("text,expected", MULTI_MATCH)
def test_multi_match_agrees_with_the_pre_centralization_parser(text, expected):
    assert parse_legacy_string(text, CLINICAL) == _pre_centralization_parse(text, CLINICAL)


SHAPES = [
    "The answer is C.",
    "\\boxed{B}",
    "Type II pneumocytes.",
    "After weighing the options, the final answer is D",
    "A",
    "I cannot provide medical advice.",
    "",
    "Neither of these fits the presentation well.",
]


@pytest.mark.parametrize("text", SHAPES + [t for t, _ in MULTI_MATCH])
def test_parity_with_the_pre_centralization_parser(text):
    """The refactor must not move any committed number, so it must match the old parser exactly."""
    assert parse_legacy_string(text, CLINICAL) == _pre_centralization_parse(text, CLINICAL)


def test_full_text_options_never_return_a_letter_derived_index_on_multi_match():
    """Directly pin the terminal branch: with two options named and a stray letter present, the
    result must be one of the named options or an abstention, never the letter's index."""
    text = "Both Goblet cells and Club cells appear. A bronchiolar lining is described."
    got = parse_mcq_choice(text, CLINICAL)
    assert got is not Abstention.REFUSAL
    if not isinstance(got, Abstention):
        assert CLINICAL[got] in ("Goblet cells", "Club cells"), CLINICAL[got]


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"])
def test_real_cache_fixture(case):
    """Frozen real Gemini replies from the committed MedQA call cache."""
    assert parse_legacy_string(case["resp"], case["options"]) == case["expected"]


def test_suite_collects_enough_cases():
    """Fail loudly if this file or the fixture set is gutted again.

    This is the guard that #342 originally deleted along with test_reproduce_parse_choice.py, which
    is precisely how the zero-collected-tests state went unnoticed.
    """
    fixture_cases = json.loads(FIXTURE.read_text())["cases"]
    assert len(fixture_cases) >= 4, "fixture set shrank"
    assert len(MULTI_MATCH) >= 3, "multi-match rows removed"
    assert len(SHAPES) >= 8, "reply-shape rows removed"
    defined = [n for n in globals() if n.startswith("test_")]
    assert len(defined) >= 6, f"only {len(defined)} tests defined in this module"
