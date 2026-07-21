"""Unit tests for the answer extraction module.

Golden-case format: each test parametrizes a table of
``(raw_text, options, expected)`` triples so failures report which
exact case broke.
"""

import pytest

from benchmaxxing.extract import Abstention, parse_mcq_choice, is_abstention


# ── is_abstention ───────────────────────────────────────────────────────

class TestIsAbstention:
    @pytest.mark.parametrize("text", [
        "As an AI, I cannot provide medical advice.",
        "I am unable to diagnose this patient.",
        "I'm sorry, but I cannot answer that.",
        "I'm sorry but I cannot answer that.",  # without comma
        "It is not appropriate for me to prescribe medicine.",
        "I am an AI language model and cannot help.",
    ])
    def test_true_positives(self, text):
        assert is_abstention(text) is True

    @pytest.mark.parametrize("text", [
        "The correct answer is A.",
        "Based on the patient's symptoms, the diagnosis is clear.",
        "no",
        "maybe",
        "",
    ])
    def test_false_negatives(self, text):
        assert is_abstention(text) is False

    def test_none_input(self):
        assert is_abstention(None) is False


# ── parse_mcq_choice — letter MCQ ──────────────────────────────────────

OPTS_5 = ("A", "B", "C", "D", "E")

GOLDEN_MCQ = [
    # ── Clean explicit declarations ──
    ("The correct answer is B.", OPTS_5, 1),
    ("The correct option is C", OPTS_5, 2),
    ("The answer is A.", OPTS_5, 0),
    ("My answer is D.", OPTS_5, 3),
    ("Answer: C", OPTS_5, 2),
    ("Answer: E.", OPTS_5, 4),

    # ── Common model phrasings ──
    ("I choose C.", OPTS_5, 2),
    ("I would choose B.", OPTS_5, 1),
    ("I'll choose A.", OPTS_5, 0),
    ("I select D.", OPTS_5, 3),

    # ── Bare letter / parenthesized ──
    ("B", OPTS_5, 1),
    ("A.", OPTS_5, 0),
    ("(C)", OPTS_5, 2),

    # ── Refusals ──
    ("As an AI I cannot answer this medical question.", OPTS_5, Abstention.REFUSAL),
    ("The question is complex. I am unable to diagnose. The answer might be A.",
     OPTS_5, Abstention.REFUSAL),

    # ── Trailing match (chain of thought concludes with a letter) ──
    ("The patient has a headache. A is migraine. B is stroke. Therefore, B.",
     OPTS_5, 1),

    # ── Length-bias trap: distractor discusses many letters, declaration wins ──
    ("A is wrong. B is wrong. Option C is a very long text that mentions A and "
     "B and D and E. The correct answer is C.",
     OPTS_5, 2),

    # ── BUG FIX: "option D" in prose should NOT hijack when a later declaration exists ──
    ("Looking at option A, it describes an ACE inhibitor. Looking at option D, "
     "it describes a diuretic. The actual answer is A.",
     OPTS_5, 0),

    # ── Conflicting double-answer → UNPARSEABLE ──
    ("It could be A or B.", OPTS_5, Abstention.UNPARSEABLE),
    ("The answer involves both A and B.", OPTS_5, Abstention.UNPARSEABLE),

    # ── Garbage / unparseable ──
    ("I think it is a viral infection.", OPTS_5, Abstention.UNPARSEABLE),
    ("", OPTS_5, Abstention.UNPARSEABLE),
    ("The answer is Z.", OPTS_5, Abstention.UNPARSEABLE),
]


class TestParseMcqChoiceLetters:
    @pytest.mark.parametrize("text,options,expected", GOLDEN_MCQ,
                             ids=[f"case_{i}" for i in range(len(GOLDEN_MCQ))])
    def test_golden_cases(self, text, options, expected):
        assert parse_mcq_choice(text, options) == expected

    def test_none_input(self):
        assert parse_mcq_choice(None, OPTS_5) is Abstention.UNPARSEABLE

    def test_empty_options(self):
        assert parse_mcq_choice("A", ()) is Abstention.UNPARSEABLE

    def test_list_options_accepted(self):
        """options can be a list, not just a tuple."""
        assert parse_mcq_choice("B", ["A", "B", "C"]) == 1


# ── parse_mcq_choice — yes / no / maybe ────────────────────────────────

OPTS_YNM = ("yes", "no", "maybe")

GOLDEN_YNM = [
    ("yes", OPTS_YNM, 0),
    ("The answer is no.", OPTS_YNM, 1),
    ("maybe", OPTS_YNM, 2),
    # "no" inside "know" / "normal" must NOT match
    ("Based on the known information, the diagnosis is normal.", OPTS_YNM, Abstention.UNPARSEABLE),
    # Conflicting
    ("It could be yes or no", OPTS_YNM, Abstention.UNPARSEABLE),
    # Refusal mentioning "yes" / "no"
    ("As an AI I cannot say yes or no.", OPTS_YNM, Abstention.REFUSAL),
]


class TestParseMcqChoiceYesNoMaybe:
    @pytest.mark.parametrize("text,options,expected", GOLDEN_YNM,
                             ids=[f"ynm_{i}" for i in range(len(GOLDEN_YNM))])
    def test_golden_cases(self, text, options, expected):
        assert parse_mcq_choice(text, options) == expected
