"""Golden regression tests for benchmaxxing.answer.parse_choice.

Pins the exact failure that a naive first-\\b[A-E]\\b parser produced: on a long reasoned response
whose prose starts with the article "A", the naive parser returned option A; the robust parser must
return the option the model actually declared.
"""
from __future__ import annotations

from benchmaxxing.answer import parse_choice

OPTS = ["Desmoplastic", "Nodular", "Acral lentiginous", "Lentigo maligna", "Superficial spreading"]


def test_boxed_letter_beats_leading_article():
    # The exact real failure (medqa-82): prose starts with "A ...", answer declared as \boxed{C}.
    resp = ("A 44-year-old African-American woman is at greatest risk for Acral lentiginous "
            "melanoma. The final answer is $\\boxed{C}$")
    assert parse_choice(resp, OPTS) == "Acral lentiginous"
    # the naive parser would have returned the article-A option:
    assert parse_choice(resp, OPTS) != "Desmoplastic"


def test_declared_answer_letter():
    resp = "After weighing the options, the answer is B."
    assert parse_choice(resp, OPTS) == "Nodular"


def test_named_option_text_last_mention():
    resp = ("Superficial spreading is common in fair skin, but for this patient the correct read "
            "is acral lentiginous.")
    assert parse_choice(resp, OPTS) == "Acral lentiginous"


def test_single_letter_reply():
    assert parse_choice("C", OPTS) == "Acral lentiginous"
    assert parse_choice("  D  ", OPTS) == "Lentigo maligna"


def test_trailing_letter():
    assert parse_choice("Reasoning ... therefore E", OPTS) == "Superficial spreading"


def test_three_option_yes_no_maybe_shape():
    opts = ["yes", "no", "maybe"]
    assert parse_choice("The evidence is inconclusive, so the answer is maybe.", opts) == "maybe"


def test_empty_and_unparseable():
    assert parse_choice("", OPTS) == ""
    assert parse_choice("I cannot answer this.", OPTS) == "I cannot answer this."
