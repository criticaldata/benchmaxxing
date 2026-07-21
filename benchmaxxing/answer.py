"""Robust extraction of a multiple-choice answer from a free-text model response.

This is the single source of truth for MCQ answer parsing. Every experiment must use
:func:`parse_choice` rather than re-implementing extraction, because a naive
``re.search(r"\\b([A-E])\\b", text)`` grabs the first standalone capital in a long reasoned
response, which is almost always the indefinite article "A" ("A 44-year-old woman ..."), and so
mis-scores the large majority of answers as option A. See ``tests/test_answer.py`` for the golden
regression cases, including that exact failure.
"""
from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\{\s*([A-E])\s*\}")
_DECLARED = re.compile(r"(?:final answer|correct answer|the answer|answer)\s*(?:is|:)?\s*\**\(?([A-E])\)?\b", re.I)
_TRAILING = re.compile(r"\b([A-E])\b\s*[.)]?\s*$")


def _letters(n: int) -> list[str]:
    return [chr(65 + i) for i in range(n)]


def parse_choice(text: str, options: list[str]):
    """Map a model completion back to one of ``options`` (returns the option's text).

    Priority, most reliable first:
      1. an explicit ``\\boxed{X}`` letter (last occurrence);
      2. an explicit "the/final/correct answer is X" letter (last occurrence);
      3. the option TEXT the model names, taking the last mention (models conclude with the answer);
      4. a trailing standalone letter at the very end of the response;
      5. a single-character reply.
    Falls back to the stripped text if nothing matches. Never uses a bare mid-sentence letter,
    which is what let the article "A" corrupt the naive parser.
    """
    if not text:
        return ""
    t = text.strip()
    letters = _letters(len(options))

    m = _BOXED.findall(t) or _DECLARED.findall(t)
    if m and m[-1].upper() in letters:
        return options[letters.index(m[-1].upper())]

    low = t.lower()
    hits = [(low.rfind(o.lower()), o) for o in options if o and o.lower() in low]
    hits = [(p, o) for p, o in hits if p >= 0]
    if hits:
        return max(hits)[1]

    m = _TRAILING.search(t.upper())
    if m and m.group(1) in letters:
        return options[letters.index(m.group(1))]

    if len(t) == 1 and t.upper() in letters:
        return options[letters.index(t.upper())]

    return t
