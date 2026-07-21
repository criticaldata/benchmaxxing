"""Answer extraction and abstention normalization for free-text model turns.

This module maps raw string outputs from models into explicit numeric answers,
standardized abstention tokens, or unparseable flags. This prevents safety
refusals or garbled logic from being misclassified as logic flips during evaluation.

TODO(Integration):
    The committee harness parses answers today via experiments._parse / _shortcut_answer.
    This extract.py should become the single source of truth those functions call.
    Wire into ``solo_evaluate`` and the committee backend so both routes share
    one normalizer (see Issue #102).
"""

from __future__ import annotations

import re
from enum import Enum


class Abstention(Enum):
    """Explicit sentinels for non-answers."""
    REFUSAL = "refusal"          # Model actively refused to answer (safety, alignment)
    UNPARSEABLE = "unparseable"  # Model output could not be mapped to an option


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

# Common refusal patterns across major model lineages (Gemini, Llama, Claude, GPT)
_REFUSAL_PATTERNS = (
    r"as an ai",
    r"i cannot provide medical",
    r"i am unable to diagnose",
    r"i cannot determine",
    r"i'm sorry,? but i cannot",
    r"i am an ai language model",
    r"i cannot fulfill this request",
    r"it is not appropriate for me to",
)
_REFUSAL_REGEX = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_abstention(text: str) -> bool:
    """Detect if *text* is a refusal to answer due to safety/alignment filters.

    Returns ``False`` for empty strings, ``None``, or legitimate short answers
    like ``"no"`` or ``"maybe"``.
    """
    if not text:
        return False
    return bool(_REFUSAL_REGEX.search(text))


# ---------------------------------------------------------------------------
# MCQ answer extraction
# ---------------------------------------------------------------------------

# Regex for explicit answer declarations.
# Ordered so that the most specific patterns are tried first.
# Captures a single letter (the answer) in group 1.
_DECLARATION_PATTERNS = (
    # "\boxed{C}" (the LaTeX final-answer box our Gemini MedQA responses end with). Most reliable,
    # and it wins even when a distractor letter sits in the trailing window that Heuristic 2 guards.
    r"\\boxed\{\s*([A-Z])\s*\}",
    # "The final answer is C", "final answer: C"
    r"final\s+answer\s+(?:is|[:=-])\s*([A-Z])\b",
    # "The correct answer is C", "the correct option is B"
    r"correct\s+(?:answer|option)\s+is\s+([A-Z])\b",
    # "The answer is C", "My answer is B"
    r"(?:the|my)\s+answer\s+is\s+([A-Z])\b",
    # "Answer: C", "Answer - C"
    r"answer\s*[:=-]\s*([A-Z])\b",
    # "I choose C", "I would choose B", "I'll choose A"
    r"i(?:'ll|\s+would)?\s+choose\s+([A-Z])\b",
    # "I select C"
    r"i\s+select\s+([A-Z])\b",
)
_DECLARATION_REGEX = re.compile(
    "|".join(_DECLARATION_PATTERNS), re.IGNORECASE
)


def _first_group(m: re.Match) -> str:
    """Return the first non-None group from a regex match with alternations."""
    for g in m.groups():
        if g is not None:
            return g.upper()
    raise ValueError("Match had no captured group")  # pragma: no cover


def parse_mcq_choice(text: str, options: tuple[str, ...] | list[str]) -> int | Abstention:
    """Extract the chosen option index from free text.

    Args:
        text: The raw generation from the model.
        options: The sequence of valid options (e.g., ``("A", "B", "C")``
                 or ``("yes", "no", "maybe")``).

    Returns:
        int: The 0-based index of the chosen option.
        Abstention.REFUSAL: If the model actively refused to answer.
        Abstention.UNPARSEABLE: If the text is ambiguous, missing, or has
            conflicting answers.
    """
    if not text:
        return Abstention.UNPARSEABLE

    if is_abstention(text):
        return Abstention.REFUSAL

    num_options = len(options)
    if num_options == 0:
        return Abstention.UNPARSEABLE

    # ── Branch: non-letter options (e.g. yes / no / maybe) ──────────────
    is_letter_options = all(len(o) == 1 and o.isalpha() for o in options)

    if not is_letter_options:
        text_lower = text.lower()
        matches: set[int] = set()
        for i, opt in enumerate(options):
            if re.search(rf"\b{re.escape(opt.lower())}\b", text_lower):
                matches.add(i)
        if len(matches) == 1:
            return matches.pop()
        return Abstention.UNPARSEABLE

    # ── Branch: standard MCQ letter options (A–E etc.) ──────────────────
    valid_letters = {chr(ord("A") + i) for i in range(min(num_options, 26))}

    # Heuristic 1: Explicit answer declarations.
    # Collect *all* declaration matches and take the **last** one.
    # This beats the length-bias trap where models ramble but conclude
    # with a clear "The answer is X."
    declarations = [
        _first_group(m)
        for m in _DECLARATION_REGEX.finditer(text)
        if _first_group(m) in valid_letters
    ]
    if declarations:
        last = declarations[-1]
        return ord(last) - ord("A")

    # Heuristic 2: Trailing standalone letter.
    # Match standalone valid letters (word-boundary enclosed).
    valid_chars = "".join(sorted(valid_letters))
    pattern = rf"\b([{valid_chars}])\b"
    all_letters = re.findall(pattern, text)

    if not all_letters:
        return Abstention.UNPARSEABLE

    # Ambiguity guard: if the last 40 characters of the text contain
    # more than one *distinct* valid letter, the answer is ambiguous.
    tail = text[-40:]
    tail_unique = {ch.upper() for ch in re.findall(pattern, tail)}
    if len(tail_unique) > 1:
        return Abstention.UNPARSEABLE

    last_letter = all_letters[-1].upper()
    if last_letter not in valid_letters:
        return Abstention.UNPARSEABLE  # defensive; shouldn't happen
    return ord(last_letter) - ord("A")
