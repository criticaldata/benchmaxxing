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
    # "\boxed{C}" and Gemini 2.5's "\boxed{\text{C. ...}}" reasoning box (also \textbf/\mathrm,
    # optional emphasis or a leading paren). The LaTeX final-answer box our Gemini responses end
    # with, and the most reliable signal: it wins even when a distractor letter sits in the trailing
    # window that Heuristic 2 guards. Only the bare-letter form was matched before, so the far more
    # common \text{} form fell through to Heuristic 2 and mis-scored decisive replies as abstentions.
    # The (?![A-Za-z]) guard is load-bearing: without it \boxed{\text{Anti-mitochondrial
    # antibodies}} captures the "A" of "Anti" and scores a full-text answer as option A. The bare
    # \boxed{C} and \boxed{\text{C. ...}} forms still match, since C is followed by } or ".".
    r"\\boxed\{\s*(?:\\(?:text|textbf|mathrm)\{)?\s*[*_$]*\(?([A-Z])(?![A-Za-z])",
    # "The final answer is C", "final answer: C"
    r"final\s+answer\s+(?:is|[:=-])\s*[*_]*\(?([A-Z])\b",
    # "The correct answer is C", "the correct option is B"
    r"correct\s+(?:answer|option)\s+is\s+[*_]*\(?([A-Z])\b",
    # "The answer is C", "My answer is B"
    r"(?:the|my)\s+answer\s+is\s+[*_]*\(?([A-Z])\b",
    # "Answer: C", "Answer - C"
    r"answer\s*[:=-]\s*[*_]*\(?([A-Z])\b",
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


def parse_yesno(text: str) -> str:
    """Robustly extract 'yes' or 'no' from a model's free-text response.
    
    Uses word-boundary matching to prevent orthographic false positives 
    (e.g., 'cannot' or 'phenomenon' matching 'no'). Searches from the end 
    of the text backwards (by taking the last match) to properly handle 
    Chain-of-Thought (CoT) where a model deliberates before concluding.
    
    Returns:
        'yes' or 'no' if found, otherwise '?'.
    """
    t = (text or "").strip().lower()
    if not t:
        return "?"
    
    matches = list(re.finditer(r"\b(yes|no)\b", t))
    if matches:
        return matches[-1].group(1)
        
    return "?"


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

    # ── Branch: non-letter options (e.g. yes / no / maybe or full text) ──────────────
    is_letter_options = all(len(o) == 1 and o.isalpha() for o in options)

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

    if not is_letter_options:
        text_lower = text.lower()
        matches: set[int] = set()
        for i, opt in enumerate(options):
            # (?<!\w)...(?!\w) rather than \b...\b: a clinical option very often ends in a
            # full stop, and \b after a literal "." can never match, since it needs a word
            # character on the far side. That made every such option invisible to this search,
            # so a reply like "B. A type 1 error occurs when ..." came back unparseable even
            # though it quotes the option verbatim. Lookarounds handle the punctuation case.
            if re.search(rf"(?<!\w){re.escape(opt.lower())}(?!\w)", text_lower):
                matches.add(i)
        if len(matches) == 1:
            return matches.pop()
        if matches:
            # Multi-match must NOT fall through to the trailing-letter scan: with options like
            # "Type II pneumocytes" a bare "A" in the prose is almost always the English article,
            # so falling through turned "... plausible. A surfactant deficiency ..." into a
            # confident vote for option A.
            ranked = sorted((text_lower.rfind(options[i].lower()), i) for i in matches)
            last_pos, last_i = ranked[-1]
            prev_pos, prev_i = ranked[-2]
            gap = text_lower[prev_pos + len(options[prev_i]):last_pos]
            if re.fullmatch(r"\s*(?:or|/|versus|vs\.?)\s*", gap):
                # An explicit disjunction ("yes or no"), not a conclusion. Abstaining is the honest
                # read and it is what this module's own yes/no/maybe goldens require.
                return Abstention.UNPARSEABLE
            # Otherwise keep the pre-centralization behaviour, the last-mentioned option, so this
            # refactor moves no committed number.
            return last_i
        # No option text present at all. Legacy still allowed a bare or trailing letter here, which
        # is how a reply of just "D" resolved against full-text options, so fall through.

    # ── Branch: standard MCQ letter options (A–E etc.) ──────────────────

    # A reply that is nothing but a single letter, in either case. The pre-centralization parser had
    # this as its last fallback (`len(t) == 1 and t.upper() in letters`) and it is the shape a model
    # uses when it answers tersely. Handled explicitly rather than by making the scan below
    # case-insensitive, because a case-insensitive standalone-letter scan over prose would match the
    # lowercase article "a" and reintroduce the bug that scan is guarded against.
    stripped = text.strip()
    if len(stripped) == 1 and stripped.upper() in valid_letters:
        return ord(stripped.upper()) - ord("A")

    # Heuristic 2: Trailing standalone letter.
    # Match standalone valid letters (word-boundary enclosed).
    valid_chars = "".join(sorted(valid_letters))
    pattern = rf"\b([{valid_chars}])\b"
    all_letters = re.findall(pattern, text)

    if not all_letters:
        return Abstention.UNPARSEABLE

    if not is_letter_options:
        # For full-text options this is the last resort, reached only when no option string appeared
        # in the reply. The pre-centralization parser anchored it at the END of the text
        # (r"\b([A-Z])\b\s*[.)]?\s*$"), which matters: scanning the whole reply instead picks up a
        # LEADING article, so "A biopsy of the mass is likely to show myxoma." resolved to option A.
        # Measured on real cached replies that accounted for every remaining disagreement with the
        # old parser. Keep the anchor.
        tail_match = re.search(rf"\b([{valid_chars}])\b\s*[.)]?\s*$", text.strip(), re.IGNORECASE)
        if not tail_match:
            return Abstention.UNPARSEABLE
        return ord(tail_match.group(1).upper()) - ord("A")

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

# ---------------------------------------------------------------------------
# Legacy Migration Wrapper
# ---------------------------------------------------------------------------

def parse_legacy_string(text: str, options: tuple[str, ...] | list[str]) -> str:
    """Wrapper around `parse_mcq_choice` that maintains the legacy string return type.
    
    In previous ad-hoc implementations (`_parse`, `_parse_choice`), an abstention or
    unparseable response implicitly returned `""` or `None`. This wrapper explicitly
    maps `Abstention` to `""`, and valid indices back to their string representation
    (`options[index]`).
    
    Args:
        text: The raw generation from the model.
        options: The sequence of valid options.
        
    Returns:
        The matched option string, or `""` if unparseable/refused.
    """
    ans = parse_mcq_choice(text, options)
    if isinstance(ans, Abstention):
        return ""
    return options[ans]
