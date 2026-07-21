"""Answer extraction and abstention normalization for free-text model turns.

This module maps raw string outputs from models into explicit numeric answers, 
standardized abstention tokens, or unparseable flags. This prevents safety 
refusals or garbled logic from being misclassified as logic flips during evaluation.

TODO(Integration):
The committee harness parses answers today via experiments._parse / _shortcut_answer.
This extract.py should become the single source of truth those functions call.
"""

from __future__ import annotations

import re
from enum import Enum


class Abstention(Enum):
    """Explicit sentinels for non-answers."""
    REFUSAL = "refusal"          # Model actively refused to answer (safety, alignment)
    UNPARSEABLE = "unparseable"  # Model output could not be mapped to an option


# Common refusal patterns across major model lineages
_REFUSAL_PATTERNS = (
    r"as an ai",
    r"i cannot provide medical",
    r"i am unable to diagnose",
    r"i cannot determine",
    r"i'm sorry, but i cannot",
    r"i am an ai language model",
    r"i cannot fulfill this request",
    r"it is not appropriate for me to",
)
_REFUSAL_REGEX = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_abstention(text: str) -> bool:
    """Detect if the text is a refusal to answer due to safety/alignment filters."""
    if not text:
        return False
        
    lower_text = text.lower()
    # Ensure we don't swallow legitimate "yes/no/maybe" or short answers as refusals
    # if they happen to contain substrings (though our regexes are fairly specific).
    # We apply the regex check.
    return bool(_REFUSAL_REGEX.search(text))


def parse_mcq_choice(text: str, options: tuple[str, ...]) -> int | Abstention:
    """Extract the chosen option index from free text.
    
    Args:
        text: The raw generation from the model.
        options: The tuple of valid options (e.g., ("A", "B", "C") or ("yes", "no")).
        
    Returns:
        int: The 0-based index of the chosen option.
        Abstention.REFUSAL: If the model actively refused to answer.
        Abstention.UNPARSEABLE: If the text is ambiguous, missing, or has conflicting answers.
    """
    if not text:
        return Abstention.UNPARSEABLE
        
    if is_abstention(text):
        return Abstention.REFUSAL

    num_options = len(options)
    if num_options == 0:
        return Abstention.UNPARSEABLE

    text_lower = text.lower()

    # 1. Handle non-letter options (e.g., yes/no/maybe)
    # If the options are not single letters (e.g. PubMedQA "yes", "no", "maybe")
    is_letter_options = all(len(o) == 1 and o.isalpha() for o in options)
    
    if not is_letter_options:
        # Check for direct verbatim quotes or explicit mentions of the option
        matches = set()
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            # If the exact option text is surrounded by boundaries
            if re.search(rf"\b{re.escape(opt_lower)}\b", text_lower):
                matches.add(i)
        
        if len(matches) == 1:
            return matches.pop()
        return Abstention.UNPARSEABLE

    # 2. Handle MCQ Letter Options (A, B, C...)
    valid_letters = "".join(chr(ord('A') + i) for i in range(min(num_options, 26)))
    
    # Heuristic 1: Explicit answer declarations.
    # Look for patterns like "The correct answer is A" or "Option B is correct".
    # Find all explicit declarations.
    declarations = re.findall(rf"(?:correct (?:answer|option) is|option)\s*([{valid_letters}])\b", text, re.IGNORECASE)
    if declarations:
        # Prefer the last explicit answer declaration (fights length bias trap)
        last_letter = declarations[-1].upper()
        return ord(last_letter) - ord('A')
        
    # Heuristic 2: Trailing letter format.
    # Often models end with "Therefore, C." or "Answer: C."
    # We look for the last standalone valid letter near the end, avoiding stray letters 
    # that might just be parts of a sentence ("Patient A...").
    
    # Check if there are conflicting standalone letters at the end
    # We find all standalone valid letters in the text
    pattern = rf"\b([{valid_letters}])\b"
    all_letters = re.findall(pattern, text)
    
    if not all_letters:
        return Abstention.UNPARSEABLE
        
    # Filter out letters that are just part of the options text if possible,
    # but the simplest robust way is to just grab the very last standalone letter
    # as the model's final conclusion. 
    # If the text explicitly has conflicting conclusions, we should return UNPARSEABLE.
    # Let's check the last few tokens. If it says "A and B", we shouldn't just guess B.
    # For now, picking the absolute last standalone letter is a good baseline, 
    # but let's see if the last 15 characters contain multiple different valid letters.
    tail = text[-30:]
    tail_letters = set(re.findall(pattern, tail))
    if len(tail_letters) > 1:
         return Abstention.UNPARSEABLE
         
    last_letter = all_letters[-1].upper()
    return ord(last_letter) - ord('A')
