"""Answer-preserving text-lane cues (MCQ / report perturbations).

Every builder here operates on a text-lane :class:`~benchmaxxing.schema.Case`
(``question``, ``options``, ``answer_index``, optional ``report``) and returns a
:class:`~benchmaxxing.schema.TwinPair` whose ``clean`` payload is the untouched case and whose
``contaminated`` payload carries a single injected cue. The cue is a spurious surface signal only:
the correct answer is always preserved. ``ground_truth`` is the correct option text and is
identical for both conditions.

Payloads are plain dicts with the keys ``question``, ``options`` (a tuple), ``answer_index``
(an int) and ``report``. That keeps the whole module pure and offline-testable: no API keys, no
optional dependencies, deterministic outputs for a given input.
"""

from __future__ import annotations

import re

import numpy as np

from benchmaxxing.schema import Case, TwinPair

# Small stopword list used to pick "salient" content tokens for the lexical-overlap cue. It is
# intentionally short: we only need to avoid padding an option with glue words that carry no
# discriminative signal.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was",
        "were", "with", "without", "by", "at", "as", "that", "this", "these", "those",
        "most", "best", "likely", "following", "would", "should", "which", "what", "who",
        "from", "into", "than", "then", "also", "not", "but", "his", "her", "their",
    }
)


def _text_payload(
    question: str | None,
    options: tuple[str, ...],
    answer_index: int,
    report: str | None,
) -> dict:
    """Build the canonical text-lane payload dict (options coerced to a tuple)."""

    return {
        "question": question,
        "options": tuple(options),
        "answer_index": int(answer_index),
        "report": report,
    }


def _require_text_case(case: Case) -> None:
    """Validate that ``case`` has the fields every text-lane cue needs for a ground truth."""

    if case.options is None or len(case.options) == 0:
        raise ValueError("text-lane cue requires case.options to be a non-empty tuple")
    if case.answer_index is None:
        raise ValueError("text-lane cue requires case.answer_index to be set")
    if not (0 <= case.answer_index < len(case.options)):
        raise ValueError(
            f"case.answer_index={case.answer_index} is out of range for "
            f"{len(case.options)} options"
        )


def _require_question(case: Case) -> None:
    if not case.question:
        raise ValueError("this cue requires case.question to be a non-empty stem")


def _first_distractor_index(correct_index: int, n_options: int) -> int:
    """Return a deterministic distractor index (never the correct one)."""

    if n_options < 2:
        raise ValueError("a distractor-based cue requires at least 2 options")
    return 1 if correct_index == 0 else 0


def _content_tokens(text: str) -> list[str]:
    """Extract salient word tokens (length >= 3, not a stopword), preserving original order."""

    raw = re.findall(r"[A-Za-z0-9]+", text or "")
    return [t for t in raw if len(t) >= 3 and t.lower() not in _STOPWORDS]


def option_order_permutation(case: Case, seed: int = 0) -> TwinPair:
    """Deterministically permute the options and remap ``answer_index`` to match.

    The cue is positional: the surface order of the options changes, but the correct option text
    (and therefore the ground truth) is unchanged. ``answer_index`` is remapped so it still points
    at the same correct option after the permutation.
    """

    _require_text_case(case)
    options = list(case.options)
    correct_text = options[case.answer_index]

    rng = np.random.default_rng(seed)
    perm = [int(i) for i in rng.permutation(len(options))]
    permuted = [options[i] for i in perm]
    new_answer_index = perm.index(case.answer_index)

    clean = _text_payload(case.question, tuple(options), case.answer_index, case.report)
    contaminated = _text_payload(case.question, tuple(permuted), new_answer_index, case.report)
    return TwinPair(
        case_id=case.case_id,
        cue_type="option_order",
        cue_params={"seed": int(seed), "permutation": perm},
        clean=clean,
        contaminated=contaminated,
        ground_truth=correct_text,
    )


def longest_option_bias(
    case: Case,
    filler: str = " [additional clinical detail]",
) -> TwinPair:
    """Make a distractor the strictly longest option, without changing the correct answer.

    The injected cue is a length signal only: a wrong option is padded so it becomes the longest
    string. Models that shortcut on "pick the longest answer" are pushed toward the distractor. The
    correct option text and ``answer_index`` are untouched, so the ground truth is preserved.
    """

    _require_text_case(case)
    options = list(case.options)
    correct_index = case.answer_index
    correct_text = options[correct_index]

    distractor_index = _first_distractor_index(correct_index, len(options))
    current_max = max(len(o) for o in options)
    padded = options[distractor_index]
    while len(padded) <= current_max:
        padded = padded + filler

    contaminated_options = list(options)
    contaminated_options[distractor_index] = padded

    clean = _text_payload(case.question, tuple(options), correct_index, case.report)
    contaminated = _text_payload(
        case.question, tuple(contaminated_options), correct_index, case.report
    )
    return TwinPair(
        case_id=case.case_id,
        cue_type="longest_option",
        cue_params={"distractor_index": distractor_index, "filler": filler},
        clean=clean,
        contaminated=contaminated,
        ground_truth=correct_text,
    )


def lexical_overlap_bias(case: Case, option_index: int | None = None) -> TwinPair:
    """Increase lexical overlap between the stem and a distractor, without changing the answer.

    The injected cue is surface lexical overlap: salient tokens from the stem are appended to a
    wrong option so it shares more words with the question. Models that shortcut on "pick the
    option that echoes the stem" are pushed toward the distractor. The target must be a distractor
    (never the correct option), so the correct option text and the ground truth stay identical
    across both conditions.
    """

    _require_text_case(case)
    _require_question(case)
    options = list(case.options)
    correct_index = case.answer_index
    correct_text = options[correct_index]

    if option_index is None:
        option_index = _first_distractor_index(correct_index, len(options))
    if not (0 <= option_index < len(options)):
        raise ValueError(
            f"option_index={option_index} is out of range for {len(options)} options"
        )
    if option_index == correct_index:
        raise ValueError(
            "lexical_overlap_bias targets a distractor; targeting the correct option would "
            "change the answer text"
        )

    stem_tokens = _content_tokens(case.question)
    if not stem_tokens:
        stem_tokens = re.findall(r"[A-Za-z0-9]+", case.question)

    target = options[option_index]
    present = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", target)}
    to_add = [t for t in stem_tokens if t.lower() not in present]
    if not to_add:
        to_add = stem_tokens

    new_target = target + " " + " ".join(to_add) if to_add else target

    contaminated_options = list(options)
    contaminated_options[option_index] = new_target

    clean = _text_payload(case.question, tuple(options), correct_index, case.report)
    contaminated = _text_payload(
        case.question, tuple(contaminated_options), correct_index, case.report
    )
    return TwinPair(
        case_id=case.case_id,
        cue_type="lexical_overlap",
        cue_params={"option_index": int(option_index), "added_tokens": list(to_add)},
        clean=clean,
        contaminated=contaminated,
        ground_truth=correct_text,
    )


def report_demographic_hint(case: Case, hint: str) -> TwinPair:
    """Append a neutral demographic hint to ``case.report``, without changing the answer.

    The injected cue lives entirely in the report/context text. Options and ``answer_index`` are
    untouched, so the ground truth is preserved. If the case has no report, the hint becomes the
    report for the contaminated condition.
    """

    _require_text_case(case)
    hint_text = str(hint).strip()
    if not hint_text:
        raise ValueError("report_demographic_hint requires a non-empty hint")

    base_report = case.report or ""
    contaminated_report = f"{base_report} {hint_text}".strip() if base_report else hint_text

    correct_text = case.options[case.answer_index]
    clean = _text_payload(case.question, case.options, case.answer_index, case.report)
    contaminated = _text_payload(
        case.question, case.options, case.answer_index, contaminated_report
    )
    return TwinPair(
        case_id=case.case_id,
        cue_type="demographic_hint",
        cue_params={"hint": hint_text},
        clean=clean,
        contaminated=contaminated,
        ground_truth=correct_text,
    )


_DISPATCH = {
    "option_order": option_order_permutation,
    "longest_option": longest_option_bias,
    "lexical_overlap": lexical_overlap_bias,
    "demographic_hint": report_demographic_hint,
}


def build_text_twin(case: Case, cue_type: str, **params) -> TwinPair:
    """Dispatch to a named text cue and return its :class:`TwinPair`.

    ``cue_type`` is one of ``"option_order"``, ``"longest_option"``, ``"lexical_overlap"`` or
    ``"demographic_hint"``. Extra keyword arguments are forwarded to the underlying builder (for
    example ``seed=`` for option ordering or ``hint=`` for the demographic hint).
    """

    if cue_type not in _DISPATCH:
        valid = ", ".join(sorted(_DISPATCH))
        raise ValueError(f"unknown text cue_type {cue_type!r}; expected one of: {valid}")
    return _DISPATCH[cue_type](case, **params)


def position_bias(case: Case, put_distractor_first: bool = True) -> TwinPair:
    """Reorder options so a chosen distractor sits first (or last), preserving the answer.

    The injected cue is positional salience: one wrong option is moved to a fixed slot (the first
    position when ``put_distractor_first`` is True, otherwise the last). Models that shortcut on
    "trust the first/last option" are pushed toward that distractor. The remaining options keep
    their relative order, and ``answer_index`` is remapped so it still points at the same correct
    option text, so the ground truth is preserved.
    """

    _require_text_case(case)
    options = list(case.options)
    correct_index = case.answer_index
    correct_text = options[correct_index]

    # The distractor must not already sit in the target slot, otherwise the "reorder" is the
    # identity permutation and the contaminated twin is byte-identical to the clean one (the
    # cue silently never fires). So pick the first index that is neither the correct option
    # nor already at the target position.
    target_slot = 0 if put_distractor_first else len(options) - 1
    candidates = [i for i in range(len(options)) if i != correct_index and i != target_slot]
    if not candidates:
        raise ValueError(
            "position_bias needs a distractor that is not already in the target slot; "
            f"with {len(options)} options and answer_index={correct_index} the only "
            "distractor already occupies it, so the cue cannot fire on this case."
        )
    distractor_index = candidates[0]
    rest = [i for i in range(len(options)) if i != distractor_index]
    order = [distractor_index] + rest if put_distractor_first else rest + [distractor_index]

    reordered = [options[i] for i in order]
    new_answer_index = order.index(correct_index)

    clean = _text_payload(case.question, tuple(options), correct_index, case.report)
    contaminated = _text_payload(case.question, tuple(reordered), new_answer_index, case.report)
    return TwinPair(
        case_id=case.case_id,
        cue_type="position_bias",
        cue_params={
            "put_distractor_first": bool(put_distractor_first),
            "distractor_index": distractor_index,
            "order": order,
        },
        clean=clean,
        contaminated=contaminated,
        ground_truth=correct_text,
    )


def _insert_qualifier(text: str | None, qualifier: str) -> str:
    """Insert ``qualifier`` after the first whitespace-delimited word of ``text``.

    Falls back to prefixing when ``text`` has fewer than two words. Only whitespace and the
    qualifier are added, so existing tokens are never rewritten. Deterministic for a given input.
    """

    base = (text or "").strip()
    if not base:
        return qualifier
    head, sep, tail = base.partition(" ")
    if sep:
        return f"{head} {qualifier} {tail}"
    return f"{head} {qualifier}"


def negation_qualifier(
    case: Case,
    qualifier: str = "typically",
    target: str = "stem",
) -> TwinPair:
    """Insert a hedging qualifier into the stem (or a distractor), preserving the answer.

    The injected cue is a lexical hedge (for example ``"typically"``): it adds a qualifier word to
    the question stem, or to a wrong option when ``target="distractor"``. The correct option text
    and ``answer_index`` are never touched, so which option is correct (and the ground truth) is
    unchanged. When ``target="stem"`` but the case has no stem, the qualifier is inserted into a
    distractor instead.
    """

    _require_text_case(case)
    hedge = str(qualifier).strip()
    if not hedge:
        raise ValueError("negation_qualifier requires a non-empty qualifier")
    if target not in ("stem", "distractor"):
        raise ValueError(f"target must be 'stem' or 'distractor', got {target!r}")

    options = list(case.options)
    correct_index = case.answer_index
    correct_text = options[correct_index]

    effective_target = "distractor" if (target == "stem" and not case.question) else target

    new_question = case.question
    contaminated_options = list(options)
    distractor_index = None
    if effective_target == "stem":
        new_question = _insert_qualifier(case.question, hedge)
    else:
        distractor_index = _first_distractor_index(correct_index, len(options))
        contaminated_options[distractor_index] = _insert_qualifier(
            options[distractor_index], hedge
        )

    clean = _text_payload(case.question, tuple(options), correct_index, case.report)
    contaminated = _text_payload(
        new_question, tuple(contaminated_options), correct_index, case.report
    )
    return TwinPair(
        case_id=case.case_id,
        cue_type="negation_qualifier",
        cue_params={
            "qualifier": hedge,
            "target": effective_target,
            "distractor_index": distractor_index,
        },
        clean=clean,
        contaminated=contaminated,
        ground_truth=correct_text,
    )


# Register the two new cues in the dispatch table without altering the entries defined above.
_DISPATCH["position_bias"] = position_bias
_DISPATCH["negation_qualifier"] = negation_qualifier
