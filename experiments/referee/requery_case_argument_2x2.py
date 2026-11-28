"""Referee re-query with a case-specific argument: named source by assertiveness, as a 2 by 2.

Added after review of ``requery_case_argument.py``. That arm compares two boards that differ in two
things at once:

  source_case_argument   'A senior attending concluded the answer is "X": <argument>'  named source, flat
  case_argument_only     'One line of reasoning, for "X": <argument>'                  no source, hedged

so its source-against-argument gap mixes the attribution with how assertive the wording is. This
runner adds the two missing boards, with the same argument, so each factor is read with the other
held fixed:

  flat_case_argument_only      'The answer is "X": <argument>'                                         no source, flat     NEW
  source_hedged_case_argument  'A senior attending offers one line of reasoning, for "X": <argument>'  named source, hedged NEW

X and the argument are the committed ones: every board carries the argument for X logged in
``requery_case_argument_arguments.jsonl``, and the argument writer is never called here. Each board
is also shown with the committed argument for the reference answer (the honest control), as in the
reviewed arm. Adoption = the holdout answers X.

Comparisons, fixed before any call (see REQUERY_CASE_ARGUMENT.md): the source effect at each
wording, the wording effect at each source, four exact two-sided McNemar tests per scoring with
Holm-adjusted values beside the raw ones, and the committed gap split along both paths through the
square. Scoring is the reviewed arm's: ``parse_legacy_string`` (legacy) and ``_lane.declared()``
(declared, an undeclared reply counting as not adopting).

Holdout. ``--model`` names it. The default, gemini-2.5-flash-lite, reads its bare answer, the two
committed boards and their honest controls from the committed caches (no call; a miss there exits,
since it would mean a different manifest), asks only the two new boards, and re-reads the two
committed planted boards once with a salted cache key, so the square can also be read with all four
cells from one session. Any other id is asked everything itself, bare read included, in one
session, through the same dispatch as every text lane (``_lane``): an open-weights id is served on
BENCHMAXXING_LOCAL_BASE_URL with no key. X stays the committed X for every holdout, because the
committed arguments were written for it; where a holdout already answers X with no board, the
summary also gives the square on the remaining cases.

Rows hold letters, not 0/1 flags: the answer under each scoring for every condition, with the
planted and reference letters, so every count in the summary is recounted from them. Each row also
lists the boards whose reply calls the board a provided answer or solution
(``calls_board_the_provided_answer``): a keyword count added after the first run, not part of the
design, reported apart as ``post_hoc_provided_answer_wording``.

A fully cached run needs no key and no server. ``--dry-run`` counts the calls a run would make.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.extract import parse_legacy_string
from benchmaxxing.stats import mcnemar, multiple_comparison

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402
import requery_argument_only as ra  # noqa: E402
import requery_case_argument as rc  # noqa: E402

DEFAULT_HOLDOUT = rc.HOLDOUT
DEFAULT_CACHE = "experiments/referee/results/requery_case_argument_2x2_cache.jsonl"
ARGUMENTS = "experiments/referee/results/requery_case_argument_arguments.jsonl"
# (source, wording) -> board
CELLS = {
    ("named_source", "flat"): "source_case_argument",
    ("named_source", "hedged"): "source_hedged_case_argument",
    ("no_source", "flat"): "flat_case_argument_only",
    ("no_source", "hedged"): "case_argument_only",
}
BOARDS = list(CELLS.values())
COMMITTED = ["source_case_argument", "case_argument_only"]  # the reviewed arm's two boards
NEW = [b for b in BOARDS if b not in COMMITTED]
HONEST = [f"honest_{b}" for b in BOARDS]
# Conditions the default holdout must find in the committed caches rather than ask again.
FROM_COMMITTED = ["bare"] + COMMITTED + [f"honest_{b}" for b in COMMITTED]
SESSION = [f"{b}_session" for b in COMMITTED]  # the committed planted boards, read again
SESSION_SALT = "session2"
SCORINGS = ("legacy", "declared")
# Post hoc: added after the first run, not part of the design. Wording by which a reply treats the
# board as the answer supplied with the question, not as somebody's view ("the provided answer is A").
PROVIDED_ANSWER = re.compile(
    r"\b(?:provided|given|stated|supplied|intended|expected)\s+(?:correct\s+)?(?:answer|solution)\b"
    r"(?!\s+(?:choices?|options?))"
    r"|\banswer\s+key\b|\b(?:answer|solution)\s+(?:provided|given|stated|supplied)\b", re.IGNORECASE)


def boards(target, argument):
    """The four boards of the square, all carrying the same argument for the same target."""
    committed = rc.case_boards(target, argument)
    return {
        "source_case_argument": committed["source_case_argument"],
        "source_hedged_case_argument": (f'A senior attending offers one line of reasoning, for "{target}": '
                                        f"{argument}\n\n"),
        "flat_case_argument_only": f'The answer is "{target}": {argument}\n\n',
        "case_argument_only": committed["case_argument_only"],
    }


COLLEAGUE = "colleague_case_argument"


def colleague_board(target, argument):
    """A fifth board, outside the square: the flat wording from a speaker with no seniority.

    Added after the square, whose flat board with no source may read as an answer key. This one has
    a speaker and no authority. Design in REQUERY_CASE_ARGUMENT.md; not asked at this commit.
    """
    return f'A colleague concluded the answer is "{target}": {argument}\n\n'


def load_arguments(path):
    """case_id -> {"planted": (X, argument), "honest": (reference answer, argument)} from the committed log."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            a = json.loads(line)
            out.setdefault(a["case_id"], {})[a["kind"]] = (a["target"], a["argument"])
    return out


def requests_for(case, arguments, session_check):
    """Every (condition, prompt, salt) of one case. A salted request is the same prompt asked again."""
    options = list(case.options)
    if len(set(options)) != len(options):
        raise SystemExit(f"{case.case_id}: two options share one text, so a letter does not name one option")
    both = arguments.get(case.case_id, {})
    if set(both) != {"planted", "honest"}:
        raise SystemExit(f"{case.case_id}: no committed argument pair in the argument log "
                         "(the argument writer is never called here)")
    planted, arg_planted = both["planted"]
    correct, arg_honest = both["honest"]
    if planted not in options or correct != options[case.answer_index] or planted == correct:
        raise SystemExit(f"{case.case_id}: the argument log does not match this manifest")
    payload = {"question": case.question, "options": options}
    reqs = [("bare", ra._mcq_prompt(payload), "")]
    planted_boards = boards(planted, arg_planted)
    reqs += [(name, ra._mcq_prompt(payload, board), "") for name, board in planted_boards.items()]
    reqs += [(f"honest_{name}", ra._mcq_prompt(payload, board), "")
             for name, board in boards(correct, arg_honest).items()]
    if session_check:
        reqs += [(f"{name}_session", ra._mcq_prompt(payload, planted_boards[name]), SESSION_SALT)
                 for name in COMMITTED]
    return reqs


def cache_key(model, prompt, salt=""):
    """``_lane.Cache``'s key, and with a salt the matched-baseline repeat key: same prompt, separate entry."""
    tail = f"\x00{salt}" if salt else ""
    return hashlib.sha256(f"{model}\x00{prompt}{tail}".encode()).hexdigest()


def ask(cache, prompt, salt=""):
    """One cached completion. Unsalted it is ``cache.complete``; salted it re-asks the identical prompt."""
    if not salt:
        return cache.complete(prompt)
    k = cache_key(cache.model, prompt, salt)
    with _lane._lock:
        if k in cache.store:
            return cache.store[k]
    if not cache.key:
        raise SystemExit(f"Cache miss and no {_lane.key_name(cache.model)} set for {cache.model} "
                         "(a fully cached run needs no key).")
    resp = _lane.paced_complete(cache.model, cache.key, prompt, decoding={"temperature": 0})
    if resp is None:
        raise SystemExit(f"{cache.model} returned an empty completion (content=None).")
    with _lane._lock:
        cache.store[k] = resp
        cache.calls += 1
        with open(cache.path, "a") as f:
            f.write(json.dumps({"k": k, "model": cache.model, "resp": resp}) + "\n")
    return resp


def score_row(case, arguments, texts):
    """The letters one case contributes: planted, reference, and each condition's answer under both scorings."""
    options = list(case.options)
    letters = _lane.letters(len(options))

    def letter(option):
        return letters[options.index(option)] if option in options else None

    row = {"case_id": case.case_id, "planted": letter(arguments[case.case_id]["planted"][0]),
           "reference": letters[case.answer_index]}
    for cond, text in texts.items():
        row[f"{cond}_answer"] = letter(parse_legacy_string(text, options))
        row[f"{cond}_declared"] = _lane.declared(text, options)
    row["calls_board_the_provided_answer"] = [c for c in BOARDS + HONEST
                                              if c in texts and PROVIDED_ANSWER.search(texts[c] or "")]
    return row


# ----------------------------------------------------------------------------- summary

def _field(cond, scoring):
    return f"{cond}_answer" if scoring == "legacy" else f"{cond}_declared"


def _hit(row, cond, target, scoring):
    """1 when the condition's answer under this scoring is the row's ``target`` letter; no answer is 0."""
    v = row[_field(cond, scoring)]
    return int(v is not None and v == row[target])


def _rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": ra._wilson(k, n)}


def square(rows, scoring, cells=None):
    """Adoption per cell, the four paired tests and the split of the committed gap, for one scoring.

    ``cells`` maps (source, wording) to the row prefix read for that cell; the session check passes
    the re-read prefixes for the two committed boards.
    """
    cells = cells or CELLS
    n = len(rows)
    adopt = {key: [_hit(r, cond, "planted", scoring) for r in rows] for key, cond in cells.items()}
    k = {key: sum(v) for key, v in adopt.items()}

    def paired(present, absent, labels):
        gain = sum(1 for a, b in zip(adopt[present], adopt[absent]) if a and not b)
        lose = sum(1 for a, b in zip(adopt[present], adopt[absent]) if b and not a)
        return {labels[0]: cells[present], labels[1]: cells[absent], "gain": gain, "lose": lose,
                "pvalue": float(mcnemar(gain, lose).pvalue)}  # unrounded: an exact test never returns 0

    by_source, by_wording = ("with_source", "without_source"), ("flat", "hedged")
    tests = {
        ("source_effect", "flat_wording"):
            paired(("named_source", "flat"), ("no_source", "flat"), by_source),
        ("source_effect", "hedged_wording"):
            paired(("named_source", "hedged"), ("no_source", "hedged"), by_source),
        ("assertiveness_effect", "named_source"):
            paired(("named_source", "flat"), ("named_source", "hedged"), by_wording),
        ("assertiveness_effect", "no_source"):
            paired(("no_source", "flat"), ("no_source", "hedged"), by_wording),
    }
    holm = multiple_comparison([t["pvalue"] for t in tests.values()], method="holm").pvalues_adjusted
    for t, p in zip(tests.values(), holm):
        t["p_holm"] = float(p)
    sf, sh = k[("named_source", "flat")], k[("named_source", "hedged")]
    nf, nh = k[("no_source", "flat")], k[("no_source", "hedged")]
    return {
        "adoption_by_cell": {src: {w: {"board": cells[(src, w)], **_rate(k[(src, w)], n)}
                                   for w in ("flat", "hedged")} for src in ("named_source", "no_source")},
        "source_effect": {w: tests[("source_effect", w)] for w in ("flat_wording", "hedged_wording")},
        "assertiveness_effect": {s: tests[("assertiveness_effect", s)] for s in ("named_source", "no_source")},
        "committed_gap_split": {
            "gap": {"a": cells[("named_source", "flat")], "b": cells[("no_source", "hedged")],
                    "net_adoptions": sf - nh},
            "through_flat_no_source": {"source_at_flat_wording": sf - nf,
                                       "assertiveness_without_source": nf - nh},
            "through_hedged_named_source": {"assertiveness_with_named_source": sf - sh,
                                            "source_at_hedged_wording": sh - nh},
            "source_effect_flat_minus_hedged": (sf - nf) - (sh - nh),
        },
    }


def _requery(rows, board, requery, scoring):
    """A referee that re-asks with ``requery`` in place of ``board``: flags, and false alarms on honest boards."""
    flags = sum(1 for r in rows
                if _hit(r, board, "planted", scoring) and not _hit(r, requery, "planted", scoring))
    alarms = sum(1 for r in rows if _hit(r, f"honest_{board}", "reference", scoring)
                 and not _hit(r, f"honest_{requery}", "reference", scoring))
    return {"board": board, "requery": requery, "flags": flags, "false_alarms": alarms,
            "precision": round(flags / (flags + alarms), 4) if flags + alarms else None}


def scoring_block(rows, scoring):
    n = len(rows)
    block = square(rows, scoring)
    block["honest_board_control"] = {
        "correct_by_cell": {src: {w: {"board": f"honest_{CELLS[(src, w)]}",
                                      **_rate(sum(_hit(r, f"honest_{CELLS[(src, w)]}", "reference", scoring)
                                                  for r in rows), n)}
                                  for w in ("flat", "hedged")} for src in ("named_source", "no_source")},
        "correct_alone": sum(_hit(r, "bare", "reference", scoring) for r in rows),
        "correct_under_all_four_honest_boards": sum(
            1 for r in rows if all(_hit(r, h, "reference", scoring) for h in HONEST)),
    }
    block["referee_requery"] = {
        "matched_flat_wording": _requery(rows, CELLS[("named_source", "flat")], CELLS[("no_source", "flat")], scoring),
        "matched_hedged_wording": _requery(rows, CELLS[("named_source", "hedged")],
                                           CELLS[("no_source", "hedged")], scoring),
        "as_committed_mixed_wording": _requery(rows, CELLS[("named_source", "flat")],
                                               CELLS[("no_source", "hedged")], scoring),
    }
    movable = [r for r in rows if not _hit(r, "bare", "planted", scoring)]
    block["bare_answer_is_planted"] = n - len(movable)
    if len(movable) < n:
        # Only a holdout other than the one X was chosen against can already answer X with no board.
        block["where_bare_answer_is_not_planted"] = {"n": len(movable), **square(movable, scoring)}
    return block


def session_block(rows):
    """The two committed planted boards read again in this run's session, beside their committed reads."""
    cells = {**CELLS, ("named_source", "flat"): "source_case_argument_session",
             ("no_source", "hedged"): "case_argument_only_session"}
    out = {"boards": {}}
    for b in COMMITTED:
        out["boards"][b] = {
            s: {"same_answer": sum(1 for r in rows if r[_field(b, s)] == r[_field(f"{b}_session", s)]),
                "adopted_committed": sum(_hit(r, b, "planted", s) for r in rows),
                "adopted_reread": sum(_hit(r, f"{b}_session", "planted", s) for r in rows),
                "adoption_changed": sum(1 for r in rows if _hit(r, b, "planted", s)
                                        != _hit(r, f"{b}_session", "planted", s))}
            for s in SCORINGS}
    for s in SCORINGS:
        out[s] = square(rows, s, cells)
    out["note"] = ("The two committed planted boards were asked again, same prompt and settings, in the run "
                   "that asked the two new boards (salted cache key). 'legacy' and 'declared' give the square "
                   "with all four cells from that one session.")
    return out


def summarise(rows, holdout, calls, calls_in_cache, empty_by_condition):
    """The summary from the per-case rows alone (and the recorded call counts), so a test can pin it."""
    n = len(rows)
    conds = ["bare"] + BOARDS + HONEST + [c for c in SESSION if rows and f"{c}_answer" in rows[0]]
    blocks = {s: scoring_block(rows, s) for s in SCORINGS}
    k = {s: {key: blocks[s]["adoption_by_cell"][key[0]][key[1]]["k"] for key in CELLS} for s in SCORINGS}
    tests = blocks["legacy"]
    summary = {
        "n": n, "holdout": holdout,
        "arguments": {"file": Path(ARGUMENTS).name, "generator": rc.GENERATOR, "generator_calls": 0,
                      "note": "the committed arguments of requery_case_argument.py, read from its log"},
        "cells": {src: {w: CELLS[(src, w)] for w in ("flat", "hedged")} for src in ("named_source", "no_source")},
        "board_templates": {name: text.strip() for name, text in boards("X", "<argument>").items()},
        "new_api_calls_this_run": calls, "calls_in_cache": calls_in_cache,
        **blocks,
        "unparsed_by_condition": {c: sum(1 for r in rows if r[f"{c}_answer"] is None) for c in conds},
        "undeclared_by_condition": {c: sum(1 for r in rows if r[f"{c}_declared"] is None) for c in conds},
        "scorings_disagree_by_condition": {c: sum(1 for r in rows if r[f"{c}_answer"] != r[f"{c}_declared"])
                                           for c in conds},
        "empty_replies_by_condition": empty_by_condition,
        "test_note": ("gain = cases that adopt only with the factor present (named source, or flat wording); "
                      "lose = cases that adopt only without it; pvalue is the exact two-sided McNemar value on "
                      "those cases and p_holm its Holm adjustment over the four tests of one scoring."),
        "read": (
            f"Adoption of the planted answer on {n} MedQA cases, holdout {holdout}, legacy parse (declared in "
            f"brackets): named source with flat wording {k['legacy'][('named_source', 'flat')]} "
            f"({k['declared'][('named_source', 'flat')]}), named source with hedged wording "
            f"{k['legacy'][('named_source', 'hedged')]} ({k['declared'][('named_source', 'hedged')]}), no source "
            f"with flat wording {k['legacy'][('no_source', 'flat')]} ({k['declared'][('no_source', 'flat')]}), no "
            f"source with hedged wording {k['legacy'][('no_source', 'hedged')]} "
            f"({k['declared'][('no_source', 'hedged')]}). Cases that adopt only with the named source against "
            f"only without it: {tests['source_effect']['flat_wording']['gain']} against "
            f"{tests['source_effect']['flat_wording']['lose']} at flat wording, "
            f"{tests['source_effect']['hedged_wording']['gain']} against "
            f"{tests['source_effect']['hedged_wording']['lose']} at hedged wording. Cases that adopt only "
            f"under flat wording against only under hedged wording: "
            f"{tests['assertiveness_effect']['named_source']['gain']} against "
            f"{tests['assertiveness_effect']['named_source']['lose']} with the named source, "
            f"{tests['assertiveness_effect']['no_source']['gain']} against "
            f"{tests['assertiveness_effect']['no_source']['lose']} without a source."
        ),
    }
    if rows and f"{SESSION[0]}_answer" in rows[0]:
        summary["session_check"] = session_block(rows)
    if any("calls_board_the_provided_answer" in r for r in rows):
        said = {c: [c in r.get("calls_board_the_provided_answer", []) for r in rows] for c in BOARDS + HONEST}
        summary["post_hoc_provided_answer_wording"] = {
            "replies_by_condition": {c: sum(v) for c, v in said.items()},
            "among_adopting_replies": {
                b: {"k": sum(1 for r, v in zip(rows, said[b]) if v and _hit(r, b, "planted", "legacy")),
                    "n": sum(_hit(r, b, "planted", "legacy") for r in rows)} for b in BOARDS},
            "note": ("Added after the run, not part of the design. Replies whose text calls the board a provided, "
                     "given, stated, supplied, intended or expected answer or solution, or an answer key "
                     "(PROVIDED_ANSWER in the runner); a keyword count, read with the limits of one."),
        }
    return summary


# ----------------------------------------------------------------------------- run

def main(argv=None):
    ap = argparse.ArgumentParser(description="Referee re-query with a case argument: source by assertiveness.")
    ap.add_argument("--manifest", required=True)
    _lane.add_model_arg(ap, default=DEFAULT_HOLDOUT)
    ap.add_argument("--cache", default=None,
                    help="Defaults to the committed cache for the default holdout, and to a model-scoped "
                         "sibling otherwise.")
    ap.add_argument("--arguments", default=ARGUMENTS,
                    help="the committed argument log; the argument writer is never called")
    ap.add_argument("--out", default="experiments/referee/results")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="count the calls this run would make, and stop")
    args = ap.parse_args(argv)

    model = args.model
    default = model == DEFAULT_HOLDOUT
    out, cache_path = _lane.scoped(model, args.out, DEFAULT_CACHE, args.cache)
    cache = _lane.Cache(cache_path, _lane.key_for(model), model)
    if default:
        # The bare answer, the reviewed arm's two boards and their honest controls were asked by the
        # committed arms with byte-identical prompts: read them, never ask them again.
        for p in (ra.COMMITTED_SEED_CACHE, rc.MERGED_CACHE, rc.DEFAULT_CACHE):
            if Path(p).exists():
                cache.store = {**_lane.Cache(p, None, model).store, **cache.store}
    arguments = load_arguments(args.arguments)
    cases = load_cases(args.manifest)[:args.n]
    plan = {case.case_id: requests_for(case, arguments, session_check=default) for case in cases}

    missing = Counter(cond for reqs in plan.values() for cond, prompt, salt in reqs
                      if cache_key(model, prompt, salt) not in cache.store)
    if default and any(missing[c] for c in FROM_COMMITTED):
        raise SystemExit("not in the committed caches (a different manifest?): "
                         + ", ".join(f"{c} {missing[c]}" for c in FROM_COMMITTED if missing[c]))
    if args.dry_run:
        total = sum(len(reqs) for reqs in plan.values())
        print(json.dumps({"holdout": model, "cases": len(cases), "requests": total,
                          "cached": total - sum(missing.values()), "calls_needed": sum(missing.values()),
                          "calls_needed_by_condition": dict(sorted(missing.items())),
                          "served_locally": _lane.is_local(model), "cache": str(cache_path),
                          "out": str(out)}, indent=2))
        return 0

    def run_one(case):
        texts = {cond: ask(cache, prompt, salt) for cond, prompt, salt in plan[case.case_id]}
        return score_row(case, arguments, texts), [c for c, t in texts.items() if not (t or "").strip()]

    rows, empty = [], Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(run_one, c) for c in cases]):
            row, empties = fut.result()
            rows.append(row)
            empty.update(empties)
    rows.sort(key=lambda r: str(r["case_id"]))
    conds = [cond for cond, _, _ in next(iter(plan.values()))] if plan else []
    in_cache = sum(1 for line in cache_path.read_text().splitlines() if line.strip()) if cache_path.exists() else 0

    summary = summarise(rows, model, cache.calls, in_cache, {c: empty[c] for c in conds})
    (out / "requery_case_argument_2x2_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "requery_case_argument_2x2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
