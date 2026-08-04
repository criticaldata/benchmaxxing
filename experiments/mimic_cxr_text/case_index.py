"""The `case_index` the de-identified MIMIC-CXR text rows are keyed on, in one place.

Why this module exists
----------------------
The lane used to commit `case_id` values of the form `mimic-cxr-text-<subject_id>-<study_id>`,
which are real MIMIC subject and study identifiers, so they were replaced with a stable
`case_index` (the same treatment #372 gave the imaging lane). Four arms
(`break_it_a.py`, `break_it_d.py`, `push_c.py`, `deliberation_framing.py`) select their hard-case
cohort by reading the committed `solo_records.jsonl` and intersecting it with the manifest, so
after the migration they needed to join on the index rather than on the id. Putting the rule here
rather than inlining it four times keeps one definition of what an index means: four copies of a
join key is how the two sides drift apart.

The rule, exactly
-----------------
An index is the rank of a case's `case_id` in the sorted set of **every** id the lane's committed
results mention, which is 633 ids, not the 600 in `solo_records.jsonl`. The referee, judge,
blind-metric and cascade arms each drew cases outside the n=600 solo sample, contributing the
other 33.

That is why the map cannot be rebuilt from a manifest alone, which is what makes
`INDEX_SPACE_SIZE` load-bearing rather than decorative: ranking the 600 solo ids gives a different
number on 599 of 600 rows, and a manifest holding only the solo sample cannot even represent the
top index (632). `build_index_map` therefore requires a manifest that covers the whole index space
and fails loudly when handed one that does not, instead of silently producing a plausible-looking
but wrong cohort. The de-identified rows on their own cannot recover the mapping (that is the
point of de-identifying them), so rebuilding it needs the credentialed manifest.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# Number of distinct case_ids across every committed result file in this lane. The index is a rank
# within this set, so a manifest has to cover all of it to reproduce the mapping.
INDEX_SPACE_SIZE = 633


class CaseIndexError(RuntimeError):
    """The committed index map cannot be reproduced from the manifest supplied."""


def build_index_map(cases) -> dict[str, int]:
    """Map each case's `case_id` to its committed `case_index`.

    `cases` is a `load_cases` result. Raises `CaseIndexError` unless the manifest covers the whole
    index space, since a partial manifest silently yields a different ranking rather than an error.
    """
    ids = sorted({c.case_id for c in cases})
    if len(ids) != INDEX_SPACE_SIZE:
        raise CaseIndexError(
            f"case_index is a rank over all {INDEX_SPACE_SIZE} case_ids this lane's committed "
            f"results mention, but this manifest holds {len(ids)}. Ranking a different set "
            "produces a different index for nearly every case, so the hard-case cohort would be "
            "silently wrong rather than absent. Pass the manifest covering the full lane "
            "(the n=600 solo manifest alone is not enough: the referee, judge, blind-metric and "
            "cascade arms drew cases outside that sample)."
        )
    return {cid: i for i, cid in enumerate(ids)}


def hard_case_indices(records_path) -> set[int]:
    """The `case_index`es the holdout got wrong on the clean prompt, from `solo_records.jsonl`.

    Accepts the de-identified rows (`case_index`) and, so an older local un-migrated file still
    works, rows that predate the migration (`case_id`); in that case the ids are returned as-is
    and `hard_cases` compares them directly.
    """
    wrong = defaultdict(list)
    for line in Path(records_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "case_index" in r:
            k = r["case_index"]
        elif "case_id" in r:
            k = r["case_id"]
        else:
            raise CaseIndexError(
                f"{records_path} has a row with neither case_index nor case_id: {sorted(r)}"
            )
        wrong[k].append(r.get("clean_correct"))
    return {k for k, flags in wrong.items() if any(f is False for f in flags)}


def hard_cases(cases, records_path, limit=None):
    """The cases in `cases` that `solo_records.jsonl` marks hard, in manifest order.

    Joins on `case_index` for de-identified rows, and on `case_id` for pre-migration rows, so the
    committed results and a local un-migrated file both select the same cohort.
    """
    hard = hard_case_indices(records_path)
    if hard and all(isinstance(k, int) for k in hard):
        index_of = build_index_map(cases)
        picked = [c for c in cases if index_of[c.case_id] in hard]
    else:
        picked = [c for c in cases if c.case_id in hard]
    return picked[:limit] if limit else picked
