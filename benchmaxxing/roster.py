"""Roster and committee construction for the same-lineage vs cross-lineage arms.

The overlap and referee experiments contrast two committee kinds:

* same-lineage committees (the control): every member shares one ``lineage`` (family/provider),
  for example all Gemini tiers. These never span more than one lineage.
* cross-lineage committees (the treatment): members span at least two lineages and include at
  least one open-weights family, so the oversight is not all coming from one backbone.

Everything here is pure and offline: it just arranges ``ModelSpec`` objects into ``Committee``
objects. No API calls, no keys.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

from benchmaxxing.schema import Committee, ModelSpec


def build_committee(members: Iterable[ModelSpec]) -> Committee:
    """Wrap an iterable of ``ModelSpec`` into a ``Committee`` (order preserved)."""
    return Committee(members=tuple(members))


def default_roster() -> list[ModelSpec]:
    """An illustrative roster for wiring and tests.

    A couple of Gemini tiers (Gemini is the default backend for now) plus one open-weights
    entry so a cross-lineage arm can actually be built. Swap in real gateway ids later.
    """
    return [
        # Gemini is the default backend for now: two tiers, one closed lineage.
        ModelSpec(name="gemini-2.5-pro", lineage="gemini", tier="pro", is_open_weights=False),
        ModelSpec(name="gemini-2.5-flash", lineage="gemini", tier="flash", is_open_weights=False),
        # An open-weights family from a different lineage, required for the cross-lineage arm.
        ModelSpec(
            name="qwen2.5-72b-instruct",
            lineage="qwen",
            tier="72b",
            is_open_weights=True,
        ),
    ]


def same_lineage_committees(roster: Iterable[ModelSpec], size: int) -> list[Committee]:
    """All committees of ``size`` members drawn from a SINGLE lineage (the control arm).

    Members are grouped by ``lineage``; each lineage with at least ``size`` members contributes
    every ``size``-sized combination. Returned committees are never cross-lineage.
    """
    if size < 1:
        raise ValueError(f"committee size must be >= 1, got {size}")

    by_lineage: dict[str, list[ModelSpec]] = defaultdict(list)
    for spec in roster:
        by_lineage[spec.lineage].append(spec)

    committees: list[Committee] = []
    for members in by_lineage.values():
        if len(members) < size:
            continue
        for combo in combinations(members, size):
            committees.append(build_committee(combo))
    return committees


def cross_lineage_committees(roster: Iterable[ModelSpec], size: int) -> list[Committee]:
    """All committees of ``size`` members that span >=2 lineages and include an open-weights family.

    If the roster has no open-weights model, a cross-lineage arm cannot be built from a single
    lineage, so this raises ``ValueError`` with an actionable message.
    """
    if size < 1:
        raise ValueError(f"committee size must be >= 1, got {size}")

    roster = list(roster)
    if not any(spec.is_open_weights for spec in roster):
        raise ValueError(
            "cannot build a cross-lineage committee: the roster has no open-weights model. "
            "The cross-lineage arm is required and needs at least one open-weights family "
            "(is_open_weights=True) from a lineage other than the closed backbone. "
            "Add one to the roster (see default_roster for an example)."
        )

    committees: list[Committee] = []
    for combo in combinations(roster, size):
        lineages = {spec.lineage for spec in combo}
        if len(lineages) < 2:
            continue
        if not any(spec.is_open_weights for spec in combo):
            continue
        committees.append(build_committee(combo))
    return committees
