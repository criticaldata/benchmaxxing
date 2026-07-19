"""Moderator-effect runner for the scrutiny panel (issue 85).

Does adding a moderator agent to the benchmark-scrutiny panel improve detection, or does it
become the single point through which misses concentrate? This is the orchestrator problem
measured a third way: the blackboard's leader is the conformity fix and a single point of
failure, and the panel's moderator has exactly the same dual character.

The scrutiny harness gives the moderator veto power only (its output is intersected with what
the seats flagged, so it can suppress but never invent). That means a moderator can never raise
the detection rate above the no-moderator union; the empirical question this runner answers is
how much detection it costs, and where the vetoed defects concentrate.

Everything is pure logic over :mod:`benchmaxxing.scrutiny`, testable offline with injected
critic and moderator callables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmaxxing.scrutiny import flaw_detection_rate, run_scrutiny_panel

__all__ = [
    "ModeratorEffect",
    "moderator_effect",
    "compare_moderators",
    "permissive_moderator",
    "strict_majority_moderator",
    "owner_deference_moderator",
]


# --------------------------------------------------------------------------- result record


@dataclass(frozen=True)
class ModeratorEffect:
    """The measured effect of one moderator on one panel over one benchmark set."""

    rate_without: float                   # flaw detection rate, no moderator (the union)
    rate_with: float                      # flaw detection rate with the moderator's vetoes
    delta: float                          # rate_with - rate_without (<= 0 by construction)
    vetoed: dict = field(default_factory=dict)   # metric_name -> planted defects vetoed
    n_vetoed: int = 0                     # total planted defects lost to the veto
    concentrates_misses: bool = False     # True when every added miss traces to the moderator

    @property
    def improved(self) -> bool:
        """A veto-only moderator can never improve detection; kept explicit for reporting."""
        return self.delta > 0


# --------------------------------------------------------------------------- canonical moderators


def permissive_moderator(benchmark, role_detections):
    """Accepts everything any seat flagged (the identity moderator, a sanity control)."""
    union: set[str] = set()
    for detected in role_detections.values():
        union.update(detected)
    return union


def strict_majority_moderator(min_roles: int = 2):
    """A moderator that accepts a defect only when at least ``min_roles`` roles flagged it.

    This is the classic consensus-seeking leader: it suppresses single-seat insights, which is
    precisely how a moderator concentrates misses, since the rare defect only one perspective
    can see is the one it vetoes.
    """

    def _moderator(benchmark, role_detections):
        counts: dict[str, int] = {}
        for detected in role_detections.values():
            for defect in detected:
                counts[defect] = counts.get(defect, 0) + 1
        return {defect for defect, n in counts.items() if n >= min_roles}

    return _moderator


def owner_deference_moderator(benchmark, role_detections):
    """A moderator that defers to the benchmark's owner: accepts only what the owner flagged.

    The pathological case: the seat with the largest blind spot for this benchmark gets the
    veto, the orchestrator problem in miniature.
    """
    owner = str(getattr(benchmark.owner_role, "value", benchmark.owner_role))
    return set(role_detections.get(owner, set()))


# --------------------------------------------------------------------------- the runner


def moderator_effect(benchmarks, critics, moderator) -> ModeratorEffect:
    """Measure what a moderator does to the panel's detection on ``benchmarks``.

    Runs the same panel twice, without and with the moderator, and reports the detection-rate
    delta plus exactly which planted defects the moderator vetoed per benchmark. Because the
    harness gives moderators veto power only, ``delta`` is never positive; the question is how
    negative, and where the losses concentrate.
    """
    benchmarks = list(benchmarks)
    without = run_scrutiny_panel(benchmarks, critics, moderator=None)
    with_mod = run_scrutiny_panel(benchmarks, critics, moderator=moderator)

    vetoed: dict[str, set[str]] = {}
    n_vetoed = 0
    all_added_misses_traced = True
    for base, modded in zip(without, with_mod):
        lost = base.detected - modded.detected
        if lost:
            vetoed[base.metric_name] = lost
            n_vetoed += len(lost)
        added_misses = modded.missed - base.missed
        if added_misses != lost:
            all_added_misses_traced = False

    rate_without = flaw_detection_rate(without)
    rate_with = flaw_detection_rate(with_mod)
    return ModeratorEffect(
        rate_without=rate_without,
        rate_with=rate_with,
        delta=rate_with - rate_without,
        vetoed=vetoed,
        n_vetoed=n_vetoed,
        concentrates_misses=bool(vetoed) and all_added_misses_traced,
    )


def compare_moderators(benchmarks, critics, moderators) -> dict:
    """Run :func:`moderator_effect` for several moderators on the same panel and benchmarks.

    ``moderators`` is a mapping ``{name: moderator_fn}``. Returns ``{name: ModeratorEffect}``
    plus a ``"ranking"`` entry listing names from least to most detection lost, so the report
    can say which moderation style is cheapest.
    """
    moderators = dict(moderators)
    effects = {name: moderator_effect(benchmarks, critics, fn) for name, fn in moderators.items()}
    ranking = sorted(effects, key=lambda name: effects[name].delta, reverse=True)
    return {**effects, "ranking": ranking}
