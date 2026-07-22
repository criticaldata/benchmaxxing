"""Stage 5 benchmark scrutiny panel (issue 19).

This module models the final review stage of benchmaxxing: before a benchmark is used to
ship or procure a model, a panel of stakeholders scrutinizes it for known failure modes.
Each benchmark is *owned* by one stakeholder role but may carry latent defects that a single
perspective is prone to miss. The panel runs a set of critics (one per seat) over the
benchmarks and reports which planted defects were caught and by which roles.

Everything here is pure logic. The critics are injected callables ``(role, benchmark) ->
set of detected defect names``; no real models are needed, so the whole panel runs offline
with no API keys and is exercised with hand-built fixtures.

Hypothesis
----------
Heterogeneous (pluralistic) panels detect more defects than homogeneous ones. A benchmark's
defects are correlated with the blind spots of its owner role: a metric a hospital cares
about hides the failures a patient would notice, and vice versa. A panel staffed by all four
stakeholder roles brings independent priors and therefore names a larger fraction of the
planted defects than a panel of one role replicated to the same size. We measure this gap
with :func:`flaw_detection_rate` rather than assume it, and expect the pluralistic panel to
lead the homogeneous control.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

__all__ = [
    "StakeholderRole",
    "ALL_ROLES",
    "DEFECTS",
    "CIRCULAR_PROVENANCE",
    "CONTAMINATED_DATING",
    "UNMOORED_PROXY",
    "COVERAGE_GAP",
    "Benchmark",
    "PanelResult",
    "plant_defects",
    "run_scrutiny_panel",
    "flaw_detection_rate",
    "compare_panels",
    "coverage_gap_estimate",
    "scrutiny_diversity_hypothesis",
]


class StakeholderRole(str, Enum):
    """The four stakeholder perspectives that can sit on a scrutiny panel."""

    PATIENT = "patient"
    PROVIDER = "provider"
    HOSPITAL = "hospital"
    PROCUREMENT = "procurement"


# The full pluralistic panel: one seat per distinct role.
ALL_ROLES: tuple[StakeholderRole, ...] = (
    StakeholderRole.PATIENT,
    StakeholderRole.PROVIDER,
    StakeholderRole.HOSPITAL,
    StakeholderRole.PROCUREMENT,
)

# Defect vocabulary: the named failure modes a panel can plant and detect.
CIRCULAR_PROVENANCE = "circular_provenance"   # metric validated against its own source
CONTAMINATED_DATING = "contaminated_dating"   # train/test dates leak the answer
UNMOORED_PROXY = "unmoored_proxy"             # proxy target drifted from the real outcome
COVERAGE_GAP = "coverage_gap"                 # whole outcome dimensions never measured

DEFECTS: frozenset[str] = frozenset(
    {CIRCULAR_PROVENANCE, CONTAMINATED_DATING, UNMOORED_PROXY, COVERAGE_GAP}
)

# A critic seat: given the seat's role and a benchmark, return the defect names it detects.
Critic = Callable[["StakeholderRole", "Benchmark"], set]


@dataclass
class Benchmark:
    """A benchmark under review: owned by one role, carrying zero or more latent defects."""

    owner_role: StakeholderRole
    metric_name: str
    defects: set[str] = field(default_factory=set)


@dataclass
class PanelResult:
    """One benchmark's scrutiny outcome: what was planted and what the panel caught."""

    metric_name: str
    owner_role: str
    planted: set[str]                    # defects planted in the benchmark
    flagged: set[str]                    # every defect the panel accepted (incl. false alarms)
    detected: set[str]                   # planted defects the panel caught
    missed: set[str]                     # planted defects nobody caught
    caught_by: dict[str, set[str]]       # defect -> set of role names that caught it
    roles: tuple[str, ...]               # panel composition (role names, with replication)


# --------------------------------------------------------------------------- helpers


def _role_key(role) -> str:
    """Canonical string name for a role, accepting a StakeholderRole or a plain string."""
    if isinstance(role, StakeholderRole):
        return role.value
    return str(role)


def _normalize_defects(defects) -> set[str]:
    """Coerce a defect spec (None, a single name, or an iterable of names) to a set."""
    if defects is None:
        return set()
    if isinstance(defects, str):
        return {defects}
    return {str(d) for d in defects}


def _panel_items(critics):
    """Yield ``(role, critic)`` pairs from a mapping or a sequence of pairs."""
    if isinstance(critics, Mapping):
        return list(critics.items())
    return list(critics)


def _normalize_panel(critics):
    """Return a list of ``(role_name, role, critic_fn)`` from the panel spec."""
    panel = []
    for role, critic_fn in _panel_items(critics):
        if not callable(critic_fn):
            raise TypeError("Each critic must be a (role, callable) pair.")
        panel.append((_role_key(role), role, critic_fn))
    if not panel:
        raise ValueError("A scrutiny panel needs at least one critic seat.")
    return panel


# --------------------------------------------------------------------------- planting


def plant_defects(benchmark: Benchmark, defects) -> Benchmark:
    """Return a new Benchmark carrying the planted defects (union with any existing).

    ``defects`` is a single defect name or an iterable of names drawn from :data:`DEFECTS`.
    Raises ValueError if any name is outside the vocabulary.
    """
    planted = _normalize_defects(defects)
    unknown = planted - DEFECTS
    if unknown:
        raise ValueError(
            f"Unknown defect(s): {sorted(unknown)}. Known defects: {sorted(DEFECTS)}."
        )
    return Benchmark(
        owner_role=benchmark.owner_role,
        metric_name=benchmark.metric_name,
        defects=set(benchmark.defects) | planted,
    )


# --------------------------------------------------------------------------- the panel


def run_scrutiny_panel(benchmarks, critics, *, moderator=None) -> list[PanelResult]:
    """Run each critic seat over every benchmark and report the planted defects caught.

    Parameters
    ----------
    benchmarks:
        Iterable of :class:`Benchmark`.
    critics:
        The panel composition, as a sequence of ``(role, critic_fn)`` pairs or a mapping
        ``{role: critic_fn}``. A role may be replicated (the same role in several seats) to
        build a homogeneous panel. Each ``critic_fn`` is called ``critic_fn(role, benchmark)``
        and returns a set of detected defect names.
    moderator:
        Optional callable ``moderator(benchmark, role_detections) -> iterable of defect
        names`` that reconciles the seats. ``role_detections`` maps each role name to the
        union of defects that role's seats detected. The moderator's return is intersected
        with what the seats actually flagged, so it can only veto, never invent. When omitted,
        the panel accepts the union of all seats' detections.

    Returns
    -------
    list[PanelResult]
        One result per benchmark, in input order.
    """
    panel = _normalize_panel(critics)
    role_names = tuple(name for name, _, _ in panel)

    results: list[PanelResult] = []
    for benchmark in benchmarks:
        planted = set(benchmark.defects)
        role_detections: dict[str, set[str]] = {}
        for role_name, role, critic_fn in panel:
            detected = _normalize_defects(critic_fn(role, benchmark))
            role_detections.setdefault(role_name, set()).update(detected)

        union: set[str] = set().union(*role_detections.values()) if role_detections else set()
        if moderator is not None:
            accepted = _normalize_defects(moderator(benchmark, role_detections)) & union
        else:
            accepted = union

        detected_planted = accepted & planted
        caught_by = {
            defect: {name for name, det in role_detections.items() if defect in det}
            for defect in detected_planted
        }
        results.append(
            PanelResult(
                metric_name=benchmark.metric_name,
                owner_role=_role_key(benchmark.owner_role),
                planted=planted,
                flagged=accepted,
                detected=detected_planted,
                missed=planted - detected_planted,
                caught_by=caught_by,
                roles=role_names,
            )
        )
    return results


def flaw_detection_rate(results) -> float:
    """Fraction of all planted defects the panel caught (micro-averaged over benchmarks).

    Returns ``nan`` when nothing was planted, so an empty benchmark set never reads as a
    perfect score.
    """
    results = list(results)
    total_planted = sum(len(r.planted) for r in results)
    total_caught = sum(len(r.detected) for r in results)
    return total_caught / total_planted if total_planted else float("nan")


def compare_panels(benchmarks, critic_by_role, *, homogeneous_role, moderator=None,
                   panel_roles=ALL_ROLES) -> dict:
    """Compare a pluralistic panel against a homogeneous one on the same benchmarks.

    The pluralistic panel seats one critic per role in ``panel_roles``. The homogeneous
    panel replicates ``homogeneous_role`` to the same seat count. Both use the per-role
    critics in ``critic_by_role`` (a mapping ``{role: critic_fn}``; keys may be
    StakeholderRole values or their string names).

    Returns a dict with each panel's :func:`flaw_detection_rate`, their difference, a
    ``pluralistic_better`` flag, and the underlying result lists.
    """
    benchmarks = list(benchmarks)
    by_name = {_role_key(role): fn for role, fn in dict(critic_by_role).items()}

    def _critic(role):
        name = _role_key(role)
        if name not in by_name:
            raise KeyError(f"No critic supplied for role {name!r}.")
        return by_name[name]

    pluralistic = [(role, _critic(role)) for role in panel_roles]
    hom_critic = _critic(homogeneous_role)
    homogeneous = [(homogeneous_role, hom_critic) for _ in panel_roles]

    pluralistic_results = run_scrutiny_panel(benchmarks, pluralistic, moderator=moderator)
    homogeneous_results = run_scrutiny_panel(benchmarks, homogeneous, moderator=moderator)
    pluralistic_rate = flaw_detection_rate(pluralistic_results)
    homogeneous_rate = flaw_detection_rate(homogeneous_results)

    return {
        "homogeneous_role": _role_key(homogeneous_role),
        "pluralistic_rate": pluralistic_rate,
        "homogeneous_rate": homogeneous_rate,
        "difference": pluralistic_rate - homogeneous_rate,
        "pluralistic_better": pluralistic_rate > homogeneous_rate,
        "pluralistic_results": pluralistic_results,
        "homogeneous_results": homogeneous_results,
    }


def coverage_gap_estimate(covered_dimensions, full_outcome_list) -> float:
    """Fraction of the held-out outcome list the panel never named.

    ``full_outcome_list`` is the full set of outcome dimensions that matter;
    ``covered_dimensions`` is what the panel actually named. The gap is the share of the
    full list left uncovered. Both inputs are deduplicated and order-insensitive; extra names
    in ``covered_dimensions`` that are not in the full list are ignored. Returns ``0.0`` when
    the full list is empty (nothing to miss).
    """
    full = {str(d) for d in full_outcome_list}
    covered = {str(d) for d in covered_dimensions}
    if not full:
        return 0.0
    return len(full - covered) / len(full)


def scrutiny_diversity_hypothesis() -> str:
    """Why a heterogeneous panel should out-detect a homogeneous one (the control)."""
    return (
        "A benchmark's defects tend to align with the blind spots of its owner role, so a "
        "panel of one role replicated inherits those blind spots and catches fewer planted "
        "defects. A heterogeneous, pluralistic panel spanning all four stakeholder roles "
        "brings independent priors and names a larger fraction of the defects. We expect the "
        "homogeneous panel to underperform the pluralistic one, and run it as the control arm "
        "to measure that gap."
    )
