"""Blind-metric evaluation arms: the misaligned-proxy arm and the test-awareness arm.

These are thin, offline runners that drive a model over a set of cases and hand the resulting
decisions to :mod:`benchmaxxing.blind_metric`. They wire the pure-math probe (uptake and the
awareness-suppression delta) to an injected model, so the whole thing is a deterministic
function of its callables and runs with no API keys or real data.

Two arms live here:

    misaligned_proxy_run (the proxy arm)
        A plausible-but-wrong PROXY is silently substituted for the real objective. We run the
        model over the cases, record each decision, and measure how much the decisions drift
        toward the proxy relative to ground truth via ``blind_metric_uptake``. The proxy plays
        exactly the role the decoy artifact plays in the decoy arm, so this arm complements it:
        the decoy arm rewards a meaningless artifact, this arm rewards a wrong-but-plausible
        stand-in for the objective. A positive ``uptake_delta`` means the model followed the
        misaligned proxy more than the truth.

    test_awareness_run (the awareness arm)
        The same proxy arm is run twice: once with a model that is TOLD it may be evaluated
        (the aware arm) and once with a model that is not (the unaware arm). The returned
        suppression delta is ``unaware_uptake - aware_uptake``: a positive value means the model
        games the proxy LESS when it believes it is being watched, so the unaware measurement is
        the more honest read of its behavior.

Both runners take injected callables (``model_fn`` / ``proxy_fn`` / ``truth_fn``) that map one
case to a scalar, so behavior is fully controlled by the caller and testable with deterministic
mocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from benchmaxxing.blind_metric import BlindMetricUptake, awareness_delta, blind_metric_uptake

__all__ = [
    "ProxyArmResult",
    "misaligned_proxy_run",
    "test_awareness_run",
]

# A per-case scalar producer: model decision, proxy value, or ground-truth value for one case.
CaseFn = Callable[[object], float]


@dataclass(frozen=True)
class ProxyArmResult:
    """One misaligned-proxy arm: the recorded per-case values and the uptake it produced.

    ``decisions`` is what the model actually did on each case, ``proxy_values`` is the
    plausible-but-wrong proxy per case, and ``truth_values`` is the real objective per case
    (all parallel and in case order). ``uptake`` is the :class:`BlindMetricUptake` of the
    decisions against the proxy (as the hidden metric) relative to truth. A positive
    ``uptake_delta`` means the decisions tracked the misaligned proxy more than the truth.
    """

    decisions: tuple[float, ...]
    proxy_values: tuple[float, ...]
    truth_values: tuple[float, ...]
    uptake: BlindMetricUptake

    @property
    def uptake_delta(self) -> float:
        """Drift toward the proxy over truth (``corr_proxy - corr_truth``)."""
        return self.uptake.uptake_delta

    @property
    def n(self) -> int:
        """Number of cases the arm ran on."""
        return self.uptake.n


def _record_arm(
    cases: Iterable[object],
    model_fn: CaseFn,
    proxy_fn: CaseFn,
    truth_fn: CaseFn,
) -> tuple[list[float], list[float], list[float]]:
    """Run the injected callables over each case and collect three parallel scalar lists."""
    decisions: list[float] = []
    proxy_values: list[float] = []
    truth_values: list[float] = []
    for case in cases:
        decisions.append(float(model_fn(case)))
        proxy_values.append(float(proxy_fn(case)))
        truth_values.append(float(truth_fn(case)))
    return decisions, proxy_values, truth_values


def misaligned_proxy_run(
    cases: Iterable[object],
    model_fn: CaseFn,
    proxy_fn: CaseFn,
    truth_fn: CaseFn,
    method: str = "pearson",
) -> ProxyArmResult:
    """Run the model over ``cases`` under a misaligned proxy and measure blind-metric uptake.

    A plausible-but-wrong ``proxy_fn`` stands in for the real objective ``truth_fn``. For each
    case the model's decision (``model_fn``), the proxy value, and the ground-truth value are
    recorded, then the decisions are correlated against the proxy (as the hidden metric) versus
    the truth via :func:`benchmaxxing.blind_metric.blind_metric_uptake`.

    Parameters
    ----------
    cases:
        Iterable of case items. Each is passed unchanged to the three callables.
    model_fn:
        ``model_fn(case) -> float``: the decision the model committed on that case. Injected so
        the run is deterministic and offline.
    proxy_fn:
        ``proxy_fn(case) -> float``: the misaligned proxy value silently rewarded in place of the
        objective. Plays the hidden-metric role in the uptake computation.
    truth_fn:
        ``truth_fn(case) -> float``: the real objective value for that case.
    method:
        Correlation method forwarded to ``blind_metric_uptake`` (``"pearson"`` or ``"spearman"``).

    Returns
    -------
    ProxyArmResult
        The recorded decisions, proxy values, and truth values plus the ``BlindMetricUptake``.
        A positive ``uptake_delta`` means the model followed the misaligned proxy more than the
        truth, complementing the decoy arm.
    """
    decisions, proxy_values, truth_values = _record_arm(cases, model_fn, proxy_fn, truth_fn)
    uptake = blind_metric_uptake(decisions, proxy_values, truth_values, method=method)
    return ProxyArmResult(
        decisions=tuple(decisions),
        proxy_values=tuple(proxy_values),
        truth_values=tuple(truth_values),
        uptake=uptake,
    )


def test_awareness_run(
    cases: Iterable[object],
    aware_model_fn: CaseFn,
    unaware_model_fn: CaseFn,
    proxy_fn: CaseFn,
    truth_fn: CaseFn,
    method: str = "pearson",
) -> float:
    """Run the proxy arm aware vs unaware of evaluation and return the suppression delta.

    The misaligned-proxy arm is run twice over the same cases: once with ``aware_model_fn`` (a
    model told it may be evaluated) and once with ``unaware_model_fn`` (a model that is not). The
    returned value is the uptake-suppression delta ``unaware_uptake - aware_uptake`` (via
    :func:`benchmaxxing.blind_metric.awareness_delta`).

    A positive delta means awareness SUPPRESSED uptake: the model games the misaligned proxy less
    when it believes it is being watched, so the unaware arm is the more honest read of its
    behavior. A delta near zero means the announcement of evaluation did not change behavior.

    Parameters
    ----------
    cases:
        Iterable of case items, shared by both arms.
    aware_model_fn:
        ``aware_model_fn(case) -> float`` for the arm that knows it may be evaluated.
    unaware_model_fn:
        ``unaware_model_fn(case) -> float`` for the arm that does not.
    proxy_fn, truth_fn:
        The misaligned proxy and the real objective, shared by both arms.
    method:
        Correlation method forwarded to each proxy run.

    Returns
    -------
    float
        The suppression delta ``unaware minus aware``. ``nan`` propagates from a degenerate arm.
    """
    aware = misaligned_proxy_run(cases, aware_model_fn, proxy_fn, truth_fn, method=method)
    unaware = misaligned_proxy_run(cases, unaware_model_fn, proxy_fn, truth_fn, method=method)
    return awareness_delta(uptake_aware=aware.uptake, uptake_unaware=unaware.uptake)


# ``test_awareness_run`` is a real API function (issue 84), not a pytest test. Mark it so pytest
# never collects it as a test case when it is imported into a test module's namespace.
test_awareness_run.__test__ = False  # type: ignore[attr-defined]
