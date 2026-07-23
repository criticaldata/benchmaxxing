"""Run-budget policy and accounting: deterministic case subsampling, a call-budget guard, and
a per-model call tally for the run summary.

The gateway (``benchmaxxing.gateway``) already handles retries and per-call rate limiting; this
module is the policy layer on top: how many cases a run touches, how many API calls it is allowed
to spend, and how those calls are accounted for so a truncated run is never mistaken for a full
one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from benchmaxxing.gateway import Backend

__all__ = [
    "RunBudget",
    "subsample_cases",
    "CallAccountant",
    "BudgetExceeded",
    "BudgetedBackend",
    "truncation_note",
]


@dataclass(frozen=True)
class RunBudget:
    """Bounds for a run: how many cases to touch and how many model calls to spend.

    ``max_cases`` and ``max_calls`` of ``None`` mean unbounded. ``seed`` drives the
    deterministic subsampling in :func:`subsample_cases`.
    """

    max_cases: int | None = None
    max_calls: int | None = None
    seed: int = 0


def subsample_cases(cases, budget: RunBudget) -> list:
    """Deterministically select up to ``budget.max_cases`` cases, seeded by ``budget.seed``.

    The same seed always draws the same permutation of ``cases``, so a smaller ``max_cases``
    is a strict prefix of a larger ``max_cases`` run over the same pool: a cheap pilot is a
    subset of the full run, not an independent (and possibly non-overlapping) sample.
    """
    cases = list(cases)
    if budget.max_cases is None or budget.max_cases >= len(cases):
        return cases
    order = np.random.default_rng(budget.seed).permutation(len(cases))
    return [cases[i] for i in order[: budget.max_cases]]


class BudgetExceeded(RuntimeError):
    """Raised by :class:`BudgetedBackend` when ``max_calls`` would be exceeded.

    A caller catches this to stop a run cleanly and flush whatever results it already has,
    instead of the run failing outright or silently continuing past its budget.
    """


@dataclass
class CallAccountant:
    """Tallies model calls (and, when a backend reports them, tokens) per model."""

    calls_per_model: dict = field(default_factory=dict)
    tokens_per_model: dict = field(default_factory=dict)
    total_calls: int = 0

    def record(self, model: str, tokens: int | None = None) -> None:
        self.calls_per_model[model] = self.calls_per_model.get(model, 0) + 1
        self.total_calls += 1
        if tokens is not None:
            self.tokens_per_model[model] = self.tokens_per_model.get(model, 0) + tokens

    def summary(self) -> dict:
        """A JSON-serializable summary for the run report."""
        return {
            "total_calls": self.total_calls,
            "calls_per_model": dict(self.calls_per_model),
            "tokens_per_model": dict(self.tokens_per_model),
        }


class BudgetedBackend(Backend):
    """Wrap a backend so every call is tallied in ``accountant``, enforcing ``max_calls``.

    Raises :class:`BudgetExceeded` instead of making the call once the shared accountant's
    ``total_calls`` would reach ``max_calls``, so a caller can stop the run and flush partial
    results rather than spending an unbounded number of calls.
    """

    def __init__(
        self,
        backend: Backend,
        accountant: CallAccountant,
        model: str,
        max_calls: int | None = None,
    ):
        self.backend = backend
        self.accountant = accountant
        self.model = model
        self.max_calls = max_calls

    def complete(self, prompt: str, image: object | None = None, decoding: dict | None = None) -> str:
        if self.max_calls is not None and self.accountant.total_calls >= self.max_calls:
            raise BudgetExceeded(
                f"max_calls={self.max_calls} reached; refusing a further call to {self.model!r}"
            )
        result = self.backend.complete(prompt, image=image, decoding=decoding)
        self.accountant.record(self.model)
        return result


def truncation_note(planned: int, completed: int) -> str | None:
    """A human-readable truncation note for the run summary, or ``None`` if nothing was cut.

    ``planned`` is how many cases the run intended to cover (post-subsampling); ``completed``
    is how many it actually finished before stopping. Never silently drop this: a truncated run
    must say so, or it reads as full coverage.
    """
    if completed >= planned:
        return None
    return f"budget truncation: completed {completed}/{planned} planned cases"
