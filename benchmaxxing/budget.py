"""Run-budget policy and accounting: deterministic case subsampling, a call-budget guard, and
a per-model call tally for the run summary.

The gateway (``benchmaxxing.gateway``) already handles retries and per-call rate limiting; this
module is the policy layer on top: how many cases a run touches, how many API calls it is allowed
to spend, and how those calls are accounted for so a truncated run is never mistaken for a full
one.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field

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


def _case_key(case) -> str:
    """A stable, content-based identity for a case, independent of its position in the pool.

    Prefers ``case.case_id`` (the schema's stable identifier); falls back to ``repr`` for the
    plain values used in tests. Never uses Python's builtin ``hash``, which is salted per process.
    """
    return str(getattr(case, "case_id", None) or repr(case))


def _rank(case, seed: int) -> str:
    """Deterministic sort rank for ``case`` under ``seed``: a hash of ``(seed, case_key)``."""
    return hashlib.blake2b(f"{seed}:{_case_key(case)}".encode()).hexdigest()


def subsample_cases(cases, budget: RunBudget) -> list:
    """Deterministically select up to ``budget.max_cases`` cases, seeded by ``budget.seed``.

    Cases are ranked by a stable hash of ``(seed, case_id)``, so the selection depends only on
    the cases' content, not on the order they arrive in. That makes a smaller ``max_cases`` a
    strict subset of a larger ``max_cases`` run over the same pool even if an upstream step
    reorders or re-filters the pool between a pilot and the full run: a cheap pilot is always a
    subset of the full run, not an independent (and possibly non-overlapping) sample.
    """
    cases = list(cases)
    if budget.max_cases is None or budget.max_cases >= len(cases):
        return cases
    # Rank by hash; the original index breaks ties so duplicate keys stay deterministic. Return
    # in rank order (not pool order) so a smaller max_cases is a strict prefix of a larger one.
    order = sorted(range(len(cases)), key=lambda i: (_rank(cases[i], budget.seed), i))
    return [cases[i] for i in order[: budget.max_cases]]


class BudgetExceeded(RuntimeError):
    """Raised by :class:`BudgetedBackend` when ``max_calls`` would be exceeded.

    A caller catches this to stop a run cleanly and flush whatever results it already has,
    instead of the run failing outright or silently continuing past its budget.
    """


@dataclass
class CallAccountant:
    """Tallies model calls (and, when a backend reports them, tokens) per model, under one
    run-wide call cap.

    ``max_calls`` is a single budget for the whole run, shared across every model that reports
    to this accountant, matching the issue's ``run_budget`` block (one scalar ``max_calls``, plus
    a per-model breakdown for the summary) rather than an independent budget per model. Models
    whose budgets must be independent get their own accountant. ``max_calls`` of ``None`` is
    unbounded.

    All mutation goes through a lock, so a shared accountant is safe for the ``ThreadPoolExecutor``
    fan-out the real experiment scripts use: the check and the record in :meth:`try_record` are
    atomic, so concurrent callers can never overspend the cap.
    """

    max_calls: int | None = None
    calls_per_model: dict = field(default_factory=dict)
    tokens_per_model: dict = field(default_factory=dict)
    total_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @classmethod
    def from_budget(cls, budget: RunBudget) -> CallAccountant:
        """Build an accountant that enforces ``budget.max_calls`` as the run-wide cap."""
        return cls(max_calls=budget.max_calls)

    def record(self, model: str, tokens: int | None = None) -> None:
        """Tally one call unconditionally, ignoring the cap. Prefer :meth:`try_record` on the
        budgeted path; this is for callers that account for calls they have already made."""
        with self._lock:
            self._record_locked(model, tokens)

    def try_record(self, model: str, tokens: int | None = None) -> None:
        """Atomically enforce the cap and tally one call, or raise :class:`BudgetExceeded`.

        The check and the increment happen under one lock, so two concurrent callers can never
        both slip past a cap that only one of them should have claimed.
        """
        with self._lock:
            if self.max_calls is not None and self.total_calls >= self.max_calls:
                raise BudgetExceeded(
                    f"max_calls={self.max_calls} reached; refusing a further call to {model!r}"
                )
            self._record_locked(model, tokens)

    def _record_locked(self, model: str, tokens: int | None) -> None:
        self.calls_per_model[model] = self.calls_per_model.get(model, 0) + 1
        self.total_calls += 1
        if tokens is not None:
            self.tokens_per_model[model] = self.tokens_per_model.get(model, 0) + tokens

    def summary(self) -> dict:
        """A JSON-serializable summary for the run report."""
        with self._lock:
            return {
                "total_calls": self.total_calls,
                "calls_per_model": dict(self.calls_per_model),
                "tokens_per_model": dict(self.tokens_per_model),
            }


class BudgetedBackend(Backend):
    """Wrap a backend so every call is tallied in ``accountant`` and the run-wide cap is enforced.

    The cap lives on the shared :class:`CallAccountant` (see its docstring), not here, so a run
    has exactly one ``max_calls`` no matter how many models report to it. Raises
    :class:`BudgetExceeded` instead of making the call once the accountant's cap is reached, so a
    caller can stop the run and flush partial results rather than spending unbounded calls.
    """

    def __init__(self, backend: Backend, accountant: CallAccountant, model: str):
        self.backend = backend
        self.accountant = accountant
        self.model = model

    def complete(self, prompt: str, image: object | None = None, decoding: dict | None = None) -> str:
        # Claim budget before spending it: raises if the cap is already reached, atomically.
        self.accountant.try_record(self.model)
        return self.backend.complete(prompt, image=image, decoding=decoding)


def truncation_note(planned: int, completed: int) -> str | None:
    """A human-readable truncation note for the run summary, or ``None`` if nothing was cut.

    ``planned`` is how many cases the run intended to cover (post-subsampling); ``completed``
    is how many it actually finished before stopping. Never silently drop this: a truncated run
    must say so, or it reads as full coverage.
    """
    if completed >= planned:
        return None
    return f"budget truncation: completed {completed}/{planned} planned cases"
