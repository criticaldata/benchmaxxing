"""Dependency-light progress reporting for long-running experiment loops.

Progress messages are written to stderr by default so stdout remains available for
machine-readable output such as final JSON summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
import time
from typing import Callable, TextIO


def _format_duration(seconds: float | None) -> str:
    """Format a duration compactly for human-readable progress logs."""
    if seconds is None:
        return "unknown"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


@dataclass
class ProgressReporter:
    """Emit periodic completed/total progress updates to a text stream.

    Parameters
    ----------
    total:
        Total number of work items expected.
    label:
        Prefix for emitted progress lines.
    every:
        Minimum seconds between non-forced updates.
    every_n:
        Minimum completed-item interval between non-forced updates. When set, progress can emit
        based on item count even if ``every`` seconds have not elapsed.
    stream:
        Destination stream. Defaults to stderr.
    time_fn:
        Clock function, injectable for deterministic tests.
    enabled:
        When false, all updates are no-ops.
    """

    total: int
    label: str = "progress"
    every: float = 10.0
    every_n: int | None = None
    stream: TextIO | None = None
    time_fn: Callable[[], float] = time.monotonic
    enabled: bool = True
    completed: int = 0
    _started_at: float = field(init=False, repr=False)
    _last_emit_at: float = field(init=False, repr=False)
    _last_completed_emit: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError("total must be non-negative")
        if self.every <= 0:
            raise ValueError("every must be positive")
        if self.every_n is not None and self.every_n <= 0:
            raise ValueError("every_n must be positive")
        if self.stream is None:
            self.stream = sys.stderr
        now = float(self.time_fn())
        self._started_at = now
        self._last_emit_at = now

    def start(self, detail: str | None = None) -> None:
        """Emit an initial run summary line."""
        if not self.enabled:
            return
        suffix = f" {detail}" if detail else ""
        print(f"{self.label}: starting total={self.total}{suffix}", file=self.stream, flush=True)
        self._last_emit_at = float(self.time_fn())

    def update(self, completed: int, *, force: bool = False) -> None:
        """Record absolute completed count and emit if enough time or items have passed."""
        if completed < 0:
            raise ValueError("completed must be non-negative")
        self.completed = completed
        if not self.enabled:
            return

        now = float(self.time_fn())
        due_by_time = (now - self._last_emit_at) >= self.every
        due_by_count = (
            self.every_n is not None
            and (completed - self._last_completed_emit) >= self.every_n
        )
        if not force and completed < self.total and not due_by_time and not due_by_count:
            return

        elapsed = max(0.0, now - self._started_at)
        rate = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
        remaining = max(0, self.total - completed)
        eta = remaining / rate if rate > 0 else None

        print(
            (
                f"{self.label}: completed={completed}/{self.total} "
                f"elapsed={_format_duration(elapsed)} "
                f"rate={rate:.2f}/s "
                f"eta={_format_duration(eta)}"
            ),
            file=self.stream,
            flush=True,
        )
        self._last_emit_at = now
        self._last_completed_emit = completed

    def increment(self, step: int = 1, *, force: bool = False) -> None:
        """Advance by ``step`` work items and maybe emit progress."""
        if step < 0:
            raise ValueError("step must be non-negative")
        self.update(self.completed + step, force=force)

    def finish(self) -> None:
        """Emit a final completed update."""
        self.update(self.total, force=True)


def progress_reporter(
    total: int,
    *,
    label: str = "progress",
    every: float = 10.0,
    every_n: int | None = None,
    stream: TextIO | None = None,
    time_fn: Callable[[], float] = time.monotonic,
    enabled: bool = True,
) -> ProgressReporter:
    """Convenience factory for a :class:`ProgressReporter`."""
    return ProgressReporter(
        total=total,
        label=label,
        every=every,
        every_n=every_n,
        stream=stream,
        time_fn=time_fn,
        enabled=enabled,
    )
