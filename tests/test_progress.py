from __future__ import annotations

import io

import pytest

from benchmaxxing.progress import ProgressReporter, _format_duration


class FakeClock:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


def test_progress_emits_completed_total_elapsed_rate_and_eta():
    clock = FakeClock()
    stream = io.StringIO()
    progress = ProgressReporter(
        total=4,
        label="solo",
        every=5.0,
        stream=stream,
        time_fn=clock,
    )

    progress.update(1)
    assert stream.getvalue() == ""

    clock.advance(5)
    progress.update(2)

    line = stream.getvalue()
    assert "solo: completed=2/4" in line
    assert "elapsed=5.0s" in line
    assert "rate=0.40/s" in line
    assert "eta=5.0s" in line


def test_progress_force_emits_before_interval():
    clock = FakeClock()
    stream = io.StringIO()
    progress = ProgressReporter(
        total=3,
        every=10.0,
        stream=stream,
        time_fn=clock,
    )

    progress.update(1, force=True)

    assert "progress: completed=1/3" in stream.getvalue()


def test_progress_finish_emits_final_update():
    clock = FakeClock()
    stream = io.StringIO()
    progress = ProgressReporter(
        total=2,
        every=10.0,
        stream=stream,
        time_fn=clock,
    )

    clock.advance(2)
    progress.increment()
    clock.advance(2)
    progress.finish()

    line = stream.getvalue().strip().splitlines()[-1]
    assert "progress: completed=2/2" in line
    assert "eta=0.0s" in line


def test_progress_can_be_disabled():
    clock = FakeClock()
    stream = io.StringIO()
    progress = ProgressReporter(
        total=2,
        stream=stream,
        time_fn=clock,
        enabled=False,
    )

    clock.advance(100)
    progress.update(1, force=True)
    progress.finish()

    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (3.2, "3.2s"),
        (65, "1m05s"),
        (3660, "1h01m"),
        (None, "unknown"),
    ],
)
def test_format_duration(seconds, expected):
    assert _format_duration(seconds) == expected


def test_progress_rejects_invalid_values():
    with pytest.raises(ValueError, match="total"):
        ProgressReporter(total=-1)

    with pytest.raises(ValueError, match="every"):
        ProgressReporter(total=1, every=0)

    progress = ProgressReporter(total=1, stream=io.StringIO())
    with pytest.raises(ValueError, match="completed"):
        progress.update(-1)

    with pytest.raises(ValueError, match="step"):
        progress.increment(-1)

def test_progress_emits_every_n_completed_items():
    clock = FakeClock()
    stream = io.StringIO()
    progress = ProgressReporter(
        total=5,
        every=999.0,
        every_n=2,
        stream=stream,
        time_fn=clock,
    )

    progress.update(1)
    assert stream.getvalue() == ""

    progress.update(2)
    assert "progress: completed=2/5" in stream.getvalue()

    stream.seek(0)
    stream.truncate(0)

    progress.update(3)
    assert stream.getvalue() == ""

    progress.update(4)
    assert "progress: completed=4/5" in stream.getvalue()


def test_progress_start_emits_initial_summary_line():
    stream = io.StringIO()
    progress = ProgressReporter(total=12, label="medqa", stream=stream)

    progress.start("dataset=medqa cases=6 twins=12")

    assert stream.getvalue() == "medqa: starting total=12 dataset=medqa cases=6 twins=12\n"

