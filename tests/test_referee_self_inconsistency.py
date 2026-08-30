from experiments.referee.referee_self_inconsistency import summarize


def test_summary_all_stable():
    rows = [
        {"temp0_flip": False},
        {"temp0_flip": False},
    ]

    result = summarize(rows)

    assert result["n"] == 2
    assert result["stable_cases"] == 2
    assert result["unstable_cases"] == 0
    assert result["temp0_self_inconsistency_rate"] == 0.0


def test_summary_all_unstable():
    rows = [
        {"temp0_flip": True},
        {"temp0_flip": True},
    ]

    result = summarize(rows)

    assert result["n"] == 2
    assert result["stable_cases"] == 0
    assert result["unstable_cases"] == 2
    assert result["temp0_self_inconsistency_rate"] == 1.0


def test_summary_mixed():
    rows = [
        {"temp0_flip": True},
        {"temp0_flip": False},
        {"temp0_flip": False},
        {"temp0_flip": True},
    ]

    result = summarize(rows)

    assert result["n"] == 4
    assert result["stable_cases"] == 2
    assert result["unstable_cases"] == 2
    assert result["temp0_self_inconsistency_rate"] == 0.5


def test_summary_contains_metadata():
    result = summarize([])

    assert result["n"] == 0
    assert result["temperature"] == 0
    assert result["cache_bypassed"] is True

from experiments.referee.referee_self_inconsistency import _DrawCache


def test_draw_cache_distinguishes_draws():
    k1 = _DrawCache.key("model", "prompt", 1)
    k2 = _DrawCache.key("model", "prompt", 2)

    assert k1 != k2
