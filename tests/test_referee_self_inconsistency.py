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


def test_summary_single_unstable_case():
    rows = [
        {
            "case_id": "medqa-1",
            "answer_1": "A",
            "answer_2": "B",
            "temp0_flip": True,
        }
    ]

    result = summarize(rows)

    assert result["n"] == 1
    assert result["unstable_cases"] == 1
    assert result["stable_cases"] == 0
    assert result["temp0_self_inconsistency_rate"] == 1.0


def test_draw_cache_same_draw_same_key():
    k1 = _DrawCache.key("model", "prompt", 1)
    k2 = _DrawCache.key("model", "prompt", 1)

    assert k1 == k2

def test_draw_cache_round_trip(tmp_path):
    cache = _DrawCache(tmp_path / "cache.jsonl")

    assert cache.get("model", "prompt", 1) is None

    cache.put("model", "prompt", 1, "answer")

    assert cache.get("model", "prompt", 1) == "answer"
    assert cache.get("model", "prompt", 2) is None


from experiments.referee.referee_self_inconsistency import build_row


def test_build_row_detects_flip():
    row = build_row("medqa-1", "A", "B")

    assert row["case_id"] == "medqa-1"
    assert row["answer_1"] == "A"
    assert row["answer_2"] == "B"
    assert row["temp0_flip"] is True

def test_draw_cache_reloads_from_disk(tmp_path):
    cache_file = tmp_path / "cache.jsonl"

    cache1 = _DrawCache(cache_file)
    cache1.put("model", "prompt", 1, "answer")

    cache2 = _DrawCache(cache_file)

    assert cache2.get("model", "prompt", 1) == "answer"
