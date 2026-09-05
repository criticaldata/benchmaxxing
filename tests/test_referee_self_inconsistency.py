from experiments.referee.referee_self_inconsistency import build_row, summarize


def test_summary_all_stable():
    rows = [
        {"temp0_flip": False, "declared_1": True, "declared_2": True},
        {"temp0_flip": False, "declared_1": True, "declared_2": True},
    ]

    result = summarize(rows)

    assert result["n"] == 2
    assert result["stable_cases"] == 2
    assert result["unstable_cases"] == 0
    assert result["temp0_self_inconsistency_rate"] == 0.0


def test_summary_all_unstable():
    rows = [
        {"temp0_flip": True, "declared_1": True, "declared_2": True},
        {"temp0_flip": True, "declared_1": True, "declared_2": True},
    ]

    result = summarize(rows)

    assert result["n"] == 2
    assert result["stable_cases"] == 0
    assert result["unstable_cases"] == 2
    assert result["temp0_self_inconsistency_rate"] == 1.0


def test_summary_mixed():
    rows = [
        {"temp0_flip": True, "declared_1": True, "declared_2": True},
        {"temp0_flip": False, "declared_1": True, "declared_2": True},
        {"temp0_flip": False, "declared_1": True, "declared_2": True},
        {"temp0_flip": True, "declared_1": True, "declared_2": True},
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

def test_summary_single_unstable_case():
    rows = [
        {
            "case_id": "medqa-1",
            "answer_1": "A",
            "answer_2": "B",
            "declared_1": True,
            "declared_2": True,
            "temp0_flip": True,
        }
    ]

    result = summarize(rows)

    assert result["n"] == 1
    assert result["unstable_cases"] == 1
    assert result["stable_cases"] == 0
    assert result["temp0_self_inconsistency_rate"] == 1.0


def test_build_row_detects_flip():
    row = build_row("medqa-1", "A", "B", True, True)

    assert row["case_id"] == "medqa-1"
    assert row["answer_1"] == "A"
    assert row["answer_2"] == "B"
    assert row["temp0_flip"] is True




def test_build_row_tracks_declaration_state():
    row = build_row(
        "medqa-1",
        "A",
        "B",
        True,
        False,
    )

    assert row["declared_1"] is True
    assert row["declared_2"] is False


def test_summary_counts_declared_pairs_only():
    rows = [
        {
            "temp0_flip": True,
            "declared_1": True,
            "declared_2": True,
        },
        {
            "temp0_flip": False,
            "declared_1": True,
            "declared_2": True,
        },
        {
            "temp0_flip": True,
            "declared_1": True,
            "declared_2": False,
        },
    ]

    result = summarize(rows)

    assert result["declared_pairs"] == 2
    assert result["undeclared_pairs"] == 1
    assert result["undeclared_draws"] == 1

    assert result["temp0_self_inconsistency_rate"] == 0.5


def test_summary_ignores_undeclared_pairs_in_rate():
    rows = [
        {
            "temp0_flip": True,
            "declared_1": False,
            "declared_2": False,
        }
    ]

    result = summarize(rows)

    assert result["declared_pairs"] == 0
    assert result["undeclared_pairs"] == 1
    assert result["undeclared_draws"] == 2
    assert result["temp0_self_inconsistency_rate"] is None
