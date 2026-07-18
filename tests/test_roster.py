"""Tests for benchmaxxing.roster (same-lineage vs cross-lineage committees)."""

from __future__ import annotations

import pytest

from benchmaxxing.roster import (
    build_committee,
    cross_lineage_committees,
    default_roster,
    same_lineage_committees,
)
from benchmaxxing.schema import Committee, ModelSpec


def _gemini_only_roster() -> list[ModelSpec]:
    return [
        ModelSpec(name="gemini-2.5-pro", lineage="gemini", tier="pro"),
        ModelSpec(name="gemini-2.5-flash", lineage="gemini", tier="flash"),
        ModelSpec(name="gemini-2.0-flash", lineage="gemini", tier="flash-2.0"),
    ]


def test_build_committee_preserves_members():
    roster = default_roster()
    committee = build_committee(roster)
    assert isinstance(committee, Committee)
    assert committee.members == tuple(roster)


def test_default_roster_has_open_weights_and_multiple_lineages():
    roster = default_roster()
    lineages = {spec.lineage for spec in roster}
    assert "gemini" in lineages
    assert len(lineages) >= 2
    assert any(spec.is_open_weights for spec in roster)


def test_same_lineage_committees_are_not_cross_lineage():
    committees = same_lineage_committees(default_roster(), size=2)
    assert committees, "expected at least one same-lineage committee"
    for committee in committees:
        assert not committee.is_cross_lineage
        assert len(committee.lineages) == 1
        assert len(committee.members) == 2


def test_same_lineage_skips_lineages_smaller_than_size():
    # Only gemini has >= 2 members here; qwen has a single member and is skipped.
    committees = same_lineage_committees(default_roster(), size=2)
    for committee in committees:
        assert committee.lineages == {"gemini"}


def test_cross_lineage_committees_span_lineages_with_open_weights():
    committees = cross_lineage_committees(default_roster(), size=2)
    assert committees, "expected at least one cross-lineage committee"
    for committee in committees:
        assert committee.is_cross_lineage
        assert len(committee.lineages) >= 2
        assert any(spec.is_open_weights for spec in committee.members)


def test_cross_lineage_raises_on_gemini_only_roster():
    with pytest.raises(ValueError, match="open-weights"):
        cross_lineage_committees(_gemini_only_roster(), size=2)


def test_same_lineage_works_on_gemini_only_roster():
    committees = same_lineage_committees(_gemini_only_roster(), size=2)
    assert committees
    for committee in committees:
        assert not committee.is_cross_lineage


def test_invalid_size_raises():
    with pytest.raises(ValueError, match="size"):
        same_lineage_committees(default_roster(), size=0)
    with pytest.raises(ValueError, match="size"):
        cross_lineage_committees(default_roster(), size=0)
