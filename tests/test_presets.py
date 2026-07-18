"""Tests for benchmaxxing.presets (the per-stage, ready-to-edit config presets).

These are pure config tests: no optional deps, no model calls, no network. They confirm that
every stage preset loads into a Config, that list_presets reports all six stages, and that the
same-lineage vs cross-lineage split the presets encode actually holds.
"""

from __future__ import annotations

import pytest

from benchmaxxing.config import Config
from benchmaxxing.presets import STAGE_PRESETS, get_preset, list_presets
from benchmaxxing.roster import default_roster

_STAGE_NAMES = [f"stage{i}" for i in range(6)]


def test_list_presets_returns_all_six_in_stage_order():
    names = list_presets()
    assert names == _STAGE_NAMES
    assert len(names) == 6


def test_list_presets_matches_stage_presets_keys():
    assert list(STAGE_PRESETS) == list_presets()


@pytest.mark.parametrize("name", _STAGE_NAMES)
def test_each_preset_loads_into_a_config(name):
    config = get_preset(name)
    assert isinstance(config, Config)
    assert isinstance(config.models, list)
    assert config.models, "a preset roster must not be empty"
    assert all(isinstance(m, str) for m in config.models)
    assert isinstance(config.dataset, str) and config.dataset
    assert isinstance(config.cue_set, str) and config.cue_set
    assert isinstance(config.seed, int)
    assert isinstance(config.out_dir, str) and config.out_dir


@pytest.mark.parametrize("name", _STAGE_NAMES)
def test_preset_config_round_trips_through_to_dict(name):
    config = get_preset(name)
    restored = Config.from_dict(config.to_dict())
    assert restored == config


@pytest.mark.parametrize("name", _STAGE_NAMES)
def test_get_preset_applies_the_stored_fields(name):
    preset = STAGE_PRESETS[name]
    config = get_preset(name)
    assert config.models == list(preset["models"])
    assert config.dataset == preset["dataset"]
    assert config.cue_set == preset["cue_set"]
    assert config.seed == preset["seed"]
    assert config.out_dir == preset["out_dir"]


def test_each_stage_writes_to_its_own_out_dir():
    out_dirs = [get_preset(name).out_dir for name in _STAGE_NAMES]
    assert len(set(out_dirs)) == len(out_dirs), "each stage should have a distinct out_dir"


def test_same_lineage_and_cross_lineage_split():
    roster = default_roster()
    lineage_of = {spec.name: spec.lineage for spec in roster}
    open_weights = {spec.name for spec in roster if spec.is_open_weights}

    # Stages 0 and 1 are the same-lineage control: a single lineage, no open-weights family.
    for name in ("stage0", "stage1"):
        models = get_preset(name).models
        lineages = {lineage_of[m] for m in models if m in lineage_of}
        assert lineages == {"gemini"}
        assert not (set(models) & open_weights)

    # Stages 2-5 are the cross-lineage arm: >1 lineage including an open-weights family.
    for name in ("stage2", "stage3", "stage4", "stage5"):
        models = get_preset(name).models
        lineages = {lineage_of[m] for m in models if m in lineage_of}
        assert len(lineages) >= 2
        assert set(models) & open_weights


def test_get_preset_is_independent_of_stored_preset():
    # Mutating the returned config must not leak back into STAGE_PRESETS.
    config = get_preset("stage0")
    config.models.append("mutated-model")
    config.out_dir = "somewhere-else"
    assert "mutated-model" not in STAGE_PRESETS["stage0"]["models"]
    assert STAGE_PRESETS["stage0"]["out_dir"] != "somewhere-else"
    # And a fresh resolve is unaffected by the mutation above.
    assert "mutated-model" not in get_preset("stage0").models


def test_unknown_preset_raises_keyerror_listing_available():
    with pytest.raises(KeyError) as excinfo:
        get_preset("stage99")
    message = str(excinfo.value)
    assert "stage99" in message
    assert "stage0" in message
