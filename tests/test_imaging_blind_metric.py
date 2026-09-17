from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.imaging_chexpert import imaging_blind_metric as metric


def test_cache_rejects_conflicting_duplicate_keys(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text(
        json.dumps({"k": "model:key", "resp": "yes"}) + "\n"
        + json.dumps({"k": "model:key", "resp": "no"}) + "\n"
    )

    with pytest.raises(ValueError, match="conflicting responses.*model:key"):
        metric._Cache(path, key=None, model="model")


def test_cache_key_is_recorded_for_each_prompt(tmp_path, monkeypatch):
    cache = metric._Cache(tmp_path / "cache.jsonl", key=None, model="model")
    image = metric.Image.new("L", (2, 2), color=0)

    key = cache.cache_key("prompt", image)
    assert key.startswith("model:")
    assert key == cache.cache_key("prompt", image)


def test_rate_excludes_unparseable_answers_from_denominator():
    rows = [
        {"blind_is_decoy": True},
        {"blind_is_decoy": False},
        {"blind_is_decoy": None},
    ]

    assert metric._rate(rows, "blind_is_decoy") == (0.5, 2)


def test_unparseable_baseline_cannot_define_a_decoy():
    with pytest.raises(ValueError, match="baseline answer is unparseable"):
        metric._decoy_for("?")


def test_committed_llama_rows_reference_all_three_cache_entries():
    root = Path(__file__).resolve().parents[1]
    cache = {
        json.loads(line)["k"]
        for line in (root / "experiments/chexpert/results/img_cache.jsonl").read_text().splitlines()
        if line.strip()
    }
    rows = [
        json.loads(line)
        for line in (root / "experiments/chexpert/results/meta_llama-3.2-90b-vision-instruct/imaging_blind_metric.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]

    assert len(rows) == 35
    for row in rows:
        assert all(row[field] in cache for field in ("base_cache_key", "blind_cache_key", "aware_cache_key"))
