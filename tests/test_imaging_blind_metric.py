from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def test_rate_limiter_waits_between_attempts(monkeypatch):
    now = [0.0]
    sleeps = []

    class FakeClock:
        @staticmethod
        def monotonic():
            return now[0]

        @staticmethod
        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

    monkeypatch.setattr(metric, "time", FakeClock, raising=False)

    limiter = metric._RateLimiter(40)
    limiter.wait()
    limiter.wait()

    assert sleeps == [pytest.approx(1.5)]


def test_rate_limiter_rejects_invalid_rpm():
    with pytest.raises(ValueError, match="rpm"):
        metric._RateLimiter(0)


def test_cache_hit_bypasses_rate_limiter(tmp_path, monkeypatch):
    cache = metric._Cache(tmp_path / "cache.jsonl", key="key", model="model", rpm=40)
    image = metric.Image.new("L", (2, 2), color=0)
    cache.store[cache.cache_key("prompt", image)] = "yes"
    waits = []
    monkeypatch.setattr(cache._limiter, "wait", lambda: waits.append(True))

    assert cache.ask("prompt", image) == "yes"
    assert waits == []


def test_nvidia_generation_config_is_explicit_without_changing_other_models():
    assert metric._generation_config("meta/llama-3.2-90b-vision-instruct") == {
        "max_tokens": 256,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
    }
    assert metric._generation_config("microsoft/phi-3-vision-128k-instruct") == {
        "max_tokens": 256,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
    }
    assert metric._generation_config("gemini-2.5-flash") == {"temperature": 0}


class _RateLimitError(Exception):
    status_code = 429


def test_retry_delay_pauses_on_rate_limit():
    assert metric._retry_delay(_RateLimitError(), attempt=1) >= 60.0


def test_run_phase_checkpoints_and_resumes_after_failure(tmp_path):
    cases = [SimpleNamespace(case_id=case_id) for case_id in ("a", "b", "c")]
    checkpoint = tmp_path / "phase.jsonl"
    seen = []

    def failing_worker(case):
        seen.append(case.case_id)
        if case.case_id == "b":
            raise RuntimeError("temporary API failure")
        return {"case_id": case.case_id, "value": 1}

    with pytest.raises(RuntimeError, match="temporary API failure"):
        metric._run_phase(cases, failing_worker, checkpoint)

    assert checkpoint.read_text().splitlines() == [json.dumps({"case_id": "a", "value": 1})]
    assert seen == ["a", "b"]

    resumed = metric._run_phase(
        cases,
        lambda case: {"case_id": case.case_id, "value": 2},
        checkpoint,
    )

    assert [row["case_id"] for row in resumed] == ["a", "b", "c"]
    assert json.loads(checkpoint.read_text().splitlines()[0])["value"] == 1
    assert len(checkpoint.read_text().splitlines()) == 3


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
