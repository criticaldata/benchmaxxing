"""Pins for experiments/lineage_report.py and the hosted-prefix exclusion it reports against.

The report script is what makes the PR body's non-per-arm claims recomputable from a clone, so the
numbers it prints are pinned here: if a future run changes them, the body is wrong rather than the
test being stale, and this is where that surfaces.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import _lane  # noqa: E402
import lineage_report as lr  # noqa: E402

MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
SLUG = "Qwen_Qwen2.5-VL-72B-Instruct"


def test_check_passes_on_the_committed_rows():
    assert lr.main(["--model", MODEL, "--check"]) == 0


def test_unseeded_accuracy_is_one_number_across_full_cohort_arms():
    arms = lr.unseeded("medqa", SLUG)
    full = [a for a in arms if a["n"] == 120]
    assert len(full) == 17, [a["arm"] for a in full]
    assert {a["unseeded_correct"] for a in full} == {90}


def test_repeat_prompt_count_behind_the_reproducibility_claim():
    import glob

    caches = glob.glob(os.path.join(lr.HERE, "medqa", "results", f"*{SLUG}*cache*.jsonl"))
    r = lr.repeats(caches)
    assert r["repeated_prompts"] == 401
    assert r["answer_changed"] == 2


@pytest.mark.parametrize("model,local", [
    ("Qwen/Qwen2.5-VL-72B-Instruct", True),
    ("openai/gpt-oss-120b", True),
    ("nvidia/nemotron-3-super-120b-a12b", False),
    ("gemini-2.5-flash", False),
])
def test_hosted_prefixes_are_never_served_locally(monkeypatch, model, local):
    """A configured local server must not answer a cache miss for a vendor-hosted id."""
    monkeypatch.setattr(_lane, "LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
    assert _lane.is_local(model) is local
