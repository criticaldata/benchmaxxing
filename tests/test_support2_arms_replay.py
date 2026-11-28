"""Every committed SUPPORT2 arm replays from its committed cache with no key, at the size it was committed
at, and reproduces its committed summary and rows. The manifest is committed, so this always runs.

The run writes into a temporary directory, so the committed files are never touched.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "gemini-2.5-flash-lite"
R = ROOT / "experiments/support2/results"
MANIFEST = R / "support2_manifest.csv"
ARMS = {a: (f"experiments/support2/{a}.py", "call_cache.jsonl", [(f"{a}_summary.json", f"{a}.jsonl")], [])
        for a in ("support2_solo", "support2_cascade", "support2_cascade_strength", "support2_referee", "support2_referee_judge")}
ARMS.update({
    "blind_metric": ("experiments/blind_metric/blind_metric.py", "blind_metric_call_cache.jsonl",
                     [("blind_metric_summary.json", "blind_metric.jsonl")], []),
    "check_warn_2x2": ("experiments/blind_metric/check_warn_2x2.py", "blind_metric_call_cache.jsonl",
                       [("check_warn_2x2_summary.json", "check_warn_2x2.jsonl")], []),
    "matched_baseline_reason": ("experiments/matched_baseline/matched_baseline_reason.py", "blind_metric_call_cache.jsonl",
                                [("matched_baseline_reason_summary.json", "matched_baseline_reason.jsonl"),
                                 ("matched_baseline_reason_repeats_summary.json", "matched_baseline_reason_repeats.jsonl")],
                                ["--lane", "text", "--repeats", "5"]),
})
PINNED = {
    "gemini-2.5-flash-lite": {"blind_metric": 40, "check_warn_2x2": 40, "matched_baseline_reason": 40, "support2_cascade": 120, "support2_cascade_strength": 120, "support2_referee": 120, "support2_referee_judge": 120, "support2_solo": 120},
    "gemini-2.5-flash": {"blind_metric": 40},
    "openai/gpt-oss-120b": {"blind_metric": 40, "matched_baseline_reason": 40, "support2_cascade": 120, "support2_cascade_strength": 120, "support2_referee": 120, "support2_referee_judge": 120, "support2_solo": 120},
    "openai/gpt-oss-20b": {"blind_metric": 40, "matched_baseline_reason": 40, "support2_cascade": 120, "support2_cascade_strength": 120, "support2_referee": 120, "support2_referee_judge": 120, "support2_solo": 120},
    "nvidia/nemotron-3-super-120b-a12b": {"blind_metric": 40, "support2_cascade": 120, "support2_cascade_strength": 120, "support2_referee": 120, "support2_referee_judge": 120, "support2_solo": 120},
    "Qwen/Qwen2.5-VL-72B-Instruct": {"blind_metric": 40, "matched_baseline_reason": 40, "support2_cascade": 120, "support2_cascade_strength": 120, "support2_referee": 120, "support2_referee_judge": 120, "support2_solo": 120},
    "nemotron-3-super-120b-a12b-fp8-local": {"blind_metric": 40, "matched_baseline_reason": 40},
}
# the solo arm also counts the temperature-1 noise resamples it made; a replay makes none and says so
VOLATILE = {"read", "new_api_calls_this_run", "new_noise_calls_this_run", "replayed_from_log"}
CASES = [(m, a, n) for m, arms in PINNED.items() for a, n in arms.items()]


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _lines(p):
    return sorted(l for l in p.read_text().splitlines() if l.strip())


def _replay(model, arm, n, manifest, tmp_path):
    script, cache, outputs, extra = ARMS[arm]
    slug = model.replace("/", "_")
    committed = R if model == DEFAULT else R / slug
    cache_path = R / (cache if model == DEFAULT else f"{slug}_{cache}")
    args = [*extra, "--manifest", str(manifest), "--cache", str(cache_path), "--out", str(tmp_path), "--n", str(n)]
    if model != DEFAULT or arm == "matched_baseline_reason":
        args += ["--model", model]
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, script, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
    fresh = tmp_path if (model == DEFAULT or arm == "matched_baseline_reason") else tmp_path / slug
    for summary, rows in outputs:
        if not (committed / summary).exists():
            continue
        after = json.loads((fresh / summary).read_text())
        assert after.get("new_api_calls_this_run", 0) == 0, f"{summary}: the committed cache does not cover the run"
        assert _strip(after) == _strip(json.loads((committed / summary).read_text())), f"{summary} differs from the committed file"
        # rows are written in completion order, so compare them as a set
        assert _lines(fresh / rows) == _lines(committed / rows), f"{rows} differs from the committed file"


@pytest.mark.parametrize("model,arm,n", CASES, ids=[f"{m.split('/')[-1]}:{a}" for m, a, _ in CASES])
def test_support2_arm_replays_keyless_at_its_committed_size(model, arm, n, tmp_path):
    _replay(model, arm, n, MANIFEST, tmp_path)
