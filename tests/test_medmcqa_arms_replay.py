"""Every committed MedMCQA arm replays from its committed cache with no key, at the size it was committed
at, and reproduces its committed summary and rows (the MedMCQA counterpart of test_medqa_arms_replay.py).

The run writes into a temporary directory, so the committed files are never touched. Arms absent from
PINNED do not replay without a key: see docs/REPLAY_AUDIT.md for each one and why.
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
R = ROOT / "experiments/medmcqa/results"
EXPECTED = "5b2ce9718e7cf421d88dd516d9c52b8b282cd886d13cbbc61ef5d7463a90a30d"  # experiments/medmcqa/fetch_dev.py
_CANDIDATES = [os.environ.get("BENCHMAXXING_MEDMCQA_MANIFEST", ""), str(ROOT / "data/medmcqa_dev.csv"),
               str(Path.home() / "Documents/DATASET/medmcqa_dev_500.csv")]
MANIFEST = next((Path(p) for p in _CANDIDATES if p and Path(p).exists()
                 and hashlib.sha256(Path(p).read_bytes()).hexdigest() == EXPECTED), None)
_MEDQA = ["attributed_tier", "authority_ladder", "committee_size_sweep", "deliberation_framing", "dose_response",
          "leader_as_auditor", "plausible_distractor", "rationale_validity", "seed_confidence", "super_additivity",
          "temperature_sensitivity", "test_awareness", "text_cue_types", "hierarchy_dominance", "majority_pressure",
          "orchestrator_failure", "true_peer_control", "unanimity_break"]
# arm -> (script, cache file name, [(summary, rows)], extra arguments)
ARMS = {a: (f"experiments/medqa/{a}.py", f"{a}_cache.jsonl", [(f"{a}_summary.json", f"{a}.jsonl")], []) for a in _MEDQA}
ARMS.update({
    "referee_deployable": ("experiments/referee/referee_deployable.py", "referee_deployable_cache.jsonl",
                           [("referee_deployable_summary.json", "referee_deployable.jsonl")], []),
    "referee_judge": ("experiments/referee/referee_judge.py", "referee_judge_cache.jsonl",
                      [("referee_judge_summary.json", "referee_judge.jsonl")], []),
    "blind_metric": ("experiments/blind_metric/blind_metric.py", "blind_metric_call_cache.jsonl",
                     [("blind_metric_summary.json", "blind_metric.jsonl")], []),
    "check_warn_2x2": ("experiments/blind_metric/check_warn_2x2.py", "blind_metric_call_cache.jsonl",
                       [("check_warn_2x2_summary.json", "check_warn_2x2.jsonl")], []),
    "matched_baseline_reason": ("experiments/matched_baseline/matched_baseline_reason.py", "blind_metric_call_cache.jsonl",
                                [("matched_baseline_reason_summary.json", "matched_baseline_reason.jsonl"),
                                 ("matched_baseline_reason_repeats_summary.json", "matched_baseline_reason_repeats.jsonl")],
                                ["--lane", "text", "--repeats", "5"]),
})
# model -> {arm: the --n it was committed at}
PINNED = {
    "gemini-2.5-flash-lite": {"attributed_tier": 120, "authority_ladder": 60, "blind_metric": 40, "check_warn_2x2": 40, "committee_size_sweep": 120, "deliberation_framing": 120, "dose_response": 120, "hierarchy_dominance": 40, "leader_as_auditor": 120, "majority_pressure": 25, "matched_baseline_reason": 40, "plausible_distractor": 120, "rationale_validity": 120, "referee_deployable": 40, "referee_judge": 40, "seed_confidence": 100, "super_additivity": 120, "temperature_sensitivity": 120, "test_awareness": 120, "text_cue_types": 120, "true_peer_control": 59, "unanimity_break": 150},
    "gemini-2.5-flash": {"blind_metric": 40},
    "openai/gpt-oss-120b": {"attributed_tier": 120, "authority_ladder": 60, "blind_metric": 40, "committee_size_sweep": 120, "deliberation_framing": 120, "dose_response": 120, "hierarchy_dominance": 40, "leader_as_auditor": 120, "majority_pressure": 25, "matched_baseline_reason": 40, "orchestrator_failure": 80, "plausible_distractor": 120, "rationale_validity": 120, "referee_deployable": 40, "referee_judge": 40, "seed_confidence": 100, "super_additivity": 120, "temperature_sensitivity": 120, "test_awareness": 120, "text_cue_types": 120, "true_peer_control": 57, "unanimity_break": 143},
    "openai/gpt-oss-20b": {"attributed_tier": 120, "authority_ladder": 60, "blind_metric": 40, "committee_size_sweep": 120, "deliberation_framing": 120, "dose_response": 120, "hierarchy_dominance": 40, "leader_as_auditor": 120, "majority_pressure": 25, "matched_baseline_reason": 40, "orchestrator_failure": 80, "plausible_distractor": 120, "rationale_validity": 120, "referee_deployable": 40, "referee_judge": 40, "seed_confidence": 100, "super_additivity": 120, "temperature_sensitivity": 120, "test_awareness": 120, "text_cue_types": 120, "true_peer_control": 57, "unanimity_break": 150},
    "nvidia/nemotron-3-super-120b-a12b": {"attributed_tier": 120, "authority_ladder": 60, "blind_metric": 40, "seed_confidence": 100, "super_additivity": 120},
    "Qwen/Qwen2.5-VL-72B-Instruct": {"attributed_tier": 120, "authority_ladder": 60, "blind_metric": 40, "committee_size_sweep": 120, "deliberation_framing": 120, "dose_response": 120, "hierarchy_dominance": 40, "leader_as_auditor": 120, "majority_pressure": 25, "matched_baseline_reason": 40, "orchestrator_failure": 80, "plausible_distractor": 120, "rationale_validity": 120, "referee_deployable": 40, "referee_judge": 40, "seed_confidence": 100, "super_additivity": 120, "temperature_sensitivity": 120, "test_awareness": 120, "text_cue_types": 120, "true_peer_control": 59, "unanimity_break": 150},
    "nemotron-3-super-120b-a12b-fp8-local": {"blind_metric": 40, "committee_size_sweep": 120, "deliberation_framing": 120, "dose_response": 120, "hierarchy_dominance": 40, "leader_as_auditor": 120, "majority_pressure": 25, "matched_baseline_reason": 40, "orchestrator_failure": 80, "plausible_distractor": 119, "rationale_validity": 120, "referee_deployable": 40, "referee_judge": 40, "temperature_sensitivity": 120, "test_awareness": 120, "text_cue_types": 120, "true_peer_control": 59, "unanimity_break": 150},
}
VOLATILE = {"read", "new_api_calls_this_run"}
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


@pytest.mark.skipif(MANIFEST is None, reason="the 500-row MedMCQA manifest (sha256 5b2ce971...) is not on this machine")
@pytest.mark.parametrize("model,arm,n", CASES, ids=[f"{m.split('/')[-1]}:{a}" for m, a, _ in CASES])
def test_medmcqa_arm_replays_keyless_at_its_committed_size(model, arm, n, tmp_path):
    _replay(model, arm, n, MANIFEST, tmp_path)
