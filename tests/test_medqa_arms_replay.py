"""Every committed Gemini MedQA break-it arm replays from its committed cache with no key, at the
size it was committed at, and reproduces its committed summary and rows.

Since the answer parser was corrected, one case (medqa-65) plants a different wrong option, so 17 of
these arms stopped on a cache miss and their committed summaries were one or two counts off what the
code now computes. They were refreshed with the missing prompts (56 calls); this file keeps them
from drifting again.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(os.environ.get("BENCHMAXXING_MEDQA_MANIFEST", str(Path.home() / "Documents/DATASET/medqa/manifest_test.csv")))
R = "experiments/medqa/results/"
# arm -> (extra arguments for the committed size, rows file stem)
ARMS = {
    "attributed_tier": (["--n", "120"], "attributed_tier"),
    "authority_ladder": (["--n", "120"], "authority_ladder"),
    "committee_size_sweep": (["--n", "120"], "committee_size_sweep"),
    "deliberation_framing": (["--n", "120"], "deliberation_framing"),
    "hierarchy_temp": (["--n", "40"], "hierarchy_temp"),
    "leader_as_auditor": (["--n", "120"], "leader_as_auditor"),
    "majority_pressure": (["--n", "40"], "majority_pressure"),
    "orchestrator_failure": (["--n", "100"], "orchestrator_failure"),
    "paraphrase_robustness": (["--n", "120"], "paraphrase_robustness"),
    "plausible_distractor": (["--n", "120"], "plausible_distractor"),
    "rationale_validity": (["--n", "120"], "rationale_validity"),
    "scale_c": ([], "scale_c_per_case"),
    "seed_confidence": (["--n", "100"], "seed_confidence"),
    "seed_timing": (["--n", "120"], "seed_timing"),
    "super_additivity": (["--n", "120"], "super_additivity"),
    "temperature_sensitivity": (["--n", "120"], "temperature_sensitivity"),
    "true_peer_control": (["--n", "150"], "true_peer_control"),
    "unanimity_break": (["--n", "150"], "unanimity_break"),
}
VOLATILE = {"read", "new_api_calls_this_run"}

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="the MedQA manifest is not on this machine")


def _strip(o):
    return {k: v for k, v in o.items() if k not in VOLATILE}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_arm_replays_keyless_at_its_committed_size(arm):
    extra, stem = ARMS[arm]
    summary, rows = ROOT / R / f"{arm}_summary.json", ROOT / R / f"{stem}.jsonl"
    saved = {p: p.read_bytes() for p in (summary, rows)}
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    try:
        r = subprocess.run([sys.executable, f"experiments/medqa/{arm}.py", "--manifest", str(MANIFEST), *extra],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
        after = json.loads(summary.read_text())
        assert after.get("new_api_calls_this_run", 0) == 0
        assert _strip(after) == _strip(json.loads(saved[summary]))
        # rows are written in completion order, so compare them as a set
        assert sorted(rows.read_text().splitlines()) == sorted(saved[rows].decode().splitlines())
    finally:
        for p, b in saved.items():
            p.write_bytes(b)
