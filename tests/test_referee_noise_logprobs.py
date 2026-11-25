"""referee_noise_logprobs.py: the committed rows replay from their caches with no server, and the helpers
behave on hand-built inputs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "referee"))

import referee_noise_logprobs as rn  # noqa: E402

MANIFEST = Path(os.environ.get("BENCHMAXXING_MEDQA_MANIFEST", str(Path.home() / "Documents/DATASET/medqa/manifest_test.csv")))
LINEAGES = {"openai/gpt-oss-120b": "openai_gpt-oss-120b",
            "nemotron-3-super-120b-a12b-fp8-local": "nemotron-3-super-120b-a12b-fp8-local",
            "Qwen/Qwen2.5-VL-72B-Instruct": "Qwen_Qwen2.5-VL-72B-Instruct"}
VOLATILE = {"new_api_calls_this_run", "calls_by_kind"}


def test_letter_distribution_reads_the_last_letter_token():
    rec = {"letter_tokens": [
        {"token": "A", "logprob": 0.0, "top": [["A", 0.0]]},
        {"token": "B", "logprob": -0.1, "top": [["B", -0.1053605], [" C", -2.302585], ["x", -1.0]]}]}
    d = rn._letter_dist(rec, ["A", "B", "C"])
    assert d["A"] == 0.0 and d["B"] == pytest.approx(0.9, abs=1e-3) and d["C"] == pytest.approx(0.1, abs=1e-3)
    assert rn._letter_dist({"letter_tokens": []}, ["A"]) is None


def test_sign_test_is_two_sided_and_exact():
    assert rn._sign_test(0, 0) is None
    assert rn._sign_test(2, 0) == 0.5
    assert rn._sign_test(5, 5) == 1.0


@pytest.mark.skipif(not MANIFEST.exists(), reason="the MedQA manifest is not on this machine")
@pytest.mark.parametrize("model", sorted(LINEAGES))
def test_committed_rows_replay_with_no_server(model):
    d = ROOT / "experiments/referee/results" / LINEAGES[model]
    files = {p: p.read_bytes() for p in (d / "referee_noise_logprobs_summary.json", d / "referee_noise_logprobs.jsonl")}
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    try:
        r = subprocess.run([sys.executable, "experiments/referee/referee_noise_logprobs.py", "--manifest", str(MANIFEST),
                            "--model", model], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
        after = json.loads((d / "referee_noise_logprobs_summary.json").read_text())
        before = json.loads(files[d / "referee_noise_logprobs_summary.json"])
        assert after["new_api_calls_this_run"] == 0
        assert {k: v for k, v in after.items() if k not in VOLATILE} == {k: v for k, v in before.items() if k not in VOLATILE}
        assert (d / "referee_noise_logprobs.jsonl").read_bytes() == files[d / "referee_noise_logprobs.jsonl"]
    finally:
        for p, b in files.items():
            p.write_bytes(b)
