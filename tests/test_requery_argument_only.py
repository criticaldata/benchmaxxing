"""The argument-only re-query arm and the pre-emptive referee arm it shares a board with replay
from their committed caches with no key, and the shared condition is the same answer in both.

The pre-emptive arm had no replay test and had stopped replaying: one case's planted option moved
when the answer parser was corrected, so its prompt was no longer in the cache. A committed result
that cannot be recomputed is not pinned by anything, hence this file.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(os.environ.get("BENCHMAXXING_MEDQA_MANIFEST", str(Path.home() / "Documents/DATASET/medqa/manifest_test.csv")))
ARMS = [
    ("experiments/medqa/pre_emptive_referee.py", "experiments/medqa/results/pre_emptive_referee"),
    ("experiments/referee/requery_argument_only.py", "experiments/referee/results/requery_argument_only"),
]
VOLATILE = {"read", "new_api_calls_this_run"}

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="the MedQA manifest is not on this machine")


def _strip(o):
    return {k: v for k, v in o.items() if k not in VOLATILE}


@pytest.mark.parametrize("script,stem", ARMS, ids=[s.split("/")[-1][:-3] for s, _ in ARMS])
def test_arm_replays_keyless(script, stem):
    summary, rows = ROOT / f"{stem}_summary.json", ROOT / f"{stem}.jsonl"
    saved = {p: p.read_bytes() for p in (summary, rows)}
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    try:
        r = subprocess.run([sys.executable, script, "--manifest", str(MANIFEST), "--n", "120"],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
        after = json.loads(summary.read_text())
        assert after["new_api_calls_this_run"] == 0
        assert _strip(after) == _strip(json.loads(saved[summary]))
        # rows are written in completion order by one of the two scripts, so compare them as a set
        assert sorted(rows.read_text().splitlines()) == sorted(saved[rows].decode().splitlines())
    finally:
        for p, b in saved.items():
            p.write_bytes(b)


def test_the_shared_condition_is_one_answer_in_both_arms():
    pre = {json.loads(l)["case_id"]: json.loads(l) for l in (ROOT / "experiments/medqa/results/pre_emptive_referee.jsonl").read_text().splitlines() if l.strip()}
    new = {json.loads(l)["case_id"]: json.loads(l) for l in (ROOT / "experiments/referee/results/requery_argument_only.jsonl").read_text().splitlines() if l.strip()}
    assert set(pre) == set(new) and len(new) == 120
    assert all(pre[k]["no_intervention_adopt"] == new[k]["source_argument_adopt"] for k in new)


def test_the_counts_in_the_summary_are_the_rows():
    s = json.loads((ROOT / "experiments/referee/results/requery_argument_only_summary.json").read_text())
    rows = [json.loads(l) for l in (ROOT / "experiments/referee/results/requery_argument_only.jsonl").read_text().splitlines() if l.strip()]
    for name, cell in s["adoption_by_condition"].items():
        assert cell["k"] == sum(r[f"{name}_adopt"] for r in rows) and cell["n"] == len(rows)
    a = s["adopters_under_source_argument"]
    assert a["n"] == s["adoption_by_condition"]["source_argument"]["k"]
    assert s["referee_flags"]["argument_only_requery"] == a["n"] - a["still_adopt_with_argument_only"]
