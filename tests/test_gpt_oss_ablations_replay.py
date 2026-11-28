"""The gpt-oss ablation outputs recompute from the committed files with no key and no server.

Every analysis here either makes no model call or reads a committed call cache, so each committed
output must come back byte-for-byte (modulo the live call counter) from a run in a scrubbed
environment. A test that only checks the file exists would not catch a stale output.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# a branch tests the lineages whose ablation outputs it carries (the two gpt-oss PRs land independently)
_CANDIDATES = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3-super-120b-a12b",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "nemotron-3-super-120b-a12b-fp8-local",
    "local/nvidia/cosmos-reason2-8b",
)
MODELS = [m for m in _CANDIDATES
          if (ROOT / "experiments/medqa/results" / m.replace("/", "_") / "stats_reconciliation_summary.json").exists()]
REASONING_MODELS = [m for m in _CANDIDATES
          if (ROOT / "experiments/medqa/results" / m.replace("/", "_") / "reasoning_effort_sensitivity_summary.json").exists()]
MANIFEST = Path(os.environ.get("BENCHMAXXING_MEDQA_MANIFEST", str(Path.home() / "Documents/DATASET/medqa/manifest_test.csv")))

DERIVED = [  # script, output path relative to the model's results dir, extra args
    ("experiments/blind_metric/counterfactual_referee.py", "blind_metric/results/{slug}/counterfactual_referee.json", []),
    ("experiments/blind_metric/reanalysis.py", "blind_metric/results/{slug}/blind_metric_ci.json", []),
    ("experiments/cascade/positional_regression.py", "cascade/results/{slug}/positional_regression.json", []),
    ("experiments/contamination/misaligned_proxy.py", "contamination/results/{slug}/misaligned_proxy.json", []),
    ("experiments/medqa/deference_reanalysis.py", "medqa/results/{slug}/deference_reanalysis.json", []),
    ("experiments/medqa/onset_distribution.py", "medqa/results/{slug}/onset_distribution.json", []),
    ("experiments/medqa/onset_distribution.py", "medmcqa/results/{slug}/onset_distribution.json",
     ["--results-dir", "experiments/medmcqa/results", "--label", "MedMCQA"]),
    ("experiments/medqa/stats_reconciliation.py", "medqa/results/{slug}/stats_reconciliation_summary.json", []),
    ("experiments/medqa/stats_reconciliation.py", "medmcqa/results/{slug}/stats_reconciliation_summary.json",
     ["--results-dir", "experiments/medmcqa/results", "--out", "experiments/medmcqa/results"]),
    ("experiments/referee/panel_vote.py", "referee/results/{slug}/panel_vote.json", []),
    ("experiments/referee/agreement.py", "referee/results/{slug}/referee_agreement.json", []),
]
VOLATILE = {"read", "new_api_calls_this_run", "lineage_note"}


def _scrubbed_env():
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    return env


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _run(script, args):
    r = subprocess.run([sys.executable, script, *args], cwd=ROOT, env=_scrubbed_env(), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-800:]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("script,rel,extra", DERIVED, ids=[f"{s.split('/')[-1][:-3]}:{r.split('/')[0]}" for s, r, _ in DERIVED])
def test_derived_output_recomputes_from_committed_rows(model, script, rel, extra, tmp_path):
    slug = model.replace("/", "_")
    out = ROOT / "experiments" / rel.format(slug=slug)
    if not out.exists():
        pytest.skip(f"{out.relative_to(ROOT)} is not committed for {model}")
    before = _strip(json.loads(out.read_text()))
    saved = out.read_bytes()
    try:
        _run(script, [*extra, "--model", model])
        after = _strip(json.loads(out.read_text()))
    finally:
        out.write_bytes(saved)
    assert after == before, f"{out.relative_to(ROOT)} does not recompute from the committed rows"


def test_every_contamination_misaligned_proxy_read_is_built_from_its_own_rows():
    """The test above leaves `read` out of its comparison. The contamination misaligned-proxy sentence
    is built from each lineage's own rows, so every file committed under
    experiments/contamination/results/ must carry exactly the sentence its own numbers produce. A file
    left with another lineage's figures fails here. No model call: the rows are only read."""
    import importlib.util

    lane = ROOT / "experiments/contamination"
    spec = importlib.util.spec_from_file_location("contamination_misaligned_proxy", lane / "misaligned_proxy.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    files = sorted((lane / "results").glob("**/misaligned_proxy.json"))
    assert files, "no contamination misaligned_proxy.json is committed"
    stale = []
    for path in files:
        committed = json.loads(path.read_text())
        # the rows the script reads (_solo_records): this lane's own copy when one was written,
        # otherwise the MedQA lane's copy in the folder of the same lineage
        rows = path.parent / "solo_records.jsonl"
        if not rows.exists():
            rows = ROOT / "experiments/medqa/results" / path.parent.name / "solo_records.jsonl"
        cases = [r for r in script._load_jsonl(rows) if r["cue"] == "longest_option"]
        result = script.misaligned_proxy_run(cases, script._model_fn, script._proxy_fn, script._truth_fn, method="pearson")
        numbers = (round(result.uptake.corr_hidden, 4), round(result.uptake.corr_ground, 4), round(result.uptake_delta, 4))
        kept = (committed["corr_decision_vs_proxy"], committed["corr_decision_vs_truth"], committed["uptake_delta"])
        assert (committed["n"], kept) == (result.n, numbers), f"{path.relative_to(ROOT)} does not recompute from its rows"
        if committed["read"] != script._read(cases, *numbers):
            stale.append(str(path.relative_to(ROOT)))
    assert not stale, "read is not the sentence these files' own rows produce: " + ", ".join(stale)


@pytest.mark.skipif(not MANIFEST.exists(), reason="MedQA manifest not on this machine")
@pytest.mark.parametrize("model", REASONING_MODELS)
def test_reasoning_effort_ladder_replays_from_its_cache(model):
    slug = model.replace("/", "_")
    d = ROOT / "experiments/medqa/results" / slug
    cache = ROOT / "experiments/medqa/results" / f"{slug}_reasoning_effort_cache.jsonl"
    summary, rows = d / "reasoning_effort_sensitivity_summary.json", d / "reasoning_effort_sensitivity.jsonl"
    before = _strip(json.loads(summary.read_text()))
    before_rows = sorted(rows.read_text().splitlines())
    saved = summary.read_bytes(), rows.read_bytes()
    try:
        _run("experiments/medqa/reasoning_effort_sensitivity.py",
             ["--manifest", str(MANIFEST), "--model", model, "--n", "120", "--cache", str(cache)])
        after = json.loads(summary.read_text())
        assert after["new_api_calls_this_run"] == 0, "the committed cache does not cover the run"
        assert _strip(after) == before
        assert sorted(rows.read_text().splitlines()) == before_rows
    finally:
        summary.write_bytes(saved[0]); rows.write_bytes(saved[1])


@pytest.mark.skipif(not MANIFEST.exists(), reason="MedQA manifest not on this machine")
@pytest.mark.parametrize("model", REASONING_MODELS)
def test_empty_answer_split_replays_keyless(model):
    slug = model.replace("/", "_")
    out = ROOT / "experiments/medqa/results" / slug / "empty_content_by_condition.json"
    before = _strip(json.loads(out.read_text()))
    saved = out.read_bytes()
    try:
        _run("experiments/medqa/empty_content_by_condition.py", ["--manifest", str(MANIFEST), "--model", model])
        after = json.loads(out.read_text())
        assert all(v == "replayed" for v in after["runner_status"].values()), after["runner_status"]
        assert _strip(after) == before
    finally:
        out.write_bytes(saved)
