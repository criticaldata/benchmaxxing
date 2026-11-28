"""The shared committee runners read ``--n`` manifest rows. The default is the size the MedQA lane's file
was committed at, so it only holds on that lane's own cache. Pointed at another lane's cache with
``--cache``, a runner whose ``--n`` is left out stops before it reads anything, instead of running the
MedQA size over a file committed at another size (MedMCQA ran four of these arms at smaller sizes).
"""
import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from benchmaxxing.data import write_manifest
from benchmaxxing.schema import Case, Modality

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import _lane  # noqa: E402

# runner -> the --n its MedQA file was committed at (docs/REPLAY_AUDIT.md, "Default flags")
HOME_N = {
    "attributed_tier": 120, "authority_ladder": 120, "committee_size_sweep": 120, "deliberation_framing": 120,
    "dose_response": 120, "hierarchy_dominance": 40, "leader_as_auditor": 120, "majority_pressure": 40,
    "orchestrator_failure": 100, "plausible_distractor": 120, "rationale_validity": 120, "seed_confidence": 100,
    "super_additivity": 120, "temperature_sensitivity": 120, "test_awareness": 120, "text_cue_types": 120,
    "true_peer_control": 150, "unanimity_break": 150,
}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(n, cache, home_n=120):
    return argparse.Namespace(n=n, cache=cache, home_n=home_n)


def test_a_stated_size_is_used_on_any_cache():
    assert _lane.resolve_n(_args(60, None)) == 60
    assert _lane.resolve_n(_args(60, "experiments/medmcqa/results/authority_ladder_cache.jsonl")) == 60


def test_the_default_is_the_home_size_on_the_home_cache():
    assert _lane.resolve_n(_args(None, None, home_n=150)) == 150


def test_a_missing_size_on_another_cache_is_an_error():
    with pytest.raises(SystemExit) as exc:
        _lane.resolve_n(_args(None, "experiments/medmcqa/results/authority_ladder_cache.jsonl"))
    assert "--n is required with --cache" in str(exc.value) and "120" in str(exc.value)


def test_add_n_arg_declares_no_default_and_keeps_the_home_size():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    _lane.add_n_arg(ap, 40)
    assert vars(ap.parse_args([])) == {"cache": None, "n": None, "home_n": 40}
    assert ap.parse_args(["--n", "25"]).n == 25


@pytest.mark.parametrize("arm", sorted(HOME_N))
def test_each_shared_runner_takes_its_size_through_the_guard(arm):
    src = (ROOT / "experiments/medqa" / f"{arm}.py").read_text()
    assert 'add_argument("--n"' not in src, "a plain --n default is back: it would apply on every lane"
    assert re.findall(r"_lane\.add_n_arg\(ap, (\d+)\)", src) == [str(HOME_N[arm])]
    # resolved straight after parsing, before the cache is opened or the output directory is made
    assert "    args = ap.parse_args()\n    args.n = _lane.resolve_n(args)\n" in src


def test_the_home_sizes_are_the_sizes_the_medqa_replay_pins():
    pinned = {arm: int(extra[extra.index("--n") + 1])
              for arm, (extra, _) in _load("test_medqa_arms_replay").ARMS.items() if "--n" in extra}
    shared = sorted(set(pinned) & set(HOME_N))
    assert len(shared) == 14
    assert {a: HOME_N[a] for a in shared} == {a: pinned[a] for a in shared}


def test_medmcqa_was_committed_at_other_sizes_for_four_arms():
    """Why the default cannot be shared: these MedMCQA files were committed at sizes below MedQA's."""
    pinned = _load("test_medmcqa_arms_replay").PINNED
    other = {arm: sorted({sizes[arm] for sizes in pinned.values() if arm in sizes and sizes[arm] != HOME_N[arm]})
             for arm in HOME_N}
    assert {a: v for a, v in other.items() if v} == {
        "authority_ladder": [60], "majority_pressure": [25], "orchestrator_failure": [80],
        "plausible_distractor": [119], "true_peer_control": [57, 59], "unanimity_break": [143]}


def _manifest(tmp_path):
    cases = [Case(case_id=f"synthetic-{i}", patient_id="", modality=Modality.TEXT,
                  question=f"Synthetic question {i}?", options=("first", "second", "third", "fourth"), answer_index=0)
             for i in range(3)]
    return write_manifest(cases, tmp_path / "manifest.csv")


def _run(arm, tmp_path, *extra):
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, f"experiments/medqa/{arm}.py", "--manifest", str(_manifest(tmp_path)),
                           "--cache", str(tmp_path / "other_lane_cache.jsonl"), "--out", str(tmp_path / "out"), *extra],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("arm", ["authority_ladder", "majority_pressure", "orchestrator_failure", "true_peer_control"])
def test_a_replay_on_another_cache_without_n_stops_before_reading_anything(arm, tmp_path):
    r = _run(arm, tmp_path)
    assert r.returncode != 0
    assert "--n is required with --cache" in r.stderr and f"({HOME_N[arm]})" in r.stderr
    assert not (tmp_path / "out").exists(), "the run got as far as making its output directory"
    assert not (tmp_path / "other_lane_cache.jsonl").exists()


def test_the_same_replay_with_n_goes_on_to_the_cache(tmp_path):
    """With the size stated the guard is passed: the run goes on and stops at the empty cache, keyless."""
    r = _run("authority_ladder", tmp_path, "--n", "2")
    assert r.returncode != 0
    assert "--n is required" not in r.stderr and "Cache miss and no GEMINI_API_KEY set" in r.stderr
