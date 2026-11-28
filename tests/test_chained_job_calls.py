"""Call counts of arms that ran across chained batch jobs (experiments/chained_job_calls.py).

A summary written by the last of several jobs counts only that job's calls. These checks pin the
count to the cache, keyless and without films, for the arms the script lists and, where the arm
has a cache of its own, for every lineage on the lane.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import chained_job_calls as cj  # noqa: E402


def _lines(path: Path) -> int:
    return sum(1 for l in path.read_text().splitlines() if l.strip())


def test_listed_arms_report_every_call_in_their_cache():
    assert cj.main(["--check"]) == 0
    got = {rel: json.loads((REPO / rel).read_text())["new_api_calls_this_run"]
           for arm in cj.ARMS for rel in arm["summaries"]}
    assert sorted(got.values()) == [210, 210, 245, 1146, 1151]


def test_matched_conditions_count_equals_its_session_cache_for_every_lineage():
    caches = sorted(REPO.glob("experiments/*/results/matched/*_session_cache.jsonl"))
    assert len(caches) >= 13
    for cache in caches:
        summary = cache.parent / cache.name[: -len("_session_cache.jsonl")] / "matched_conditions_summary.json"
        if summary.exists():
            assert json.loads(summary.read_text())["new_api_calls_this_run"] == _lines(cache), cache


def test_cue_strength_count_equals_its_own_cache_for_every_vision_lineage():
    # Lineage folders only: the Gemini file one level up predates the per-arm strength cache.
    checked = 0
    for summary in sorted(REPO.glob("experiments/imaging/results/*/imaging_strength_cascade_summary.json")):
        cache = summary.parent / "img_strength_cache.jsonl"
        if cache.exists():
            assert json.loads(summary.read_text())["new_api_calls_this_run"] == _lines(cache), summary
            checked += 1
    assert checked >= 3


def test_rewrite_changes_only_the_count(tmp_path, monkeypatch):
    lane = tmp_path / "res"
    lane.mkdir()
    before = {"n": 2, "model": "m", "new_api_calls_this_run": 1, "rate": 0.5}
    (lane / "s.json").write_text(json.dumps(before, indent=2))
    (lane / "cache.jsonl").write_text("".join(json.dumps({"k": str(i), "resp": "yes"}) + "\n" for i in range(3)))
    (lane / "reps.jsonl").write_text("".join(json.dumps({"case_id": c, "answers": ["yes"] * 4}) + "\n" for c in "ab"))
    (lane / "r.json").write_text(json.dumps({"new_api_calls_this_run": 2}, indent=2))
    monkeypatch.setattr(cj, "REPO", tmp_path)
    monkeypatch.setattr(cj, "ARMS", [
        {"name": "own cache", "count": "lines", "summaries": ["res/s.json"], "cache": "res/cache.jsonl"},
        {"name": "shared cache", "count": "repeat_reads", "summaries": ["res/r.json"],
         "rows": "res/reps.jsonl", "cache": "res/cache.jsonl"},
    ])
    assert cj.main(["--check"]) == 1
    assert cj.main([]) == 0
    assert json.loads((lane / "s.json").read_text()) == {**before, "new_api_calls_this_run": 3}
    assert json.loads((lane / "r.json").read_text()) == {"new_api_calls_this_run": 8}
    assert (lane / "s.json").read_text() == json.dumps({**before, "new_api_calls_this_run": 3}, indent=2)
    assert cj.main(["--check"]) == 0
