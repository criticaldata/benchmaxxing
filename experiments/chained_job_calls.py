"""Restore the call count of arms that ran across chained batch jobs. No model calls.

Every runner writes ``new_api_calls_this_run``: the calls its own process made. When a batch job
hits its time limit, the next job reruns the arm, which reads the calls already made from its
cache and asks only for the rest. The summary written by that last job then counts only the
calls of the last job, although the cache holds every call the arm made. Every other lineage ran these arms in one process, so its field equals the arm's cache
lines. This script sets the field to that count for the arms listed below and changes nothing
else in their summaries (``json.dumps(indent=2)``, the runners' own format).

How each arm's count is read from its cache:

  lines        the arm has a cache of its own (the matched-conditions session cache, the
               cue-strength cache): every line is one call the arm made.
  repeat_reads the arm shares the lane cache (the matched control with --repeats): every read in
               ``matched_baseline_reason_repeats.jsonl`` is a key that only this arm asks for (the
               control prompt, then the same prompt salted ``repeat1`` to ``repeatN``), so the
               arm wrote one cache line per read. With ``--image-root`` the script recomputes
               those keys from the films and checks that each is in the cache exactly once. The
               keys are over PNG bytes, so this check needs the image library the run used
               (Pillow 12; Pillow 13 encodes the same film to different bytes).

Usage:
    python experiments/chained_job_calls.py            # rewrite the listed summaries
    python experiments/chained_job_calls.py --check    # exit 1 if any summary disagrees
    python experiments/chained_job_calls.py --check --image-root <NIH images/>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GLM = "zai-org_GLM-5.3-Flash"
NIH = f"experiments/imaging/results/{GLM}"

# GLM-5.3-Flash imaging battery (PR #471): the four arms that a job time limit split across
# ORCD jobs 30344588, 30378097 and 30407030.
ARMS = [
    {"name": "NIH matched conditions", "count": "lines",
     "summaries": [f"experiments/imaging/results/matched/{GLM}/matched_conditions_summary.json"],
     "cache": f"experiments/imaging/results/matched/{GLM}_session_cache.jsonl"},
    {"name": "CheXpert matched conditions", "count": "lines",
     "summaries": [f"experiments/chexpert/results/matched/{GLM}/matched_conditions_summary.json"],
     "cache": f"experiments/chexpert/results/matched/{GLM}_session_cache.jsonl"},
    {"name": "NIH cue strength", "count": "lines",
     "summaries": [f"{NIH}/imaging_strength_cascade_summary.json"],
     "cache": f"{NIH}/img_strength_cache.jsonl"},
    {"name": "NIH matched control with repeats", "count": "repeat_reads",
     "summaries": [f"{NIH}/matched_baseline_reason_summary.json",
                   f"{NIH}/matched_baseline_reason_repeats_summary.json"],
     "rows": f"{NIH}/matched_baseline_reason_repeats.jsonl",
     "cache": f"{NIH}/img_cache.jsonl",
     "manifest": "experiments/imaging/results/nih_manifest.csv", "model": "zai-org/GLM-5.3-Flash"},
]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _film_keys(arm: dict, root: Path, repo: Path) -> list[str]:
    """The control arm's cache keys, computed by the runner's own key function."""
    sys.path.insert(0, str(repo / "experiments"))
    sys.path.insert(0, str(repo / "experiments" / "matched_baseline"))
    import matched_baseline_reason as mbr  # noqa: E402
    from PIL import Image

    rows = _jsonl(repo / arm["rows"])
    repeats = len(rows[0]["answers"]) - 1
    by_id = {c.case_id: c for c in mbr.load_cases(str(repo / arm["manifest"]))}
    cache = mbr._Cache(repo / arm["cache"], None, arm["model"])
    keys = []
    for r in rows:
        case = by_id[r["case_id"]]
        img = Image.open(root / case.image_ref).convert("L")
        _, reason_p, _ = mbr.imaging_prompts(case.label.split("|")[0].strip())
        keys.append(cache._k(reason_p, img))
        keys += [cache._k(reason_p, img, salt=f"repeat{i}") for i in range(1, repeats + 1)]
    return keys


def arm_calls(arm: dict, repo: Path = REPO, image_root: Path | None = None) -> int:
    """The number of calls the arm made across all its jobs, read from its cache."""
    cache = repo / arm["cache"]
    if arm["count"] == "lines":
        return len(_jsonl(cache))
    n = sum(len(r["answers"]) for r in _jsonl(repo / arm["rows"]))
    if image_root is not None:
        lines = Counter(r["k"] for r in _jsonl(cache))
        keys = _film_keys(arm, Path(image_root), repo)
        if len(set(keys)) != n or any(lines[k] != 1 for k in keys):
            missing = sum(1 for k in set(keys) if lines[k] == 0)
            raise SystemExit(f"{arm['name']}: {missing} of {len(set(keys))} keys not in {arm['cache']} "
                             "(wrong films, or an image library that encodes PNGs differently)")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if a summary disagrees")
    ap.add_argument("--image-root", default=None,
                    help="NIH images: also check the shared-cache arm's keys against the cache")
    args = ap.parse_args(argv)
    bad = 0
    for arm in ARMS:
        want = arm_calls(arm, REPO, Path(args.image_root) if args.image_root else None)
        for rel in arm["summaries"]:
            path = REPO / rel
            summary = json.loads(path.read_text())
            have = summary["new_api_calls_this_run"]
            print(f"{arm['name']}: {rel}: {have} -> {want}" if have != want else
                  f"{arm['name']}: {rel}: {have} (matches the cache)")
            if have != want:
                bad += 1
                if not args.check:
                    summary["new_api_calls_this_run"] = want
                    path.write_text(json.dumps(summary, indent=2))
    return 1 if (bad and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
