"""The per-run reproducibility bundle: what a finished run leaves on disk (issue 105).

A number nobody can re-derive is not a result. Every run writes one directory that carries the
resolved config, the environment it ran in, the run manifest, the structured results, a readable
summary, and (for the cascade) the transcripts::

    runs/cascade-medqa-20260723/
      config.json         resolved config: models, cue set, seed, limits
      versions.json       package version, git SHA, python, platform, pinned library versions
      run_manifest.json   the RunManifest (model ids, prompt versions, dataset revision, ...)
      results.json        the runner's structured output, plus the plan, estimates and family
      summary.md          the artifact a collaborator reads first
      transcripts/        one saved transcript per case and condition, for replay

The point of the transcripts is :func:`replay_cascade`: the headline cascade numbers can be
recomputed from what is on disk, with no model calls and no key, so a reviewer can check the
reported adoption rather than take it on faith.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

from benchmaxxing.manifest import git_sha, library_versions
from benchmaxxing.onset import adoption_rate
from benchmaxxing.report import write_html_report
from benchmaxxing.runstore import RunStore

__all__ = [
    "BUNDLE_FILES",
    "bundle_dir_name",
    "environment",
    "load_bundle",
    "replay_cascade",
    "render_bundle_report",
    "write_versions",
]

# The files every bundle carries. A missing one means the run did not finish.
BUNDLE_FILES = ("config.json", "versions.json", "run_manifest.json", "results.json", "summary.md")

TRANSCRIPTS_DIR = "transcripts"


def bundle_dir_name(stage: str, dataset: str, date: str) -> str:
    """The conventional bundle directory name, ``<stage>-<dataset>-<YYYYMMDD>``."""
    return f"{stage}-{dataset}-{date}"


def environment() -> dict:
    """The environment a run happened in: package version, revision, interpreter, libraries."""
    import benchmaxxing

    return {
        "benchmaxxing": benchmaxxing.__version__,
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": library_versions(),
    }


def write_versions(out_dir) -> Path:
    """Write ``versions.json`` into a bundle directory and return its path."""
    path = Path(out_dir) / "versions.json"
    path.write_text(json.dumps(environment(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_bundle(bundle_dir) -> dict:
    """Read a bundle from disk into ``{config, versions, run_manifest, results, summary}``.

    Raises ``FileNotFoundError`` naming every file that is missing, so a half-written bundle
    fails with the whole list rather than one file at a time.
    """
    root = Path(bundle_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {root}")
    missing = [name for name in BUNDLE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{root} is not a complete run bundle; missing: {missing}")

    def read_json(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    return {
        "dir": str(root),
        "config": read_json("config.json"),
        "versions": read_json("versions.json"),
        "run_manifest": read_json("run_manifest.json"),
        "results": read_json("results.json"),
        "summary": (root / "summary.md").read_text(encoding="utf-8"),
    }


def replay_cascade(bundle_dir) -> dict:
    """Recompute the cascade adoption numbers from the saved transcripts alone.

    Reads the planted answer of each case out of ``results.json`` and then recounts adoption from
    ``transcripts/`` with :func:`benchmaxxing.onset.adoption_rate`, so the returned numbers come
    from the transcripts rather than from the reported summary. Offline: no model, no key.

    Returns ``{"per_case", "shared_adoption", "isolated_adoption", "adoption_delta", "n_cases",
    "skipped"}``, shaped like the cascade stage's own output so the two can be compared directly.
    ``skipped`` lists the reported cases whose transcript pair is missing from the bundle: the
    replay cannot cover them, so a caller can refuse to certify a bundle that lost part of its
    evidence rather than passing on the surviving subset's means.
    """
    root = Path(bundle_dir)
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    if results.get("stage") != "cascade":
        raise ValueError(
            f"replay_cascade needs a cascade bundle; {root} holds stage {results.get('stage')!r}"
        )
    store_dir = root / TRANSCRIPTS_DIR
    if not store_dir.is_dir():
        raise FileNotFoundError(f"No {TRANSCRIPTS_DIR}/ in {root}: nothing to replay")

    store = RunStore(store_dir)
    seeded_by_case = {
        row["case_id"]: row["seeded_answer"] for row in results["results"].get("per_case", [])
    }

    per_case = []
    skipped = []
    for case_id, seeded in sorted(seeded_by_case.items()):
        keys = {condition: f"{case_id}::{condition}" for condition in ("shared", "isolated")}
        if not all(store.has(key) for key in keys.values()):
            skipped.append(case_id)
            continue
        per_case.append(
            {
                "case_id": case_id,
                "seeded_answer": seeded,
                "shared_adoption": adoption_rate(store.load(keys["shared"]), seeded),
                "isolated_adoption": adoption_rate(store.load(keys["isolated"]), seeded),
            }
        )

    shared = [row["shared_adoption"] for row in per_case]
    isolated = [row["isolated_adoption"] for row in per_case]
    mean = lambda values: sum(values) / len(values) if values else float("nan")  # noqa: E731
    return {
        "per_case": per_case,
        "n_cases": len(per_case),
        "skipped": skipped,
        "shared_adoption": mean(shared),
        "isolated_adoption": mean(isolated),
        "adoption_delta": mean(shared) - mean(isolated) if per_case else float("nan"),
    }


def render_bundle_report(bundle_dir, out_path=None) -> Path:
    """Render a bundle into a standalone HTML report and return the file written.

    Everything comes from the bundle on disk, so a report can be regenerated months later from
    the directory alone. Defaults to ``report.html`` inside the bundle.
    """
    bundle = load_bundle(bundle_dir)
    results = bundle["results"]
    report = {
        "run": {
            "run_id": results.get("run_id"),
            "stage": results.get("stage"),
            "bundle": bundle["dir"],
        },
        "plan": results.get("plan", {}),
        "estimates": results.get("estimates", {}),
        "family correction": results.get("family_correction", {}),
        "results": results.get("results", {}),
        "config": bundle["config"],
        "environment": bundle["versions"],
        "run manifest": bundle["run_manifest"],
    }
    out = Path(out_path) if out_path else Path(bundle["dir"]) / "report.html"
    title = f"benchmaxxing {results.get('stage', 'run')}: {results.get('run_id', '')}".strip()
    return write_html_report(report, out, title=title)
