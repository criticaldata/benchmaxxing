"""Drive the MIMIC-CXR imaging battery: the committed path from per-arm manifest to summary (#343).

Every number under ``results/`` is produced by the dataset-agnostic runners in
``experiments/imaging/`` (the same code the NIH lane runs), pointed at the per-arm manifests that
``build_subset.py select`` writes. This module is the committed record of *how* they were pointed:
which manifest, which output directory, which cache, and in what order. Before it existed the
invocations lived only in the README prose, so the committed summaries could not be regenerated or
audited from the repo.

Forking copies of the imaging runners into this directory would duplicate the exact bug class that
#332/#333/#338 had to fix seven times over. The arms below call the shared, already-fixed runners
instead, and :func:`clean_read_planters` refuses to launch an arm whose runner has regressed to the
old plant-against-the-clean-read design.

Three things the README's prose commands got wrong, all fixed here:

- **cache**: every runner except ``imaging_solo`` defaults ``--cache`` into
  ``experiments/imaging/results/`` (``img_cache.jsonl``, or ``img_strength_cache.jsonl`` for the
  strength sweep), which are the NIH lane's *committed* caches. A MIMIC run that does not pass
  ``--cache`` appends credentialed-report-derived reads into a tracked file.
- **truncation**: ``--n`` defaults to the NIH arm sizes (35 or 40), so an unpinned run silently
  measures the first 35 images of an 834-image arm.
- **the peer contrast**: ``imaging_system_flag`` reads ``imaging_cascade.jsonl`` from its own
  ``--out`` to contrast an automated flag against a peer assertion. The cascade arm writes to
  ``cascade_150/``, so without staging that transcript the ``vs_peer_assertion_cascade`` block
  drops out of the summary with no warning.

The ``judge`` arm exists because it did not (#393): the same-lineage judge was the one detector in
the cross-dataset table with no MIMIC-CXR number, purely because nothing here ever invoked it. It
replays the ``referee_300`` transcript the ``referee`` arm already replays, so ordering it into the
same battery run costs one text pass and closes the window where that gitignored transcript is
produced, used once, and cleaned up before the judge ever sees it.

Usage (see README section 4 for the full recipe):

    python -m experiments.mimic_cxr_image.run_battery --image-root ~/mimic-cxr-raw
    python -m experiments.mimic_cxr_image.run_battery --image-root ~/mimic-cxr-raw --arm cascade
    python -m experiments.mimic_cxr_image.run_battery --image-root . --dry-run

``--dry-run`` prints the commands and calls nothing, so the recipe is checkable without a key,
without images, and without spending API calls.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
IMAGING_DIR = REPO_ROOT / "experiments" / "imaging"

# The shared runners cap the case list at --n and default to the NIH arm sizes (35 cascade, 40
# solo). Every MIMIC arm wants its manifest whole, so pass a bound no arm will ever reach rather
# than a per-arm count that would go stale the next time build_subset picks different sizes.
WHOLE_MANIFEST = 1_000_000

# One cache for the whole battery. Keys are sha256(image bytes + prompt) under a fixed model and
# the arms nest (blind 100 < cascade 150 < referee 300 < solo 600), so a single file lets a later
# arm reuse an earlier arm's identical clean reads instead of paying for them twice.
CACHE_NAME = "img_cache.jsonl"

# The judge keeps its own cache file. Since #408 it opens the film, so its keys are
# sha256(image bytes + prompt) like the image arms rather than prompt-only, and the file is
# DUA-sensitive on the same footing; results/**/*.jsonl already excludes it. Keeping it separate
# still matters, because the judge's prompt set is tiny and mixing it into the shared cache would
# make the image arms' hit rate unreadable.
JUDGE_CACHE_NAME = "judge_cache.jsonl"

# The cascade family used to plant `wrong = flip(clean_read)`, which coincides with ground truth
# whenever the holdout's own read was already wrong (#332). MIMIC is the worst case: ~80% of its
# finding-present cases planted the truth, because gemini-2.5-flash reads pneumothorax poorly.
# #333/#338 replaced it with a constant `wrong = "no"` against the always-"yes" ground truth.
_CLEAN_READ_PLANT = re.compile(r'wrong\s*=\s*"(?:yes|no)"\s+if\s+clean\s*==')


@dataclass(frozen=True)
class Arm:
    """One battery arm: a shared imaging runner bound to a manifest and an output directory."""

    name: str
    module: str
    manifest: str
    out: str = ""  # subdirectory of --results; "" is the results root
    extra: tuple[str, ...] = ()  # appended verbatim; "{results}" expands to the results root
    stage: tuple[str, str] | None = None  # (src, dst) relative to --results, copied before the run
    takes_n: bool = True
    takes_manifest: bool = True  # False for runners that replay a transcript and open no image
    cache: str = CACHE_NAME
    needs: str = ""  # transcript this arm reads, relative to --results; checked before launching


# Order matters twice: system_flag stages the cascade transcript, and the referee consumes the
# referee_300 cascade transcript. Both dependencies are encoded below, not left to the operator.
ARMS: tuple[Arm, ...] = (
    Arm("solo", "imaging_solo", "solo_600.csv"),
    Arm("nih_match_solo", "imaging_solo", "nih_match_35.csv", out="nih_match_35"),
    Arm("cascade", "imaging_cascade", "cascade_150.csv", out="cascade_150"),
    Arm(
        "system_flag",
        "imaging_system_flag",
        "solo_600.csv",
        stage=("cascade_150/imaging_cascade.jsonl", "imaging_cascade.jsonl"),
    ),
    Arm("strength_cascade", "imaging_strength_cascade", "solo_600.csv"),
    Arm("blind_metric", "imaging_blind_metric", "blind_metric_100.csv"),
    Arm("referee_cascade", "imaging_cascade", "referee_300.csv", out="referee_300"),
    Arm(
        "referee",
        "imaging_referee",
        "referee_300.csv",
        out="referee_300",
        extra=("--cascade-jsonl", "{results}/referee_300/imaging_cascade.jsonl"),
        takes_n=False,  # imaging_referee replays a transcript and has no --n
        needs="referee_300/imaging_cascade.jsonl",
    ),
    Arm(
        "judge",
        "imaging_judge_referee",
        "referee_300.csv",
        out="referee_300",
        extra=("--cascade-jsonl", "{results}/referee_300/imaging_cascade.jsonl"),
        takes_n=False,
        # #408 changed the contract under this arm: the judge now opens the film so its verdict is
        # not pinned to the naive gate, which means it needs the manifest (to map case_id to
        # image_ref) and --image-root. Without them imaging_judge_referee exits on a usage error,
        # so this arm could not run as originally written.
        takes_manifest=True,
        cache=JUDGE_CACHE_NAME,
        needs="referee_300/imaging_cascade.jsonl",
    ),
)

ARM_NAMES = tuple(a.name for a in ARMS)


def clean_read_planters(imaging_dir: Path = IMAGING_DIR) -> list[str]:
    """Runner module names still planting against the clean read. Empty means all are fixed.

    Source inspection rather than a call, because the plant is a local inside each runner's
    per-case closure. The check is cheap and the bug recurred across seven sibling files, so it is
    worth enforcing at launch instead of trusting the next author to remember #332.
    """
    found = []
    for module in sorted({a.module for a in ARMS}):
        path = imaging_dir / f"{module}.py"
        if path.is_file() and _CLEAN_READ_PLANT.search(path.read_text(encoding="utf-8")):
            found.append(module)
    return found


def writes(transcript: str) -> str:
    """Name of the arm that produces ``transcript`` (a path relative to --results).

    Matched on output directory plus runner module, since every imaging runner names its per-case
    file after itself; ``""`` when nothing in ARMS writes it.
    """
    out, _, name = transcript.rpartition("/")
    return next((a.name for a in ARMS if a.out == out and a.module == name.removesuffix(".jsonl")), "")


def build_command(arm: Arm, manifests: Path, image_root: Path, results: Path,
                  python: str = sys.executable) -> list[str]:
    """The exact argv for one arm. Pure: no filesystem writes, so tests can assert on it."""
    cmd = [python, "-m", f"experiments.imaging.{arm.module}"]
    if arm.takes_manifest:
        cmd += ["--manifest", str(manifests / arm.manifest), "--image-root", str(image_root)]
    cmd += [
        "--out", str(results / arm.out if arm.out else results),
        "--cache", str(results / arm.cache),
    ]
    if arm.takes_n:
        cmd += ["--n", str(WHOLE_MANIFEST)]
    return cmd + [a.format(results=results) for a in arm.extra]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the MIMIC-CXR imaging battery through the shared imaging runners.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image-root", required=True, help="dir holding the downloaded MIMIC images")
    ap.add_argument("--manifests", default=str(HERE / "manifests"),
                    help="dir of per-arm manifests written by build_subset.py select")
    ap.add_argument("--results", default=str(HERE / "results"), help="output dir for summaries and transcripts")
    ap.add_argument("--arm", action="append", choices=ARM_NAMES,
                    help="repeatable; default runs every arm in dependency order")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and call nothing")
    args = ap.parse_args()

    manifests = Path(args.manifests).expanduser()
    results = Path(args.results).expanduser()
    image_root = Path(args.image_root).expanduser()
    selected = [a for a in ARMS if not args.arm or a.name in args.arm]

    blocked = sorted({a.module for a in selected} & set(clean_read_planters()))
    if blocked:
        raise SystemExit(
            f"refusing to run: {', '.join(blocked)} still plants against the model's clean read, "
            "not ground truth (#332). Rebase onto main, which carries the #333/#338 fix, before "
            "regenerating these arms."
        )

    # Only the arms that pass --manifest: a transcript-replay arm has no use for the file, and
    # failing it here would point the operator at build_subset when its real input is missing.
    missing = sorted({a.manifest for a in selected
                      if a.takes_manifest and not (manifests / a.manifest).is_file()})
    if missing and not args.dry_run:
        raise SystemExit(
            f"missing manifest(s) under {manifests}: {', '.join(missing)}. "
            "Run `build_subset.py select` first (README section 1)."
        )

    for arm in selected:
        out = results / arm.out if arm.out else results
        cmd = build_command(arm, manifests, image_root, results)
        print(f"# {arm.name}\n+ {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        if arm.needs and not (results / arm.needs).is_file():
            # results/**/*.jsonl is gitignored, so a transcript-replay arm run against a fresh
            # checkout, or after a cleanup, finds nothing. Say which arm rewrites it (#393).
            raise SystemExit(
                f"{arm.name} reads {results / arm.needs}, which does not exist. It is written by "
                f"the '{writes(arm.needs)}' arm and is gitignored, so it never comes from a "
                f"checkout: run that arm first (or the whole battery, which orders them)."
            )
        if arm.stage:
            src, dst = (results / p for p in arm.stage)
            if not src.is_file():
                raise SystemExit(f"{arm.name} needs {src}; run the arm that writes it first.")
            shutil.copyfile(src, dst)
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
