"""Deterministic subset tooling for CheXpert imaging experiments.

Mirrors ``experiments/mimic_cxr_image/build_subset.py`` (#329): instead of duplicating the
imaging runners (solo, cascade, referee, etc.), this script builds nested per-arm manifests
from a single seed and the shared runners (``experiments/imaging/imaging_*.py``) are invoked
directly with ``--manifest`` pointed at the relevant arm CSV.

Two modes:

    select   build the full manifest via the chexpert adapter, rank cases deterministically,
             write per-arm manifests into --out-dir
    verify   after confirming images exist on disk, write provenance.json with sha256 per image

CheXpert is image-level (one case per image), not study-level like MIMIC, so the nesting is
simpler: every smaller arm is a strict prefix of every larger arm under the same seed.

Finding-absent subgroup: ``--include-no-finding`` writes a parallel set of manifests for
"no finding" cases, kept as a separate subgroup per the Issue #331 discussion.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from benchmaxxing.budget import RunBudget, subsample_cases
from benchmaxxing.data import load_cases
from benchmaxxing.datasets import chexpert
from benchmaxxing.datasets.base import finalize

# Arm -> number of cases. Every smaller arm is a strict prefix of every larger arm.
DEFAULT_SIZES = {
    "solo": 600,
    "referee": 300,
    "cascade": 150,
    "nih_match": 35,
}


def _is_finding(case) -> bool:
    """True when the case carries a real finding label (not 'no finding')."""
    return bool(case.label) and case.label.strip().lower() not in ("no finding", "", "none")


def select_arms(cases, seed: int, sizes: dict[str, int],
                require_finding: bool = True) -> dict[str, list]:
    """Rank cases deterministically by ``seed`` and return nested per-arm case lists.

    Because every arm ranks the same pool with the same seed, the smaller arms are strict
    prefixes of the larger ones (the prefix property of ``subsample_cases``).
    """
    eligible = [c for c in cases if not require_finding or _is_finding(c)]

    arms: dict[str, list] = {}
    for arm, n in sizes.items():
        picked = subsample_cases(eligible, RunBudget(max_cases=n, seed=seed))
        arms[arm] = picked
    return arms




def _build_full_cases(raw_root: str, full_manifest: str | None):
    """Return every case in the release. Reuses ``--full-manifest`` when given (fast reruns),
    otherwise builds it once into a temp file (discarded) via the registered adapter."""
    if full_manifest and Path(full_manifest).is_file():
        return load_cases(full_manifest)
    with tempfile.TemporaryDirectory() as tmp:
        mpath = chexpert.build_manifest(raw_root, Path(tmp) / "full_manifest.csv")
        return load_cases(mpath)


def do_select(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = dict(DEFAULT_SIZES)

    all_cases = _build_full_cases(args.raw_root, args.full_manifest)

    if args.exclude:
        import csv
        with open(args.exclude) as f:
            exclude_ids = {row["case_id"] for row in csv.DictReader(f)}
        all_cases = [c for c in all_cases if c.case_id not in exclude_ids]

    if args.target == "support_devices":
        from dataclasses import replace
        # Exclude uncertain cases (Issue #331 methodology critique)
        all_cases = [c for c in all_cases if c.meta.get("labels", {}).get("Support Devices") != "-1.0"]
        remapped_cases = []
        for c in all_cases:
            new_meta = dict(c.meta)
            new_meta["comorbidity"] = c.label
            new_label = "Support Devices" if c.meta.get("support_devices") else "no finding"
            remapped_cases.append(replace(c, label=new_label, meta=new_meta))
        all_cases = remapped_cases

    # Finding-present arms (main experiment).
    arms = select_arms(all_cases, seed=args.seed, sizes=sizes, require_finding=True)

    written = {}
    for arm, cases in arms.items():
        path = out_dir / f"{arm}_{sizes[arm]}.csv"
        finalize(cases, path)
        written[arm] = {"manifest": str(path), "images": len(cases)}

    # Label distribution for the largest arm (sanity check against monoculture).
    from collections import Counter
    superset = arms.get("solo", [])
    label_dist = dict(Counter(c.label.split("|")[0].strip() for c in superset).most_common())

    summary = {
        "seed": args.seed,
        "arms": written,
        "label_distribution_solo": label_dist,
        "recommended_commands": {
            arm: f"python experiments/imaging/imaging_{'solo' if 'solo' in arm else arm.split('_')[0]}.py --manifest {written[arm]['manifest']} --n {sizes[arm]}"
            for arm in written
        }
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def do_verify(args) -> None:
    image_root = Path(args.image_root)
    cases = load_cases(args.manifest)
    # Provenance checksums are now handled by staging.stage_dataset
    # Here we just verify images exist for the subset.
    missing = [c.image_ref for c in cases if not (image_root / c.image_ref).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(cases)} images not found under {image_root} "
            f"(first few: {missing[:5]}). Check --image-root."
        )
    print(json.dumps({"n_cases": len(cases), "all_present": True}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    sel = sub.add_parser("select", help="rank cases and write per-arm manifests")
    sel.add_argument("--raw-root", required=True,
                     help="CheXpert train.csv/valid.csv, or a directory containing it")
    sel.add_argument("--out-dir", required=True, help="output dir for per-arm manifests")
    sel.add_argument("--seed", type=int, default=20240724,
                     help="selection seed (fixes the nested case set)")
    sel.add_argument("--full-manifest", default=None,
                     help="reuse a prebuilt full manifest instead of rebuilding")
    sel.add_argument("--exclude", default=None,
                     help="Path to manifest CSV to exclude cases from")
    sel.add_argument("--target", default=None,
                     help="override the primary label to this target (e.g. 'support_devices')")
    sel.set_defaults(func=do_select)

    ver = sub.add_parser("verify",
                         help="check images resolve on disk")
    ver.add_argument("--manifest", required=True,
                     help="a per-arm manifest (usually the solo_600 superset)")
    ver.add_argument("--image-root", required=True,
                     help="dir holding the CheXpert images")
    ver.set_defaults(func=do_verify)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
