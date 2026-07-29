"""Build a provenance-checked manifest for the imaging experiments (#162, #331).

Parses the dataset release via the registered adapter, keeps only cases
whose image actually resolves on disk (only a subset of the dataset might be downloaded)
and that carry a real finding label, records a sha256 checksum per kept image so the
exact pixels used are auditable, and writes a manifest consumable by ``benchmaxxing.data.load_cases``.

By default the first ``--n`` eligible cases (in release order) are kept. Pass ``--case-ids-file``
(one case_id per line) to instead pin an exact, already-used case set - this is how the committed
``results/nih_manifest.csv`` for the solo/cascade/referee experiments was produced, so a fresh
manifest build reproduces the identical 35-case set those results were computed on.

Pass ``--stratify`` for round-robin selection over primary findings, seeded by ``--seed``. Useful for
building a NEW cohort with better finding coverage, but deliberately not the default: as a default it
silently changes which cases a rebuild selects, which breaks the audit trail for published numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import registry
from benchmaxxing.datasets.base import finalize


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", required=True, help="Data CSV, or a directory containing it")
    ap.add_argument("--image-root", required=True, help="directory holding the downloaded images/")
    ap.add_argument("--out", required=True, help="output manifest CSV path")
    ap.add_argument("--provenance", required=True, help="output provenance JSON path (checksums)")
    ap.add_argument("--dataset", default="nih_cxr14", help="dataset adapter to use (default: nih_cxr14)")
    ap.add_argument("--n", type=int, default=35, help="cases to keep if --case-ids-file is not given")
    ap.add_argument("--stratify", action="store_true",
                    help="round-robin over primary findings instead of release order. Opt-in on "
                         "purpose: it changes which cases a rebuild selects.")
    ap.add_argument("--seed", type=int, default=42, help="random seed, only used with --stratify")
    ap.add_argument("--case-ids-file", default=None, help="optional file of case_ids (one per line) to pin exactly")
    args = ap.parse_args()

    image_root = Path(args.image_root)
    dataset_module = registry.get(args.dataset)

    with tempfile.TemporaryDirectory() as tmp:
        kwargs = {}
        if args.stratify and "label_format" in inspect.signature(
                dataset_module.build_manifest).parameters:
            kwargs["label_format"] = "stratified"
        full_manifest = dataset_module.build_manifest(args.raw_root, Path(tmp) / "full_manifest.csv", **kwargs)
        all_cases = load_cases(full_manifest)

    eligible = [
        c for c in all_cases
        if c.label and c.label.lower() not in ("no finding", "", "none")
        and c.image_ref and (image_root / c.image_ref).exists()
    ]

    if args.case_ids_file:
        wanted = [line.strip() for line in Path(args.case_ids_file).read_text().splitlines() if line.strip()]
        by_id = {c.case_id: c for c in eligible}
        missing = [cid for cid in wanted if cid not in by_id]
        if missing:
            raise SystemExit(f"case_ids not eligible (no image on disk, or no real finding label): {missing}")
        kept = [by_id[cid] for cid in wanted]
    elif not args.stratify:
        # Release order, the default. This is what the committed manifests were built with, so a
        # rebuild reproduces the exact cohort every published number was computed on.
        kept = eligible[: args.n]
    else:
        # Stratify by primary finding using round-robin
        by_finding = defaultdict(list)
        for c in eligible:
            primary_finding = c.label.split("|")[0].strip().lower()
            by_finding[primary_finding].append(c)

        rng = random.Random(args.seed)
        for group in by_finding.values():
            rng.shuffle(group)
        
        kept = []
        findings = list(by_finding.keys())
        rng.shuffle(findings)

        while len(kept) < args.n and any(by_finding.values()):
            for f in findings:
                if len(kept) >= args.n:
                    break
                if by_finding[f]:
                    kept.append(by_finding[f].pop(0))

    finalize(kept, Path(args.out))

    provenance = {
        "source_csv": Path(args.raw_root).name,
        "n_cases": len(kept),
        "cases": [
            {
                "case_id": c.case_id,
                "image_ref": c.image_ref,
                "label": c.label,
                "sha256": _sha256(image_root / c.image_ref),
                "bytes": (image_root / c.image_ref).stat().st_size,
            }
            for c in kept
        ],
    }
    Path(args.provenance).write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"n_cases": len(kept), "manifest": str(args.out), "provenance": str(args.provenance)}, indent=2))


if __name__ == "__main__":
    main()
