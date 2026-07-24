"""Build a provenance-checked CheXpert manifest for the imaging experiments.

Parses the CheXpert release via the registered ``chexpert`` adapter, keeps only cases
whose image actually resolves on disk and that carry a real finding label, records a sha256 checksum per kept image so the
exact pixels used are auditable, and writes a manifest consumable by ``benchmaxxing.data.load_cases``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import chexpert
from benchmaxxing.datasets.base import finalize


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", required=True, help="CheXpert train.csv/valid.csv, or a directory containing it")
    ap.add_argument("--image-root", required=True, help="directory holding the downloaded images/")
    ap.add_argument("--out", required=True, help="output manifest CSV path")
    ap.add_argument("--provenance", required=True, help="output provenance JSON path (checksums)")
    ap.add_argument("--n", type=int, default=35, help="cases to keep if --case-ids-file is not given")
    ap.add_argument("--case-ids-file", default=None, help="optional file of case_ids (one per line) to pin exactly")
    args = ap.parse_args()

    image_root = Path(args.image_root)
    with tempfile.TemporaryDirectory() as tmp:
        full_manifest = chexpert.build_manifest(args.raw_root, Path(tmp) / "full_manifest.csv")
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
    else:
        kept = eligible[: args.n]

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
