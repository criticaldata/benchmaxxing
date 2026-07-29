"""Build a provenance-checked SUPPORT2 manifest for the tabular experiments.

Parses the raw SUPPORT2 CSV through the registered ``support2`` adapter, selects the case set the
experiments run on, and records a checksum of the source CSV plus a per-case checksum of the exact
rendered vignette, so the stimulus every number was computed on is auditable without shipping
patient rows into the repo.

By default the set is outcome-balanced: equal numbers of survivors and in-hospital deaths, taken in
file order. SUPPORT2's in-hospital mortality is roughly a quarter of the cohort, so a plain first-N
slice makes "always answer survives" look accurate; balancing keeps the clean-accuracy number
interpretable. Pass ``--no-balance`` for a straight slice, or ``--case-ids-file`` (one case_id per
line) to pin an exact, already-used case set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import support2
from benchmaxxing.datasets.base import finalize


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced(cases, n):
    """Alternate survivors and deaths in file order until ``n`` cases are taken."""
    by_outcome = {0: [c for c in cases if c.answer_index == 0],
                  1: [c for c in cases if c.answer_index == 1]}
    per_arm = n // 2
    kept = by_outcome[0][:per_arm] + by_outcome[1][:per_arm]
    if len(kept) < n:                       # one arm ran out: top up from whichever has spares
        spare = by_outcome[0][per_arm:] + by_outcome[1][per_arm:]
        kept += spare[: n - len(kept)]
    return sorted(kept, key=lambda c: c.case_id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", required=True, help="support2.csv, or a directory containing it")
    ap.add_argument("--out", required=True, help="output manifest CSV path")
    ap.add_argument("--provenance", required=True, help="output provenance JSON path (checksums)")
    ap.add_argument("--target", default="hospdead", choices=sorted(support2.TARGETS))
    ap.add_argument("--n", type=int, default=120, help="cases to keep unless --case-ids-file")
    ap.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True,
                    help="keep equal numbers of each outcome (default: balanced)")
    ap.add_argument("--case-ids-file", default=None,
                    help="optional file of case_ids (one per line) to pin exactly")
    args = ap.parse_args()

    source = support2.resolve_csv(args.raw_root)
    with tempfile.TemporaryDirectory() as tmp:
        full = support2.build_manifest(source, Path(tmp) / "full.csv", target=args.target)
        all_cases = load_cases(full)

    if args.case_ids_file:
        wanted = [line.strip() for line in Path(args.case_ids_file).read_text().splitlines()
                  if line.strip()]
        by_id = {c.case_id: c for c in all_cases}
        missing = [cid for cid in wanted if cid not in by_id]
        if missing:
            raise SystemExit(f"case_ids not present in the parsed SUPPORT2 cohort: {missing}")
        kept = [by_id[cid] for cid in wanted]
    elif args.balance:
        kept = _balanced(all_cases, args.n)
    else:
        kept = all_cases[: args.n]

    finalize(kept, Path(args.out))

    provenance = {
        "source_csv": source.name,
        "source_sha256": _sha256_file(source),
        "target": args.target,
        "n_cohort": len(all_cases),
        "n_cases": len(kept),
        "n_by_outcome": {"survives": sum(1 for c in kept if c.answer_index == 0),
                         "dies": sum(1 for c in kept if c.answer_index == 1)},
        "excluded_columns": list(support2.EXCLUDED),
        "cases": [
            {
                "case_id": c.case_id,
                "answer_index": c.answer_index,
                "label": c.label,
                "n_fields": len(c.meta.get("fields", [])),
                "n_absent": len(c.meta.get("absent", [])),
                "vignette_sha256": hashlib.sha256(c.question.encode("utf-8")).hexdigest(),
            }
            for c in kept
        ],
    }
    Path(args.provenance).write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: provenance[k] for k in
                      ("source_csv", "source_sha256", "n_cohort", "n_cases", "n_by_outcome")},
                     indent=2))


if __name__ == "__main__":
    main()
