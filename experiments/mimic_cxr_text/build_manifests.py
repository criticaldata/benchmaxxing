"""Rebuild the MIMIC-CXR text manifests this lane's committed rows are keyed on.

The manifests hold report text, so they are never committed. This script rebuilds them from a
credentialed copy of the dataset and checks the result against the known checksums, so anyone
with PhysioNet access gets the same cohort the committed results were run on.

The 633-case file is not a plain build. It is made in three steps:

1. ``build_manifest`` over the whole dataset gives the full manifest (4,901 cases).
2. That file is read back with ``load_cases``, which strips leading and trailing whitespace
   from every report. The published files were written after this step, so a manifest written
   straight from step 1 has the same cases but a different checksum.
3. The index space is the union of three draws from the full list, kept in manifest order:
   the 600 solo cases (``random.Random(0).sample(cases[:1500], 600)``), the 20 cascade boards
   (``random.Random(0).sample(cases[:777], 20)``) and the first 40 cases (blind metric and both
   referee arms).

The arms then read the prefix files: solo from the 1500 file, cascade from the 777 file, and
the prefix arms from the first 40 rows of the 633 file.

Inputs: the report tree (``files/pXX/pYYYYYYYY/sZZZZZZZZ.txt``) from MIMIC-CXR and
``mimic-cxr-2.0.0-chexpert.csv[.gz]`` from MIMIC-CXR-JPG. Versions 2.0.0 and 2.1.0 of both
carry the same reports and labels.

Usage
-----
    python -m experiments.mimic_cxr_text.build_manifests \
        --reports /path/to/mimic-cxr/files \
        --labels /path/to/mimic-cxr-2.0.0-chexpert.csv.gz \
        --out /path/outside/the/repo

Write the output outside the repository: these files must not be committed.

Platforms
---------
The same command works on Linux, macOS and Windows (in PowerShell or cmd, put it on one line).
The output bytes do not depend on the operating system: every file is read and written as
UTF-8, rows always end in CRLF as the csv module writes them, and a report tree that was
extracted with Windows line endings yields the same manifest because line endings inside
reports are normalised on read. Row order follows the label file, not the file system.
Checked against the reference files on Linux; `tests/test_mimic_cxr_text_build_manifests.py`
pins the line-ending behaviour on every platform.

If a checksum does not match
----------------------------
The script prints what it read. Compare with the reference build: 227,827 label rows, label
file sha256 starting ``efb1eac787b0b65f`` (uncompressed), 4,901 cases in the full manifest.
A lower case count means reports are missing from the tree (the builder skips a study whose
report file it cannot find); a different label checksum means a different label file.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import platform
import random
import sys
from pathlib import Path

from benchmaxxing.data import load_cases, write_manifest
from benchmaxxing.datasets.mimic_cxr_text import _resolve_labels_csv, build_manifest
from experiments.mimic_cxr_text.case_index import INDEX_SPACE_SIZE

SOLO_PREFIX, SOLO_N = 1500, 600
CASCADE_PREFIX, CASCADE_N = 777, 20
PREFIX_ARMS_N = 40

EXPECTED_LABEL_ROWS = 227_827
EXPECTED_LABELS_SHA = "efb1eac787b0b65f"  # the uncompressed CheXpert label file
EXPECTED_FULL_CASES = 4_901

# First 16 hex characters of each file's sha256.
EXPECTED = {
    "full": "9b475fc497259c1e",
    "633": "1d60ae9902d1a71c",
    "1500": "8c3f72079cb47f54",
    "777": "0898bcedc641dbb2",
}


def select_index_space(cases):
    """The cases the committed rows are keyed on, in manifest order."""
    solo = random.Random(0).sample(cases[:SOLO_PREFIX], SOLO_N)
    cascade = random.Random(0).sample(cases[:CASCADE_PREFIX], CASCADE_N)
    keep = {c.case_id for c in (*solo, *cascade, *cases[:PREFIX_ARMS_N])}
    return [c for c in cases if c.case_id in keep]


def checksum(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def describe_inputs(labels, n_cases) -> str:
    """What was read, for comparing against the reference build when a checksum differs."""
    path = _resolve_labels_csv(labels)
    raw = gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    rows = max(raw.count(b"\n") - 1, 0)
    return (
        f"platform        {platform.system()} {platform.release()}, Python {platform.python_version()}\n"
        f"label rows      {rows:,} (reference {EXPECTED_LABEL_ROWS:,})\n"
        f"label sha256    {sha} (reference {EXPECTED_LABELS_SHA})\n"
        f"full manifest   {n_cases:,} cases (reference {EXPECTED_FULL_CASES:,})"
    )


def build_all(reports, labels, out) -> dict[str, Path]:
    """Write the four manifests; raises ValueError when the full build is too small to sample."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / f"mimic_cxr_text_{name}.csv" for name in EXPECTED}
    build_manifest(reports, labels, paths["full"])
    cases = load_cases(paths["full"])  # strips report whitespace, see step 2 above
    if len(cases) < SOLO_PREFIX:
        raise ValueError(
            f"the full manifest has {len(cases):,} cases, fewer than the {SOLO_PREFIX:,} the "
            "cohort is drawn from: the report tree is incomplete or the label file is not the "
            "CheXpert one"
        )
    write_manifest(select_index_space(cases), paths["633"])
    write_manifest(cases[:SOLO_PREFIX], paths["1500"])
    write_manifest(cases[:CASCADE_PREFIX], paths["777"])
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", required=True, help="MIMIC-CXR report tree (the files/ directory)")
    parser.add_argument("--labels", required=True, help="mimic-cxr-2.0.0-chexpert.csv or .csv.gz")
    parser.add_argument("--out", required=True, help="output directory, outside the repository")
    args = parser.parse_args(argv)

    full_path = Path(args.out) / "mimic_cxr_text_full.csv"
    try:
        paths = build_all(args.reports, args.labels, args.out)
    except ValueError as error:
        print(f"Cannot build the cohort: {error}.")
        ok, paths = False, {}
    else:
        ok = True
    for name, path in paths.items():
        got = checksum(path)
        match = got == EXPECTED[name]
        ok = ok and match
        print(f"{name:>5}  {got}  {'OK' if match else 'MISMATCH, expected ' + EXPECTED[name]}")
    if paths and len(load_cases(paths["633"])) != INDEX_SPACE_SIZE:
        ok = False
        print(f"the index space does not have {INDEX_SPACE_SIZE} cases")
    if not ok:
        print("\nInputs as read (see 'If a checksum does not match' in the module docstring):")
        print(describe_inputs(args.labels, len(load_cases(full_path))))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
