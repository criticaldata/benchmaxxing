"""Rebuild data/medmcqa_dev.csv from the public MedMCQA validation split, and check it.

``data/`` is gitignored, so every MedMCQA arm needs this manifest staged locally before it replays.
The public copy on the Hugging Face hub (openlifescienceai/medmcqa) is a Parquet file, while
``benchmaxxing datasets stage medmcqa`` reads the original release's ``dev.json`` (one JSON object
per line). This script downloads the Parquet split, writes it out row for row as that ``dev.json``,
stages the 500-row manifest the README specifies, and checks its sha256 against the manifest every
committed MedMCQA cache was built on. A mismatch exits non-zero rather than leaving a manifest that
would silently pick different cases.

    python experiments/medmcqa/fetch_dev.py            # writes data/medmcqa_dev.csv
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ("https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/main/"
           "data/validation-00000-of-00001.parquet")
# sha256 of the 500-row manifest the committed MedMCQA caches replay against
EXPECTED = "5b2ce9718e7cf421d88dd516d9c52b8b282cd886d13cbbc61ef5d7463a90a30d"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data/medmcqa_dev.csv"))
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    import pandas as pd  # parquet reading needs pandas with pyarrow

    raw = urllib.request.urlopen(PARQUET, timeout=120).read()
    df = pd.read_parquet(io.BytesIO(raw))
    with tempfile.TemporaryDirectory() as td:
        dev = Path(td) / "dev.json"
        with dev.open("w") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps({k: (None if v != v else v) if isinstance(v, float) else v
                                    for k, v in rec.items()}) + "\n")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "benchmaxxing.cli", "datasets", "stage", "medmcqa",
                        "--raw-root", td, "--out", str(out), "--limit", str(args.limit)],
                       check=True, cwd=ROOT)
    got = sha256(out)
    if args.limit == 500 and got != EXPECTED:
        sys.exit(f"{out}: sha256 {got} does not match the committed caches' manifest {EXPECTED}")
    print(f"{out}: {args.limit} rows, sha256 {got}" + (" (matches)" if got == EXPECTED else ""))


if __name__ == "__main__":
    main()
