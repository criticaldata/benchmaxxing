"""Complete a lineage's CheXpert image_provenance.json from hashes already committed. No calls.

A lineage's ``image_provenance.json`` records the sha256 of every CheXpert film it read. The
GLM-5.3-Flash record hashed only the 35 films of ``nih_match_35.csv``, but the lineage also read
the three 150-study cohorts (device_absent, natural_cues, natural_independent; the system-flag arm
reads natural_independent). This script adds each missing film with the hash another committed
record gives for exactly the same film id (``CheXpert-v1.0-small/...`` path), and marks each added
entry with ``hash_source`` so it is never mistaken for a hash taken from this run's own copies.

It also compares every entry with every other committed CheXpert hash record: the other lineages'
``image_provenance.json`` and the sha256 record committed with the Gemini cohort rows on 29 Jul
2026 (``imaging_chexpert/results/natural_independent/manifest_provenance.json``, which covers the
device_absent and natural_independent cohorts). Any disagreement is an error.

Usage:
    python experiments/chexpert/provenance_by_film_id.py            # add the missing films
    python experiments/chexpert/provenance_by_film_id.py --check    # exit 1 if any film is missing
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHEX = "experiments/chexpert/results"
IMCH = "experiments/imaging_chexpert/results"
LINEAGE = "zai-org_GLM-5.3-Flash"
DONOR = "local_nvidia_cosmos-reason2-8b"
GEMINI_JULY = f"{IMCH}/natural_independent/manifest_provenance.json"
# The manifests the lineage's CheXpert arms read, by cohort.
MANIFESTS = {
    "nih_match_35": f"{IMCH}/nih_match_35.csv",
    "device_absent": f"{IMCH}/device_absent/cascade_150.csv",
    "natural_cues": f"{IMCH}/natural_cues/cascade_150.csv",
    "natural_independent": f"{IMCH}/natural_independent/cascade_150.csv",
}


def _sha(entry) -> str:
    return entry["sha256"] if isinstance(entry, dict) else entry


def records(repo: Path = REPO) -> dict[str, dict[str, str]]:
    """Every committed CheXpert hash record: name -> {film id: sha256}."""
    out = {}
    for p in sorted((repo / CHEX).glob("*/image_provenance.json")):
        out[p.parent.name] = {k: _sha(v) for k, v in json.loads(p.read_text())["images"].items()}
    out["gemini_29_jul"] = {k: _sha(v) for k, v in json.loads((repo / GEMINI_JULY).read_text()).items()}
    return out


def films_read(repo: Path = REPO) -> dict[str, list[str]]:
    return {name: [r["image_ref"] for r in csv.DictReader(open(repo / rel, newline=""))]
            for name, rel in MANIFESTS.items()}


def problems(repo: Path = REPO, lineage: str = LINEAGE) -> list[str]:
    own = json.loads((repo / CHEX / lineage / "image_provenance.json").read_text())["images"]
    recs = records(repo)
    out = []
    for cohort, films in films_read(repo).items():
        out += [f"{cohort}: {f} has no hash" for f in films if f not in own]
    for name, rec in recs.items():
        out += [f"{f}: {lineage} {_sha(own[f])[:12]} vs {name} {rec[f][:12]}"
                for f in own if f in rec and rec[f] != _sha(own[f])]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    path = REPO / CHEX / LINEAGE / "image_provenance.json"
    doc = json.loads(path.read_text())
    donor = json.loads((REPO / CHEX / DONOR / "image_provenance.json").read_text())["images"]
    if not args.check:
        added = 0
        for films in films_read().values():
            for f in films:
                if f not in doc["images"]:
                    doc["images"][f] = {**donor[f], "hash_source": f"{CHEX}/{DONOR}/image_provenance.json, "
                                        "same film id; not re-hashed from this run's copies"}
                    added += 1
        path.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"added {added} films")
    bad = problems()
    for b in bad:
        print(b)
    own = doc["images"]
    recs = records()
    for cohort, films in films_read().items():
        hashed = sum(1 for f in films if f in own and "hash_source" not in own[f])
        july = sum(1 for f in films if f in recs["gemini_29_jul"])
        print(f"{cohort}: {len(films)} films, {hashed} hashed from this run's copies, "
              f"{july} in the Gemini 29 Jul record")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
