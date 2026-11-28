"""CheXpert film hashes: every committed record agrees, and GLM-5.3-Flash's covers what it read.

Keyless and film-free: compares the committed sha256 records only
(experiments/chexpert/provenance_by_film_id.py).
"""

import csv
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "chexpert"))
import provenance_by_film_id as pv  # noqa: E402


def _refs(rel):
    return {r["image_ref"] for r in csv.DictReader(open(REPO / rel, newline=""))}


def test_every_film_hashed_in_two_records_has_one_hash():
    recs = pv.records()
    assert {"gemini_29_jul", pv.DONOR, pv.LINEAGE} <= set(recs)
    for (a, ra), (b, rb) in itertools.combinations(recs.items(), 2):
        for f in set(ra) & set(rb):
            assert ra[f] == rb[f], (f, a, b)


def test_glm_record_covers_every_chexpert_film_it_read():
    assert pv.problems() == []
    own = json.loads((REPO / pv.CHEX / pv.LINEAGE / "image_provenance.json").read_text())["images"]
    read = set().union(*map(set, pv.films_read().values()))
    assert set(own) == read and len(read) == 407
    # Only the 35 nih_match_35 films were hashed from the run's own copies; the rest say where
    # their hash comes from.
    assert {f for f, e in own.items() if "hash_source" not in e} == _refs(pv.MANIFESTS["nih_match_35"])


def test_gemini_july_record_is_the_two_independent_cohorts_and_not_the_35():
    july = set(pv.records()["gemini_29_jul"])
    assert july == _refs(pv.MANIFESTS["device_absent"]) | _refs(pv.MANIFESTS["natural_independent"])
    assert len(july) == 300
    assert not july & _refs(pv.MANIFESTS["nih_match_35"])
    assert len(july & _refs(pv.MANIFESTS["natural_cues"])) == 56
