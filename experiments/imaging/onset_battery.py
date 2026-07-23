"""Contagion-index and deference-rate, imaging lane (#174, part b). Activates
`benchmaxxing.onset.contagion_index`/`deference_rate` (the plan's flagship reusable metrics,
exported and unit-tested but never reported on committed data) on the already-committed
`imaging_cascade*.jsonl` files, one holdout per case treated as the population; zero API calls.

Each case has exactly one holdout agent (not a full committee roster), so the "population" for
both metrics is the set of 35 cases rather than a set of agents within one transcript:

    contagion_index: solo_flip = iso_adopt (did the holdout adopt the wrong read alone),
                      committee_flip = shared_adopt (did it adopt under peer pressure),
                      seed_present = True for every case (every case was exposed to the peer's
                      asserted wrong read in the shared condition).
                      Fraction of shared-adopters whose adoption cannot be explained by their own
                      solo behavior (they did NOT adopt when reading alone).

    deference_rate: solo_correct = iso == "yes" (ground truth is always "yes" on this manifest,
                     so an isolated "yes" read is correct), committee_answer = shared,
                     seeded_answer = wrong.
                     Fraction of solo-correct cases where the shared answer matches the peer's
                     wrong assertion - abandoning a correct solo read for the shortcut.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.onset import contagion_index, deference_rate

CUE_FILES = {
    "watermark": "imaging_cascade.jsonl",
    "cable": "imaging_cascade_cable.jsonl",
    "corner_tag": "imaging_cascade_corner_tag.jsonl",
    "laterality": "imaging_cascade_laterality.jsonl",
}


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _cue_metrics(rows):
    solo_flip = [bool(r["iso_adopt"]) for r in rows]
    committee_flip = [bool(r["shared_adopt"]) for r in rows]
    seed_present = [True] * len(rows)
    ci = contagion_index(solo_flip, committee_flip, seed_present)

    solo_correct = [r["iso"] == "yes" for r in rows]
    committee_answer = [r["shared"] for r in rows]
    seeded_answer = [r["wrong"] for r in rows]
    dr = deference_rate(solo_correct, committee_answer, seeded_answer)

    return {
        "n": len(rows),
        "contagion_index": round(ci, 4),
        "deference_rate": round(dr, 4),
        "n_shared_adopters": sum(committee_flip),
        "n_solo_correct": sum(solo_correct),
    }


def main():
    results_dir = Path(__file__).parent / "results"
    out = {"cues": {}}
    for cue, filename in CUE_FILES.items():
        rows = _load_jsonl(results_dir / filename)
        out["cues"][cue] = _cue_metrics(rows)

    out["read"] = (
        "contagion_index ranges 0.65-0.80 across the four cues: a majority but not the entirety "
        "of shared-condition adoptions are cases that did NOT adopt in isolation, so most "
        "adoption is attributable to the peer board rather than the cue's own solo potency, "
        "though a real minority (roughly a fifth to a third) also flipped solo and would have "
        "adopted regardless of the board. This is directionally consistent with #185's "
        "case-driven-not-cue-driven story (Spearman rho=-1.0) but is a distinct, less extreme "
        "quantity than a perfect attribution - the two should not be read as restating the same "
        "number. deference_rate is high for every cue (0.95-1.0): nearly every solo-correct case "
        "still abandons its correct read once the peer board asserts the wrong one, matching the "
        "near-total conformity already reported in #177's harm/rescue decomposition (harm rate "
        "0.95-1.0) far more closely than contagion_index does."
    )
    (results_dir / "onset_battery.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
