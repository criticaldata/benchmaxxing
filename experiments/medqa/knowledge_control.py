"""Knowledge control for MedQA deference: does a lineage adopt the planted wrong answer only on cases it
does not know?

Every MedQA arm on the full 120-case cohort whose rows carry ``bare`` (the model's own unseeded answer)
and ``ground_truth`` gives, per lineage, the cases it answers correctly alone. Adoption is pooled over
every adoption column of those arms (``*_adopt``; the control columns ``control_adopt`` and
``neither_adopt``, which plant nothing, and the ``*_declared_adopt`` rescores are left out) and read on
the cases every lineage knows, so case difficulty is held fixed across lineages.

A case is "known" when the bare answer is correct in EVERY arm (the default, ``--known every``). A
lineage whose bare answer changes between arms (gpt-oss-120b: 15 of 120 cases) would otherwise count a
case as known on one arm's reading and unknown on another's. The two looser readings are reported beside
it: ``union`` (correct in at least one arm) and ``majority`` (correct in more than half of the arms that
ran the case). Lineages whose bare answers agree across arms get the same set under all three.

Zero model calls; reads committed rows only.
Usage: python -m experiments.medqa.knowledge_control [--out experiments/medqa/results/knowledge_control.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "medqa" / "results"
# lineage label -> results subdirectory ("" for the committed Gemini rows at the lane root)
LINEAGES = {
    "gemini-2.5-flash-lite": "",
    "Qwen/Qwen2.5-VL-72B-Instruct": "Qwen_Qwen2.5-VL-72B-Instruct",
    "openai/gpt-oss-120b": "openai_gpt-oss-120b",
    "local/nvidia/cosmos-reason2-8b": "local_nvidia_cosmos-reason2-8b",
}
ARMS = ["attributed_tier", "authority_ladder", "committee_size_sweep", "deliberation_channel", "deliberation_framing",
        "dose_response", "leader_as_auditor", "paraphrase_robustness", "pre_emptive_referee", "rationale_validity",
        "seed_timing", "super_additivity", "test_awareness", "text_cue_types"]
CONTROL = ("control_adopt", "neither_adopt")


def _rows(slug: str, arm: str) -> list[dict]:
    path = (RESULTS / slug / f"{arm}.jsonl") if slug else (RESULTS / f"{arm}.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def known_sets(rows_by_arm: dict[str, list[dict]]) -> dict[str, set[str]]:
    per_arm = {arm: {r["case_id"]: r.get("bare") == r.get("ground_truth") for r in rows if "bare" in r}
               for arm, rows in rows_by_arm.items()}
    cases = set().union(*(set(a) for a in per_arm.values()))
    ran = {c: [a[c] for a in per_arm.values() if c in a] for c in cases}
    return {"every": {c for c, v in ran.items() if all(v)},
            "union": {c for c, v in ran.items() if any(v)},
            "majority": {c for c, v in ran.items() if sum(v) * 2 > len(v)}}


def pooled_adoption(rows_by_arm: dict[str, list[dict]], cases: set[str]) -> tuple[float | None, int]:
    cells = [int(bool(v)) for rows in rows_by_arm.values() for r in rows if r["case_id"] in cases
             for col, v in r.items()
             if col.endswith("_adopt") and col not in CONTROL and not col.endswith("_declared_adopt") and v is not None]
    return (sum(cells) / len(cells) if cells else None), len(cells)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(RESULTS / "knowledge_control.json"))
    args = ap.parse_args(argv)
    rows = {label: {arm: _rows(slug, arm) for arm in ARMS} for label, slug in LINEAGES.items()}
    known = {label: known_sets(r) for label, r in rows.items()}
    out: dict = {"arms": ARMS, "default_definition": "every", "definitions": {}}
    for definition in ("every", "union", "majority"):
        common = set.intersection(*(known[label][definition] for label in LINEAGES))
        per = {}
        for label in LINEAGES:
            rate, n = pooled_adoption(rows[label], common)
            per[label] = {"cases_known_alone": len(known[label][definition]), "adoption_on_cases_all_know": rate,
                          "cells": n}
        out["definitions"][definition] = {"cases_known_by_all": len(common), "per_lineage": per}
    # cases whose bare answer is correct in some arms and wrong in others
    out["bare_disagrees_across_arms"] = {label: len(known[label]["union"] - known[label]["every"]) for label in LINEAGES}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for definition, d in out["definitions"].items():
        rates = ", ".join(f"{k} {v['adoption_on_cases_all_know']:.3f}" for k, v in d["per_lineage"].items())
        print(f"{definition}: {d['cases_known_by_all']} cases known by all; {rates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
