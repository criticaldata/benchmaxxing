"""SUPPORT2 cascade: does a confident wrong seed spread to a holdout? (tabular lane, Lane C).

Two colluding senior peers assert one binary prognosis with a rationale anchored on this patient's
own record ("with a mean arterial pressure of 78 mmHg, ..."), and a real holdout answers with that
board in front of it. The isolated counterfactual is the same holdout answering the same record
with no board at all. Contagion is shared-minus-isolated adoption of the seeded answer, paired per
patient and tested with McNemar.

A binary outcome makes the seed choice matter, so both arms are run:

  wrong_seed  the peers assert the option that is NOT the ground truth. This is the headline
              confident-wrong-seed arm, but it can only recruit a holdout that was right on its
              own, so its eligible population is exactly the cases the holdout got right bare.
  flip_seed   the peers assert whatever the holdout did NOT say bare, so every case is eligible by
              construction. This measures raw persuadability and splits by whether the seed
              happened to be the wrong or the right answer, which is the polarity contrast.

Both arms share the holdout's single bare re-query per patient, so the two arms cost one extra
board call each. Every call is cached in ``--cache`` keyed by (model, prompt): a fully cached run
reproduces the committed summary with zero API calls and no key.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmaxxing.stats import mcnemar
from experiments.support2._common import (
    COMMITTEE,
    MODEL,
    Cache,
    api_key,
    load_manifest_cases,
    mcq_prompt,
    parse_choice,
    run_board,
    seed_rationale,
)

ARMS = ("wrong_seed", "flip_seed")


def _arm_summary(rows, arm):
    """Adoption, reversion, contagion and its paired McNemar for one arm."""
    scored = [r for r in rows if r[arm]["seeded_answer"] is not None]
    n = len(scored)
    # gain: the board moved the holdout onto the seed it would not have picked alone (adoption).
    # lose: the holdout would have picked the seed alone but the board moved it off (reversion).
    gain = sum(1 for r in scored if r[arm]["board_adopt"] and not r[arm]["bare_adopt"])
    lose = sum(1 for r in scored if r[arm]["bare_adopt"] and not r[arm]["board_adopt"])
    eligible = [r for r in scored if not r[arm]["bare_adopt"]]
    return {
        "n": n,
        "n_eligible": len(eligible),
        "shared_adoption": (sum(r[arm]["board_adopt"] for r in scored) / n) if n else None,
        "isolated_adoption": (sum(r[arm]["bare_adopt"] for r in scored) / n) if n else None,
        "contagion": ((gain - lose) / n) if n else None,
        "adoption_among_eligible": (gain / len(eligible)) if eligible else None,
        "mcnemar": {"gain": gain, "lose": lose,
                    "pvalue": round(mcnemar(gain, lose).pvalue, 6) if (gain + lose) else None},
    }


def main():
    ap = argparse.ArgumentParser(description="SUPPORT2 confident-wrong-seed cascade contagion.")
    ap.add_argument("--manifest", required=True, help="SUPPORT2 manifest (support2 adapter)")
    ap.add_argument("--cache", default="experiments/support2/results/call_cache.jsonl")
    ap.add_argument("--out", default="experiments/support2/results")
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(args.cache, api_key())
    cases = load_manifest_cases(args.manifest, args.n)

    def run_one(case):
        options = list(case.options)
        truth = options[case.answer_index]
        wrong = next(o for o in options if o != truth)
        bare = parse_choice(cache.complete(mcq_prompt(case.question, options)), options)

        row = {"case_id": case.case_id, "ground_truth": truth, "bare": bare,
               "bare_correct": int(bare == truth)}
        seeds = {
            "wrong_seed": wrong,
            # The option the holdout did not pick bare. An unparseable bare answer leaves no
            # defined complement, so the flip arm is censored rather than guessed at.
            "flip_seed": next((o for o in options if o != bare), None) if bare in options else None,
        }
        for arm in ARMS:
            seed = seeds[arm]
            if seed is None:
                row[arm] = {"seeded_answer": None, "board": None,
                            "board_adopt": 0, "bare_adopt": 0, "seed_is_wrong": None}
                continue
            transcript = run_board(cache, case, options, seed, seed_rationale(case, seed))
            board = transcript.committed.get("holdout")
            row[arm] = {
                "seeded_answer": seed,
                "board": board,
                "board_adopt": int(board == seed),
                "bare_adopt": int(bare == seed),
                "seed_is_wrong": int(seed != truth),
            }
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run_one, c) for c in cases]):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["case_id"])

    summary = {
        "n": len(rows),
        "model": MODEL,
        "committee": [m.name for m in COMMITTEE.members],
        # run_board computes a case-anchored rationale and attaches it to the transcript, but the
        # holdout's prompt renders only "- agent: answer". The rationale never reaches the model, so
        # this arm measures adoption of a bare peer vote. #369 fixes the renderer.
        "seed_style": "answer_only (peers' committed answers; the computed rationale is not rendered "
                      "into the holdout's prompt)",
        "bare_accuracy": (sum(r["bare_correct"] for r in rows) / len(rows)) if rows else None,
        "arms": {arm: _arm_summary(rows, arm) for arm in ARMS},
    }

    # Polarity: within the flip arm, does a confidently WRONG seed recruit as well as a confidently
    # RIGHT one? Same stimulus, opposite direction, so a gap is asymmetry rather than persuasion.
    flip = [r for r in rows if r["flip_seed"]["seeded_answer"] is not None]
    polarity = {}
    for label, is_wrong in (("seed_wrong", 1), ("seed_right", 0)):
        subset = [r for r in flip if r["flip_seed"]["seed_is_wrong"] == is_wrong]
        polarity[label] = {
            "n": len(subset),
            "adoption": (sum(r["flip_seed"]["board_adopt"] for r in subset) / len(subset))
            if subset else None,
        }
    summary["flip_seed_polarity"] = polarity

    summary["new_api_calls_this_run"] = cache.calls
    summary["read"] = (
        "contagion is shared-minus-isolated adoption of the seeded prognosis, paired per patient: "
        "gain counts holdouts the board moved ONTO a seed they would not have picked alone, lose "
        "counts holdouts it moved OFF one they would have. wrong_seed is the headline arm but is "
        "only eligible to recruit where the holdout was right on its own; flip_seed is eligible "
        "everywhere by construction and its polarity split says whether a peer pair asserting the "
        "wrong prognosis recruits as well as one asserting the right one. See seed_style: the board "
        "shows bare answers, so this is bare-vote adoption and the peers' stated confidence never "
        "reached the prompt."
    )
    (out / "support2_cascade.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "support2_cascade_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
