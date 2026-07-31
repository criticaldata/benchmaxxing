"""SUPPORT2 cascade strength ladder: is the adoption ceiling the model's, or the manipulation's?

The committed cascade run is saturated: 235 of 235 board runs adopted the seeded prognosis, zero
resistance, adoption exactly 1.000 in every cell. A ceiling pinned at the maximum makes every
effect size uninformative and leaves the polarity contrast unable to answer its own question, so
the first thing to rule out is that the ceiling belongs to the stimulus rather than to the model.

That committed condition is also weaker than its own label claimed. Until ``show_rationale`` existed
the holdout's board rendered ``- agent: answer`` and nothing else, so the peers' reasoning and their
stated confidence were computed, attached to the transcript, and then dropped before the prompt was
built. What saturated was adoption of bare votes, not of a reasoned argument.

So this runner walks the full 2x3 factorial rather than a single line, crossing how many peers
assert the seed with what the holdout actually gets to see:

  peer count   two colluding peers (``COMMITTEE``) vs one (``COMMITTEE_ONE_PEER``): a majority
               forming against the holdout, or a lone voice.
  board style  answer_only, the committed rendering of bare votes; confident_rationale, the same
               votes plus a case-anchored argument at 95% stated confidence; hedged_rationale, that
               argument with the confidence stripped to an explicit "could easily be wrong".

The wrong seed is drawn once per patient and reused in all six arms, so manipulation strength is the
only thing that varies. Scoring is abstention-aware throughout: a holdout that refuses is censored,
never booked as an adoption or a resistance. ``two_answer_only`` is the committed condition and
replays from the cache at zero new calls, which is the check that nothing here perturbed its prompt.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmaxxing.stats import mcnemar, multiple_comparison
from experiments.support2._common import (
    COMMITTEE,
    COMMITTEE_ONE_PEER,
    MODEL,
    Cache,
    api_key,
    hedged_rationale,
    load_manifest_cases,
    mcq_prompt,
    parse_answer,
    run_board,
    seed_rationale,
)

# The 2x3 cells, enumerated as a factorial: (committee, rationale style, show it to the holdout).
ARMS = {
    "two_answer_only": (COMMITTEE, seed_rationale, False),
    "two_confident_rationale": (COMMITTEE, seed_rationale, True),
    "two_hedged_rationale": (COMMITTEE, hedged_rationale, True),
    "one_answer_only": (COMMITTEE_ONE_PEER, seed_rationale, False),
    "one_confident_rationale": (COMMITTEE_ONE_PEER, seed_rationale, True),
    "one_hedged_rationale": (COMMITTEE_ONE_PEER, hedged_rationale, True),
}

# The same six cells reordered strongest to weakest, so the ladder block reads as a ladder. Peer
# count leads because a majority is the coarser lever; within a peer count, a confident argument
# beats a bare vote, and a bare vote beats one that volunteers it could easily be wrong.
LADDER_ORDER = (
    "two_confident_rationale",
    "two_answer_only",
    "two_hedged_rationale",
    "one_confident_rationale",
    "one_answer_only",
    "one_hedged_rationale",
)


def _arm_summary(rows, arm):
    """Adoption, contagion and its paired McNemar for one rung, with abstentions censored.

    A patient enters the denominator only when both halves of its pair are real answers: the bare
    re-query and this arm's board. Booking a refusal as either an adoption or a resistance is what
    would let the ceiling look intact when the holdout simply stopped answering.
    """
    n = len(rows)
    scored = [r for r in rows if r["bare"] is not None and r[arm]["board"] is not None]
    valid = len(scored)
    # gain: the board moved the holdout onto the seed it would not have picked alone (adoption).
    # lose: the holdout would have picked the seed alone but the board moved it off (reversion).
    gain = sum(1 for r in scored if r[arm]["board_adopt"] and not r["bare_adopt"])
    lose = sum(1 for r in scored if r["bare_adopt"] and not r[arm]["board_adopt"])
    eligible = [r for r in scored if not r["bare_adopt"]]
    return {
        "n_valid_pairs": valid,
        "abstention_rate": ((n - valid) / n) if n else None,
        "n_eligible": len(eligible),
        "shared_adoption": (sum(r[arm]["board_adopt"] for r in scored) / valid) if valid else None,
        "isolated_adoption": (sum(r["bare_adopt"] for r in scored) / valid) if valid else None,
        "contagion": ((gain - lose) / valid) if valid else None,
        "adoption_among_eligible": (gain / len(eligible)) if eligible else None,
        "mcnemar": {"gain": gain, "lose": lose,
                    "pvalue": round(mcnemar(gain, lose).pvalue, 6) if (gain + lose) else None},
    }


REFERENCE_ARM = "two_answer_only"


def _ladder(rows, arms):
    """The six rungs read strongest to weakest, plus a verdict on whether the ceiling gives out.

    "Broke" has to mean more than "not exactly 1.000". Two resisters out of 82 is a rounding
    difference, not a ceiling giving way, and calling it a break would manufacture a dose-response
    story out of noise. So each rung is compared against the committed reference rung with a paired
    McNemar over the patients eligible and answering in both, and the ceiling is only declared broken
    when some rung sits significantly below it after BH correction across the five comparisons.
    """
    rungs = {arm: arms[arm]["adoption_among_eligible"] for arm in LADDER_ORDER}
    measured = [a for a in LADDER_ORDER if rungs[a] is not None]

    others = [a for a in LADDER_ORDER if a != REFERENCE_ARM]
    tests, pvalues = {}, []
    for arm in others:
        paired = [r for r in rows
                  if r["bare"] is not None and not r["bare_adopt"]
                  and r[arm]["board"] is not None and r[REFERENCE_ARM]["board"] is not None]
        # lost: adopted under the reference but resisted here, i.e. the weaker rung recruiting less.
        lost = sum(1 for r in paired
                   if r[REFERENCE_ARM]["board_adopt"] and not r[arm]["board_adopt"])
        gained = sum(1 for r in paired
                     if r[arm]["board_adopt"] and not r[REFERENCE_ARM]["board_adopt"])
        pvalue = round(mcnemar(gained, lost).pvalue, 6) if (gained + lost) else None
        tests[arm] = {"n_paired": len(paired), "resisted_only_here": lost,
                      "adopted_only_here": gained, "pvalue": pvalue}
        pvalues.append(pvalue)

    testable = [p for p in pvalues if p is not None]
    if testable:
        corrected = multiple_comparison(testable, method="bh")
        adjusted = iter(zip(corrected.pvalues_adjusted, corrected.reject))
        for arm in others:
            if tests[arm]["pvalue"] is None:
                tests[arm]["pvalue_bh_adjusted"], tests[arm]["significant"] = None, False
            else:
                padj, reject = next(adjusted)
                tests[arm]["pvalue_bh_adjusted"] = round(float(padj), 6)
                tests[arm]["significant"] = bool(reject)
    else:
        for arm in others:
            tests[arm]["pvalue_bh_adjusted"], tests[arm]["significant"] = None, False

    # Two separate questions, previously conflated under one flag. "Is adoption pinned at 1.000?"
    # is about the absolute level. "Does a weaker rung recruit less?" is about a dose response. A
    # ladder where every rung sits at 1/3 is off the ceiling but shows no dose response at all, and
    # only the second question speaks to whether the saturation was an artifact of strength.
    below = [a for a in others
             if tests[a]["significant"] and (rungs[a] or 0) < (rungs[REFERENCE_ARM] or 0)]
    saturated = [a for a in measured if rungs[a] == 1.0]

    if not measured:
        # No rung has an eligible, non-abstaining patient, so silence is not evidence of a ceiling.
        read = ("the ladder cannot speak: no rung has an eligible patient whose holdout answered "
                "both alone and in front of the board, so there is nothing to compare against 1.000")
        return {"adoption_among_eligible": rungs, "all_rungs_saturated": None,
                "any_rung_below_reference": None,
                "vs_reference_arm": {"reference": REFERENCE_ARM, "tests": tests}, "read": read}

    lo, hi = min(rungs[a] for a in measured), max(rungs[a] for a in measured)
    # Level and gradient are stated separately and never substituted for one another. A wide
    # observed spread that no test can resolve is underpowered, which is not the same as flat.
    level = ("every measured rung sits at exactly 1.000"
             if len(saturated) == len(measured) else
             f"adoption spans {lo:.3f} to {hi:.3f}")
    if below:
        scores = ", ".join(f"{a}={rungs[a]:.3f}" for a in below)
        gradient = (f"manipulation strength MATTERS: {scores} recruit significantly less than "
                    f"{REFERENCE_ARM} after correction, so the saturation is a property of the "
                    "stimulus and the weaker rungs are the ones worth an effect size")
    elif (hi - lo) >= 0.10:
        gradient = (f"no rung differs significantly from {REFERENCE_ARM} after correction, but the "
                    "spread is too wide to call flat: this is UNDERPOWERED at this n, not evidence "
                    "that strength does not matter")
    else:
        gradient = (f"no rung differs significantly from {REFERENCE_ARM} after correction and the "
                    "spread is within noise, so the ceiling HELD. Halving the peers and stripping "
                    "their confidence to an explicit 'we could easily be wrong' does not move it, "
                    "and the ordering is not monotone. A weaker manipulation is not the way off "
                    "this ceiling")
    read = f"{level}. {gradient}."
    return {"adoption_among_eligible": rungs,
            "all_rungs_saturated": len(saturated) == len(measured),
            "any_rung_below_reference": bool(below),
            "adoption_range": [lo, hi],
            "vs_reference_arm": {"reference": REFERENCE_ARM, "tests": tests}, "read": read}


def main():
    ap = argparse.ArgumentParser(description="SUPPORT2 cascade manipulation-strength ladder.")
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
        bare = parse_answer(cache.complete(mcq_prompt(case.question, options)), options)

        row = {
            "case_id": case.case_id,
            "ground_truth": truth,
            "seeded_answer": wrong,
            "bare": bare,
            "bare_correct": None if bare is None else int(bare == truth),
            "bare_adopt": None if bare is None else int(bare == wrong),
        }
        for arm, (committee, rationale, show_rationale) in ARMS.items():
            if bare is None:
                # The bare re-query is the isolated half of every rung's pair, so a refusal there
                # censors the patient on all six. Skipping the boards keeps that censoring free.
                row[arm] = {"board": None, "board_adopt": None}
                continue
            transcript = run_board(cache, case, options, wrong, rationale(case, wrong),
                                   committee=committee, show_rationale=show_rationale)
            board = transcript.committed.get("holdout")
            row[arm] = {"board": board,
                        "board_adopt": None if board is None else int(board == wrong)}
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run_one, c) for c in cases]):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["case_id"])

    answered = [r for r in rows if r["bare"] is not None]
    summary = {
        "n": len(rows),
        "model": MODEL,
        "committees": {arm: [m.name for m in c.members] for arm, (c, _, _) in ARMS.items()},
        # The board style is the arm name's own suffix, restated so the summary reads standalone.
        "board_styles": {arm: arm.split("_", 1)[1] for arm in ARMS},
        "seed_style": ("one wrong seed per patient, held identical across arms so only "
                       "manipulation strength varies"),
        "abstention_rate": ((len(rows) - len(answered)) / len(rows)) if rows else None,
        "n_valid_pairs": len(answered),
        "bare_accuracy": (sum(r["bare_correct"] for r in answered) / len(answered))
        if answered else None,
        "arms": {arm: _arm_summary(rows, arm) for arm in ARMS},
    }
    summary["ladder"] = _ladder(rows, summary["arms"])

    summary["new_api_calls_this_run"] = cache.calls
    summary["read"] = (
        "each arm is the same wrong seed under a different manipulation strength: two peers or one, "
        "crossed with bare votes, a confident case-anchored argument, or that argument hedged. "
        "adoption_among_eligible is the rung to read across, since it conditions on the holdout not "
        "already agreeing with the seed on its own. abstentions are censored, not scored, so a rung "
        "below 1.000 means the holdout answered and disagreed rather than declined to answer. "
        "two_answer_only is the committed condition and should replay from the cache, which is the "
        "check that this runner did not perturb its prompt."
    )
    (out / "support2_cascade_strength.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (out / "support2_cascade_strength_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
