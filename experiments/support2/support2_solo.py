"""SUPPORT2 solo shortcut susceptibility (tabular lane, Lane C).

For each SUPPORT2 patient, ask the binary prognostic question on the CLEAN record and on each
CONTAMINATED record (one answer-preserving tabular cue injected: field order, unit rescale,
precision inflation, redundant restatement, missingness recode, administrative hint). Five of the
six are information-identical, so the contaminated record states exactly the same clinical facts.

**What a flip has to beat.** A raw flip rate overstates susceptibility, because a model does not
answer identically under any perturbation whatsoever. This runner reports two separate comparators
and does not conflate them:

``whitespace_null``
    The primary control, and the one a susceptibility claim must clear. Re-renders the record
    changing one space per line: same fields, same order, same values, same units. Deterministic
    temperature 0 on both sides, exactly like the cue contrasts, so the comparison is like for like.
    ``flip_above_null`` and ``vs_null_mcnemar`` are the headline numbers.

``noise_floor``
    A secondary diagnostic answering a *different* question: how much the answer moves when the
    decoding temperature changes rather than when the bytes change. It resamples the clean read at
    temperature > 0, so it is not comparable to a temperature-0 cue contrast and must not be
    subtracted from one. Reported because it is informative about the model, labelled because the
    earlier version of this lane used it as the control and thereby understated every cue.

Reproduction: every deterministic call is cached in ``--cache`` keyed by (model, prompt). The
temperature > 0 resamples go to a separate ``--noise-log`` and are replayed from it, so a committed
noise floor reproduces instead of being redrawn. ``--refresh-noise`` draws fresh samples.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmaxxing.cues.tabular import INFORMATION_IDENTICAL, build_tabular_twin
from benchmaxxing.stats import achieved_power, mcnemar, multiple_comparison
from experiments.support2._common import (
    MODEL,
    Cache,
    api_key,
    load_manifest_cases,
    mcq_prompt,
    parse_answer,
)

ADMIN_HINT = "insurance class: private; chart flagged for utilization review"
NULL_CUE = "whitespace_null"
CUES = list(INFORMATION_IDENTICAL) + ["administrative_hint"]
ALL_CUES = CUES + [NULL_CUE]
CUE_PARAMS = {"administrative_hint": {"hint": ADMIN_HINT}}


def _rate(numerator, denominator):
    return (numerator / denominator) if denominator else None


def _cue_stats(rows, cue):
    """Flip rate for one cue over the pairs where neither side abstained."""
    seen = [r for r in rows if f"{cue}_flip" in r or f"{cue}_abstained" in r]
    valid = [r for r in rows if f"{cue}_flip" in r]
    flips = sum(r[f"{cue}_flip"] for r in valid)
    return {
        "flip_rate_excl_abstentions": _rate(flips, len(valid)),
        "n_valid_pairs": len(valid),
        "abstention_rate": _rate(len(seen) - len(valid), len(seen)),
        "n_skipped": sum(1 for r in rows if f"{cue}_skipped" in r),
    }


def main():
    ap = argparse.ArgumentParser(description="SUPPORT2 solo shortcut susceptibility.")
    ap.add_argument("--manifest", required=True, help="SUPPORT2 manifest (support2 adapter)")
    ap.add_argument("--cache", default="experiments/support2/results/call_cache.jsonl")
    ap.add_argument("--noise-log", default="experiments/support2/results/noise_resamples.jsonl")
    ap.add_argument("--out", default="experiments/support2/results")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--noise-temperature", type=float, default=1.0)
    ap.add_argument("--refresh-noise", action="store_true",
                    help="draw fresh temperature>0 samples instead of replaying the noise log")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(args.cache, api_key(), noise_path=args.noise_log,
                  refresh_noise=args.refresh_noise)
    cases = load_manifest_cases(args.manifest, args.n)

    def run_one(case):
        options = list(case.options)
        truth = options[case.answer_index]
        clean = parse_answer(cache.complete(mcq_prompt(case.question, options)), options)
        row = {"case_id": case.case_id, "clean": clean, "ground_truth": truth}
        if clean is None:
            # No usable baseline, so no pair on this record is scorable. Censor rather than guess.
            row["clean_abstained"] = 1
            return row
        row["clean_correct"] = int(clean == truth)
        for cue in ALL_CUES:
            try:
                twin = build_tabular_twin(case, cue, **CUE_PARAMS.get(cue, {}))
            except ValueError as exc:
                # A cue that cannot fire on this record (no absent field, no convertible unit) is
                # skipped for this case only; scoring it as a non-flip would dilute the rate.
                row[f"{cue}_skipped"] = str(exc)[:80]
                continue
            payload = twin.contaminated
            answer = parse_answer(
                cache.complete(mcq_prompt(payload["question"], list(payload["options"]))), options
            )
            row[cue] = answer
            if answer is None:
                row[f"{cue}_abstained"] = 1
            else:
                row[f"{cue}_flip"] = int(answer != clean)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run_one, c) for c in cases]):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["case_id"])

    scorable = [r for r in rows if not r.get("clean_abstained")]
    summary = {
        "n": len(rows),
        "model": MODEL,
        "n_clean_abstained": sum(1 for r in rows if r.get("clean_abstained")),
        "clean_accuracy_excl_abstentions": _rate(
            sum(r["clean_correct"] for r in scorable), len(scorable)
        ),
        "information_identical_cues": list(INFORMATION_IDENTICAL),
        "null_control": NULL_CUE,
        "cues": {cue: _cue_stats(scorable, cue) for cue in CUES},
        "null_control_stats": _cue_stats(scorable, NULL_CUE),
    }

    # Primary contrast: each cue against the temperature-0 null control, paired per patient over the
    # records where both sides produced a usable answer.
    null_rate = summary["null_control_stats"]["flip_rate_excl_abstentions"]
    for cue in CUES:
        rate = summary["cues"][cue]["flip_rate_excl_abstentions"]
        summary["cues"][cue]["flip_above_null"] = (
            (rate - null_rate) if (rate is not None and null_rate is not None) else None
        )
        paired = [r for r in scorable if f"{cue}_flip" in r and f"{NULL_CUE}_flip" in r]
        gain = sum(1 for r in paired if r[f"{cue}_flip"] and not r[f"{NULL_CUE}_flip"])
        lose = sum(1 for r in paired if r[f"{NULL_CUE}_flip"] and not r[f"{cue}_flip"])
        summary["cues"][cue]["vs_null_mcnemar"] = {
            "gain": gain, "lose": lose, "n_paired": len(paired),
            "pvalue": round(mcnemar(gain, lose).pvalue, 6) if (gain + lose) else None,
        }

    # Six cue-vs-null tests are one family, so a nominal 0.05 is not a result on its own. Corrected
    # here rather than left to a reader, and paired with the power each test actually had, so a null
    # is not mistaken for evidence of absence.
    ordered = list(CUES)
    pvalues = [summary["cues"][c]["vs_null_mcnemar"]["pvalue"] for c in ordered]
    if all(p is not None for p in pvalues):
        corrected = multiple_comparison(pvalues, method="bh")
        for cue, adjusted, reject in zip(ordered, corrected.pvalues_adjusted, corrected.reject):
            stats = summary["cues"][cue]["vs_null_mcnemar"]
            stats["pvalue_bh_adjusted"] = round(float(adjusted), 6)
            stats["survives_bh"] = bool(reject)
            # Power for the paired test that was actually run, so both arguments come from the same
            # discordant pairs. flip_above_null is the wrong effect to feed here: it is a difference
            # of two marginal rates over two different denominators, so it can exceed the discordant
            # proportion, and achieved_power rejects |effect| > psi with a ValueError that would take
            # the whole run down after the calls were already paid for.
            paired = stats["n_paired"]
            discordant = (stats["gain"] + stats["lose"]) / paired if paired else 0.0
            effect = (stats["gain"] - stats["lose"]) / paired if paired else 0.0
            stats["paired_effect"] = round(effect, 6)
            stats["achieved_power"] = (
                round(achieved_power(paired, discordant, effect), 4)
                if paired and discordant and effect else None
            )
        summary["family_correction"] = {
            "method": "bh",
            "alpha": 0.05,
            "n_tests": len(pvalues),
            "n_surviving": int(sum(corrected.reject)),
            "note": (
                "one family: each cue against the whitespace_null control. The administrative_hint "
                "cue is included because it is tested against the same control, even though it is "
                "the one cue that adds information rather than restating it."
            ),
        }

    # Secondary diagnostic: clean-read self-inconsistency under a temperature change. A different
    # question from the cue contrasts, so it is reported but never subtracted from them.
    if cache.key or cache.noise_store:
        def noise(case):
            options = list(case.options)
            resampled = cache.resample(mcq_prompt(case.question, options), args.noise_temperature)
            return case.case_id, parse_answer(resampled, options)

        resample_by_id, noise_error = {}, None
        try:
            with ThreadPoolExecutor(max_workers=4) as ex:
                for fut in as_completed([ex.submit(noise, c) for c in cases]):
                    case_id, answer = fut.result()
                    resample_by_id[case_id] = answer
        except SystemExit as exc:
            # A partial noise log with no key can serve some records and not others. The floor is a
            # secondary diagnostic, so it degrades to "skipped" rather than taking the run down and
            # losing the cue results, which need no key at all.
            noise_error = str(exc)
            resample_by_id = {}
        for row in rows:
            row["clean_resample"] = resample_by_id.get(row["case_id"])
            if row.get("clean") is not None and row["clean_resample"] is not None:
                row["noise_flip"] = int(row["clean_resample"] != row["clean"])

        noise_valid = [r for r in rows if "noise_flip" in r]
        summary["noise_floor_temperature_change"] = {
            "rate": _rate(sum(r["noise_flip"] for r in noise_valid), len(noise_valid)),
            "n_valid_pairs": len(noise_valid),
            "temperature": args.noise_temperature,
            # Reports what happened, not what was asked for. Keying this off --refresh-noise made a
            # fresh 120-call draw claim it had been replayed.
            "replayed_from_log": cache.noise_calls == 0,
            "note": (
                "clean read resampled at temperature>0 against a temperature-0 baseline. This "
                "measures decoding sensitivity, NOT surface-form sensitivity, so it is not "
                "comparable to the cue contrasts and must not be subtracted from them. Use "
                "null_control_stats for that."
            ),
        }
        if noise_error:
            summary["noise_floor_temperature_change"]["skipped_reason"] = noise_error[:200]
    else:
        summary["noise_floor_temperature_change"] = {
            "rate": None,
            "note": "skipped: no key and no committed noise log to replay",
        }

    summary["new_api_calls_this_run"] = cache.calls
    summary["new_noise_calls_this_run"] = cache.noise_calls
    summary["read"] = (
        "flip_rate_excl_abstentions is the fraction of patients whose prognosis changed when the "
        "record was re-rendered, over the pairs where the model gave a usable answer both times. "
        "A cue earns a susceptibility claim only if it clears the whitespace_null control, which "
        "changes bytes but nothing semantic: that is flip_above_null and vs_null_mcnemar. "
        "achieved_power is the power of that paired McNemar and is computed against paired_effect, "
        "(gain - lose) / n_paired, not against flip_above_null: the latter differences two marginal "
        "rates over two different denominators and is not the quantity the paired test estimates."
    )
    (out / "support2_solo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "support2_solo_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
