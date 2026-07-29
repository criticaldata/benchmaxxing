"""SUPPORT2 solo shortcut susceptibility + noise floor (tabular lane, Lane C).

For each SUPPORT2 patient, ask the binary prognostic question on the CLEAN record and on each
CONTAMINATED record (one answer-preserving tabular cue injected: field order, unit rescale,
precision inflation, redundant restatement, missingness recode, administrative hint). Five of the
six cues are information-identical, so the contaminated record states exactly the same clinical
facts; a change in the prognosis is shortcut evidence, not evidence updating.

A raw flip rate overstates susceptibility, because a model is not perfectly self-consistent even on
an unchanged input. The noise floor is computed in-script: each clean read is resampled once at
temperature > 0 with the cache bypassed, so the floor is clean-read self-inconsistency. The honest
susceptibility is flip-above-noise (per-cue flip rate minus the floor), tested per cue with a
paired McNemar against that resample.

Reproduction: every deterministic call is cached in ``--cache`` keyed by (model, prompt), so the
flip pass reproduces the committed summary with zero API calls and no key. The noise floor is the
one uncached step; it needs ``GEMINI_API_KEY`` and is recorded as None (skipped) without one.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmaxxing.cues.tabular import INFORMATION_IDENTICAL, build_tabular_twin
from benchmaxxing.stats import mcnemar
from experiments.support2._common import (
    MODEL,
    Cache,
    api_key,
    load_manifest_cases,
    mcq_prompt,
    parse_choice,
)

ADMIN_HINT = "insurance class: private; chart flagged for utilization review"
CUES = list(INFORMATION_IDENTICAL) + ["administrative_hint"]
CUE_PARAMS = {"administrative_hint": {"hint": ADMIN_HINT}}


def main():
    ap = argparse.ArgumentParser(description="SUPPORT2 solo shortcut susceptibility + noise floor.")
    ap.add_argument("--manifest", required=True, help="SUPPORT2 manifest (support2 adapter)")
    ap.add_argument("--cache", default="experiments/support2/results/call_cache.jsonl")
    ap.add_argument("--out", default="experiments/support2/results")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--noise-temperature", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(args.cache, api_key())
    cases = load_manifest_cases(args.manifest, args.n)

    def run_one(case):
        options = list(case.options)
        truth = options[case.answer_index]
        clean = parse_choice(cache.complete(mcq_prompt(case.question, options)), options)
        row = {"case_id": case.case_id, "clean": clean, "ground_truth": truth,
               "clean_correct": int(clean == truth)}
        for cue in CUES:
            try:
                twin = build_tabular_twin(case, cue, **CUE_PARAMS.get(cue, {}))
            except ValueError as exc:
                # A cue that cannot fire on this record (no absent field, no convertible unit) is
                # skipped for this case only; scoring it as a non-flip would dilute the rate.
                row[f"{cue}_skipped"] = str(exc)[:80]
                continue
            payload = twin.contaminated
            answer = parse_choice(
                cache.complete(mcq_prompt(payload["question"], list(payload["options"]))), options
            )
            row[cue] = answer
            row[f"{cue}_flip"] = int(answer != clean)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run_one, c) for c in cases]):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["case_id"])

    summary = {
        "n": len(rows),
        "model": MODEL,
        "clean_accuracy": (sum(r["clean_correct"] for r in rows) / len(rows)) if rows else None,
        "information_identical_cues": list(INFORMATION_IDENTICAL),
        "cues": {},
    }
    for cue in CUES:
        flips = [r[f"{cue}_flip"] for r in rows if f"{cue}_flip" in r]
        summary["cues"][cue] = {
            "flip_rate": (sum(flips) / len(flips)) if flips else None,
            "n": len(flips),
            "n_skipped": sum(1 for r in rows if f"{cue}_skipped" in r),
        }

    # Noise floor: resample each CLEAN read once at temperature > 0 with the cache bypassed, so the
    # floor measures self-inconsistency rather than any cue effect. Skipped without a key so the
    # deterministic flip pass above still reproduces.
    if cache.key:
        def noise(case):
            options = list(case.options)
            prompt = mcq_prompt(case.question, options)
            resampled = cache.complete_uncached(prompt, args.noise_temperature)
            return case.case_id, parse_choice(resampled, options)

        resample_by_id = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            for fut in as_completed([ex.submit(noise, c) for c in cases]):
                case_id, answer = fut.result()
                resample_by_id[case_id] = answer
        for row in rows:
            row["clean_resample"] = resample_by_id.get(row["case_id"])
            row["noise_flip"] = int(row["clean_resample"] != row["clean"])

        noise_flips = [r["noise_flip"] for r in rows]
        noise_floor = (sum(noise_flips) / len(noise_flips)) if noise_flips else None
        summary["noise_floor"] = noise_floor
        summary["noise_floor_n"] = len(noise_flips)
        summary["noise_temperature"] = args.noise_temperature
        for cue in CUES:
            paired = [r for r in rows if f"{cue}_flip" in r]
            rate = summary["cues"][cue]["flip_rate"]
            summary["cues"][cue]["flip_above_noise"] = (
                (rate - noise_floor) if (rate is not None and noise_floor is not None) else None
            )
            gain = sum(1 for r in paired if r[f"{cue}_flip"] and not r["noise_flip"])
            lose = sum(1 for r in paired if r["noise_flip"] and not r[f"{cue}_flip"])
            summary["cues"][cue]["vs_noise_mcnemar"] = {
                "gain": gain, "lose": lose,
                "pvalue": round(mcnemar(gain, lose).pvalue, 6) if (gain + lose) else None,
            }
    else:
        summary["noise_floor"] = None
        summary["noise_floor_note"] = (
            "skipped: no key (the noise floor is an uncached temperature>0 resample)"
        )

    summary["new_api_calls_this_run"] = cache.calls
    summary["read"] = (
        "Per-cue flip rate is the fraction of patients whose prognosis changed when the record was "
        "re-rendered without changing what it says. The noise floor is the same model disagreeing "
        "with itself on an unchanged record at temperature>0; flip_above_noise subtracts it and "
        "vs_noise_mcnemar tests the same contrast pairwise. Only information_identical_cues "
        "support the strong claim (identical facts, different surface form); administrative_hint "
        "adds a line to the record and is the weaker comparator."
    )
    (out / "support2_solo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "support2_solo_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
