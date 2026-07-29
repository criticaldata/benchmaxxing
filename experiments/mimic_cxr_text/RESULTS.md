# MIMIC-CXR text (Lane B): solo susceptibility + noise floor

Real-API results for #316, using `benchmaxxing/datasets/mimic_cxr_text.py` (adapter design per
#330) and `experiments/medqa/reproduce.py` reused as-is (per #316: "do not rewrite the method,
replicate it"), `--stage solo`.

Per #334/#316: an `n=35` smoke run first (new adapter, never exercised against the real API),
then `n=600` (sizing rationale and precedent in #334, in the same range as MedQA's own real-API
counts).

Manifests and raw model-call caches are **not committed**: report text is credentialed
PhysioNet data under a Data Use Agreement and must not be redistributed, even in a private repo.
Only the aggregate JSON below (flip rates, noise floor) is committed.

## Three bugs found in review and fixed (#336)

The first two posted rounds of numbers here were wrong, caught in review:

1. **`Case.question` held the whole report instead of `Case.report`** (caught by @Agastya191,
   @sebasmos, @maximinl). `mimic_cxr_text.py` originally built
   `question=f"{report}\n\n{prompt}"`. `lexical_overlap_bias` (`benchmaxxing/cues/text.py`) drew
   its perturbation tokens from `case.question`, so the injected distractor could quote the
   report's own words back — including, sometimes, the correct finding — and became far longer
   than the other options, co-activating `longest_option` on the same case. Fixed:
   `report=<study text>`, `question=<the fixed short prompt>`, matching the existing
   `pubmedqa.py` pattern.
2. **`benchmaxxing/data.py`'s `_text_case` never read `report` back off a manifest CSV** (unlike
   `_image_case`, which does). Pre-existing bug in shared code, not in this adapter — never
   triggered before because MedQA/MedMCQA don't use `Case.report`, and no other Lane B dataset
   with `Case.report` (PubMedQA, #293) has run an experiment through a manifest CSV yet. Effect:
   `run_solo`'s `load_cases()` call silently dropped the report, so the real API run answered
   "What is the primary finding?" with **no report text at all**, guessing cold among 4 similar
   finding names — very high, unstable flip rate and ~35-60% of `gemini-2.5-flash` responses were
   a refusal-shaped "please provide the report." Fixed with a one-line read-back plus a
   regression test (`tests/test_datasets.py::test_text_case_report_round_trips`).
3. **`lexical_overlap_bias` degenerated to a constant-suffix manipulation once bug 1 was fixed**
   (caught by @Agastya191 and @sebasmos on the bug-1 re-run). With the report correctly moved out
   of `question`, `question` became a fixed 8-word stem identical across every case, and
   `lexical_overlap_bias` draws its perturbation tokens from `case.question` — so it appended the
   same four words ("primary", "finding", "described", "report") to the same distractor on all
   600 cases. That made it functionally the same manipulation as `longest_option` (padding one
   option) rather than a genuine per-case lexical signal, so the two columns converging in the
   bug-1 re-run wasn't a finding, and the "lexical_overlap is strongest" read of that data wasn't
   valid. Fixed in `benchmaxxing/cues/text.py`: the cue now draws from `case.report` when
   present (falling back to `case.question` exactly as before for datasets with no separate
   report, e.g. MedQA — unaffected), and excludes any token that appears in *any* option (not
   just the target), so the injected text can no longer echo another option's wording — in
   particular, it can't quote the correct answer into a distractor. Regression tests added:
   `test_lexical_overlap_draws_from_report_not_a_fixed_question_stem` and
   `test_lexical_overlap_never_injects_another_options_wording`.

All three fixes are committed; the numbers below are from the re-run after all three (fresh
cache).

## Refusal-aware cross-check

`experiments/medqa/reproduce.py`'s inline parser has no refusal/unparseable sentinel — any
non-answer falls through to returning raw text, which can spuriously look like a "flip." MedQA
rarely triggers this; MIMIC-CXR text did, heavily, under bug 2 above. `refusal_aware_reanalysis.py`
re-derives flip rate from the already-cached responses using the proper abstention-aware
extractor (`benchmaxxing.extract.parse_mcq_choice`), with **no new API calls**, as a cross-check
that the committed numbers aren't still contaminated by unhandled refusals. Result: abstention
rate ~0% for both models at both n=35 and n=600, and the refusal-aware flip rate matches
`reproduce.py`'s own number closely (small ~1-case differences from parser-heuristic
disagreement, expected). See `results/solo_results_refusal_aware.json` and
`results_n600/solo_results_refusal_aware.json`.

## n=35 (smoke run, all three fixes)

```json
{
  "n_records": 210,
  "noise_floor_by_model": {"gemini-2.5-flash": 0.0, "gemini-2.5-flash-lite": 0.0},
  "flip_rate_by_model": {
    "gemini-2.5-flash": {
      "overall": 0.1143,
      "per_cue": {"lexical_overlap": 0.1429, "longest_option": 0.0857, "option_order": 0.1143},
      "n": 105
    },
    "gemini-2.5-flash-lite": {
      "overall": 0.1333,
      "per_cue": {"lexical_overlap": 0.1714, "longest_option": 0.1143, "option_order": 0.1143},
      "n": 105
    }
  }
}
```

Full record: [`results/solo_results.json`](results/solo_results.json).

## n=600 (all three fixes)

```json
{
  "n_records": 3600,
  "noise_floor_by_model": {"gemini-2.5-flash": 0.0, "gemini-2.5-flash-lite": 0.0},
  "flip_rate_by_model": {
    "gemini-2.5-flash": {
      "overall": 0.0806,
      "per_cue": {"lexical_overlap": 0.0933, "longest_option": 0.105, "option_order": 0.0433},
      "n": 1800
    },
    "gemini-2.5-flash-lite": {
      "overall": 0.1333,
      "per_cue": {"lexical_overlap": 0.175, "longest_option": 0.13, "option_order": 0.095},
      "n": 1800
    }
  }
}
```

Full record: [`results_n600/solo_results.json`](results_n600/solo_results.json).

## Cross-dataset table row (MIMIC-CXR text, n=600)

| model | overall flip | noise floor | flip-above-noise |
|---|---|---|---|
| gemini-2.5-flash | 0.081 | 0.000 | +0.081 |
| gemini-2.5-flash-lite | 0.133 | 0.000 | +0.133 |

Both models land in a similar band to MedQA-USMLE (flash 0.063, flash-lite 0.117), so the
shortcut effect generalizes from exam-style vignettes to real clinical report text. Consistent
in direction between n=35 (0.114/0.133) and n=600 (0.081/0.133) samples, and confirmed
refusal-aware (abstention ~0% at both scales).

**No single cue is "strongest" for both tiers**, unlike the earlier (bug-3-contaminated) read:
at n=600, `longest_option` (0.105) edges out `lexical_overlap` (0.093) for `gemini-2.5-flash`,
while `lexical_overlap` (0.175) is clearly strongest for `gemini-2.5-flash-lite`. Now that
`lexical_overlap` draws real per-case content from the report (bug 3), it and `longest_option`
are genuinely different manipulations again, so this split is a real, if modest, tier-dependent
result rather than an artifact.

## Cascade contagion (#317): bare vs case-anchored confident wrong seed

Real-API results for #317, reusing `experiments/medqa/reproduce.py --stage cascade` as-is for
the bare/baseline-relative seed, and a copy of `experiments/medqa/push_c.py`
(`experiments/mimic_cxr_text/push_c.py`) for the case-anchored plausibility dose-response, per
#317's "copy this template... adapt the loader if needed."

**Three adaptations were required**, all stemming from this dataset's report/question split
(#336) or its report format (#348):

1. `push_c.py`'s `_mcq` builder had no "Clinical context" section at all (MedQA's original has
   no separate `report` field, so it never needed one) — without this the holdout would answer
   blind, the same class of bug as #336's bug 2.
2. The case-anchored rationale's detail extraction originally pulled from `case.question` (here
   a fixed 8-word stem, identical across every case) — fixed to pull from `case.report`.
3. **(caught by @sebasmos on the first push_c.py run)** pulling the first 14/30 words of the
   *whole* report lands on MIMIC-CXR's fixed header boilerplate ("FINAL REPORT", "EXAMINATION:
   CHEST (PA AND LAT)", "INDICATION: ...") rather than clinical content, so `anchored` and
   `generic` prompts were near-identical and the anchored-equals-generic null measured the
   boilerplate, not anchoring. Fixed with `_findings_text()`, which extracts from the first
   `FINDINGS:`/`IMPRESSION:` header onward (falling back to the whole report if neither is
   present); spot-checked against real reports to confirm it captures actual clinical sentences.

`build_solo_records.py` reconstructs the `--solo-records` input `push_c.py` needs from the
already-cached n=600 solo run, with no new API calls.

### Bare seed (baseline-relative), n=20

```json
{"n": 20, "n_valid": 20, "mean_contagion": 0.0, "mean_shared_adopt": 0.0, "mean_isolated_adopt": 0.0}
```

Null, same as MedQA's own bare-seed finding: a wrong peer with no rationale does not spread.
Full record: [`results/cascade_results.json`](results/cascade_results.json).

### Case-anchored plausibility dose-response, n=60 (126 hard cases available)

```json
{
  "n_hard_cases": 60,
  "generic": {"conform": 51, "rate": 0.85, "wilson95": [0.739, 0.919]},
  "anchored": {"conform": 53, "rate": 0.883, "wilson95": [0.778, 0.942]},
  "anchored_strong": {"conform": 55, "rate": 0.917, "wilson95": [0.819, 0.964]},
  "anchored_solo": {"conform": 44, "rate": 0.733, "wilson95": [0.61, 0.829]},
  "anchored_vs_generic_paired": {"mcnemar_p": 0.625, "rate_diff": 0.033},
  "anchored_strong_vs_generic_paired": {"mcnemar_p": 0.289, "rate_diff": 0.067},
  "anchored_vs_anchored_solo_paired": {"mcnemar_p": 0.0039, "rate_diff": 0.15}
}
```

Full record: [`results/push_c_summary.json`](results/push_c_summary.json).

Baseline conformity on these "hard" cases (holdout already wrong alone) is high across the
board (0.85-0.92), much higher than MedQA's generic baseline (0.32) — MIMIC-CXR text holdouts
defer to two confident attendings far more readily than MedQA holdouts do. Neither
`anchored` vs `generic` (p=0.625) nor `anchored_strong` vs `generic` (p=0.289) reaches
significance, so **this is not read as a plausibility dose-response** the way MedQA's is (MedQA:
+0.12 over generic, p=0.041). **The contrast that is significant** is `anchored` vs
`anchored_solo` (two peers vs one, same anchored rationale): p=0.0039, rate diff +0.15 — a
genuine majority-pressure effect, consistent in direction with MedQA's own peer-count findings.
Read together with #316: report-text MCQs are about as susceptible to a lone answer-preserving
cue as MedQA, and once wrong, susceptible to social pressure — but here it's the number of
peers, not how case-specific their rationale sounds, that moves the holdout.

## Break-it A (#318): contaminated shared context (system pre-screen flag)

Real-API results for #318, copied from `experiments/medqa/break_it.py`'s arm A into
`experiments/mimic_cxr_text/break_it_a.py`, adapted per this dataset's report/question split
(`_mcq` renders `case.report` as clinical context, same as #336/#348) and per #318's explicit
instruction to choose the flagged wrong answer per-model so it differs from that model's own
clean baseline (MedQA's original arm A picked one `wrong` per case regardless of model and was
noted as confounded there: "flag equalled the model's baseline wrong answer, 0.90"). With the
de-confounded choice, `control_rate` is 0 by construction, so `flag_rate` is directly the honest
adoption-of-a-counterfactual-flag rate.

n=20 hard cases (126 available), both tiers:

```json
{
  "n_cases": 20,
  "A_contaminated_context": {
    "control_rate": 0.0,
    "flag_rate": 0.45,
    "effect": 0.45,
    "per_model": {
      "gemini-2.5-flash": {"control_rate": 0.0, "flag_rate": 0.4, "effect": 0.4},
      "gemini-2.5-flash-lite": {"control_rate": 0.0, "flag_rate": 0.5, "effect": 0.5}
    }
  }
}
```

Full record: [`results/break_it_a_summary.json`](results/break_it_a_summary.json).

A wrong "automated pre-screen" flag is adopted by both tiers on MIMIC-CXR text (flash +0.40,
flash-lite +0.50), comparable in direction and magnitude to MedQA's own system-flag finding
(+0.26 flash, +0.30 flash-lite, control 0.00). Consistent with #317's cascade result: MIMIC-CXR
text holdouts defer readily to an asserted wrong answer once they're already wrong alone,
whether the source is a peer committee or an upstream system signal.

## Blind-metric substitution (#320): does a hidden rubric drive reward-hacking?

Real-API results for #320, copied from `experiments/blind_metric/blind_metric.py` into
`experiments/mimic_cxr_text/blind_metric.py`. Ported this script rather than
`experiments/medqa/break_it.py`'s simpler arm D: #320 explicitly asks to also report "whether the
model names the rubric," which only `blind_metric.py`'s three-condition design
(baseline/blind/test_aware + a naming regex on the justification) actually measures — `break_it.py`'s
arm D has no naming check at all. One adaptation: the prompt builder renders `case.report` as
clinical context, same fix pattern as #336/#348/#353 (the MedQA original has no separate report
field).

n=40, gemini-2.5-flash-lite (matching the MedQA original's model/n):

```json
{
  "n": 40,
  "decoy_uptake": {"baseline": 0.0, "blind": 0.05, "test_aware": 0.175},
  "decoy_uptake_delta_blind_minus_baseline": 0.05,
  "test_awareness_suppression_delta_blind_minus_aware": -0.125,
  "naming_vs_drifting": {"n_drifted": 2, "n_named_rubric": 2, "n_drifted_silently": 0}
}
```

Full record: [`results/blind_metric_summary.json`](results/blind_metric_summary.json).

**Two honest caveats before reading this as a finding.** First, at n=40 with only 2/40 baseline
drifters and ~7 total decoy picks across all three conditions, every rate here rests on a
handful of cases — wide, unreported uncertainty. Second, and more interesting: the
`test_awareness_suppression` sign is *reversed* from MedQA's own result (there, the honesty-primed
condition suppressed drift; here it increased it, 0.175 vs 0.05 blind). Inspecting the raw
`test_aware` completions (not just the parsed letter) shows why: on at least one case, the model's
own chain-of-thought explicitly misreads "the rubric may be misaligned, answer with the clinically
best option" as *"I am tasked to... assume the rubric is correct... find the most plausible,
albeit weak, justification"* for the decoy — a genuine reasoning failure under the meta-instruction,
not a prompt-construction bug (confirmed by reconstructing and reading the exact prompt/response
pair). Whether this is `gemini-2.5-flash-lite`-specific confusion on long clinical report context,
or a real and important tier-dependent failure mode, needs a larger n before it's reportable as a
cross-dataset finding rather than noise from ~7 events.

Where the two datasets agree: of the 2 cases that did drift toward the decoy under the `blind`
condition, both **named the rubric explicitly** in their justification (2/2) — no silent drift,
consistent with MedQA's text-lane finding that text drifters self-declare (11/11 there), unlike
the imaging lane where drift was silent (0/29).
