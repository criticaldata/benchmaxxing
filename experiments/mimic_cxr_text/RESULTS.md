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

## Two bugs found in review and fixed (#336)

The first posted numbers here were wrong, caught by @Agastya191, @sebasmos, and @maximinl in
review:

1. **`Case.question` held the whole report instead of `Case.report`.** `mimic_cxr_text.py`
   originally built `question=f"{report}\n\n{prompt}"`. `lexical_overlap_bias`
   (`benchmaxxing/cues/text.py`) draws its perturbation tokens from `case.question`, so the
   injected distractor could quote the report's own words back — including, sometimes, the
   correct finding — and became far longer than the other options, co-activating
   `longest_option` on the same case. Fixed: `report=<study text>`, `question=<the fixed short
   prompt>`, matching the existing `pubmedqa.py` pattern. `experiments/medqa/reproduce.py`'s
   `_mcq_prompt` already renders `report` as "Clinical context: ..." ahead of the question, so
   the model sees the same information either way — only the cue-injection logic changes what
   it can touch.
2. **`benchmaxxing/data.py`'s `_text_case` never read `report` back off a manifest CSV** (unlike
   `_image_case`, which does). This is a pre-existing bug in shared code, not in this adapter —
   it was never triggered before because MedQA/MedMCQA don't use `Case.report`, and no other Lane
   B dataset with `Case.report` (e.g. PubMedQA, #293) has run an experiment through a manifest
   CSV yet. Its effect here: `experiments/medqa/reproduce.py`'s `run_solo` calls `load_cases()`,
   so every real run was silently answering "What is the primary finding?" with **no report
   text at all** — the model was guessing cold among 4 similar-sounding finding names. That
   produced a very high, unstable flip rate and ~35-60% of `gemini-2.5-flash` responses were a
   refusal-shaped "please provide the report" (confirmed by inspecting raw cached completions).
   Fixed with a one-line addition (`report=_none_or_str(_get(row, "report"))`) plus a regression
   test (`tests/test_datasets.py::test_text_case_report_round_trips`).

Both fixes are committed; the numbers below are from the **corrected** re-run (fresh cache,
after both fixes).

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

## n=35 (smoke run, corrected)

```json
{
  "n_records": 210,
  "noise_floor_by_model": {"gemini-2.5-flash": 0.0667, "gemini-2.5-flash-lite": 0.0},
  "flip_rate_by_model": {
    "gemini-2.5-flash": {"overall": 0.0857, "n": 105},
    "gemini-2.5-flash-lite": {"overall": 0.0952, "n": 105}
  }
}
```

Full record: [`results/solo_results.json`](results/solo_results.json).

## n=600 (corrected)

```json
{
  "n_records": 3600,
  "noise_floor_by_model": {"gemini-2.5-flash": 0.0667, "gemini-2.5-flash-lite": 0.0},
  "flip_rate_by_model": {
    "gemini-2.5-flash": {
      "overall": 0.0894,
      "per_cue": {"lexical_overlap": 0.12, "longest_option": 0.105, "option_order": 0.0433},
      "n": 1800
    },
    "gemini-2.5-flash-lite": {
      "overall": 0.1144,
      "per_cue": {"lexical_overlap": 0.1183, "longest_option": 0.13, "option_order": 0.095},
      "n": 1800
    }
  }
}
```

Full record: [`results_n600/solo_results.json`](results_n600/solo_results.json).

## Cross-dataset table row (MIMIC-CXR text, n=600, corrected)

| model | overall flip | noise floor | flip-above-noise |
|---|---|---|---|
| gemini-2.5-flash | 0.089 | 0.067 | +0.022 |
| gemini-2.5-flash-lite | 0.114 | 0.000 | +0.114 |

Both models land in the same narrow band as MedQA-USMLE (flash 0.063, flash-lite 0.117), so the
shortcut effect generalizes from exam-style vignettes to real clinical report text. Consistent
between n=35 (0.086/0.095) and n=600 (0.089/0.114) samples, and confirmed refusal-aware
(abstention ~0% at both scales). `lexical_overlap` is the strongest cue for both tiers at n=600,
though the margin over `longest_option` is now small (unlike the pre-fix numbers, where it looked
artificially dominant).

## Cascade contagion (#317): bare vs case-anchored confident wrong seed

Real-API results for #317, reusing `experiments/medqa/reproduce.py --stage cascade` as-is for
the bare/baseline-relative seed, and a copy of `experiments/medqa/push_c.py`
(`experiments/mimic_cxr_text/push_c.py`) for the case-anchored plausibility dose-response, per
#317's "copy this template... adapt the loader if needed."

**Two adaptations were required**, both stemming from this dataset's report/question split
(#336): `push_c.py`'s `_mcq` builder had no "Clinical context" section at all (MedQA's original
has no separate `report` field, so it never needed one — without this fix the holdout would be
answering blind, the same class of bug as #336's bug 2), and the case-anchored rationale's detail
extraction pulled from `case.question`, which here is a fixed 8-word stem identical across every
case (would make every "anchored" rationale generic) — fixed to pull from `case.report`, the
actual clinical narrative. `build_solo_records.py` reconstructs the `--solo-records` input
`push_c.py` needs (which model) from the already-cached n=600 solo run, with no new API calls.

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
  "anchored": {"conform": 51, "rate": 0.85, "wilson95": [0.739, 0.919]},
  "anchored_strong": {"conform": 55, "rate": 0.917, "wilson95": [0.819, 0.964]},
  "anchored_solo": {"conform": 43, "rate": 0.717, "wilson95": [0.592, 0.815]},
  "anchored_vs_generic_paired": {"anchored_only": 3, "generic_only": 3, "mcnemar_p": 1.0, "rate_diff": 0.0}
}
```

Full record: [`results/push_c_summary.json`](results/push_c_summary.json).

Unlike MedQA (generic 0.32 baseline conformity), MIMIC-CXR text's baseline conformity is already
very high (0.85) — on these "hard" cases (holdout already wrong alone), two confident attendings
asserting *any* wrong answer, anchored or not, draw strong deference. `anchored` adds nothing
over `generic` here (McNemar p=1.0, unlike MedQA's significant +0.12 lift), but `anchored_strong`
(a longer, more specific rationale) does push conformity higher, and `anchored_solo` (a single
peer instead of two) drops it — a majority-pressure effect, consistent in direction with MedQA.
Read together with #316: report-text MCQs are about as susceptible to a lone answer-preserving
cue as MedQA, but more susceptible to social pressure once wrong, independent of whether that
pressure cites case-specific detail.
