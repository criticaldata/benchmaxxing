# MedQA (Lane B) experiments: first real-model runs

This folder holds the sanitized reproduction script (`reproduce.py`) and a saved copy of the
results (`results/`) for the first real-model runs on MedQA-USMLE with same-lineage Gemini
committees. It closes #93 (stage-0 MedQA pilot) and #96 (cascade experiment on real data).

No secrets are committed: the script reads `GEMINI_API_KEY` from the environment, and `.env`
files are git-ignored.

## Reproduce

```bash
export GEMINI_API_KEY=...            # never commit this
# build the MedQA manifest first with the adapter (benchmaxxing/datasets/medqa.py, PR #66):
#   python -c "from benchmaxxing.datasets import medqa; medqa.build_manifest('path/to/medqa', 'medqa_manifest.csv')"
python -m experiments.medqa.reproduce --manifest medqa_manifest.csv \
    --out experiments/medqa/results --stage all --solo-n 100 --cascade-n 20
```

Every model call is cached under `--out/call_cache.jsonl` (keyed by model+prompt), so re-runs
reproduce the numbers with no new API calls and an interrupted run resumes for free. The cache
and per-run transcripts are regenerable and are not committed.

## What the results show

**Solo susceptibility (Story 2): strong.** 100 cases, gemini-2.5-flash / flash-lite,
answer-preserving MCQ cues. Noise floor 0.000 (deterministic at temperature 0), so flips are
genuine cue effects, not answer instability.

| model | overall flip | noise floor | flip-above-noise |
|---|---|---|---|
| gemini-2.5-flash | 0.79 | 0.00 | +0.79 |
| gemini-2.5-flash-lite | 0.89 | 0.00 | +0.89 |

`option_order` is the strongest cue (flash 0.88, flash-lite 0.99).

**Cascade (Story 1): null, and the setup was the problem, not the thesis.** The first
cascade used a first-distractor seed that coincided with the committee's own baseline answer
in 16/20 cases (no counterfactual gap). Re-running with a **baseline-relative seed** (the
planted answer differs from each committee's isolated baseline) gives contagion at or near
zero: same-lineage Gemini committees hold their independent answer even against a confident
wrong peer on the shared board. Why, and how the design changes (voiceable cue-rationale
seeds, hard-case selection, majority pressure, a natural-cue imaging cascade), is tracked in
issues #115, #116, #117, #120, #121.

## Files

- `reproduce.py`, the runnable script (solo + noise floor + baseline-relative cascade).
- `results/RESULTS.md`, consolidated results with tables and interpretation.
- `results/solo_and_cascade_v1_results.json`, `results/run_summary.md`, the full run output.
- `results/cascade_v2_summary.json`, `results/cascade_v2_per_case.jsonl`, the valid cascade.
- `results/solo_records.jsonl`, per-twin solo records (the raw result data).
- `results/config.json`, `results/versions.json`, run config and provenance.

## Note

`reproduce.py` uses only modules on `main`. Building the manifest needs the MedQA adapter from
PR #66 (dataset adapters); once that merges, the whole flow runs from `main`.
