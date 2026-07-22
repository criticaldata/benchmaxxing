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

Every model call is cached under `--out/call_cache.jsonl` (keyed by model+prompt). A fully
cached reproduction needs **no key**: `reproduce.py` reads the committed cache and returns the
committed solo/cascade numbers with zero API calls (verified). A key is required only to fill a
cache miss and for the uncached noise-floor control (skipped without a key). The committed cache
and transcripts ARE included so the numbers are verifiable from the committed code.

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

## Cascade-onset distribution (#174, part a)

`onset_distribution.py` activates `benchmaxxing.onset.cascade_onset` (the plan's flagship
reusable artifact, exported and unit-tested but never reported on committed data) on the
baseline-relative arm's transcripts. Uses `_repro_shared.jsonl` (post-parser-fix), not the plain
`medqa-*_shared.jsonl` the issue names, since those predate the fix and were never regenerated;
`cascade_results.json`'s own `onset` field is already computed from the same `_repro_` transcripts,
confirmed here to match exactly.

```bash
python -m experiments.medqa.onset_distribution
```

**Result:** onset is detected (not censored) on all 20 of 20 cases, and every detected onset lands
on the exact same turn index (2) - immediately after the seeded agent's one-off wrong turn, which
it then abandons for the rest of the run. This is the trivial detection of a single-turn blip that
reverts immediately, not a genuine sustained tipping point: consistent with the already-established
robust null on this arm, and does not indicate the onset metric is broken - there is simply no
sustained regime change for a single change-point detector to find here.

Also reports the `onset` field already present in `cascade_v2_per_case.jsonl` (the anchored-seed
arm, onsets 1/2/5, real spread), explicitly flagged as **confirmed tainted** by the pre-fix
answer-parser bug and not treated as trustworthy; included because the issue asks for it, not as a
validated finding. The two arms' seed-invalidity rates are NOT the same measurement as the
previously reported "16/20 invalid seed" figure (that describes the older v1 first-distractor
design, not either arm analyzed here) and are not conflated with it.

## Files

- `reproduce.py`, the runnable script (solo + noise floor + baseline-relative cascade).
- `results/RESULTS.md`, consolidated results with tables and interpretation.
- `results/solo_and_cascade_v1_results.json`, `results/run_summary.md`, the full run output.
- `results/cascade_v2_summary.json`, `results/cascade_v2_per_case.jsonl`, the valid cascade.
- `results/solo_records.jsonl`, per-twin solo records (the raw result data).
- `results/config.json`, `results/versions.json`, run config and provenance.
- `onset_distribution.py`, `results/onset_distribution.json` - #174's cascade-onset distribution.

## Note

`reproduce.py` uses only modules on `main`. Building the manifest needs the MedQA adapter from
PR #66 (dataset adapters); once that merges, the whole flow runs from `main`.
