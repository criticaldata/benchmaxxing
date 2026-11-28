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

## Plausible versus implausible planted distractor on self-hosted nemotron FP8 (#212)

`plausible_distractor.py` asks whether the holdout adopts a senior attending's wrong answer more
when that answer is the holdout's own second choice (plausible) than when it is its own
least-likely option (implausible). This section is the run on
`nemotron-3-super-120b-a12b-fp8-local` (the FP8 weights served locally): temperature 0, 120 cases
asked, 101 scored. Both commands replay from the committed cache with no key and no server.

```bash
python -m experiments.medqa.plausible_distractor --manifest <medqa_manifest.csv> \
    --model nemotron-3-super-120b-a12b-fp8-local --n 120
python -m experiments.medqa.declared_only_audit_plausible_distractor \
    --manifest <medqa_manifest.csv> --model nemotron-3-super-120b-a12b-fp8-local
```

| rows | second choice adopted | least likely adopted | discordant pairs | exact McNemar P |
|---|---|---|---|---|
| all 101 scored rows | 0.1782 (18) | 0.0594 (6) | 15 vs 3 | 0.007538 |
| the 80 rows with no empty reply | 0.175 (14) | 0.0375 (3) | 12 vs 1 | 0.003418 |
| the 21 rows with an empty reply all scored against the result | 0.1386 (14) | 0.2376 (24) | 12 vs 22 | 0.121449 |

**Read, with the caveat that goes beside the number.** Adoption is 0.1782 for the second-choice
answer against 0.0594 for the least-likely one (15 vs 3 discordant pairs, exact McNemar
P = 0.007538), but 21 of the 101 scored rows carry an empty reply: the gap holds on the 80 rows
with none (0.175 against 0.0375, 12 vs 1, P = 0.003418) and reverses, without reaching
significance, when those 21 rows are all scored against it (0.1386 against 0.2376, 12 vs 22,
P = 0.121). Wherever 0.1782 against 0.0594 is quoted, the rest of that sentence goes with it.

Where the empty replies are. 50 of the arm's 562 cached replies came back with no answer text: 21
second-choice ranking prompts, 19 least-likely ranking prompts, 6 bare questions, 3 second-choice
boards and 1 least-likely board. An empty ranking reply makes the runner fall back to the first
eligible wrong option, so that distractor is not the model's own ranking, and an empty board reply
is scored as no adoption. 27 of the 50 sit in the 21 scored rows above. The other 23 sit in 14 of
the 19 unscored cases, where the fallback left both seeds on the same option and no board was
asked; those cases are outside every row of the table, and counting them as 14 more rows against
the result gives 12 vs 36 (P = 0.000717).

The first row is `results/nemotron-3-super-120b-a12b-fp8-local/plausible_distractor_summary.json`.
The second is `plausible_distractor_declared_audit.json` in the same folder: its 21 undeclared rows
are exactly the rows with an empty reply. The third adds those 21 rows to the second as "least
likely adopted, second choice not adopted". `tests/test_nemotron_fp8_plausible_empty_replies.py`
recomputes every figure here from the committed rows and cache, and fails if a paragraph of this
file quotes the first row without the other two.

## Majority-pressure (Asch) variant (#117)

Does a wrong 2-of-3 MAJORITY move the holdout more than a single wrong peer? The classic Asch
conformity effect needs a majority, not a lone dissenter. This seeds `k` of the committee's peers
with the same case-anchored wrong answer (plus a persuasive rationale) and measures the non-seeded
flash-lite holdout's adoption at k=0 (isolated), k=1, and k=2. Genuinely exercises
`run_committee`'s multi-spec `seed_turn` (a sequence of specs plants several slots) on real
committee data for the first time.

```bash
python -m experiments.medqa.majority_pressure \
    --manifest <medqa_manifest.csv> --out experiments/medqa/results --n 40
```

| seeded wrong peers | holdout adoption |
|---|---|
| 0 (isolated) | 0.0 |
| 1 | 0.125 |
| 2 | 0.15 |

1-peer vs 2-peer, paired: McNemar gain=2, lose=1, **p=1.0 (not significant)**.

**Read.** A wrong 2-of-3 majority does not move the flash-lite holdout meaningfully more than a
single wrong peer, and adoption stays low throughout (≤0.15) even with a persuasive case-anchored
rationale on every seeded peer. The classic Asch prediction fails for a safety-trained
same-lineage committee: adding a second confident wrong voice buys essentially nothing. This
mirrors the imaging lane's majority-pressure result (#172) from the opposite direction: there
adoption saturated at one peer (1-peer == 2-peer == 0.97, near-total); here it saturates at one
peer too but at a low floor (~0.13). Both lanes agree the increment from 1 to 2 wrong peers is
null; they differ only in the floor a single peer already sets. Holdout is flash-lite (the
susceptible tier, matching `referee_deployable.py`/`scale_c`), the peers are flash. 3 API calls
per case (isolated + k=1 + k=2 shared); verified keyless afterward (0 new calls).

## Files

- `reproduce.py`, the runnable script (solo + noise floor + baseline-relative cascade).
- `results/RESULTS.md`, consolidated results with tables and interpretation.
- `results/solo_and_cascade_v1_results.json`, `results/run_summary.md`, the full run output.
- `results/cascade_v2_summary.json`, `results/cascade_v2_per_case.jsonl`, the valid cascade.
- `results/solo_records.jsonl`, per-twin solo records (the raw result data).
- `results/config.json`, `results/versions.json`, run config and provenance.
- `onset_distribution.py`, `results/onset_distribution.json` - #174's cascade-onset distribution.
- `plausible_distractor.py`, `declared_only_audit_plausible_distractor.py`,
  `results/nemotron-3-super-120b-a12b-fp8-local/plausible_distractor*` - #212 on self-hosted
  nemotron FP8, with its empty-reply caveat.
- `majority_pressure.py`, `results/majority_pressure.jsonl` / `_summary.json` - #117's Asch variant.

## Note

`reproduce.py` uses only modules on `main`. Building the manifest needs the MedQA adapter from
PR #66 (dataset adapters); once that merges, the whole flow runs from `main`.
