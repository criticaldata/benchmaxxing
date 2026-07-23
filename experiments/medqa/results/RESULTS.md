# benchmaxxing MedQA (Lane B) results, consolidated

_Analysis artifact for ongoing review. Run bundle: `full_v1/`. Dataset: MedQA-USMLE test. Models: gemini-2.5-flash, gemini-2.5-flash-lite (same-lineage control arm). git 3273c011 (feat/adapters-runners), started 2026-07-20 00:45:30. Temperature 0; deterministic. Every model call is cached in `call_cache.jsonl` so re-runs reproduce these numbers with zero API calls; cascade transcripts are in `transcripts/` for offline replay._

## 0. Provenance and reproduction map

The committed bundle in `results/` is the saved output of the first real run (git 3273c011, 2026-07-20). `reproduce.py` is the single consolidated verifier: it re-reads the committed `call_cache.jsonl` and recomputes the same numbers, writing to its OWN filenames so it never overwrites the saved bundle. (The earlier per-stage runners that produced the original bundle are not committed; `reproduce.py` supersedes them and is the one script a reviewer needs.) Verified 2026-07-21: a keyless `--stage all` run reproduced every number below with zero new API calls.

| committed artifact (saved run) | `reproduce.py` output (verifier, regenerable) | number to check |
|---|---|---|
| `solo_and_cascade_v1_results.json`, `solo_records.jsonl` | `solo_results.json` | solo flip 0.787 / 0.893 |
| `cascade_v2_summary.json`, `cascade_v2_per_case.jsonl` | `cascade_results.json` | mean contagion -0.05, shared 0.025 / isolated 0.075 |
| `transcripts/{case}_v2_{shared,isolated}.jsonl` | `transcripts/{case}_repro_{shared,isolated}.jsonl` | per-turn replay (distinct `_repro_` names, no overwrite) |

To verify: run the reproduce command in the footer, then diff each verifier output against the committed artifact in the same row. A key is needed only to fill a cache miss or the uncached noise-floor control (skipped without one).

## 1. Solo susceptibility (Story 2 building block) - STRONG, defensible

Flip = the model's answer changes between the clean twin and the cue-injected twin. Noise floor = self-inconsistency when the same clean case is run twice with no cue (uncached control, 15 cases/model). Flip-above-noise is the honest susceptibility.

| model | overall flip | noise floor | flip-above-noise |
|---|---|---|---|
| gemini-2.5-flash | 0.787 | 0.000 | +0.787 |
| gemini-2.5-flash-lite | 0.893 | 0.000 | +0.893 |

### per model x cue flip rate
| model | lexical_overlap | longest_option | option_order |
|---|---|---|---|
| gemini-2.5-flash | 0.74 | 0.74 | 0.88 |
| gemini-2.5-flash-lite | 0.85 | 0.84 | 0.99 |

**Read:** cues bite hard and genuinely (noise floor 0.000, so flips are real, not instability). option_order is the strongest cue. This is the headline text-lane result.

## 2. Same-lineage overlap (control)

- within-lineage phi = 0.5581129694273874, p = nan (cross term undefined with a single lineage, p=nan by design). A real cross value needs the open-weights cross-lineage arm (cluster).

## 3. Cascade / contagion (Story 1) - two seed designs compared

Contagion = (fraction of NON-seed committee members who adopt the planted shortcut answer in the SHARED run) minus (the same in the ISOLATED run). Positive = the shared board spreads the shortcut.

| seed design | cases | mean contagion | mean shared adopt | mean isolated adopt | valid? |
|---|---|---|---|---|---|
| v1: first distractor (`_shortcut_answer`) | 20 | -0.006 | - | - | **NO** (seed == committee baseline in 16/20 cases, no counterfactual gap) |
| v2: baseline-relative (issue #104 fix) | 20 | -0.050 | 0.025 | 0.075 | yes |

**Read (v2, the valid one):** contagion is essentially zero, and shared adoption (0.025) is actually LOWER than isolated (0.075). Same-lineage Gemini committees on MedQA MCQ do NOT cascade on a single confident wrong peer; they hold their independent answer. Honest null for the text-lane cascade story. (The onset count is not meaningful here: the detector fires on the planted seed turn itself, not on genuine downstream adoption.)

### v2 per-case (baseline-relative seed)
| case | shared adopt | isolated adopt | contagion |
|---|---|---|---|
| medqa-1033 | 0.00 | 0.00 | +0.00 |
| medqa-1047 | 0.00 | 0.00 | +0.00 |
| medqa-1090 | 0.50 | 1.00 | -0.50 |
| medqa-1194 | 0.00 | 0.00 | +0.00 |
| medqa-1266 | 0.00 | 0.00 | +0.00 |
| medqa-194 | 0.00 | 0.00 | +0.00 |
| medqa-285 | 0.00 | 0.00 | +0.00 |
| medqa-286 | 0.00 | 0.00 | +0.00 |
| medqa-447 | 0.00 | 0.00 | +0.00 |
| medqa-513 | 0.00 | 0.00 | +0.00 |
| medqa-530 | 0.00 | 0.00 | +0.00 |
| medqa-577 | 0.00 | 0.00 | +0.00 |
| medqa-621 | 0.00 | 0.00 | +0.00 |
| medqa-733 | 0.00 | 0.00 | +0.00 |
| medqa-788 | 0.00 | 0.50 | -0.50 |
| medqa-82 | 0.00 | 0.00 | +0.00 |
| medqa-829 | 0.00 | 0.00 | +0.00 |
| medqa-861 | 0.00 | 0.00 | +0.00 |
| medqa-976 | 0.00 | 0.00 | +0.00 |
| medqa-995 | 0.00 | 0.00 | +0.00 |

## 4. Interpretation and next steps

- **Solo susceptibility stands on its own** as a strong, clean result (Story 2). Report as flip-above-noise.
- **Cascade contagion is a null in this setup.** The phenomenon (Story 1) most likely needs one or more of: the imaging lane (richer, more ambiguous stimulus), weaker / more suggestible models, MULTIPLE colluding seeded peers (not one), or a deference-inducing committee prompt. The harness and the (now valid) metric are ready to test each of these.
- **Residual measurement note for v2:** the seed was chosen to differ from each committee's MAJORITY baseline; two cases still showed isolated adoption because an individual agent's baseline coincided with the seed. Tightening the seed to differ from EVERY agent's isolated baseline would remove that.
- **Cross-lineage overlap** still needs the open-weights arm (cluster) for a real cross term.

_Reproduce: `python -m experiments.medqa.reproduce --manifest <medqa_manifest.csv> --out experiments/medqa/results --stage all` re-reads `call_cache.jsonl` and reproduces the solo and cascade numbers here with NO new API calls and NO key (verified: solo flip 0.787 / 0.893 from cache, zero new calls). Only the uncached noise-floor control needs a key; it is skipped without one. reproduce.py writes its own cascade transcripts under `transcripts/*_repro_*.jsonl` so it never overwrites the committed run's transcripts._
