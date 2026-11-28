# Flip scoring audit (28 Nov 2026)

What a solo cue "flip" counted, what it counts now, every committed number that moves, every file
that could not be regenerated here, and what merging with the other open pull requests needs. No
model call was made for anything in this file: the regenerated files are keyless replays of
committed caches, and the rest is recomputed from committed rows.

## The defect

A flip was `clean_answer != contaminated_answer`, a comparison of two answer strings, where each
answer is the text of the option the model chose on that twin. Two of the three text cues rewrite
the text of one wrong option: `longest_option` appends a filler to it and `lexical_overlap` appends
words from the stem. A model that picks that option on the clean twin and picks it again on the cued
twin has not changed its answer, but it returns two different strings, and the comparison counted a
flip. `option_order` only moves the options, so it was never affected.

Every such case is wrong on the clean twin (the rewritten option is always a wrong one), so the
defect inflates the flip rate of exactly the cases a model already gets wrong, and it grows with a
dataset's difficulty.

## The fix

A flip is now a different option on the cued twin than on the clean twin:
`benchmaxxing.analysis.chose_same_option` maps each answer to the option it names, as that option's
position in the clean option list, whichever twin the answer came from (an option the cue moved is
followed by its text, an option the cue rewrote by its position, two options with the same text are
one option). An answer that names no option (an abstention, a yes/no read, an image label) is
compared as before.

The earlier count is kept everywhere under `text_changed`, so no number disappears:
`FlipRecord.text_changed`, a `text_changed` column in `solo_records.jsonl`,
`text_changed_rate_by_model` in `solo_results.json`, `text_changed` (rate and counts) in each cue
row of the cross-dataset result, and `text_changed_rate_excl_abstentions` in the MIMIC-CXR text
re-analysis. A changed option is always a changed text, so `flipped <= text_changed` in every row.

## Blast radius

The string comparison lived in three places, and all three now call the same function:

| where | used by |
|---|---|
| `benchmaxxing/analysis.py`, `solo_evaluate` | the cross-dataset arm, `benchmaxxing run --stage pilot/solo`, the offline smoke |
| `experiments/medqa/reproduce.py`, `run_solo` | the MedQA, MedMCQA and MIMIC-CXR text solo stages |
| `experiments/mimic_cxr_text/refusal_aware_reanalysis.py` | the MIMIC-CXR text refusal-aware solo rates |

Two scripts read the stored `flipped` column and so follow whatever the records file says:
`experiments/contamination/contamination_audit.py` and `experiments/contamination/misaligned_proxy.py`.
Two more hold copies of the Gemini contamination numbers as constants:
`experiments/effect_sizes/effect_sizes_text.py` (5/11, 7/89, 14/22, 7/78) and
`experiments/family_correction/family_correction.py` (two Fisher p values).

**Affected results**

| lane | files | here |
|---|---|---|
| cross-dataset | 8 result files (Gemini 500, 150 and 50; five lineages at 50) | 7 regenerated, 1 not |
| MedQA solo | 6 `solo_results.json`, the rerun file, 6 `solo_records.jsonl`, the contamination copy of the Gemini rows | 4 lineages and the rerun file regenerated; Gemini first run and hosted Nemotron not |
| MedMCQA solo | 7 `solo_results.json`, 7 `solo_records.jsonl` | 4 lineages regenerated; Gemini and both Nemotron files not |
| contamination | 6 audit summaries, 6 misaligned-proxy files | 4 lineages regenerated; Gemini and hosted Nemotron not |
| effect sizes, family-wise correction | `effect_sizes_text.json`, `family_correction.json` | not regenerated (constants in the scripts) |
| MIMIC-CXR text solo | `solo_results.json` and `solo_results_refusal_aware.json` under `results/` and `results_n600/` for every lineage, `solo_results_600.json`, the four `cross_lineage_summary.json`, `RESULTS.md` | not regenerated (no cache, no manifest, no answers in the rows) |
| generic runner | `experiments/runs/pilot-medqa-seed0/` | not regenerated (no rows, no cache) |

**Not affected, and why**

- **`option_order`, everywhere.** The cue moves options and rewrites none, so a changed text is a
  changed option. Its counts are identical before and after in every file below.
- **Every count restricted to cases answered correctly on the clean twin.** A model that is right on
  the clean twin chose the correct option, whose text no cue rewrites. Checked on the rows: of the
  1,737 regenerated solo rows that are right on the clean twin, the two comparisons differ on 0; the
  `clean_correct` blocks of the cross-dataset files are unchanged.
- **`clean_correct` and `contaminated_correct`.** Compared with the text of the correct option,
  which is never rewritten. So the hard-case cohorts of `break_it`, `clean_a`, `push_c` and the
  MIMIC-CXR text arms, which are picked on `clean_correct` alone, do not move.
- **SUPPORT2 solo.** The tabular cues rewrite the record, never the options: both twins are built
  with `case.options`, and every committed answer is one of the same two strings.
- **Imaging solo (NIH, CheXpert, MIMIC-CXR image).** Yes/no reads of two films, no option text.
- **Committee and cascade arms, `text_cue_types`, the matched-baseline `reason_flip`, the
  temperature flips.** They compare answers given over one unchanged option list.

## Regenerated files

All at 0 model calls, with no key in the environment, on the MedQA manifest the adapter builds from
the public `test.jsonl` (sha256 `0f2076a4...`) and the 500-row MedMCQA manifest (`5b2ce971...`).
In every regenerated file the new `text_changed` numbers equal the flip numbers the file had before.

| file | cue | MedQA flips, old to new | MedMCQA flips, old to new | Fisher p, old to new | correct on clean, old to new | Fisher p, old to new |
|---|---|---|---|---|---|---|
| Gemini 2.5 Flash, 500 | `option_order` | 31/500 | 64/499 | 0.00035 | 14/451 vs 26/401 | 0.0229 |
| Gemini 2.5 Flash, 500 | `longest_option` | 41/500 to 29/500 | 73/499 to 55/499 | 0.0014 to 0.0030 | 17/451 vs 19/401 | 0.500 |
| Gemini 2.5 Flash, 500 | `lexical_overlap` | 38/500 to 29/500 | 84/499 to 62/499 | 7.7e-06 to 0.00027 | 13/451 vs 28/401 | 0.0061 |
| Gemini 2.5 Flash, 150 | `option_order` | 8/150 | 18/150 | 0.0630 | 4/140 vs 9/123 | 0.152 |
| Gemini 2.5 Flash, 150 | `longest_option` | 9/150 to 7/150 | 22/150 to 15/150 | 0.0216 to 0.119 | 4/140 vs 8/123 | 0.236 |
| Gemini 2.5 Flash, 150 | `lexical_overlap` | 8/150 to 5/150 | 25/150 to 15/150 | 0.0027 to 0.0349 | 3/140 vs 9/123 | 0.0722 |
| Qwen2.5-VL-72B, 50 | `option_order` | 4/50 | 6/50 | 0.741 | 3/38 vs 2/24 | 1.000 |
| Qwen2.5-VL-72B, 50 | `longest_option` | 3/50 to 2/50 | 12/50 to 7/50 | 0.0226 to 0.160 | 1/38 vs 1/24 | 1.000 |
| Qwen2.5-VL-72B, 50 | `lexical_overlap` | 2/50 to 1/50 | 15/50 to 12/50 | 0.00091 to 0.0018 | 0/38 vs 2/24 | 0.146 |
| Cosmos-Reason2-8B, 50 | `option_order` | 12/50 | 15/50 | 0.653 | 4/30 vs 5/25 | 0.716 |
| Cosmos-Reason2-8B, 50 | `longest_option` | 12/50 to 8/50 | 16/50 to 6/50 | 0.504 to 0.774 | 5/30 vs 4/25 | 1.000 |
| Cosmos-Reason2-8B, 50 | `lexical_overlap` | 8/50 | 22/50 to 13/50 | 0.0041 to 0.326 | 2/30 vs 7/25 | 0.0636 |
| Nemotron-3-Super-120B (hosted), 50 | `option_order` | 3/50 | 7/50 | 0.318 | 2/46 vs 2/31 | 1.000 |
| Nemotron-3-Super-120B (hosted), 50 | `longest_option` | 3/50 | 14/50 to 9/50 | 0.0064 to 0.121 | 2/46 vs 4/31 | 0.213 |
| Nemotron-3-Super-120B (hosted), 50 | `lexical_overlap` | 2/50 | 13/50 to 10/50 | 0.0038 to 0.0277 | 2/46 vs 4/31 | 0.213 |
| gpt-oss-120b, 50 | `option_order` | 1/50 | 8/50 | 0.0309 | 1/46 vs 3/30 | 0.294 |
| gpt-oss-120b, 50 | `longest_option` | 2/50 | 9/50 to 8/50 | 0.0511 to 0.0916 | 1/46 vs 1/30 | 1.000 |
| gpt-oss-120b, 50 | `lexical_overlap` | 4/50 | 12/50 to 8/50 | 0.0538 to 0.357 | 3/46 vs 3/30 | 0.675 |
| gpt-oss-20b, 50 | `option_order` | 6/50 | 15/50 | 0.0479 | 1/38 vs 6/29 | 0.0371 |
| gpt-oss-20b, 50 | `longest_option` | 8/50 | 15/50 to 13/50 | 0.153 to 0.326 | 2/38 vs 6/29 | 0.0684 |
| gpt-oss-20b, 50 | `lexical_overlap` | 7/50 | 14/50 to 10/50 | 0.140 to 0.595 | 2/38 vs 4/29 | 0.391 |

`option_order` does not change in any file. The correct-on-clean columns do not change in any file
(for the five lineage files that block is new; its counts are the same under both comparisons).
Five of the new correct-on-clean Fisher tests return exactly 1.0 on small, nearly equal counts;
each has an entry in `tests/degeneracy_exemptions.json` with its table.

| lineage | file | cue | flips, old to new | rate, old to new |
|---|---|---|---|---|
| gemini-2.5-flash | `solo_results_rerun_2026-07-29.json` | `option_order` | 5/100 | 0.0500 |
| gemini-2.5-flash | `solo_results_rerun_2026-07-29.json` | `longest_option` | 5/100 to 3/100 | 0.0500 to 0.0300 |
| gemini-2.5-flash | `solo_results_rerun_2026-07-29.json` | `lexical_overlap` | 6/100 to 3/100 | 0.0600 to 0.0300 |
| gemini-2.5-flash | `solo_results_rerun_2026-07-29.json` | all three cues | 16/300 to 11/300 | 0.0533 to 0.0367 |
| gemini-2.5-flash-lite | `solo_results_rerun_2026-07-29.json` | `option_order` | 9/100 | 0.0900 |
| gemini-2.5-flash-lite | `solo_results_rerun_2026-07-29.json` | `longest_option` | 12/100 to 9/100 | 0.1200 to 0.0900 |
| gemini-2.5-flash-lite | `solo_results_rerun_2026-07-29.json` | `lexical_overlap` | 16/100 to 11/100 | 0.1600 to 0.1100 |
| gemini-2.5-flash-lite | `solo_results_rerun_2026-07-29.json` | all three cues | 37/300 to 29/300 | 0.1233 to 0.0967 |
| Qwen/Qwen2.5-VL-72B-Instruct | `Qwen_Qwen2.5-VL-72B-Instruct/solo_results.json` | `option_order` | 11/100 | 0.1100 |
| Qwen/Qwen2.5-VL-72B-Instruct | `Qwen_Qwen2.5-VL-72B-Instruct/solo_results.json` | `longest_option` | 9/100 to 3/100 | 0.0900 to 0.0300 |
| Qwen/Qwen2.5-VL-72B-Instruct | `Qwen_Qwen2.5-VL-72B-Instruct/solo_results.json` | `lexical_overlap` | 12/100 to 10/100 | 0.1200 to 0.1000 |
| Qwen/Qwen2.5-VL-72B-Instruct | `Qwen_Qwen2.5-VL-72B-Instruct/solo_results.json` | all three cues | 32/300 to 24/300 | 0.1067 to 0.0800 |
| local/nvidia/cosmos-reason2-8b | `local_nvidia_cosmos-reason2-8b/solo_results.json` | `option_order` | 24/100 | 0.2400 |
| local/nvidia/cosmos-reason2-8b | `local_nvidia_cosmos-reason2-8b/solo_results.json` | `longest_option` | 22/100 to 10/100 | 0.2200 to 0.1000 |
| local/nvidia/cosmos-reason2-8b | `local_nvidia_cosmos-reason2-8b/solo_results.json` | `lexical_overlap` | 28/100 to 25/100 | 0.2800 to 0.2500 |
| local/nvidia/cosmos-reason2-8b | `local_nvidia_cosmos-reason2-8b/solo_results.json` | all three cues | 74/300 to 59/300 | 0.2467 to 0.1967 |
| openai/gpt-oss-120b | `openai_gpt-oss-120b/solo_results.json` | `option_order` | 4/100 | 0.0400 |
| openai/gpt-oss-120b | `openai_gpt-oss-120b/solo_results.json` | `longest_option` | 8/100 to 5/100 | 0.0800 to 0.0500 |
| openai/gpt-oss-120b | `openai_gpt-oss-120b/solo_results.json` | `lexical_overlap` | 8/100 to 5/100 | 0.0800 to 0.0500 |
| openai/gpt-oss-120b | `openai_gpt-oss-120b/solo_results.json` | all three cues | 20/300 to 14/300 | 0.0667 to 0.0467 |
| openai/gpt-oss-20b | `openai_gpt-oss-20b/solo_results.json` | `option_order` | 12/100 | 0.1200 |
| openai/gpt-oss-20b | `openai_gpt-oss-20b/solo_results.json` | `longest_option` | 11/100 to 8/100 | 0.1100 to 0.0800 |
| openai/gpt-oss-20b | `openai_gpt-oss-20b/solo_results.json` | `lexical_overlap` | 14/100 to 12/100 | 0.1400 to 0.1200 |
| openai/gpt-oss-20b | `openai_gpt-oss-20b/solo_results.json` | all three cues | 37/300 to 32/300 | 0.1233 to 0.1067 |

Within-lineage phi in the rerun file (flash against flash-lite failure vectors): 0.2719 to 0.1162.
Its `noise_floor_by_model` and `rerun_provenance` are kept; a `flip_scoring` note says which fields
moved. The per-lineage `noise_floor_by_model` values are kept as committed: the floor is an uncached
live control that a keyless replay cannot measure.

| lineage | file | cue | flips, old to new | rate, old to new |
|---|---|---|---|---|
| Qwen/Qwen2.5-VL-72B-Instruct | `Qwen_Qwen2.5-VL-72B-Instruct/solo_results.json` | `option_order` | 19/100 | 0.1900 |
| Qwen/Qwen2.5-VL-72B-Instruct | `Qwen_Qwen2.5-VL-72B-Instruct/solo_results.json` | `longest_option` | 15/100 to 7/100 | 0.1500 to 0.0700 |
| Qwen/Qwen2.5-VL-72B-Instruct | `Qwen_Qwen2.5-VL-72B-Instruct/solo_results.json` | `lexical_overlap` | 20/100 to 14/100 | 0.2000 to 0.1400 |
| Qwen/Qwen2.5-VL-72B-Instruct | `Qwen_Qwen2.5-VL-72B-Instruct/solo_results.json` | all three cues | 54/300 to 40/300 | 0.1800 to 0.1333 |
| local/nvidia/cosmos-reason2-8b | `local_nvidia_cosmos-reason2-8b/solo_results.json` | `option_order` | 34/100 | 0.3400 |
| local/nvidia/cosmos-reason2-8b | `local_nvidia_cosmos-reason2-8b/solo_results.json` | `longest_option` | 26/100 to 13/100 | 0.2600 to 0.1300 |
| local/nvidia/cosmos-reason2-8b | `local_nvidia_cosmos-reason2-8b/solo_results.json` | `lexical_overlap` | 37/100 to 25/100 | 0.3700 to 0.2500 |
| local/nvidia/cosmos-reason2-8b | `local_nvidia_cosmos-reason2-8b/solo_results.json` | all three cues | 97/300 to 72/300 | 0.3233 to 0.2400 |
| openai/gpt-oss-120b | `openai_gpt-oss-120b/solo_results.json` | `option_order` | 8/100 | 0.0800 |
| openai/gpt-oss-120b | `openai_gpt-oss-120b/solo_results.json` | `longest_option` | 12/100 to 8/100 | 0.1200 to 0.0800 |
| openai/gpt-oss-120b | `openai_gpt-oss-120b/solo_results.json` | `lexical_overlap` | 16/100 to 12/100 | 0.1600 to 0.1200 |
| openai/gpt-oss-120b | `openai_gpt-oss-120b/solo_results.json` | all three cues | 36/300 to 28/300 | 0.1200 to 0.0933 |
| openai/gpt-oss-20b | `openai_gpt-oss-20b/solo_results.json` | `option_order` | 16/100 | 0.1600 |
| openai/gpt-oss-20b | `openai_gpt-oss-20b/solo_results.json` | `longest_option` | 23/100 to 15/100 | 0.2300 to 0.1500 |
| openai/gpt-oss-20b | `openai_gpt-oss-20b/solo_results.json` | `lexical_overlap` | 31/100 to 23/100 | 0.3100 to 0.2300 |
| openai/gpt-oss-20b | `openai_gpt-oss-20b/solo_results.json` | all three cues | 70/300 to 54/300 | 0.2333 to 0.1800 |

| lineage | per-record flip rate | ever-flipped rate | flip rate, wrong on clean | flip rate, right on clean | Fisher p, right against wrong | misaligned proxy: corr(flip, answer length) | corr(flip, correct) | uptake delta |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-72B | 0.1067 to 0.08 | 0.21 to 0.19 | 0.4444 to 0.3704 | 0.1233 | 0.0015 to 0.0090 | 0.1739 to -0.1477 | -0.4509 to -0.163 | 0.6248 to 0.0154 |
| Cosmos-Reason2-8B | 0.2467 to 0.1967 | 0.39 to 0.38 | 0.7333 to 0.7111 | 0.1091 | 1.1e-10 to 4.4e-10 | 0.3487 to 0.053 | -0.3795 to -0.054 | 0.7283 to 0.107 |
| gpt-oss-120b | 0.0667 to 0.0467 | 0.11 to 0.08 | 0.5 to 0.2 | 0.0667 | 0.0012 to 0.182 | 0.1087 to 0.025 | -0.6032 to -0.3593 | 0.7119 to 0.3843 |
| gpt-oss-20b | 0.1233 to 0.1067 | 0.22 to 0.2 | 0.7059 to 0.5882 | 0.1205 | 2.2e-06 to 9.6e-05 | 0.1519 to 0.0903 | -0.6066 to -0.4553 | 0.7585 to 0.5456 |

The audit summaries also record `new_api_calls_this_run` 0 where they recorded 200: they are now the
output of a replay. The misaligned-proxy arm scores `decision = flipped` on the `longest_option`
rows, and its proxy is the length of the cued answer text; a same-option pick is by construction the
longest text, which is why the correlation with length falls once those picks stop counting as flips.

## Not regenerated

Nothing below was edited. Where the committed rows carry the two answers, the numbers the fix gives
are recomputed from them with `experiments/medqa/rescore_solo_records.py` (no cache, no model) and
shown; the same script, with `--out-records` and `--out-summary`, writes the rescored file if that
is wanted.

| file | why it does not replay here | what a rerun needs |
|---|---|---|
| `experiments/cross_dataset/results/medqa_vs_medmcqa.json` (Gemini, 50 cases) | 211 of its 400 prompts are not in its cache on today's manifests: its 50 MedQA cases are not the first 50 of the test manifest (200 prompts), and 11 MedMCQA `lexical_overlap` prompts changed with the cue builder | a Gemini key and 211 calls, which gives a new MedQA sample, not this one |
| `experiments/medqa/results/solo_results.json`, `solo_records.jsonl`, and the copy in `experiments/contamination/results/` (Gemini, first run) | the `lexical_overlap` cue builder changed after this run, so 32 of the 100 `lexical_overlap` prompts it asked are no longer the ones the builder makes (`solo_manifest_provenance.json`: 68 of 100 replay); the rerun file beside it does replay and is regenerated above | nothing: the rows carry the answers (table below) |
| `experiments/medqa/results/nvidia_nemotron-3-super-120b-a12b/` | 361 of the 400 solo prompts are not in the committed cache | an NVIDIA key and 361 calls, or the rows (table below) |
| `experiments/medmcqa/results/solo_results.json`, `solo_records.jsonl` (Gemini) | 17 `lexical_overlap` prompts per tier are not in the cache (same cue builder change) | a Gemini key and 34 calls, which changes those 17 cued prompts, or the rows (table below) |
| `experiments/medmcqa/results/nvidia_nemotron-3-super-120b-a12b/`, `nemotron-3-super-120b-a12b-fp8-local/` | no cache is committed for the MedMCQA lane of either | 400 calls each (a key for the hosted one, the local server for the other), or the rows (table below) |
| `experiments/contamination/results/contamination_audit_summary.json`, `misaligned_proxy.json`, and the same two under `nvidia_nemotron-3-super-120b-a12b/` | they replay, but they read the solo rows above, which are unchanged | rescored rows, then a keyless rerun of the two scripts |
| `experiments/effect_sizes/results/effect_sizes_text.json`, `experiments/family_correction/results/family_correction.json` | their contamination inputs are constants in the scripts | new constants, then a rerun (no model) |
| MIMIC-CXR text: `solo_results.json` and `solo_results_refusal_aware.json` under `results/` and `results_n600/` (Gemini and every lineage), `Qwen_Qwen2.5-VL-72B-Instruct/solo_results_600.json`, the four `cross_lineage_summary.json`, the tables in `RESULTS.md` | neither the caches nor the manifests are in the repository (PhysioNet terms), and the committed rows hold `clean_correct` only | on the host that holds the data: a keyless `reproduce.py --stage solo` at 35 and 600 cases per lineage, `refusal_aware_reanalysis.py`, then `cross_lineage_table.py`. The size of the change there is unknown |
| `experiments/runs/pilot-medqa-seed0/` | a run bundle with rates only: no rows, no cache | a Gemini key and 120 calls |
| `experiments/medqa/results/solo_and_cascade_v1_results.json` | superseded before this audit (its rates come from the parser defect that `RESULTS.md` describes) | none intended |

The 50-case Gemini pilot cannot be recomputed exactly from anything in the repository either.
Rebuilding its twins from the prompts stored in its cache, with the parser of the published tag or
with today's, gives 3/50 against 4/50, 4/50 against 7/50 and 4/50 against 8/50 under the string
comparison, where the file has 4, 5, 4 against 4, 8, 8: one reply per cue is read differently now.
On that rebuilt basis the option comparison gives 3/50 against 4/50 (`option_order`), 2/50 against
6/50 (`longest_option`) and 2/50 against 7/50 (`lexical_overlap`). No test in that file is below
0.05 before or after.

| lineage | file | cue | flips in the file | flips under the fix | matched by prefix |
|---|---|---|---|---|---|
| gemini-2.5-flash | `experiments/medqa/results/solo_results.json` | `option_order` | 5/100 | 5/100 | 0 |
| gemini-2.5-flash | `experiments/medqa/results/solo_results.json` | `longest_option` | 5/100 | 3/100 | 0 |
| gemini-2.5-flash | `experiments/medqa/results/solo_results.json` | `lexical_overlap` | 9/100 | 6/100 | 3 |
| gemini-2.5-flash | same file | all three cues | 19/300 (0.0633) | 14/300 (0.0467) | |
| gemini-2.5-flash-lite | `experiments/medqa/results/solo_results.json` | `option_order` | 9/100 | 9/100 | 0 |
| gemini-2.5-flash-lite | `experiments/medqa/results/solo_results.json` | `longest_option` | 12/100 | 9/100 | 0 |
| gemini-2.5-flash-lite | `experiments/medqa/results/solo_results.json` | `lexical_overlap` | 14/100 | 10/100 | 1 |
| gemini-2.5-flash-lite | same file | all three cues | 35/300 (0.1167) | 28/300 (0.0933) | |
| nvidia/nemotron-3-super-120b-a12b | `experiments/medqa/results/nvidia_nemotron-3-super-120b-a12b/solo_results.json` | `option_order` | 3/100 | 3/100 | 0 |
| nvidia/nemotron-3-super-120b-a12b | `experiments/medqa/results/nvidia_nemotron-3-super-120b-a12b/solo_results.json` | `longest_option` | 6/100 | 3/100 | 0 |
| nvidia/nemotron-3-super-120b-a12b | `experiments/medqa/results/nvidia_nemotron-3-super-120b-a12b/solo_results.json` | `lexical_overlap` | 6/100 | 3/100 | 0 |
| nvidia/nemotron-3-super-120b-a12b | same file | all three cues | 15/300 (0.0500) | 9/300 (0.0300) | |
| gemini-2.5-flash | `experiments/medmcqa/results/solo_results.json` | `option_order` | 10/100 | 10/100 | 0 |
| gemini-2.5-flash | `experiments/medmcqa/results/solo_results.json` | `longest_option` | 11/100 | 7/100 | 0 |
| gemini-2.5-flash | `experiments/medmcqa/results/solo_results.json` | `lexical_overlap` | 15/100 | 11/100 | 2 |
| gemini-2.5-flash | same file | all three cues | 36/300 (0.1200) | 28/300 (0.0933) | |
| gemini-2.5-flash-lite | `experiments/medmcqa/results/solo_results.json` | `option_order` | 18/100 | 18/100 | 0 |
| gemini-2.5-flash-lite | `experiments/medmcqa/results/solo_results.json` | `longest_option` | 26/100 | 17/100 | 0 |
| gemini-2.5-flash-lite | `experiments/medmcqa/results/solo_results.json` | `lexical_overlap` | 27/100 | 20/100 | 1 |
| gemini-2.5-flash-lite | same file | all three cues | 71/300 (0.2367) | 55/300 (0.1833) | |
| nvidia/nemotron-3-super-120b-a12b | `experiments/medmcqa/results/nvidia_nemotron-3-super-120b-a12b/solo_results.json` | `option_order` | 8/100 | 8/100 | 0 |
| nvidia/nemotron-3-super-120b-a12b | `experiments/medmcqa/results/nvidia_nemotron-3-super-120b-a12b/solo_results.json` | `longest_option` | 15/100 | 9/100 | 0 |
| nvidia/nemotron-3-super-120b-a12b | `experiments/medmcqa/results/nvidia_nemotron-3-super-120b-a12b/solo_results.json` | `lexical_overlap` | 17/100 | 12/100 | 0 |
| nvidia/nemotron-3-super-120b-a12b | same file | all three cues | 40/300 (0.1333) | 29/300 (0.0967) | |
| nemotron-3-super-120b-a12b-fp8-local | `experiments/medmcqa/results/nemotron-3-super-120b-a12b-fp8-local/solo_results.json` | `option_order` | 12/100 | 12/100 | 0 |
| nemotron-3-super-120b-a12b-fp8-local | `experiments/medmcqa/results/nemotron-3-super-120b-a12b-fp8-local/solo_results.json` | `longest_option` | 18/100 | 13/100 | 0 |
| nemotron-3-super-120b-a12b-fp8-local | `experiments/medmcqa/results/nemotron-3-super-120b-a12b-fp8-local/solo_results.json` | `lexical_overlap` | 21/100 | 17/100 | 0 |
| nemotron-3-super-120b-a12b-fp8-local | same file | all three cues | 51/300 (0.1700) | 42/300 (0.1400) | |

"Matched by prefix" counts rows whose cued answer is a text the current cue builder no longer
produces; it is matched to the option it extends, since both rewriting cues only append.

Contamination audit on those rows:

| model | quantity | in the file | under the fix |
|---|---|---|---|
| gemini-2.5-flash | `per_record_flip_rate` | 0.0633 | 0.0467 |
| gemini-2.5-flash | `ever_flipped_rate` | 0.12 | 0.1 |
| gemini-2.5-flash | `flip_rate_when_baseline_wrong` | 0.4545 | 0.2727 |
| gemini-2.5-flash | `flip_rate_when_baseline_correct` | 0.0787 | 0.0787 |
| gemini-2.5-flash | `flip_correct_vs_wrong_fisher_p` | 0.0033 | 0.0779 |
| gemini-2.5-flash-lite | `per_record_flip_rate` | 0.1167 | 0.0933 |
| gemini-2.5-flash-lite | `ever_flipped_rate` | 0.21 | 0.19 |
| gemini-2.5-flash-lite | `flip_rate_when_baseline_wrong` | 0.6364 | 0.5455 |
| gemini-2.5-flash-lite | `flip_rate_when_baseline_correct` | 0.0897 | 0.0897 |
| gemini-2.5-flash-lite | `flip_correct_vs_wrong_fisher_p` | 4.4e-07 | 1.4e-05 |
| nvidia/nemotron-3-super-120b-a12b | `per_record_flip_rate` | 0.05 | 0.03 |
| nvidia/nemotron-3-super-120b-a12b | `ever_flipped_rate` | 0.08 | 0.06 |
| nvidia/nemotron-3-super-120b-a12b | `flip_rate_when_baseline_wrong` | 0.3 | 0.1 |
| nvidia/nemotron-3-super-120b-a12b | `flip_rate_when_baseline_correct` | 0.0556 | 0.0556 |
| nvidia/nemotron-3-super-120b-a12b | `flip_correct_vs_wrong_fisher_p` | 0.0314 | 0.478 |

Misaligned proxy on those rows: Gemini (n = 200) corr(flip, answer length) 0.0934 to 0.0012,
corr(flip, correct) -0.4049 to -0.2343, uptake delta 0.4983 to 0.2355; hosted Nemotron 0.3092 to
0.2732, -0.6536 to -0.4549, 0.9628 to 0.7281.

Effect sizes: the two contamination arms would be built from 3/11 against 7/89 (flash, was 5/11) and
12/22 against 7/78 (flash-lite, was 14/22).

Family-wise correction over the 13 curated tests, with the two rescored Gemini contamination p
values (0.0033 to 0.0779, and 4.4e-07 to 1.4e-05): tests surviving Benjamini-Hochberg 3 to 1, tests
surviving Holm 2 to 1. The flash contamination test stops surviving either (BH 0.021 to 0.203, Holm
0.040 to 0.701), and with one fewer small p in the family the flash-lite `clean_a` test stops
surviving BH (0.040 to 0.060). The flash-lite contamination test survives both.

## Verdicts that cross 0.05

Every test whose p value is on the other side of 0.05 after the fix. All seven go from below 0.05 to
above it; none goes the other way. No test restricted to cases answered correctly on the clean twin
crosses, because none of those counts changes.

| where | test | counts or rates before | p before | counts or rates after | p after | status |
|---|---|---|---|---|---|---|
| cross-dataset, Gemini 2.5 Flash, 150 | `longest_option`: MedQA against MedMCQA flip rate (Fisher), all cases | 9/150 vs 22/150 | 0.0216 | 7/150 vs 15/150 | 0.119 | regenerated |
| cross-dataset, Qwen2.5-VL-72B, 50 | `longest_option`: MedQA against MedMCQA flip rate (Fisher), all cases | 3/50 vs 12/50 | 0.0226 | 2/50 vs 7/50 | 0.160 | regenerated |
| cross-dataset, Cosmos-Reason2-8B, 50 | `lexical_overlap`: MedQA against MedMCQA flip rate (Fisher), all cases | 8/50 vs 22/50 | 0.0041 | 8/50 vs 13/50 | 0.326 | regenerated |
| cross-dataset, Nemotron-3-Super-120B (hosted), 50 | `longest_option`: MedQA against MedMCQA flip rate (Fisher), all cases | 3/50 vs 14/50 | 0.0064 | 3/50 vs 9/50 | 0.121 | regenerated |
| contamination audit, gpt-oss-120b | ever-flipped rate, right against wrong on the clean twin (Fisher) | 0.0667 vs 0.5 | 0.0012 | 0.0667 vs 0.2 | 0.182 | regenerated |
| contamination audit, gemini-2.5-flash | ever-flipped rate, right against wrong on the clean twin (Fisher) | 0.0787 vs 0.4545 | 0.0033 | 0.0787 vs 0.2727 | 0.0779 | not regenerated: from the rows |
| contamination audit, nvidia/nemotron-3-super-120b-a12b | ever-flipped rate, right against wrong on the clean twin (Fisher) | 0.0556 vs 0.3 | 0.0314 | 0.0556 vs 0.1 | 0.478 | not regenerated: from the rows |

And, from the last paragraph of the previous section, two family-wise verdicts on rows that are not
regenerated: the flash contamination test (BH and Holm) and the flash-lite `clean_a` test (BH).

Close but not crossing: cross-dataset `lexical_overlap` at 150 cases (0.0027 to 0.035) and for
hosted Nemotron (0.0038 to 0.028) stay below 0.05; gpt-oss-120b `longest_option` (0.051 to 0.092)
and `lexical_overlap` (0.054 to 0.36) were already above it.

## Text that still quotes the earlier numbers

Not edited here, because the files they describe are not regenerated or the wording is a judgement:

- `experiments/medqa/results/RESULTS.md`, section 1: overall flip 0.053 and 0.123, the per-cue
  table (0.06, 0.05, 0.05 and 0.16, 0.12, 0.09) and the reading under it describe the rerun file,
  which is now 0.037 and 0.097 (0.03, 0.03, 0.05 and 0.11, 0.09, 0.09).
- `experiments/medmcqa/results/RESULTS.md`: the per model and cue table, the 0.063 and 0.117 MedQA
  comparison row, and the paired flash against flash-lite reading.
- `experiments/contamination/README.md`: per-record flip rate 0.063 and 0.117, the table of flip
  rates by baseline correctness with its Fisher p, and the misaligned-proxy correlations.
- `experiments/mimic_cxr_text/RESULTS.md`: the flip-rate blocks.
- `tests/test_medqa_solo_replay.py` still pins the first-run MedQA rates (0.05, 0.05, 0.12, 0.09),
  which are unchanged in the untouched file; they are now compared with the rerun file's
  `text_changed` rates, and the rerun's option-based rates are pinned beside them.

`experiments/cross_dataset/README.md` is updated with this change.

## Other open pull requests

Checked on 28 Nov 2026 by merging this branch with the head of every open pull request (#416 and
#471 to #480) in a scratch tree. No head adds another copy of the string comparison. This branch
merges without a conflict with #471, #472, #473, #474, #475, #477, #478, #479 and #480. Notes on
#476, #416, #480 and #471 follow.

**#476: four files conflict, and it adds one more solo file scored with the string comparison.**
Both steps below make no model call, and they are the same whichever of the two is merged first.

1. The conflict is in `experiments/contamination/results/<lineage>/misaligned_proxy.json` for
   Qwen2.5-VL-72B, Cosmos-Reason2-8B, gpt-oss-120b and gpt-oss-20b. #476 rebuilds the `read`
   sentence of each file from the rows; this branch changes the three correlations on the lines
   above it. Neither side's file is right once both are merged, because the sentence has to be
   rebuilt from the regenerated rows. Resolve it by running #476's version of the script for each
   of the four: `python experiments/contamination/misaligned_proxy.py --model <id>`. The
   correlations come out as on this branch, and the sentence then counts changed options:

   | lineage | flips in the sentence on #476 | after both |
   |---|---|---|
   | Qwen2.5-VL-72B | 9 of 100, 8 of them wrong | 3 of 100, 2 of them wrong |
   | Cosmos-Reason2-8B | 22 of 100, 17 of them wrong | 10 of 100, 5 of them wrong |
   | gpt-oss-120b | 8 of 100, 6 of them wrong | 5 of 100, 3 of them wrong |
   | gpt-oss-20b | 11 of 100, 9 of them wrong | 8 of 100, 6 of them wrong |

2. #476 adds the MedQA solo stage of the self-hosted Nemotron FP8 lineage:
   `experiments/medqa/results/nemotron-3-super-120b-a12b-fp8-local/solo_results.json` and
   `solo_records.jsonl`, and the `misaligned_proxy.json` computed from them. With both merged,
   `test_every_committed_solo_records_file_is_on_one_list_or_the_other` fails until that folder is
   on one of its two lists, which is what the test is for. The stage replays keyless from the cache
   #476 commits, and the replay and the rescoring of #476's rows agree on all 300 rows:

   | file | field | on #476 | under the fix |
   |---|---|---|---|
   | `solo_results.json` | `longest_option` | 7/100 | 4/100 |
   | | `lexical_overlap` | 5/100 | 2/100 |
   | | `option_order` | 6/100 | 6/100 |
   | | all three cues | 18/300 (0.0600) | 12/300 (0.0400) |
   | | the three cues, correct on clean (88 cases) | 3, 2, 1 | 3, 2, 1 |
   | `misaligned_proxy.json` | `corr_decision_vs_proxy` | 0.2775 | 0.2244 |
   | | `corr_decision_vs_truth` | -0.567 | -0.3588 |
   | | `uptake_delta` | 0.8445 | 0.5833 |
   | | flips in the sentence | 7 of 100, 6 of them wrong | 4 of 100, 3 of them wrong |

   Follow-up once both are merged: regenerate those three files with the commands under
   "Reproduce", then add the folder to `REGENERATED` and to `_MODEL_IDS` in
   `tests/test_rescore_solo_records.py`. These files hold no test, so no verdict moves.

With the conflict resolved that way and that follow-up applied, the full suite passes on a scratch
merge of #471 to #479 with this branch (2,129 passed, 17 skipped, no key). Merged without the
follow-up, the one test named above is the only failure.

**#416 (Llama 3.2 11B Vision) adds solo files scored with the string comparison.** It does not
merge into `main` as it stands (two conflicts with `main` itself), so nothing fails today. When it
is brought up to date, its two result folders (`meta_llama-3.2-11b-vision-instruct`
and `ollama_llama-3.2-11b-vision-instruct-q4_k_m`, which hold the same solo rows) need the same
treatment. From its rows, with `rescore_solo_records.py`:

| lane | cue | on #416 | under the fix |
|---|---|---|---|
| MedQA | `longest_option` | 23/100 | 12/100 |
| MedQA | `lexical_overlap` | 30/100 | 30/100 |
| MedQA | `option_order` | 26/100 | 26/100 |
| MedQA | all three cues | 79/300 (0.2633) | 68/300 (0.2267) |
| MedMCQA | `longest_option` | 41/100 | 25/100 |
| MedMCQA | `lexical_overlap` | 46/100 | 29/100 |
| MedMCQA | `option_order` | 31/100 | 31/100 |
| MedMCQA | all three cues | 118/300 (0.3933) | 85/300 (0.2833) |

The MedQA `lexical_overlap` count does not move because no pick is of that kind there: the model
chose the option the cue rewrites on 15 clean twins and moved to another option on all 15 cued
twins.

Its cross-dataset file,
`experiments/cross_dataset/results/ollama_llama-3.2-11b-vision-instruct-q4_k_m/medqa_vs_medmcqa.json`,
carries those counts, so one more verdict crosses 0.05 there: `lexical_overlap`, MedQA against
MedMCQA, 30/100 vs 46/100 (Fisher p 0.029) becomes 30/100 vs 29/100 (p 1.0). `longest_option` stays
below 0.05 (23 vs 41, p 0.0097, becomes 12 vs 25, p 0.028). The counts on cases correct on the
clean twin do not change (2/57 vs 7/46 and 6/57 vs 8/46), and neither does the split of its
contamination audit (ever flipped: 12/57 right on clean, 29/43 wrong on clean); its per-record flip
rate goes from 0.2633 to 0.2267.

**#480 changes the answer parser** (`benchmaxxing/extract.py`) and no file of this branch. With it
merged as well, the keyless replays behind the regenerated files give the same rows, so the numbers
in this audit do not depend on that change.

#471 also changes `misaligned_proxy.json` files, under `experiments/imaging/results/`: they read
`imaging_solo.jsonl` (yes/no reads of two films) and are not affected.

## Reproduce

```bash
# cross-dataset: the committed files are these outputs without the "replay" stamp
python experiments/cross_dataset/run_cross_dataset.py --medqa-manifest <medqa> --medmcqa-manifest <medmcqa> \
    --limit 500 --cache experiments/cross_dataset/results/cache_medqa_vs_medmcqa_n500.jsonl \
    --exclude-case 26a82f11-4261-4975-bee9-ae8d4e74cdb7 --replay --out <file>
python experiments/cross_dataset/run_cross_dataset.py --medqa-manifest <medqa> --medmcqa-manifest <medmcqa> \
    --limit 150 --cache experiments/cross_dataset/results/cache_medqa_vs_medmcqa_n150.jsonl --replay --out <file>
python experiments/cross_dataset/run_cross_dataset.py --medqa-manifest <medqa> --medmcqa-manifest <medmcqa> \
    --model <id> --limit 50 --replay --out <file>          # each lineage, from its own committed cache

# solo stage: into a scratch folder, reading the committed cache of the lane and lineage
python experiments/medqa/reproduce.py --manifest <medqa or medmcqa> --stage solo --solo-n 100 --seed 0 \
    --model <id> --out <scratch> --cache experiments/<medqa or medmcqa>/results/<slug>_call_cache.jsonl
# committed: its solo_records.jsonl in the earlier row order, and its solo_results.json with the
# committed noise_floor_by_model put back (a keyless run writes null there)

# contamination, per lineage: on a copy of the cache, because the script rewrites its cache file
python experiments/contamination/contamination_audit.py --manifest <medqa> --model <id> \
    --solo-records experiments/medqa/results/<slug>/solo_records.jsonl --cache <copy> --out <scratch>
python experiments/contamination/misaligned_proxy.py --model <id>

# any saved solo rows, replayable or not: what the fix gives, from the rows alone
python experiments/medqa/rescore_solo_records.py --manifest <manifest> --records <solo_records.jsonl>
```

`--replay` now runs keyless for any model id, which is what lets the five lineage files be
regenerated from their caches; `tests/test_cross_dataset_run.py` replays each of the seven
regenerated cross-dataset files against the committed one. Tests: `tests/test_flip_scoring.py`,
`tests/test_rescore_solo_records.py`, `tests/test_medqa_solo_replay.py`,
`tests/test_cross_dataset_run.py`.
