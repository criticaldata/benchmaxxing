# Ollama Llama 3.2 Vision 11B Q4_K_M imaging results

- **Lineage id:** `ollama_llama-3.2-11b-vision-instruct-q4_k_m`
- **Hosted NIM slug:** `meta_llama-3.2-11b-vision-instruct`
- **Ollama tag:** `llama3.2-vision:11b` (`Q4_K_M`)
- **Audited source head:** `6ace8cc5ba2363a4c6fbbe325c3f8cfa583f722f`; result commit `1bb6828a750e15a91043dd2f2432ef5d93f83d2f`.

The Q4 result directories use the Ollama lineage id; the API model alias in row payloads is retained as `meta/llama-3.2-11b-vision-instruct`. It does not merge these observations with hosted NIM. The 30 image-arm cohorts in the audit case-set file all match their result case IDs exactly.

The NIH ChestX-ray14 shared reference arm is **1 of 35 correct (1/35; 2.9%)**, as requested. The table reports each arm’s own clean baseline because some arms were independently re-asked with different prompt/temperature settings; those per-arm baselines can vary. For `imaging_multi_round` and `imaging_referee`, the baseline is taken from the exact same-ID clean comparator shown in the JSON baseline file. CheXpert hidden-note baseline is reported separately as **6/35 correct (17.1%)**.

## NIH ChestX-ray14 — 22 audited arms

| Imaging arm | Main result numbers | Clean baseline accuracy for this arm |
|---|---|---:|
| `imaging_solo` | Cue flips: cable 0/35, corner tag 0/35, watermark 2/35, laterality 0/35; clean-read noise floor 7/35. | 1/35 (2.9%) |
| `imaging_solo_lite` | Cue flips: cable 0/35, corner tag 0/35, watermark 2/35, laterality 0/35; clean-read noise floor 8/35. | 1/35 (2.9%) |
| `imaging_noise_floor` | Repeated clean-read disagreement: 7/35. | 1/35 (2.9%) |
| `imaging_noise_floor_lite` | Repeated clean-read disagreement: 8/35. | 1/35 (2.9%) |
| `imaging_cascade` | Shared adoption 35/35; isolated adoption 34/35; contagion 1/35. | 1/35 (2.9%) |
| `imaging_cascade_cable` | Shared adoption 35/35; isolated adoption 34/35; contagion 1/35. | 1/35 (2.9%) |
| `imaging_cascade_corner_tag` | Shared adoption 35/35; isolated adoption 34/35; contagion 1/35. | 1/35 (2.9%) |
| `imaging_cascade_laterality` | Shared adoption 35/35; isolated adoption 34/35; contagion 1/35. | 1/35 (2.9%) |
| `imaging_cascade_scale` | Shared adoption 150/150; isolated adoption 140/150; contagion 10/150. | 6/150 (4.0%) |
| `imaging_cue_combo` | Eligible cases: 1/35. Watermark flip 1/1, corner-tag flip 0/1, combined flip 1/1. | 1/35 (2.9%) |
| `imaging_cue_families` | Cue flips: rotation 2/35, compression 1/35, brightness 0/35, soft-tissue 0/35; clean-read noise floor 3/35. | 1/35 (2.9%) |
| `imaging_cue_families_noise` | Repeated clean-read disagreement: 3/35. | 1/35 (2.9%) |
| `imaging_judge_referee (legacy text-only)` | Judge TP 0, FP 0, FN 1, TN 34; naive gate TP 1, FP 34, FN 0, TN 0. The audit marks this as a legacy text-only arm, not a judge measurement. | 1/35 (2.9%) |
| `imaging_judge_referee_film` | Judge sees the film: TP 0, FP 0, FN 1, TN 34; naive gate TP 1, FP 34, FN 0, TN 0. | 1/35 (2.9%) |
| `imaging_majority_pressure` | A single wrong peer was adopted on 35/35 cases. | 1/35 (2.9%) |
| `imaging_matched_temp` | Watermark signal flips: temp 0 = 2/35; temp 1 = 10/35. At temp 1, clean-read floor 7/35 and matched signal above floor 3/35. | 1/35 (2.9%) |
| `imaging_multi_round` | Shared adoption by rounds 1–3: 35/35, 35/35, 35/35. Isolated adoption by rounds 1–3: 35/35, 29/35, 28/35. | 1/35 (2.9%) |
| `imaging_peer_size_curve` | Adoption by wrong-peer count: 1 peer 35/35; 2 peers 35/35; 4 peers 35/35. | 0/35 (0.0%) |
| `imaging_polarity` | Positive finding: 0/35 clean-correct; absent finding: 35/35 clean-correct. Cue flips: positive yes→no 0/0 eligible; negative no→yes 0/140. | Present 0/35 (0.0%); absent 35/35 (100.0%) |
| `imaging_referee` | Referee TP 1, FP 7, FN 0, TN 27; naive gate TP 1, FP 34, FN 0, TN 0. | 1/35 (2.9%) |
| `imaging_strength_cascade` | Opacity 0.15: solo flips 1/35, isolated/shared adoption 35/35, contagion 0/35. Opacity 0.30: same. Opacity 0.45: solo flips 2/35, isolated 34/35, shared 35/35, contagion 1/35. | 1/35 (2.9%) |
| `imaging_system_flag` | Shared adoption 35/35; isolated adoption 35/35; contagion 0/35. | 0/35 (0.0%) |

## CheXpert — 8 audited arms

| Imaging arm | Main result numbers | Clean baseline accuracy for this arm |
|---|---|---:|
| `imaging_solo` | Cue flips: cable 1/35, corner tag 1/35, watermark 0/35, laterality 1/35; clean-read noise floor 12/35. | 2/35 (5.7%) |
| `imaging_cascade` | Shared adoption 35/35; isolated adoption 32/35; contagion 3/35. | 3/35 (8.6%) |
| `imaging_matched_temp` | Watermark signal flips: temp 0 = 0/35; temp 1 = 9/35. Temp-1 clean-read floor 12/35; matched signal above floor −3/35. | 2/35 (5.7%) |
| `imaging_noise_floor` | Repeated clean-read disagreement: 12/35. | 2/35 (5.7%) |
| `device_absent_cascade` | Shared adoption 150/150; isolated adoption 145/150; contagion 5/150; contagion among baseline-correct cases 5/5. | 5/150 (3.3%) |
| `natural_cues_cascade` | Shared adoption 143/150; isolated adoption 33/150; contagion 110/150; contagion among baseline-correct cases 110/117. | 117/150 (78.0%) |
| `natural_independent_cascade` | Shared adoption 148/150; isolated adoption 30/150; contagion 118/150; contagion among baseline-correct cases 118/120. | 120/150 (80.0%) |
| `system_flag` | Shared adoption 145/150; isolated adoption 24/150; contagion 121/150. | 126/150 (84.0%) |

## Hidden-note and matched-baseline imaging outcomes

| Imaging arm | Main result numbers | Clean baseline accuracy for this arm |
|---|---|---:|
| `NIH blind metric` | Baseline decoy 0/35; blind-note decoy uptake 20/35 (57.1%); test-aware uptake 0/35. | 1/35 (2.9%) |
| `NIH matched baseline/reason` | Reason-only decoy 2/35; blind-note decoy 20/35; reason request flip 9/35; repeated-read disagreement on 9/35 cases. | 1/35 (2.9%) |
| `CheXpert hidden-note blind metric` | Blind-note decoy uptake 28/35 (80.0%); validation-message uptake 28/35 (80.0%). | 6/35 (17.1%) |
| `CheXpert matched baseline/reason` | Reason-only decoy 16/35; blind-note decoy 28/35; reason request flip 27/35; repeated-read disagreement on 8/35 cases. | 6/35 (17.1%) |
| `NIH matched multi-condition prompts` | First repeat: reason-only decoy 7/35; note decoy 21/35; paired net note-minus-reason 14/35. Across five repeats: reason 33/175; note 117/175. | 1/35 (2.9%) |
| `CheXpert matched multi-condition prompts` | First repeat: reason-only decoy 15/35; note decoy 31/35; paired net note-minus-reason 16/35. Across five repeats: reason 74/175; note 152/175. | 8/35 (22.9%) |

## Provenance and scope

- Per-arm baselines and exact result/case-set paths: `experiments/model_metadata/q4_image_baseline_accuracy.json`.
- Source audit and cohort IDs: `experiments/model_metadata/q4_final_audit.md`, `experiments/model_metadata/q4_image_case_sets.json`.
- CheXpert cross-model hidden-note definitions: `docs/ollama_q4_chexpert_cross_model_rates.md`.
- Matched multi-condition summaries: `experiments/imaging/results/matched/ollama_llama-3.2-11b-vision-instruct-q4_k_m/matched_conditions_summary.json` and `experiments/chexpert/results/matched/ollama_llama-3.2-11b-vision-instruct-q4_k_m/matched_conditions_summary.json`.
- MIMIC-CXR is excluded; no MIMIC-CXR results or gap declarations are imported.
- Percentages are count/denominator values; each baseline in the table is the model’s clean read on that arm’s matching cohort. The polarity arm reports separate positive-finding and absent-finding baselines.
