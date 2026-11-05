# Llama 3.2 Vision 11B Q4_K_M parity run — final audit

**Status:** complete; no push or merge performed.
**Run branch:** `experiments/llama32-vision-q4km-parity`
**Base:** `origin/main` at `0eb8f23d9a62c32f3b0eac52914aac9b32382d4a` (fetch confirmed branch HEAD is the same commit).
**Model:** Ollama `llama3.2-vision:11b`, quantization `Q4_K_M`; alias `meta/llama-3.2-11b-vision-instruct`. Manifest digest: `sha256:6f2f9757ae97e8a3f8ea33d6adb2b11d93d9a35bef277cd2c0b1b5af8e8d0b1e`; weight-layer digest: `sha256:9999d473417a8e179d993498195be5f42cab963acc75f4a6b15d981e8b68abed`. Full model metadata is in `meta_llama-3.2-11b-vision-instruct.json`.

## Completed scope

| Battery | Completed arms | Verified cohort/result scope |
|---|---:|---|
| Hidden-note | 13 | MedQA n=40 and n=100; NIH n=35; CheXpert n=35; requested controls and companion prompts. IDs match the comparator sets. |
| MedQA text committee | 31/31 | All requested arms, including per-case reproduction files. |
| MedMCQA text committee | 25/25 | All requested arms. Model-scoped result rows are under `experiments/medmcqa/results/meta_llama-3.2-11b-vision-instruct/`; reusable cache files are model-prefixed in the MedQA cache location used by the lane helper. |
| NIH ChestX-ray14 imaging | 22/22 | All requested arms and variants. Every output ID set matched its saved cohort; n=35 except the scale arm n=150. |
| CheXpert imaging | 8/8 | Shared-cohort arms n=35; `device_absent`, `natural_cues`, `natural_independent`, and `system_flag` n=150. The overwritten base cascade was reconstructed from its 105-entry call cache, with 35 exact case rows. |
| SUPPORT2 | 5/5 | All n=120 arms; referee control rows were retained as designed. |
| Referee | 5/5 | All n=40 arms; deployable rows include the `::clean` control for each case. |
| Multi-round cascade | 2/2 | n=40 and n=120; the n=40 result remains the canonical `multi_round.jsonl`/`multi_round_summary.json`. |
| Contamination audit | 1/1 | n=100 exact selected cases. |
| Model dependence | 1/1 | `cascade_C_flash`, 22 selected comparator IDs. |

MIMIC-CXR was excluded as requested. Total requested arms verified: 113. No unverified old SUPPORT2 outputs were reused; the five Q4 arms were regenerated. The 35-case CheXpert cross-model rates and definitions are recorded in `../chexpert/results/chexpert_cross_model_rates.md`.

## Cohorts, caches, and provenance

- The 19 follow-on output groups (CheXpert 5, SUPPORT2 5, Referee 5, multi-round 2, contamination 1, model-dependence 1) passed exact case-ID and row-multiplicity checks. Every required Q4 call-cache file for those groups exists and is nonempty. The CheXpert base cascade was replayed from cache without inference.
- All 22 NIH imaging and all 8 CheXpert imaging result files matched their corresponding case-ID manifests exactly. The pre-existing `imaging_solo_lite` result and cache remain complete at n=35; a later redundant attempt exited 143 before replacing them.
- Earlier run-log failures for NIH judge/referee text-only and majority pressure were rerun successfully. The judge/referee arms have both the text-only and film variants; majority pressure uses the comparator's `watermark` cue.
- Across model-specific result/call JSONL files, 267 files and 50,354 nonblank rows parsed with zero malformed lines. A credential-pattern scan found no Hugging Face, OpenAI-style, Google, or Slack tokens.
- Existing degeneracy labels remain: `reread_control_rate_DEGENERATE` in MedQA and MedMCQA `clean_a` summaries, and `referees_vs_adoption_planted_only_DEGENERATE` in the MedMCQA deployable-referee summary. No new declared gaps were added.
- Historical `phase2_11b/phase2_n600/` contains only `run.log`, so there were no n=600 result files to move. The historical n=40 Llama result used in the CheXpert comparison is explicitly labeled build/runtime unverified and kept separate from the current Q4 lineage.

## Verification and resource cleanup

- Final `.venv/bin/python -m pytest tests/test_lineage_parity.py -q`: **2 passed**. No new declared gaps.
- `git diff --check`: passed.
- Final lineage test initially caught an unnecessary `multi_round_summary_n40.json` alias. The duplicate was removed, the run scripts were adjusted so they do not recreate it, and the test then passed without modifying the test or adding a gap.
- The H100 supervisor completed the queued follow-on battery at `2026-11-04T13:56:43Z UTC` and released GPU 1. Final lease state was free, with 0 MiB on GPU 1; no Ollama model remained loaded there.
