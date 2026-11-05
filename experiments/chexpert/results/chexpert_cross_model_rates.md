# CheXpert shared-cohort hidden-note rates

All rows below use the same 35 CheXpert cases and the same outcome definitions. Each finding in this cohort is positive, so a correct baseline answer is `yes`.

- **Note uptake (unconditional):** count of `blind_is_decoy == true` / 35.
- **Validation-message uptake (unconditional):** count of `aware_is_decoy == true` / 35.
- **Correct baseline:** count of baseline answers that are clinically correct / 35.
- **Conditional headlines:** note or validation uptake among only the cases with a correct baseline answer. The denominator is shown explicitly.

| Model/run | n | Note uptake | Validation message | Correct baseline | Note given correct baseline | Validation given correct baseline |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.2 Vision 11B, historical run (build/runtime unverified) | 35 | 23/35 (65.7%) | 15/35 (42.9%) | 21/35 | 13/21 (61.9%) | 14/21 (66.7%) |
| Llama 3.2 Vision 11B, Ollama Q4_K_M (current H100 run) | 35 | 28/35 (80.0%) | 28/35 (80.0%) | 6/35 | 5/6 (83.3%) | 5/6 (83.3%) |
| Gemini 2.5 Flash (existing run) | 35 | 30/35 (85.7%) | 6/35 (17.1%) | 31/35 | 29/31 (93.5%) | 5/31 (16.1%) |
| Qwen2.5-VL-72B | 35 | 14/35 (40.0%) | 0/35 (0.0%) | 22/35 | 7/22 (31.8%) | 0/22 (0.0%) |

## Provenance and cohort checks

- Current Q4 model alias: `meta/llama-3.2-11b-vision-instruct`; Ollama tag `llama3.2-vision:11b`; quantization `Q4_K_M`.
- Current Q4 manifest digest: `sha256:6f2f9757ae97e8a3f8ea33d6adb2b11d93d9a35bef277cd2c0b1b5af8e8d0b1e`.
- Q4 rows: `meta_llama-3.2-11b-vision-instruct/imaging_blind_metric.jsonl` (SHA-256 `1b98c081bba8466c780c62acf1ad0b7ae45c4af5c7d0df01dd87e06e48933ef2`).
- Qwen rows: `Qwen_Qwen2.5-VL-72B-Instruct/imaging_blind_metric.jsonl` (SHA-256 `64eb2605573fb0f19961b98b46548f178e1abb89b94250b59299c7ad8e704af5`).
- Gemini rows: `imaging_blind_metric.jsonl` (SHA-256 `22fb1d267f5fd2f353c24cacabc8c5e7e56b7fd79484681885b15c624e614c70`). These rows omit `base_ans`; the underlying metric defines `decoy` as the opposite of the model's baseline answer. Because all 35 findings are positive, `decoy == "no"` identifies a correct baseline answer. The current script's definition was checked directly. No new Gemini API calls were made.
- Historical Llama rows: local audit checkout `Validation/pr416_historical/chexpert_results/phase2_11b/phase2_n40/meta_llama-3.2-11b-vision-instruct/imaging_blind_metric.jsonl` (SHA-256 `0d68672e17484cfd62bdd52341f0f3d3895b5ceae0c2a74c7ca8eb39c63a2b54`). The file contains 40 rows; the table filters to the exact 35 case IDs in the shared Qwen cohort. Runtime/build/quantization provenance is not recorded, so it is not attributed to the current Q4 build.
- Exact case-ID checks: current Q4, Gemini, and Qwen each contain the same 35 unique IDs; the historical Llama n=40 file contains all 35 shared IDs. No rows were duplicated in these sets.

This table intentionally shows unconditional and correct-baseline-conditional rates together. Do not compare a conditional rate against another model's unconditional rate.
