# Keyless replay audit (24 Nov 2026)

Every committed arm of the MedMCQA, SUPPORT2, MIMIC-CXR text and imaging lanes (NIH, CheXpert,
MIMIC-CXR image), for every lineage that has committed results, was rerun with no API key at its
committed size and compared with its committed summary (and rows, as a set, where a rows file exists).
The audited tree is `main` at `1cb76ad`. No model call was made: a cache miss stops a run, it never
calls out. The MedQA lane was audited earlier and is pinned by `tests/test_medqa_arms_replay.py`.

Where each lane ran:

- MedMCQA and SUPPORT2: on a laptop (Python 3.11, Pillow 13.0.0), with the 500-row MedMCQA manifest
  (sha256 `5b2ce971...`, the one `experiments/medmcqa/fetch_dev.py` checks) and the committed SUPPORT2
  manifest.
- MIMIC-CXR text and image: on ORCD only (patient data stays there), with the gitignored manifests and
  caches of the ORCD clone linked into an exported tree. Only statuses and field names left ORCD.
- NIH and CheXpert imaging: on ORCD (Python 3.12, Pillow 12.3.0) with the films in the shared
  `benchmaxxing-films` folder. The image caches key on the PNG bytes the runner re-encodes, so a cache
  only replays on a machine whose image stack produces the same bytes. The same NIH Qwen arms that all
  missed on the laptop (Pillow 13.0.0) replay exactly on ORCD (Pillow 12.3.0, the version of the node
  that made them); the Gemini imaging caches miss on both machines. Those misses are recorded as
  cache key mismatches, not as drift.

## Summary

| Lane | exact | equal on committed fields | equal up to row order | equal except live noise floor | equal except bootstrap ci | drift | cache miss | no committed cache | no cache found | env missing package | not assessed films not found | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medmcqa | 103 | 1 | 3 | 3 | 0 | 15 | 8 | 28 | 0 | 0 | 0 | 161 |
| support2 | 35 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 38 |
| mimic_text | 14 | 0 | 0 | 0 | 0 | 0 | 37 | 0 | 1 | 0 | 0 | 52 |
| nih | 40 | 1 | 0 | 7 | 2 | 0 | 19 | 0 | 0 | 0 | 0 | 69 |
| chexpert | 10 | 3 | 0 | 2 | 0 | 0 | 11 | 0 | 0 | 1 | 0 | 27 |
| mimic_image | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 12 |
| all | 203 | 8 | 3 | 12 | 2 | 15 | 75 | 28 | 1 | 1 | 11 | 359 |

Status meanings: *exact* means every committed field and every row came back identical; *equal on
committed fields* means the code now also writes a field or call counter the committed file lacks;
*equal up to row order* means the same per-case rows in another order; *equal except live noise floor*
means everything matched except a noise floor that is an uncached temperature-1 control by design, so
a keyless run leaves it empty; *equal except bootstrap CI* means the point estimates match and one
bootstrap bound differs in the third decimal; *cache miss* means a prompt the run needs is not in the
cache it reads (for imaging, a cache key mismatch); *no committed cache* means the lineage's cache for
that arm is not in the repository.

## Findings

1. **Fifteen Gemini MedMCQA arms did not match their committed files.** Each replays the same way on
   three runs in a row (identical summaries and rows), so they were not changing between runs; they
   disagreed with what was committed. The summaries were committed on 30 Jul (`1f28ce4`, `3f27fbc`) and
   the caches on 1 Aug (`f872470`); the same disagreement remains with four earlier versions of
   `benchmaxxing/extract.py`, so the committed rows were not computed from these cached replies. On
   this branch the fifteen summaries and rows are rewritten from the committed caches (0 calls), and
   `stats_reconciliation_summary.json` and the hosted nemotron `cross_lineage_summary.json` are
   regenerated from them. The Holm and BH survivors do not change. Against `main` this changes 221
   leaf fields in 20 JSON files, 174 of them numbers: 98 in the 15 summaries, 29 in the
   reconciliation and 47 in the `gemini` column of the hosted nemotron comparison file (see "Files and
   fields that change against `main`"). `referee_deployable` only gains the
   `extra_requery_calls_needed` field and its current `note` wording.
2. **Gemini MedMCQA `orchestrator_failure` cannot be refreshed without calls.** It replays up to
   `--n 41` and then needs a `gemini-2.5-flash` prompt that is not cached, for a case the committed rows
   do not hold. It stays as committed and is not pinned.
3. **The hosted nemotron MedMCQA lane has no committed cache for 14 committee arms** (only
   `attributed_tier`, `authority_ladder`, `blind_metric`, `seed_confidence` and `super_additivity`
   caches are in the repository). Its other arms are committed results that cannot be replayed.
4. **Gemini MedMCQA solo and `clean_a` need `gemini-2.5-flash` prompts, and `scale_c` needs
   `gemini-2.5-flash-lite` prompts, that are not in the committed `call_cache.jsonl`** (`break_it`
   replays exactly with the new `--cache` flag); the Gemini MedMCQA cascade replays but differs on two of 20 boards
   (mean contagion 0.025 committed, 0.05 replayed). Not refreshed here: its transcripts feed
   `onset_distribution.json`.
5. **MIMIC-CXR text:** the gpt-oss-120b and gpt-oss-20b arms replay exactly (14 arms). The Gemini cache
   on ORCD (`results/call_cache.jsonl`) does not cover the committed Gemini prompts, the per-arm Gemini
   caches (`break_it_d`, `deliberation_framing`, `referee_call_cache`) are not on ORCD, and nemotron and
   Qwen have no MIMIC-CXR text cache there. No `results_n600` cache exists on ORCD, so no n=600 file
   replays, and the abstention-aware n=600 file for Qwen cannot be produced without model calls.
6. **MIMIC-CXR image:** the manifests' JPGs were not found under the ORCD roots probed
   (`lceli_shared/data/mimic-cxr-images` and four others), so the image arms are not assessed at this
   depth; the runs that started covered fewer cases than committed and are not counted as findings.
   The text-only judge arm replays exactly.
7. **Imaging:** on ORCD 40 NIH and 10 CheXpert arm runs replay exactly, nearly all of them Qwen2.5-VL
   and Qwen3.8; the Gemini image caches miss (cache key mismatch). `imaging_judge_referee.py` and
   `matched_baseline_reason.py` now write extra fields, and the 150-film scale bootstrap CI bound moves
   in the third decimal on this Python stack.

## Default flags (MedQA home lane)

Each shared script was run with only `--manifest` (its default `--n` and cache), writing to a temporary
directory. Four defaults did not match the MedQA lane's committed sizes; they are aligned on this branch.
`empty_content_by_condition.py`, which runs every runner at its default size, now passes the earlier
sizes explicitly so its committed outputs keep replaying. MedMCQA and SUPPORT2 reuse the same scripts at
other sizes, so their exact flags are pinned in `tests/test_medmcqa_arms_replay.py` and
`tests/test_support2_arms_replay.py`.

| Script | Default --n on main | Status on main with only --manifest | Committed size | On this branch |
|---|---|---|---|---|
| `experiments/medqa/attributed_tier.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/authority_ladder.py` | 60 | drift | 120 | default set to 120, replays exactly |
| `experiments/medqa/committee_size_sweep.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/deliberation_framing.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/hierarchy_temp.py` | 40 | exact | 40 | unchanged |
| `experiments/medqa/leader_as_auditor.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/majority_pressure.py` | 25 | drift | 40 | default set to 40, replays exactly |
| `experiments/medqa/orchestrator_failure.py` | 80 | drift | 100 | default set to 100, replays exactly |
| `experiments/medqa/paraphrase_robustness.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/plausible_distractor.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/rationale_validity.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/seed_confidence.py` | 100 | exact | 100 | unchanged |
| `experiments/medqa/seed_timing.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/super_additivity.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/temperature_sensitivity.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/true_peer_control.py` | 60 | drift | 150 | default set to 150, replays exactly |
| `experiments/medqa/unanimity_break.py` | 150 | exact | 150 | unchanged |
| `experiments/medqa/dose_response.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/text_cue_types.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/test_awareness.py` | 120 | exact | 120 | unchanged |
| `experiments/medqa/hierarchy_dominance.py` | 40 | exact | 40 | unchanged |
| `experiments/referee/referee_deployable.py` | 40 | exact | 40 | unchanged |
| `experiments/referee/referee_judge.py` | 40 | exact | 40 | unchanged |
| `experiments/referee/requery_argument_only.py` | 120 | exact | 120 | unchanged |
| `experiments/blind_metric/blind_metric.py` | 40 | exact | 40 | unchanged |
| `experiments/blind_metric/check_warn_2x2.py` | 40 | exact | 40 | unchanged |

The review's list of 62 replay commands was not available to this audit, so its 26 failing commands
could not be matched one by one; this table and the pinned tests replace it.

## Every arm

| Lane | Lineage | Arm | Size or flags | Committed value (main) | Replayed value | Status | Machine |
|---|---|---|---|---|---|---|---|
| support2 | gemini-2.5-flash-lite | support2_solo | --n 120 | `support2_solo_summary.json` clean_accuracy_excl_abstentions = 0.7130434782608696 | same | replays exactly | laptop |
| support2 | gemini-2.5-flash-lite | support2_cascade | --n 120 | `support2_cascade_summary.json` bare_accuracy_excl_abstentions = 0.7130434782608696 | same | replays exactly | laptop |
| support2 | gemini-2.5-flash-lite | support2_cascade_strength | --n 120 | `support2_cascade_strength_summary.json` abstention_rate = 0.041666666666666664 | same | replays exactly | laptop |
| support2 | gemini-2.5-flash-lite | support2_referee | --n 120 | `support2_referee_summary.json` abstention_rate = 0.04166666666666663 | same | replays exactly | laptop |
| support2 | gemini-2.5-flash-lite | support2_referee_judge | --n 120 | `support2_referee_judge_summary.json` abstention_rate = 0.041666666666666664 | same | replays exactly | laptop |
| support2 | gemini-2.5-flash-lite | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| support2 | gemini-2.5-flash-lite | check_warn_2x2 | --n 40 | `check_warn_2x2_summary.json` cells.none.decoy_uptake = 0.4194 | same | replays exactly | laptop |
| support2 | gemini-2.5-flash-lite | matched_baseline_reason | --n 40 | `matched_baseline_reason_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | laptop |
| support2 | gemini-2.5-flash | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-120b | support2_solo | --n 120 | `support2_solo_summary.json` clean_accuracy_excl_abstentions = 0.6666666666666666 | same | equal on every committed field (the code now writes extra fields or call counters) | laptop |
| support2 | openai/gpt-oss-120b | support2_cascade | --n 120 | `support2_cascade_summary.json` bare_accuracy_excl_abstentions = 0.6666666666666666 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-120b | support2_cascade_strength | --n 120 | `support2_cascade_strength_summary.json` abstention_rate = 0.0 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-120b | support2_referee | --n 120 | `support2_referee_summary.json` abstention_rate = 0.0 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-120b | support2_referee_judge | --n 120 | `support2_referee_judge_summary.json` abstention_rate = 0.0 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-120b | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-120b | matched_baseline_reason | --n 40 | `matched_baseline_reason_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-20b | support2_solo | --n 120 | `support2_solo_summary.json` clean_accuracy_excl_abstentions = 0.7 | same | equal on every committed field (the code now writes extra fields or call counters) | laptop |
| support2 | openai/gpt-oss-20b | support2_cascade | --n 120 | `support2_cascade_summary.json` bare_accuracy_excl_abstentions = 0.7 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-20b | support2_cascade_strength | --n 120 | `support2_cascade_strength_summary.json` abstention_rate = 0.0 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-20b | support2_referee | --n 120 | `support2_referee_summary.json` abstention_rate = 0.0 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-20b | support2_referee_judge | --n 120 | `support2_referee_judge_summary.json` abstention_rate = 0.016666666666666666 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-20b | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| support2 | openai/gpt-oss-20b | matched_baseline_reason | --n 40 | `matched_baseline_reason_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | laptop |
| support2 | nvidia/nemotron-3-super-120b-a12b | support2_solo | --n 120 | `support2_solo_summary.json` clean_accuracy_excl_abstentions = 0.7166666666666667 | same | replays exactly | laptop |
| support2 | nvidia/nemotron-3-super-120b-a12b | support2_cascade | --n 120 | `support2_cascade_summary.json` bare_accuracy_excl_abstentions = 0.7166666666666667 | same | replays exactly | laptop |
| support2 | nvidia/nemotron-3-super-120b-a12b | support2_cascade_strength | --n 120 | `support2_cascade_strength_summary.json` abstention_rate = 0.0 | same | replays exactly | laptop |
| support2 | nvidia/nemotron-3-super-120b-a12b | support2_referee | --n 120 | `support2_referee_summary.json` abstention_rate = 0.0 | same | replays exactly | laptop |
| support2 | nvidia/nemotron-3-super-120b-a12b | support2_referee_judge | --n 120 | `support2_referee_judge_summary.json` abstention_rate = 0.0 | same | replays exactly | laptop |
| support2 | nvidia/nemotron-3-super-120b-a12b | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| support2 | Qwen/Qwen2.5-VL-72B-Instruct | support2_solo | --n 120 | `support2_solo_summary.json` clean_accuracy_excl_abstentions = 0.7166666666666667 | same | equal on every committed field (the code now writes extra fields or call counters) | laptop |
| support2 | Qwen/Qwen2.5-VL-72B-Instruct | support2_cascade | --n 120 | `support2_cascade_summary.json` bare_accuracy_excl_abstentions = 0.7166666666666667 | same | replays exactly | laptop |
| support2 | Qwen/Qwen2.5-VL-72B-Instruct | support2_cascade_strength | --n 120 | `support2_cascade_strength_summary.json` abstention_rate = 0.0 | same | replays exactly | laptop |
| support2 | Qwen/Qwen2.5-VL-72B-Instruct | support2_referee | --n 120 | `support2_referee_summary.json` abstention_rate = 0.0 | same | replays exactly | laptop |
| support2 | Qwen/Qwen2.5-VL-72B-Instruct | support2_referee_judge | --n 120 | `support2_referee_judge_summary.json` abstention_rate = 0.0 | same | replays exactly | laptop |
| support2 | Qwen/Qwen2.5-VL-72B-Instruct | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| support2 | Qwen/Qwen2.5-VL-72B-Instruct | matched_baseline_reason | --n 40 | `matched_baseline_reason_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | laptop |
| support2 | nemotron-3-super-120b-a12b-fp8-local | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| support2 | nemotron-3-super-120b-a12b-fp8-local | matched_baseline_reason | --n 40 | `matched_baseline_reason_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | laptop |
| medmcqa | gemini-2.5-flash-lite | attributed_tier | --n 120 | `attributed_tier_summary.json` adoption_by_attribution.unlabeled = 0.8417 | 0.825 | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | authority_ladder | --n 60 | `authority_ladder_summary.json` control_adoption = 0.0 | 0.0 (headline unchanged); differs in adjacent_rung_mcnemar.automated_system_vs_clinical_guideline.gain, adjacent_rung_mcnemar.automated_system_vs_clinical_guideline.pvalue | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | committee_size_sweep | --n 120 | `committee_size_sweep_summary.json` adoption_by_committee_size.s0 = 0.7417 | 0.725 | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | deliberation_framing | --n 120 | `deliberation_framing_summary.json` adoption_by_framing.none = 0.7417 | 0.725 | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | dose_response | --n 120 | `dose_response_summary.json` adoption_by_dose.l1_faint = 0.2083 | 0.2083 (headline unchanged); differs in adoption_by_dose.l2_lean, adoption_by_dose.l3_assert | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | leader_as_auditor | --n 120 | `leader_as_auditor_summary.json` adoption_by_role.peer = 0.7417 | 0.725 | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | plausible_distractor | --n 120 | `plausible_distractor_summary.json` adoption.plausible = 0.8219 | 0.8169 | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | rationale_validity | --n 120 | `rationale_validity_summary.json` adoption_by_rationale.bare = 0.7083 | 0.7 | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | seed_confidence | --n 100 | `seed_confidence_summary.json` confident_adoption = 0.44 | 0.41 | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | super_additivity | --n 120 | `super_additivity_summary.json` adoption.neither = 0.0 | 0.0 (headline unchanged); differs in adoption.both, adoption.peer | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | temperature_sensitivity | --n 120 | `temperature_sensitivity_summary.json` adoption_rate_by_temperature.t0.0 = 0.7417 | 0.725 | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | test_awareness | --n 120 | `test_awareness_summary.json` adoption_by_framing.neutral = 0.7417 | 0.725 | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | text_cue_types | --n 120 | `text_cue_types_summary.json` adoption_by_cue.baseline = 0.7417 | 0.725 | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | hierarchy_dominance | --n 40 | `hierarchy_dominance_summary.json` cases_with_order_independent_dominant_agent = 40 | same | replays exactly | laptop |
| medmcqa | gemini-2.5-flash-lite | majority_pressure | --n 25 | `majority_pressure_summary.json` adoption_curve_by_seeded_peer_count.0_isolated = 0.0 | same | replays exactly | laptop |
| medmcqa | gemini-2.5-flash-lite | orchestrator_failure | --n 80 | `orchestrator_failure_summary.json` wrong_peer_poisons_output = 0.2903 | not replayed | cache miss | laptop |
| medmcqa | gemini-2.5-flash-lite | true_peer_control | --n 59 | `true_peer_control_summary.json` wrong_peer_adoption = 0.2 | same | replays exactly | laptop |
| medmcqa | gemini-2.5-flash-lite | unanimity_break | --n 150 | `unanimity_break_summary.json` adoption_unanimous_two_wrong_peers = 0.4211 | 0.3947 | drift; refreshed on this branch | laptop |
| medmcqa | gemini-2.5-flash-lite | referee_deployable | --n 40 | `referee_deployable_summary.json` referees_vs_adoption_planted_only_DEGENERATE.naive_gate (shared-only, no re-query).precision = 0.575 | same | equal on every committed field (the code now writes extra fields or call counters) | laptop |
| medmcqa | gemini-2.5-flash-lite | referee_judge | --n 40 | `referee_judge_summary.json` same_lineage_judge_vs_adoption.precision = 1.0 | same | replays exactly | laptop |
| medmcqa | gemini-2.5-flash-lite | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| medmcqa | gemini-2.5-flash-lite | check_warn_2x2 | --n 40 | `check_warn_2x2_summary.json` cells.none.decoy_uptake = 0.425 | same | replays exactly | laptop |
| medmcqa | gemini-2.5-flash-lite | matched_baseline_reason | --n 40 | `matched_baseline_reason_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | laptop |
| medmcqa | gemini-2.5-flash | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | attributed_tier | --n 120 | `attributed_tier_summary.json` adoption_by_attribution.unlabeled = 0.4 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | authority_ladder | --n 60 | `authority_ladder_summary.json` control_adoption = 0.0 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | committee_size_sweep | --n 120 | `committee_size_sweep_summary.json` adoption_by_committee_size.s0 = 0.3 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | deliberation_framing | --n 120 | `deliberation_framing_summary.json` adoption_by_framing.none = 0.325 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | dose_response | --n 120 | `dose_response_summary.json` adoption_by_dose.l1_faint = 0.0917 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | leader_as_auditor | --n 120 | `leader_as_auditor_summary.json` adoption_by_role.peer = 0.3 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | plausible_distractor | --n 120 | `plausible_distractor_summary.json` adoption.plausible = 0.3678 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | rationale_validity | --n 120 | `rationale_validity_summary.json` adoption_by_rationale.bare = 0.2167 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | seed_confidence | --n 100 | `seed_confidence_summary.json` confident_adoption = 0.05 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | super_additivity | --n 120 | `super_additivity_summary.json` adoption.neither = 0.0 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | temperature_sensitivity | --n 120 | `temperature_sensitivity_summary.json` adoption_rate_by_temperature.t0.0 = 0.2917 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | test_awareness | --n 120 | `test_awareness_summary.json` adoption_by_framing.neutral = 0.3167 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | text_cue_types | --n 120 | `text_cue_types_summary.json` adoption_by_cue.baseline = 0.3167 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | hierarchy_dominance | --n 40 | `hierarchy_dominance_summary.json` cases_with_order_independent_dominant_agent = 40 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | majority_pressure | --n 25 | `majority_pressure_summary.json` adoption_curve_by_seeded_peer_count.0_isolated = 0.0 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | orchestrator_failure | --n 80 | `orchestrator_failure_summary.json` wrong_peer_poisons_output = 0.1475 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | true_peer_control | --n 57 | `true_peer_control_summary.json` wrong_peer_adoption = 0.125 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | unanimity_break | --n 143 | `unanimity_break_summary.json` adoption_unanimous_two_wrong_peers = 0.1429 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | referee_deployable | --n 40 | `referee_deployable_summary.json` extra_requery_calls_needed = 40 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | referee_judge | --n 40 | `referee_judge_summary.json` same_lineage_judge_vs_adoption.precision = 1.0 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-120b | matched_baseline_reason | --n 40 | `matched_baseline_reason_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | attributed_tier | --n 120 | `attributed_tier_summary.json` adoption_by_attribution.unlabeled = 0.3583 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | authority_ladder | --n 60 | `authority_ladder_summary.json` control_adoption = 0.0 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | committee_size_sweep | --n 120 | `committee_size_sweep_summary.json` adoption_by_committee_size.s0 = 0.3167 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | deliberation_framing | --n 120 | `deliberation_framing_summary.json` adoption_by_framing.none = 0.3417 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | dose_response | --n 120 | `dose_response_summary.json` adoption_by_dose.l1_faint = 0.125 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | leader_as_auditor | --n 120 | `leader_as_auditor_summary.json` adoption_by_role.peer = 0.3333 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | plausible_distractor | --n 120 | `plausible_distractor_summary.json` adoption.plausible = 0.3867 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | rationale_validity | --n 120 | `rationale_validity_summary.json` adoption_by_rationale.bare = 0.2 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | seed_confidence | --n 100 | `seed_confidence_summary.json` confident_adoption = 0.15 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | super_additivity | --n 120 | `super_additivity_summary.json` adoption.neither = 0.0 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | temperature_sensitivity | --n 120 | `temperature_sensitivity_summary.json` adoption_rate_by_temperature.t0.0 = 0.3417 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | test_awareness | --n 120 | `test_awareness_summary.json` adoption_by_framing.neutral = 0.325 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | text_cue_types | --n 120 | `text_cue_types_summary.json` adoption_by_cue.baseline = 0.3333 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | hierarchy_dominance | --n 40 | `hierarchy_dominance_summary.json` cases_with_order_independent_dominant_agent = 39 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | majority_pressure | --n 25 | `majority_pressure_summary.json` adoption_curve_by_seeded_peer_count.0_isolated = 0.0 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | orchestrator_failure | --n 80 | `orchestrator_failure_summary.json` wrong_peer_poisons_output = 0.1091 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | true_peer_control | --n 57 | `true_peer_control_summary.json` wrong_peer_adoption = 0.25 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | unanimity_break | --n 150 | `unanimity_break_summary.json` adoption_unanimous_two_wrong_peers = 0.2083 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | referee_deployable | --n 40 | `referee_deployable_summary.json` extra_requery_calls_needed = 40 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | referee_judge | --n 40 | `referee_judge_summary.json` same_lineage_judge_vs_adoption.precision = 0.8888888888888888 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| medmcqa | openai/gpt-oss-20b | matched_baseline_reason | --n 40 | `matched_baseline_reason_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | attributed_tier | --n 120 | `attributed_tier_summary.json` adoption_by_attribution.unlabeled = 0.4833 | same | replays exactly | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | authority_ladder | --n 60 | `authority_ladder_summary.json` control_adoption = 0.0 | same | replays exactly | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | committee_size_sweep | --n 120 | `committee_size_sweep_summary.json` adoption_by_committee_size.s0 = 0.3667 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | deliberation_framing | --n 120 | `deliberation_framing_summary.json` adoption_by_framing.none = 0.4083 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | dose_response | --n 120 | `dose_response_summary.json` adoption_by_dose.l1_faint = 0.1083 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | leader_as_auditor | --n 120 | `leader_as_auditor_summary.json` adoption_by_role.peer = 0.425 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | plausible_distractor | --n 119 | `plausible_distractor_summary.json` adoption.plausible = 0.5057 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | rationale_validity | --n 120 | `rationale_validity_summary.json` adoption_by_rationale.bare = 0.275 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | seed_confidence | --n 100 | `seed_confidence_summary.json` confident_adoption = 0.23 | same | replays exactly | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | super_additivity | --n 120 | `super_additivity_summary.json` adoption.neither = 0.0 | same | replays exactly | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | temperature_sensitivity | --n 120 | `temperature_sensitivity_summary.json` adoption_rate_by_temperature.t0.0 = 0.425 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | test_awareness | --n 120 | `test_awareness_summary.json` adoption_by_framing.neutral = 0.4083 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | text_cue_types | --n 120 | `text_cue_types_summary.json` adoption_by_cue.baseline = 0.425 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | hierarchy_dominance | --n 40 | `hierarchy_dominance_summary.json` cases_with_order_independent_dominant_agent = 40 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | majority_pressure | --n 25 | `majority_pressure_summary.json` adoption_curve_by_seeded_peer_count.0_isolated = 0.0 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | orchestrator_failure | --n 80 | `orchestrator_failure_summary.json` wrong_peer_poisons_output = 0.082 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | true_peer_control | --n 59 | `true_peer_control_summary.json` wrong_peer_adoption = 0.25 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | unanimity_break | --n 143 | `unanimity_break_summary.json` adoption_unanimous_two_wrong_peers = 0.16 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | referee_deployable | --n 40 | `referee_deployable_summary.json` extra_requery_calls_needed = 40 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | referee_judge | --n 40 | `referee_judge_summary.json` same_lineage_judge_vs_adoption.precision = 1.0 | not replayed | no committed cache | laptop |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | attributed_tier | --n 120 | `attributed_tier_summary.json` adoption_by_attribution.unlabeled = 0.5083 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | authority_ladder | --n 60 | `authority_ladder_summary.json` control_adoption = 0.0 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | committee_size_sweep | --n 120 | `committee_size_sweep_summary.json` adoption_by_committee_size.s0 = 0.5 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | deliberation_framing | --n 120 | `deliberation_framing_summary.json` adoption_by_framing.none = 0.5 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | dose_response | --n 120 | `dose_response_summary.json` adoption_by_dose.l1_faint = 0.1917 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | leader_as_auditor | --n 120 | `leader_as_auditor_summary.json` adoption_by_role.peer = 0.5 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | plausible_distractor | --n 120 | `plausible_distractor_summary.json` adoption.plausible = 0.6022 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | rationale_validity | --n 120 | `rationale_validity_summary.json` adoption_by_rationale.bare = 0.2333 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | seed_confidence | --n 100 | `seed_confidence_summary.json` confident_adoption = 0.24 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | super_additivity | --n 120 | `super_additivity_summary.json` adoption.neither = 0.0 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | temperature_sensitivity | --n 120 | `temperature_sensitivity_summary.json` adoption_rate_by_temperature.t0.0 = 0.5 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | test_awareness | --n 120 | `test_awareness_summary.json` adoption_by_framing.neutral = 0.5 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | text_cue_types | --n 120 | `text_cue_types_summary.json` adoption_by_cue.baseline = 0.5 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | hierarchy_dominance | --n 40 | `hierarchy_dominance_summary.json` cases_with_order_independent_dominant_agent = 40 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | majority_pressure | --n 25 | `majority_pressure_summary.json` adoption_curve_by_seeded_peer_count.0_isolated = 0.0 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | orchestrator_failure | --n 80 | `orchestrator_failure_summary.json` wrong_peer_poisons_output = 0.2075 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | true_peer_control | --n 59 | `true_peer_control_summary.json` wrong_peer_adoption = 0.4 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | unanimity_break | --n 150 | `unanimity_break_summary.json` adoption_unanimous_two_wrong_peers = 0.3922 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | referee_deployable | --n 40 | `referee_deployable_summary.json` extra_requery_calls_needed = 40 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | referee_judge | --n 40 | `referee_judge_summary.json` same_lineage_judge_vs_adoption.precision = 0.425 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | matched_baseline_reason | --n 40 | `matched_baseline_reason_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | committee_size_sweep | --n 120 | `committee_size_sweep_summary.json` adoption_by_committee_size.s0 = 0.35 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | deliberation_framing | --n 120 | `deliberation_framing_summary.json` adoption_by_framing.none = 0.425 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | dose_response | --n 120 | `dose_response_summary.json` adoption_by_dose.l1_faint = 0.075 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | leader_as_auditor | --n 120 | `leader_as_auditor_summary.json` adoption_by_role.peer = 0.3333 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | plausible_distractor | --n 119 | `plausible_distractor_summary.json` adoption.plausible = 0.5325 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | rationale_validity | --n 120 | `rationale_validity_summary.json` adoption_by_rationale.bare = 0.2833 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | temperature_sensitivity | --n 120 | `temperature_sensitivity_summary.json` adoption_rate_by_temperature.t0.0 = 0.4167 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | test_awareness | --n 120 | `test_awareness_summary.json` adoption_by_framing.neutral = 0.3917 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | text_cue_types | --n 120 | `text_cue_types_summary.json` adoption_by_cue.baseline = 0.3667 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | hierarchy_dominance | --n 40 | `hierarchy_dominance_summary.json` cases_with_order_independent_dominant_agent = 40 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | majority_pressure | --n 25 | `majority_pressure_summary.json` adoption_curve_by_seeded_peer_count.0_isolated = 0.0 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | orchestrator_failure | --n 80 | `orchestrator_failure_summary.json` wrong_peer_poisons_output = 0.069 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | true_peer_control | --n 59 | `true_peer_control_summary.json` wrong_peer_adoption = 0.1053 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | unanimity_break | --n 150 | `unanimity_break_summary.json` adoption_unanimous_two_wrong_peers = 0.2353 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | referee_deployable | --n 40 | `referee_deployable_summary.json` extra_requery_calls_needed = 40 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | referee_judge | --n 40 | `referee_judge_summary.json` same_lineage_judge_vs_adoption.precision = 0.9090909090909091 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | blind_metric | --n 40 | `blind_metric_summary.json` declared_only.baseline.decoy_uptake = 0.0 | same | replays exactly | laptop |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | matched_baseline_reason | --n 40 | `matched_baseline_reason_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | laptop |
| medmcqa | gemini-2.5-flash-lite | reproduce_solo | --stage solo | `solo_results.json` noise_floor_by_model.gemini-2.5-flash = 0.06666666666666667 | not replayed | cache miss | laptop (branch tree, needs the new --cache flag) |
| medmcqa | gemini-2.5-flash-lite | reproduce_cascade | --stage cascade | `cascade_results.json` mean_contagion = 0.025 | 0.05 | drift | laptop (branch tree, needs the new --cache flag) |
| medmcqa | gemini-2.5-flash-lite | clean_a | --solo-records | `clean_a_summary.json` gemini-2.5-flash.flag_rate = 0.26666666666666666 | not replayed | cache miss | laptop (branch tree, needs the new --cache flag) |
| medmcqa | gemini-2.5-flash-lite | break_it | --solo-records | `break_it_summary.json` A_contaminated_context.flag_adopt = 22 | same | replays exactly | laptop (branch tree, needs the new --cache flag) |
| medmcqa | gemini-2.5-flash-lite | scale_c |  | `scale_c_summary.json` generic.conform = 79 | not replayed | cache miss | laptop (branch tree, needs the new --cache flag) |
| medmcqa | openai/gpt-oss-120b | reproduce_solo | --stage solo | `solo_results.json` noise_floor_by_model.openai/gpt-oss-120b = 0.06666666666666667 | same | equal except the noise floor, an uncached live control by design | laptop (branch tree, needs the new --cache flag) |
| medmcqa | openai/gpt-oss-120b | reproduce_cascade | --stage cascade | `cascade_results.json` mean_contagion = 0.0 | same | equal up to the order of per-case rows | laptop (branch tree, needs the new --cache flag) |
| medmcqa | openai/gpt-oss-120b | clean_a | --solo-records | `clean_a_summary.json` openai/gpt-oss-120b.flag_rate = 0.3 | not replayed | cache miss | laptop (branch tree, needs the new --cache flag) |
| medmcqa | openai/gpt-oss-120b | break_it | --solo-records | `break_it_summary.json` A_contaminated_context.flag_adopt = 8 | not replayed | cache miss | laptop (branch tree, needs the new --cache flag) |
| medmcqa | openai/gpt-oss-120b | scale_c |  | `scale_c_summary.json` generic.conform = 32 | same | replays exactly | laptop (branch tree, needs the new --cache flag) |
| medmcqa | openai/gpt-oss-20b | reproduce_solo | --stage solo | `solo_results.json` noise_floor_by_model.openai/gpt-oss-20b = 0.13333333333333333 | same | equal except the noise floor, an uncached live control by design | laptop (branch tree, needs the new --cache flag) |
| medmcqa | openai/gpt-oss-20b | reproduce_cascade | --stage cascade | `cascade_results.json` mean_contagion = 0.025 | same | equal up to the order of per-case rows | laptop (branch tree, needs the new --cache flag) |
| medmcqa | openai/gpt-oss-20b | clean_a | --solo-records | `clean_a_summary.json` openai/gpt-oss-20b.flag_rate = 0.26666666666666666 | not replayed | cache miss | laptop (branch tree, needs the new --cache flag) |
| medmcqa | openai/gpt-oss-20b | break_it | --solo-records | `break_it_summary.json` A_contaminated_context.flag_adopt = 10 | not replayed | cache miss | laptop (branch tree, needs the new --cache flag) |
| medmcqa | openai/gpt-oss-20b | scale_c |  | `scale_c_summary.json` generic.conform = 51 | same | replays exactly | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | reproduce_solo | --stage solo | `solo_results.json` noise_floor_by_model.nvidia/nemotron-3-super-120b-a12b = 0.06666666666666667 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | reproduce_cascade | --stage cascade | `cascade_results.json` mean_contagion = 0.0 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | clean_a | --solo-records | `clean_a_summary.json` nvidia/nemotron-3-super-120b-a12b.flag_rate = 0.45454545454545453 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | break_it | --solo-records | `break_it_summary.json` A_contaminated_context.flag_adopt = 11 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | push_c | --solo-records | `push_c_summary.json` generic.conform = 12 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nvidia/nemotron-3-super-120b-a12b | scale_c |  | `scale_c_summary.json` generic.conform = 44 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | reproduce_solo | --stage solo | `solo_results.json` noise_floor_by_model.Qwen/Qwen2.5-VL-72B-Instruct = 0.0 | same | equal except the noise floor, an uncached live control by design | laptop (branch tree, needs the new --cache flag) |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | reproduce_cascade | --stage cascade | `cascade_results.json` mean_contagion = 0.0 | same | equal up to the order of per-case rows | laptop (branch tree, needs the new --cache flag) |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | clean_a | --solo-records | `clean_a_summary.json` Qwen/Qwen2.5-VL-72B-Instruct.flag_rate = 0.3793103448275862 | same | replays exactly | laptop (branch tree, needs the new --cache flag) |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | break_it | --solo-records | `break_it_summary.json` A_contaminated_context.flag_adopt = 10 | same | replays exactly | laptop (branch tree, needs the new --cache flag) |
| medmcqa | Qwen/Qwen2.5-VL-72B-Instruct | scale_c |  | `scale_c_summary.json` generic.conform = 56 | same | replays exactly | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | reproduce_solo | --stage solo | `solo_results.json` noise_floor_by_model.nemotron-3-super-120b-a12b-fp8-local = 0.0 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | reproduce_cascade | --stage cascade | `cascade_results.json` mean_contagion = 0.0 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | clean_a | --solo-records | `clean_a_summary.json` nemotron-3-super-120b-a12b-fp8-local.flag_rate = 0.5 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | break_it | --solo-records | `break_it_summary.json` A_contaminated_context.flag_adopt = 11 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | push_c | --solo-records | `push_c_summary.json` generic.conform = 10 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| medmcqa | nemotron-3-super-120b-a12b-fp8-local | scale_c |  | `scale_c_summary.json` generic.conform = 38 | not replayed | no committed cache | laptop (branch tree, needs the new --cache flag) |
| mimic_text | gemini-2.5-flash-lite | solo_35 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text | gemini-2.5-flash-lite | cascade_20 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text | gemini-2.5-flash-lite | refusal_aware_35 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text | gemini-2.5-flash-lite | blind_metric |  | `blind_metric_summary.json` decoy_uptake.baseline = 0.0 | not replayed | cache miss | ORCD |
| mimic_text | gemini-2.5-flash-lite | break_it_a |  | `break_it_a_summary.json` A_contaminated_context.flag_adopt = 18 | not replayed | cache miss | ORCD |
| mimic_text | gemini-2.5-flash-lite | break_it_d |  | `break_it_d_summary.json` incent_decoy = 3 | not replayed | cache miss | ORCD |
| mimic_text | gemini-2.5-flash-lite | deliberation_framing |  | `deliberation_framing_summary.json` adoption_by_framing.none = 0.7167 | not replayed | cache miss | ORCD |
| mimic_text | gemini-2.5-flash-lite | push_c |  | `push_c_summary.json` generic.conform = 51 | not replayed | cache miss | ORCD |
| mimic_text | gemini-2.5-flash-lite | referee_deployable |  | `referee_deployable_summary.json` extra_requery_calls_needed = 40 | not replayed | cache miss | ORCD |
| mimic_text | gemini-2.5-flash-lite | referee_judge |  | `referee_judge_summary.json` same_lineage_judge_vs_adoption.precision = 1.0 | not replayed | cache miss | ORCD |
| mimic_text | openai/gpt-oss-120b | cascade_20 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text | openai/gpt-oss-120b | blind_metric |  | `blind_metric_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-120b | break_it_a |  | `break_it_a_summary.json` A_contaminated_context.flag_adopt = 11 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-120b | break_it_d |  | `break_it_d_summary.json` incent_decoy = 4 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-120b | deliberation_framing |  | `deliberation_framing_summary.json` adoption_by_framing.none = 0.45 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-120b | push_c |  | `push_c_summary.json` generic.conform = 32 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-120b | referee_deployable |  | `referee_deployable_summary.json` extra_requery_calls_needed = 40 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-120b | referee_judge |  | `referee_judge_summary.json` same_lineage_judge_vs_adoption.precision = 1.0 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-20b | cascade_20 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text | openai/gpt-oss-20b | blind_metric |  | `blind_metric_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-20b | break_it_a |  | `break_it_a_summary.json` A_contaminated_context.flag_adopt = 5 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-20b | break_it_d |  | `break_it_d_summary.json` incent_decoy = 1 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-20b | deliberation_framing |  | `deliberation_framing_summary.json` adoption_by_framing.none = 0.45 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-20b | push_c |  | `push_c_summary.json` generic.conform = 27 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-20b | referee_deployable |  | `referee_deployable_summary.json` extra_requery_calls_needed = 40 | same | replays exactly | ORCD |
| mimic_text | openai/gpt-oss-20b | referee_judge |  | `referee_judge_summary.json` same_lineage_judge_vs_adoption.precision = 0.7272727272727273 | same | replays exactly | ORCD |
| mimic_text | nvidia/nemotron-3-super-120b-a12b | solo_35 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text | nvidia/nemotron-3-super-120b-a12b | cascade_20 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text | nvidia/nemotron-3-super-120b-a12b | blind_metric |  | `blind_metric_summary.json` decoy_uptake.baseline = 0.0 | not replayed | cache miss | ORCD |
| mimic_text | nvidia/nemotron-3-super-120b-a12b | break_it_a |  | `break_it_a_summary.json` A_contaminated_context.flag_adopt = 5 | not replayed | cache miss | ORCD |
| mimic_text | nvidia/nemotron-3-super-120b-a12b | break_it_d |  | `break_it_d_summary.json` incent_decoy = 5 | not replayed | cache miss | ORCD |
| mimic_text | nvidia/nemotron-3-super-120b-a12b | deliberation_framing |  | `deliberation_framing_summary.json` adoption_by_framing.none = 0.45 | not replayed | cache miss | ORCD |
| mimic_text | nvidia/nemotron-3-super-120b-a12b | push_c |  | `push_c_summary.json` generic.conform = 34 | not replayed | cache miss | ORCD |
| mimic_text | nvidia/nemotron-3-super-120b-a12b | referee_deployable |  | `referee_deployable_summary.json` extra_requery_calls_needed = 40 | not replayed | cache miss | ORCD |
| mimic_text | nvidia/nemotron-3-super-120b-a12b | referee_judge |  | `referee_judge_summary.json` same_lineage_judge_vs_adoption.precision = 0.8333333333333334 | not replayed | cache miss | ORCD |
| mimic_text | Qwen/Qwen2.5-VL-72B-Instruct | solo_35 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text | Qwen/Qwen2.5-VL-72B-Instruct | cascade_20 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text | Qwen/Qwen2.5-VL-72B-Instruct | blind_metric |  | `blind_metric_summary.json` decoy_uptake.baseline = 0.0 | not replayed | cache miss | ORCD |
| mimic_text | Qwen/Qwen2.5-VL-72B-Instruct | break_it_a |  | `break_it_a_summary.json` A_contaminated_context.flag_adopt = 8 | not replayed | cache miss | ORCD |
| mimic_text | Qwen/Qwen2.5-VL-72B-Instruct | break_it_d |  | `break_it_d_summary.json` incent_decoy = 4 | not replayed | cache miss | ORCD |
| mimic_text | Qwen/Qwen2.5-VL-72B-Instruct | deliberation_framing |  | `deliberation_framing_summary.json` adoption_by_framing.none = 0.1667 | not replayed | cache miss | ORCD |
| mimic_text | Qwen/Qwen2.5-VL-72B-Instruct | push_c |  | `push_c_summary.json` generic.conform = 41 | not replayed | cache miss | ORCD |
| mimic_text | Qwen/Qwen2.5-VL-72B-Instruct | referee_deployable |  | `referee_deployable_summary.json` extra_requery_calls_needed = 40 | not replayed | cache miss | ORCD |
| mimic_text | Qwen/Qwen2.5-VL-72B-Instruct | referee_judge |  | `referee_judge_summary.json` same_lineage_judge_vs_adoption.precision = 0.3333333333333333 | not replayed | cache miss | ORCD |
| mimic_text_n600 | gemini-2.5-flash-lite | solo_600 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text_n600 | gemini-2.5-flash-lite | refusal_aware_600 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text_n600 | openai/gpt-oss-120b | solo_600 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text_n600 | openai/gpt-oss-120b | refusal_aware_600 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text_n600 | openai/gpt-oss-20b | solo_600 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text_n600 | openai/gpt-oss-20b | refusal_aware_600 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text_n600 | nvidia/nemotron-3-super-120b-a12b | solo_600 |  | (see lane folder) | not replayed | cache miss | ORCD |
| mimic_text_n600 | nvidia/nemotron-3-super-120b-a12b | refusal_aware_600 |  | (see lane folder) | not replayed | no cache found | ORCD |
| nih | gemini-2.5-flash | solo |  | `imaging_solo_summary.json` cues.cable.flip_rate = 0.2 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | solo_lite |  | `imaging_solo_lite_summary.json` cues.cable.flip_rate = 0.08571428571428572 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | cascade_watermark |  | `imaging_cascade_cable_summary.json` shared_adopt = 1.0 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | cascade_cable |  | `imaging_cascade_cable_summary.json` shared_adopt = 1.0 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | cascade_corner_tag |  | `imaging_cascade_corner_tag_summary.json` shared_adopt = 0.9714 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | cascade_laterality |  | `imaging_cascade_laterality_summary.json` shared_adopt = 0.9714 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | scale |  | `imaging_cascade_scale_summary.json` shared_adopt = 0.9733 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | cue_combo |  | `imaging_cue_combo_summary.json` flip_rates.watermark = 0.3182 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | cue_families |  | `imaging_cue_families_summary.json` cues.rotation.flip_rate = 0.3142857142857143 | same | equal except the noise floor, an uncached live control by design | ORCD |
| nih | gemini-2.5-flash | majority_pressure |  | `imaging_majority_pressure_summary.json` adoption_curve_by_seeded_peer_count.0 = 0.4286 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | matched_temp |  | `imaging_matched_temp_summary.json` two_by_two.temp0_floor = 0.0 | same | replays exactly | ORCD |
| nih | gemini-2.5-flash | multi_round |  | `imaging_multi_round_summary.json` K = 3 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | peer_size_curve |  | `imaging_peer_size_curve_summary.json` adoption_by_peer_count.1 = 0.9714 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | polarity |  | `imaging_polarity_summary.json` positive_polarity_yes_to_no.flips = 22 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | strength_cascade |  | `imaging_strength_cascade_summary.json` strength_x_cascade.opacity_0.15.solo_flip = 0.3143 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | system_flag |  | `imaging_system_flag_summary.json` shared_adopt = 0.6 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | blind_metric |  | `imaging_blind_metric_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | ORCD |
| nih | gemini-2.5-flash | referee |  | `imaging_referee_summary.json` peer_driven_adoptions_gt = 19 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | gemini-2.5-flash | judge_text_only |  | `imaging_judge_referee_summary.json` peer_driven_adoptions_gt = 19 | same | equal on every committed field (the code now writes extra fields or call counters) | ORCD |
| nih | gemini-2.5-flash | matched_baseline_reason |  | `matched_baseline_reason_repeats_summary.json` reads_per_case = 6 | same | replays exactly | ORCD |
| nih | gemini-2.5-flash-lite | blind_metric |  | `imaging_blind_metric_summary.json` decoy_uptake.baseline = 0.0 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | solo |  | `imaging_solo_summary.json` cues.cable.flip_rate = 0.05714285714285714 | same | equal except the noise floor, an uncached live control by design | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | solo_lite |  | `imaging_solo_lite_summary.json` cues.cable.flip_rate = 0.05714285714285714 | same | equal except the noise floor, an uncached live control by design | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | cascade_watermark |  | `imaging_cascade_summary.json` shared_adopt = 1.0 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | cascade_cable |  | `imaging_cascade_cable_summary.json` shared_adopt = 1.0 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | cascade_corner_tag |  | `imaging_cascade_corner_tag_summary.json` shared_adopt = 1.0 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | cascade_laterality |  | `imaging_cascade_laterality_summary.json` shared_adopt = 1.0 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | scale |  | `imaging_cascade_scale_summary.json` shared_adopt = 0.9933 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | cue_combo |  | `imaging_cue_combo_summary.json` flip_rates.watermark = 0.0 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | cue_families |  | `imaging_cue_families_summary.json` cues.rotation.flip_rate = 0.14285714285714285 | same | equal except the noise floor, an uncached live control by design | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | majority_pressure |  | `imaging_majority_pressure_summary.json` adoption_curve_by_seeded_peer_count.0 = 0.4857 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | matched_temp |  | `imaging_matched_temp_summary.json` two_by_two.temp0_floor = 0.0 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | multi_round |  | `imaging_multi_round_summary.json` K = 3 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | peer_size_curve |  | `imaging_peer_size_curve_summary.json` adoption_by_peer_count.1 = 0.9714 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | polarity |  | `imaging_polarity_summary.json` positive_polarity_yes_to_no.flips = 6 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | strength_cascade |  | `imaging_strength_cascade_summary.json` strength_x_cascade.opacity_0.15.solo_flip = 0.0857 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | system_flag |  | `imaging_system_flag_summary.json` shared_adopt = 0.8 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | blind_metric |  | `imaging_blind_metric_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | referee |  | `imaging_referee_summary.json` peer_driven_adoptions_gt = 18 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | judge_text_only |  | `imaging_judge_referee_summary.json` peer_driven_adoptions_gt = 18 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | judge_film |  | `imaging_judge_referee_summary_film.json` peer_driven_adoptions_gt = 18 | same | replays exactly | ORCD |
| nih | Qwen/Qwen2.5-VL-72B-Instruct | matched_baseline_reason |  | `matched_baseline_reason_repeats_summary.json` reads_per_case = 6 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | solo |  | `imaging_solo_summary.json` cues.cable.flip_rate = 0.14285714285714285 | same | equal except the noise floor, an uncached live control by design | ORCD |
| nih | Qwen/Qwen3.8-27B | solo_lite |  | `imaging_solo_lite_summary.json` cues.cable.flip_rate = 0.08571428571428572 | same | equal except the noise floor, an uncached live control by design | ORCD |
| nih | Qwen/Qwen3.8-27B | cascade_watermark |  | `imaging_cascade_summary.json` shared_adopt = 0.8857 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | cascade_cable |  | `imaging_cascade_cable_summary.json` shared_adopt = 0.9143 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | cascade_corner_tag |  | `imaging_cascade_corner_tag_summary.json` shared_adopt = 0.9429 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | cascade_laterality |  | `imaging_cascade_laterality_summary.json` shared_adopt = 0.8571 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | scale |  | `imaging_cascade_scale_summary.json` shared_adopt = 0.9267 | not replayed | point estimates equal; a bootstrap CI bound differs in the third decimal | ORCD |
| nih | Qwen/Qwen3.8-27B | cue_combo |  | `imaging_cue_combo_summary.json` flip_rates.watermark = 0.1905 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | cue_families |  | `imaging_cue_families_summary.json` cues.rotation.flip_rate = 0.11428571428571428 | same | equal except the noise floor, an uncached live control by design | ORCD |
| nih | Qwen/Qwen3.8-27B | majority_pressure |  | `imaging_majority_pressure_summary.json` adoption_curve_by_seeded_peer_count.0 = 0.4571 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | matched_temp |  | `imaging_matched_temp_summary.json` two_by_two.temp0_floor = 0.0 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | multi_round |  | `imaging_multi_round_summary.json` K = 3 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | peer_size_curve |  | `imaging_peer_size_curve_summary.json` adoption_by_peer_count.1 = 0.9429 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | polarity |  | `imaging_polarity_summary.json` positive_polarity_yes_to_no.flips = 13 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | strength_cascade |  | `imaging_strength_cascade_summary.json` strength_x_cascade.opacity_0.15.solo_flip = 0.0571 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | system_flag |  | `imaging_system_flag_summary.json` shared_adopt = 0.5143 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | blind_metric |  | `imaging_blind_metric_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | referee |  | `imaging_referee_summary.json` peer_driven_adoptions_gt = 15 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | judge_text_only |  | `imaging_judge_referee_summary.json` peer_driven_adoptions_gt = 15 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | judge_film |  | `imaging_judge_referee_summary_film.json` peer_driven_adoptions_gt = 15 | same | replays exactly | ORCD |
| nih | Qwen/Qwen3.8-27B | matched_baseline_reason |  | `matched_baseline_reason_repeats_summary.json` reads_per_case = 6 | same | replays exactly | ORCD |
| nih_n500 | gemini-2.5-flash | scale_500 |  | `imaging_cascade_scale_summary.json` shared_adopt = 0.892 | not replayed | point estimates equal; a bootstrap CI bound differs in the third decimal | ORCD |
| nih_n500 | gemini-2.5-flash | blind_metric_500 |  | `imaging_blind_metric_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | ORCD |
| nih_n500 | gemini-2.5-flash | matched_baseline_reason_500 |  | `matched_baseline_reason_repeats_summary.json` reads_per_case = 6 | same | replays exactly | ORCD |
| nih_n80 | gemini-2.5-flash | cascade_80 |  | `imaging_cascade_summary.json` shared_adopt = 0.975 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih_n80 | gemini-2.5-flash | referee_80 |  | `imaging_judge_referee_summary.json` peer_driven_adoptions_gt = 47 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| nih_n80 | gemini-2.5-flash | judge_80 |  | `imaging_judge_referee_summary.json` peer_driven_adoptions_gt = 47 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert35_blind | gemini-2.5-flash | blind_metric |  | `imaging_blind_metric_summary.json` decoy_uptake.baseline = 0.0 | not replayed | not run: google-genai is not installed in the ORCD Python | ORCD |
| chexpert35_blind | gemini-2.5-flash | matched_baseline_reason |  | `matched_baseline_reason_repeats_summary.json` reads_per_case = 6 | same | replays exactly | ORCD |
| chexpert35_blind | Qwen/Qwen2.5-VL-72B-Instruct | blind_metric |  | `imaging_blind_metric_summary.json` decoy_uptake.baseline = 0.0 | same | replays exactly | ORCD |
| chexpert35_blind | Qwen/Qwen2.5-VL-72B-Instruct | matched_baseline_reason |  | `matched_baseline_reason_repeats_summary.json` reads_per_case = 6 | same | equal on every committed field (the code now writes extra fields or call counters) | ORCD |
| chexpert35_blind | Qwen/Qwen3.8-27B | blind_metric |  | `imaging_blind_metric_summary.json` decoy_uptake.baseline = 0.0 | same | equal on every committed field (the code now writes extra fields or call counters) | ORCD |
| chexpert35_blind | Qwen/Qwen3.8-27B | matched_baseline_reason |  | `matched_baseline_reason_repeats_summary.json` reads_per_case = 6 | same | equal on every committed field (the code now writes extra fields or call counters) | ORCD |
| chexpert35_cascade | gemini-2.5-flash | cascade |  | `imaging_cascade_summary.json` shared_adopt = 0.8571 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert35_cascade | gemini-2.5-flash | solo |  | `imaging_solo_summary.json` cues.cable.flip_rate = 0.08571428571428572 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert35_cascade | gemini-2.5-flash | matched_temp |  | `imaging_matched_temp_summary.json` two_by_two.temp0_floor = 0.0 | same | replays exactly | ORCD |
| chexpert35_cascade | Qwen/Qwen2.5-VL-72B-Instruct | cascade |  | `imaging_cascade_summary.json` shared_adopt = 0.9714 | same | replays exactly | ORCD |
| chexpert35_cascade | Qwen/Qwen2.5-VL-72B-Instruct | solo |  | `imaging_solo_summary.json` cues.cable.flip_rate = 0.11428571428571428 | same | equal except the noise floor, an uncached live control by design | ORCD |
| chexpert35_cascade | Qwen/Qwen2.5-VL-72B-Instruct | matched_temp |  | `imaging_matched_temp_summary.json` two_by_two.temp0_floor = 0.0 | same | replays exactly | ORCD |
| chexpert35_cascade | Qwen/Qwen3.8-27B | cascade |  | `imaging_cascade_summary.json` shared_adopt = 0.9429 | same | replays exactly | ORCD |
| chexpert35_cascade | Qwen/Qwen3.8-27B | solo |  | `imaging_solo_summary.json` cues.cable.flip_rate = 0.05714285714285714 | same | equal except the noise floor, an uncached live control by design | ORCD |
| chexpert35_cascade | Qwen/Qwen3.8-27B | matched_temp |  | `imaging_matched_temp_summary.json` two_by_two.temp0_floor = 0.0 | same | replays exactly | ORCD |
| chexpert35_system_flag | gemini-2.5-flash | system_flag_none |  | `imaging_system_flag_summary.json` shared_adopt = 0.0067 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert35_system_flag | Qwen/Qwen2.5-VL-72B-Instruct | system_flag_none |  | `imaging_system_flag_summary.json` shared_adopt = 0.12 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert35_system_flag | Qwen/Qwen3.8-27B | system_flag_none |  | `imaging_system_flag_summary.json` shared_adopt = 0.4267 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert150_device_absent | gemini-2.5-flash | cascade_none_150 |  | `imaging_cascade_none_summary.json` shared_adopt = 0.94 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert150_device_absent | Qwen/Qwen2.5-VL-72B-Instruct | cascade_none_150 |  | `imaging_cascade_none_summary.json` shared_adopt = 0.9867 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert150_device_absent | Qwen/Qwen3.8-27B | cascade_none_150 |  | `imaging_cascade_none_summary.json` shared_adopt = 0.9467 | same | replays exactly | ORCD |
| chexpert150_natural_cues | gemini-2.5-flash | cascade_none_150 |  | `imaging_cascade_none_summary.json` shared_adopt = 0.6067 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert150_natural_cues | Qwen/Qwen2.5-VL-72B-Instruct | cascade_none_150 |  | `imaging_cascade_none_summary.json` shared_adopt = 0.46 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert150_natural_cues | Qwen/Qwen3.8-27B | cascade_none_150 |  | `imaging_cascade_none_summary.json` shared_adopt = 0.8 | same | replays exactly | ORCD |
| chexpert150_natural_independent | gemini-2.5-flash | cascade_none_150 |  | `imaging_cascade_none_summary.json` shared_adopt = 0.6133 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert150_natural_independent | Qwen/Qwen2.5-VL-72B-Instruct | cascade_none_150 |  | `imaging_cascade_none_summary.json` shared_adopt = 0.4333 | not replayed | cache key mismatch (image bytes on this machine differ from the ones that made the cache) | ORCD |
| chexpert150_natural_independent | Qwen/Qwen3.8-27B | cascade_none_150 |  | `imaging_cascade_none_summary.json` shared_adopt = 0.7933 | same | replays exactly | ORCD |
| mimic_image | gemini-2.5-flash | solo |  | `imaging_solo_summary.json` cues.cable.flip_rate = 0.2014388489208633 | not replayed | not assessed: the MIMIC-CXR JPGs of the manifests were not found under the ORCD roots probed | ORCD |
| mimic_image | gemini-2.5-flash | nih_match_solo |  | (see lane folder) | not replayed | not assessed: the MIMIC-CXR JPGs of the manifests were not found under the ORCD roots probed | ORCD |
| mimic_image | gemini-2.5-flash | cascade |  | `imaging_strength_cascade_summary.json` strength_x_cascade.opacity_0.15.solo_flip = 0.2014 | not replayed | not assessed: the MIMIC-CXR JPGs of the manifests were not found under the ORCD roots probed | ORCD |
| mimic_image | gemini-2.5-flash | system_flag |  | `imaging_system_flag_summary.json` shared_adopt = 0.9245 | not replayed | not assessed: the MIMIC-CXR JPGs of the manifests were not found under the ORCD roots probed | ORCD |
| mimic_image | gemini-2.5-flash | strength_cascade |  | `imaging_strength_cascade_summary.json` strength_x_cascade.opacity_0.15.solo_flip = 0.2014 | not replayed | not assessed: the MIMIC-CXR JPGs of the manifests were not found under the ORCD roots probed | ORCD |
| mimic_image | gemini-2.5-flash | blind_metric |  | `imaging_blind_metric_summary.json` decoy_uptake.baseline = 0.0 | not replayed | not assessed: the MIMIC-CXR JPGs of the manifests were not found under the ORCD roots probed | ORCD |
| mimic_image | gemini-2.5-flash | matched_baseline_reason |  | `matched_baseline_reason_summary.json` decoy_uptake.baseline = 0.0 | not replayed | not assessed: the MIMIC-CXR JPGs of the manifests were not found under the ORCD roots probed | ORCD |
| mimic_image | gemini-2.5-flash | matched |  | `matched_baseline_reason_repeats_summary.json` reads_per_case = 6 | not replayed | not assessed: the MIMIC-CXR JPGs of the manifests were not found under the ORCD roots probed | ORCD |
| mimic_image | gemini-2.5-flash | referee_cascade |  | (see lane folder) | not replayed | not assessed: the MIMIC-CXR JPGs of the manifests were not found under the ORCD roots probed | ORCD |
| mimic_image | gemini-2.5-flash | referee |  | (see lane folder) | not replayed | not assessed: the MIMIC-CXR JPGs of the manifests were not found under the ORCD roots probed | ORCD |
| mimic_image | gemini-2.5-flash | judge |  | (see lane folder) | same | replays exactly | ORCD |
| mimic_image | gemini-2.5-flash | judge_with_image |  | (see lane folder) | not replayed | not assessed: the MIMIC-CXR JPGs of the manifests were not found under the ORCD roots probed | ORCD |

## Files and fields that change against `main`

Counted leaf by leaf (every JSON value, list elements separately) between `main` `1cb76ad` and this
branch. An earlier version of this section said 123; that count covered only the 98 numeric fields of
the 15 Gemini summaries plus `stats_reconciliation_summary.json` with each confidence interval counted
as one field (25), and it left out the 47 mirrored cells of the hosted nemotron comparison file. The
complete count is 221 changed leaf fields in 20 JSON files, of which 174 are numbers
(98 rates or estimates, 49 counts, 17 p-values, 9 CI bounds, 1 added
field). No change is formatting only: every changed number differs in value, and every changed text
differs in its numbers, its verdict, or (the `note` of `referee_deployable_summary.json`) the current
script's wording.

| File | Changed leaf fields | Rates or estimates | Counts | p-values | CI bounds | Field added | Field removed | Call counter | Text |
|---|---|---|---|---|---|---|---|---|---|
| `attributed_tier_summary.json` | 10 | 2 | 4 | 2 | 0 | 0 | 0 | 1 | 1 |
| `authority_ladder_summary.json` | 13 | 6 | 4 | 1 | 0 | 0 | 0 | 1 | 1 |
| `committee_size_sweep_summary.json` | 7 | 2 | 3 | 0 | 0 | 0 | 0 | 1 | 1 |
| `deliberation_framing_summary.json` | 10 | 2 | 5 | 1 | 0 | 0 | 0 | 1 | 1 |
| `dose_response_summary.json` | 10 | 3 | 4 | 1 | 0 | 0 | 0 | 1 | 1 |
| `leader_as_auditor_summary.json` | 10 | 2 | 6 | 0 | 0 | 0 | 0 | 1 | 1 |
| `plausible_distractor_summary.json` | 8 | 2 | 3 | 1 | 0 | 0 | 0 | 1 | 1 |
| `rationale_validity_summary.json` | 8 | 2 | 3 | 1 | 0 | 0 | 0 | 1 | 1 |
| `seed_confidence_summary.json` | 6 | 3 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| `super_additivity_summary.json` | 9 | 4 | 2 | 1 | 0 | 0 | 0 | 1 | 1 |
| `temperature_sensitivity_summary.json` | 7 | 5 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `test_awareness_summary.json` | 9 | 3 | 2 | 2 | 0 | 0 | 0 | 1 | 1 |
| `text_cue_types_summary.json` | 12 | 2 | 5 | 3 | 0 | 0 | 0 | 1 | 1 |
| `unanimity_break_summary.json` | 6 | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
| `referee_deployable_summary.json` | 3 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| **(a) the 15 Gemini MedMCQA summaries** | **128** | **40** | **43** | **14** | **0** | **1** | **0** | **15** | **15** |
| `stats_reconciliation_summary.json` | 29 | 11 | 6 | 3 | 9 | 0 | 0 | 0 | 0 |
| `nvidia_nemotron-3-super-120b-a12b/cross_lineage_summary.json` | 47 | 47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **(b) derived and mirror files** | **76** | **58** | **6** | **3** | **9** | **0** | **0** | **0** | **0** |
| `tests/degeneracy_exemptions.json` | 15 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 14 |
| `Qwen_Qwen2.5-VL-72B-Instruct/unanimity_break_summary.json` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `openai_gpt-oss-20b/unanimity_break_summary.json` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| **(c) text-only files** | **17** | **0** | **0** | **0** | **0** | **0** | **1** | **0** | **16** |
| **all changed JSON** | **221** | **98** | **49** | **17** | **9** | **1** | **1** | **15** | **31** |

Why each file outside the 15 Gemini summaries changes:

- `stats_reconciliation_summary.json` is regenerated by `experiments/medqa/stats_reconciliation.py
  --results-dir experiments/medmcqa/results`, which reads eight arm rows files; five of them were
  refreshed, so five of its eight contrasts move. Its Holm and BH survivors do not change.
- `nvidia_nemotron-3-super-120b-a12b/cross_lineage_summary.json` is regenerated by
  `experiments/cross_lane_table.py --lane medmcqa --model nvidia/nemotron-3-super-120b-a12b`; it places
  each Gemini value beside the hosted nemotron value, so 47 cells of its `gemini` column change and its
  `model` column (the nemotron numbers) does not.
- `tests/degeneracy_exemptions.json`: 14 exemption reasons quote McNemar counts and p-values from the
  refreshed summaries and are updated to the new counts; one exemption
  (`hardcoded_verdict|experiments/medqa/unanimity_break.py|main:NOT significant`) is removed because the
  verdict it excused no longer exists.
- `Qwen_Qwen2.5-VL-72B-Instruct/unanimity_break_summary.json` and
  `openai_gpt-oss-20b/unanimity_break_summary.json` (MedMCQA): only their `read` text changes.
  `experiments/medqa/unanimity_break.py` wrote "NOT significant" whatever the p-value; it now derives
  the verdict from the test, and these two files (p = 6.1e-05 and 0.007812) and the Gemini one
  (p = 0.000122) were the only committed summaries whose text was wrong. Every other committed
  `unanimity_break_summary.json` replays to the same text. 0 calls.

The 15 rows files are rewritten by the same zero-call runs: 84 row fields change in 14 of them, and
`referee_deployable.jsonl` only changes row order.

| Rows file | Rows (main / branch) | Changed row fields |
|---|---|---|
| `attributed_tier.jsonl` | 120 / 120 | 3 |
| `authority_ladder.jsonl` | 60 / 60 | 5 |
| `committee_size_sweep.jsonl` | 120 / 120 | 3 |
| `deliberation_framing.jsonl` | 120 / 120 | 3 |
| `dose_response.jsonl` | 120 / 120 | 5 |
| `leader_as_auditor.jsonl` | 120 / 120 | 5 |
| `plausible_distractor.jsonl` | 73 / 71 | 17 |
| `rationale_validity.jsonl` | 120 / 120 | 3 |
| `seed_confidence.jsonl` | 100 / 100 | 4 |
| `super_additivity.jsonl` | 120 / 120 | 5 |
| `temperature_sensitivity.jsonl` | 120 / 120 | 23 |
| `test_awareness.jsonl` | 120 / 120 | 4 |
| `text_cue_types.jsonl` | 120 / 120 | 3 |
| `unanimity_break.jsonl` | 38 / 38 | 1 |
| `referee_deployable.jsonl` | 80 / 80 | 0 |
| **total** | | **84** |

### Every changed number in the 15 Gemini summaries and the reconciliation (127 fields)

| File | Field | main `1cb76ad` | this branch |
|---|---|---|---|
| `attributed_tier_summary.json` | adoption_by_attribution.human_senior | 0.8083 | 0.8 |
| `attributed_tier_summary.json` | adoption_by_attribution.unlabeled | 0.8417 | 0.825 |
| `attributed_tier_summary.json` | unlabeled_vs_human_senior.gain | 3 | 4 |
| `attributed_tier_summary.json` | unlabeled_vs_human_senior.pvalue | 0.34375 | 0.548828 |
| `attributed_tier_summary.json` | unlabeled_vs_junior_model.gain | 1 | 2 |
| `attributed_tier_summary.json` | unlabeled_vs_junior_model.lose | 52 | 51 |
| `attributed_tier_summary.json` | unlabeled_vs_senior_model.gain | 6 | 8 |
| `attributed_tier_summary.json` | unlabeled_vs_senior_model.pvalue | 0.005925 | 0.02412 |
| `authority_ladder_summary.json` | adjacent_rung_mcnemar.automated_system_vs_clinical_guideline.gain | 16 | 17 |
| `authority_ladder_summary.json` | adjacent_rung_mcnemar.automated_system_vs_clinical_guideline.pvalue | 0.004425 | 0.002577 |
| `authority_ladder_summary.json` | adjacent_rung_mcnemar.colleague_vs_senior_attending.gain | 31 | 33 |
| `authority_ladder_summary.json` | adjacent_rung_mcnemar.senior_attending_vs_automated_system.gain | 4 | 5 |
| `authority_ladder_summary.json` | adjacent_rung_mcnemar.senior_attending_vs_automated_system.lose | 5 | 6 |
| `authority_ladder_summary.json` | adoption_by_rung.automated_system | 0.7 | 0.6833 |
| `authority_ladder_summary.json` | adoption_by_rung.colleague | 0.2 | 0.15 |
| `authority_ladder_summary.json` | adoption_by_rung.senior_attending | 0.7167 | 0.7 |
| `authority_ladder_summary.json` | rungs_low_to_high[0][1] | 0.2 | 0.15 |
| `authority_ladder_summary.json` | rungs_low_to_high[1][1] | 0.7 | 0.6833 |
| `authority_ladder_summary.json` | rungs_low_to_high[2][1] | 0.7167 | 0.7 |
| `committee_size_sweep_summary.json` | adoption_by_committee_size.s0 | 0.7417 | 0.725 |
| `committee_size_sweep_summary.json` | adoption_by_committee_size.s1 | 0.2667 | 0.2583 |
| `committee_size_sweep_summary.json` | s0_vs_s1.lose | 60 | 59 |
| `committee_size_sweep_summary.json` | s0_vs_s2.lose | 67 | 65 |
| `committee_size_sweep_summary.json` | s0_vs_s4.lose | 68 | 66 |
| `deliberation_framing_summary.json` | adoption_by_framing.collaborative | 0.5583 | 0.55 |
| `deliberation_framing_summary.json` | adoption_by_framing.none | 0.7417 | 0.725 |
| `deliberation_framing_summary.json` | none_vs_collaborative.gain | 5 | 6 |
| `deliberation_framing_summary.json` | none_vs_collaborative.pvalue | 0.000113 | 0.000324 |
| `deliberation_framing_summary.json` | none_vs_critical.gain | 2 | 3 |
| `deliberation_framing_summary.json` | none_vs_critical.lose | 57 | 56 |
| `deliberation_framing_summary.json` | none_vs_independent.gain | 3 | 4 |
| `deliberation_framing_summary.json` | none_vs_independent.lose | 48 | 47 |
| `dose_response_summary.json` | adoption_by_dose.l2_lean | 0.4583 | 0.4417 |
| `dose_response_summary.json` | adoption_by_dose.l3_assert | 0.7667 | 0.7583 |
| `dose_response_summary.json` | adoption_by_dose.l4_emphatic | 0.55 | 0.5333 |
| `dose_response_summary.json` | faint_vs_assert.gain | 70 | 69 |
| `dose_response_summary.json` | faint_vs_emphatic.gain | 44 | 42 |
| `dose_response_summary.json` | lean_vs_emphatic.gain | 20 | 19 |
| `dose_response_summary.json` | lean_vs_emphatic.lose | 9 | 8 |
| `dose_response_summary.json` | lean_vs_emphatic.pvalue | 0.061428 | 0.052239 |
| `leader_as_auditor_summary.json` | adoption_by_role.peer | 0.7417 | 0.725 |
| `leader_as_auditor_summary.json` | adoption_by_role.signoff | 0.4917 | 0.4667 |
| `leader_as_auditor_summary.json` | auditor_vs_signoff.gain | 40 | 38 |
| `leader_as_auditor_summary.json` | auditor_vs_signoff.lose | 4 | 5 |
| `leader_as_auditor_summary.json` | peer_vs_auditor.gain | 1 | 2 |
| `leader_as_auditor_summary.json` | peer_vs_auditor.lose | 67 | 66 |
| `leader_as_auditor_summary.json` | peer_vs_signoff.gain | 4 | 5 |
| `leader_as_auditor_summary.json` | peer_vs_signoff.lose | 34 | 36 |
| `plausible_distractor_summary.json` | adoption.implausible | 0.7397 | 0.7042 |
| `plausible_distractor_summary.json` | adoption.plausible | 0.8219 | 0.8169 |
| `plausible_distractor_summary.json` | n | 73 | 71 |
| `plausible_distractor_summary.json` | plausible_vs_implausible.gain | 9 | 10 |
| `plausible_distractor_summary.json` | plausible_vs_implausible.lose | 3 | 2 |
| `plausible_distractor_summary.json` | plausible_vs_implausible.pvalue | 0.145996 | 0.038574 |
| `rationale_validity_summary.json` | adoption_by_rationale.bare | 0.7083 | 0.7 |
| `rationale_validity_summary.json` | adoption_by_rationale.named_fallacy | 0.3917 | 0.375 |
| `rationale_validity_summary.json` | bare_vs_named_fallacy.lose | 41 | 42 |
| `rationale_validity_summary.json` | bare_vs_valid_wrong.lose | 45 | 44 |
| `rationale_validity_summary.json` | valid_wrong_vs_named_fallacy.gain | 16 | 14 |
| `rationale_validity_summary.json` | valid_wrong_vs_named_fallacy.pvalue | 0.711071 | 1.0 |
| `seed_confidence_summary.json` | confidence_elasticity | 0.26 | 0.24 |
| `seed_confidence_summary.json` | confident_adoption | 0.44 | 0.41 |
| `seed_confidence_summary.json` | confident_vs_hedged_mcnemar.gain | 26 | 24 |
| `seed_confidence_summary.json` | hedged_adoption | 0.18 | 0.17 |
| `super_additivity_summary.json` | adoption.both | 0.85 | 0.8333 |
| `super_additivity_summary.json` | adoption.peer | 0.7417 | 0.725 |
| `super_additivity_summary.json` | adoption.system | 0.7 | 0.6917 |
| `super_additivity_summary.json` | both_vs_stronger_single.gain | 19 | 20 |
| `super_additivity_summary.json` | both_vs_stronger_single.lose | 6 | 7 |
| `super_additivity_summary.json` | both_vs_stronger_single.pvalue | 0.014633 | 0.019157 |
| `super_additivity_summary.json` | interaction_both_minus_sum_of_singles | -0.5917 | -0.5834 |
| `temperature_sensitivity_summary.json` | adoption_rate_by_temperature.t0.0 | 0.7417 | 0.725 |
| `temperature_sensitivity_summary.json` | adoption_rate_by_temperature.t0.7 | 0.7528 | 0.7417 |
| `temperature_sensitivity_summary.json` | adoption_rate_by_temperature.t1.0 | 0.7139 | 0.6972 |
| `temperature_sensitivity_summary.json` | temp_gt0_within_case_flip_fraction.t0.7 | 0.275 | 0.2833 |
| `temperature_sensitivity_summary.json` | temp_gt0_within_case_flip_fraction.t1.0 | 0.2667 | 0.275 |
| `test_awareness_summary.json` | adoption_by_framing.accuracy_eval | 0.575 | 0.5667 |
| `test_awareness_summary.json` | adoption_by_framing.agreement_eval | 0.6167 | 0.6083 |
| `test_awareness_summary.json` | adoption_by_framing.neutral | 0.7417 | 0.725 |
| `test_awareness_summary.json` | neutral_vs_accuracy_eval.gain | 6 | 7 |
| `test_awareness_summary.json` | neutral_vs_accuracy_eval.pvalue | 0.000535 | 0.001319 |
| `test_awareness_summary.json` | neutral_vs_agreement_eval.gain | 8 | 9 |
| `test_awareness_summary.json` | neutral_vs_agreement_eval.pvalue | 0.010674 | 0.020062 |
| `text_cue_types_summary.json` | adoption_by_cue.baseline | 0.7417 | 0.725 |
| `text_cue_types_summary.json` | adoption_by_cue.qualifier | 0.75 | 0.7417 |
| `text_cue_types_summary.json` | baseline_vs_negation.gain | 23 | 25 |
| `text_cue_types_summary.json` | baseline_vs_negation.pvalue | 3e-06 | 1e-06 |
| `text_cue_types_summary.json` | baseline_vs_primacy.gain | 6 | 7 |
| `text_cue_types_summary.json` | baseline_vs_primacy.lose | 27 | 26 |
| `text_cue_types_summary.json` | baseline_vs_primacy.pvalue | 0.000324 | 0.001319 |
| `text_cue_types_summary.json` | baseline_vs_qualifier.gain | 9 | 11 |
| `text_cue_types_summary.json` | baseline_vs_qualifier.lose | 8 | 9 |
| `text_cue_types_summary.json` | baseline_vs_qualifier.pvalue | 1.0 | 0.823803 |
| `unanimity_break_summary.json` | adoption_unanimous_two_wrong_peers | 0.4211 | 0.3947 |
| `unanimity_break_summary.json` | dissenter_reduction | 0.3948 | 0.3684 |
| `unanimity_break_summary.json` | unanimous_vs_dissenter_mcnemar.gain | 15 | 14 |
| `unanimity_break_summary.json` | unanimous_vs_dissenter_mcnemar.pvalue | 6.1e-05 | 0.000122 |
| `referee_deployable_summary.json` | extra_requery_calls_needed | <absent> | 40 |
| `stats_reconciliation_summary.json` | contrasts[0].delta | 0.7167 | 0.7667 |
| `stats_reconciliation_summary.json` | contrasts[0].delta_ci95[0] | 0.6 | 0.65 |
| `stats_reconciliation_summary.json` | contrasts[0].delta_ci95[1] | 0.8333 | 0.8667 |
| `stats_reconciliation_summary.json` | contrasts[0].mcnemar_gain | 43 | 46 |
| `stats_reconciliation_summary.json` | contrasts[0].rate_a | 0.2 | 0.15 |
| `stats_reconciliation_summary.json` | contrasts[1].delta | 0.26 | 0.24 |
| `stats_reconciliation_summary.json` | contrasts[1].delta_ci95[0] | 0.18 | 0.16 |
| `stats_reconciliation_summary.json` | contrasts[1].delta_ci95[1] | 0.35 | 0.33 |
| `stats_reconciliation_summary.json` | contrasts[1].mcnemar_gain | 26 | 24 |
| `stats_reconciliation_summary.json` | contrasts[1].pvalue | 3e-08 | 1.2e-07 |
| `stats_reconciliation_summary.json` | contrasts[1].rate_a | 0.18 | 0.17 |
| `stats_reconciliation_summary.json` | contrasts[1].rate_b | 0.44 | 0.41 |
| `stats_reconciliation_summary.json` | contrasts[2].delta | -0.3417 | -0.3333 |
| `stats_reconciliation_summary.json` | contrasts[2].delta_ci95[0] | -0.4417 | -0.4333 |
| `stats_reconciliation_summary.json` | contrasts[2].delta_ci95[1] | -0.25 | -0.2417 |
| `stats_reconciliation_summary.json` | contrasts[2].mcnemar_lose | 45 | 44 |
| `stats_reconciliation_summary.json` | contrasts[2].rate_a | 0.7083 | 0.7 |
| `stats_reconciliation_summary.json` | contrasts[3].delta_ci95[0] | 0.0333 | 0.025 |
| `stats_reconciliation_summary.json` | contrasts[3].mcnemar_gain | 19 | 20 |
| `stats_reconciliation_summary.json` | contrasts[3].mcnemar_lose | 6 | 7 |
| `stats_reconciliation_summary.json` | contrasts[3].pvalue | 0.0146333 | 0.01915729 |
| `stats_reconciliation_summary.json` | contrasts[3].rate_a | 0.7417 | 0.725 |
| `stats_reconciliation_summary.json` | contrasts[3].rate_b | 0.85 | 0.8333 |
| `stats_reconciliation_summary.json` | contrasts[5].delta | 0.3947 | 0.3684 |
| `stats_reconciliation_summary.json` | contrasts[5].delta_ci95[0] | 0.2368 | 0.2105 |
| `stats_reconciliation_summary.json` | contrasts[5].delta_ci95[1] | 0.5526 | 0.5263 |
| `stats_reconciliation_summary.json` | contrasts[5].mcnemar_gain | 15 | 14 |
| `stats_reconciliation_summary.json` | contrasts[5].pvalue | 6.104e-05 | 0.00012207 |
| `stats_reconciliation_summary.json` | contrasts[5].rate_b | 0.4211 | 0.3947 |

### The 47 mirrored cells in the hosted nemotron comparison file

| Row | Arm | Metric | `gemini` on main | `gemini` on this branch | `model` (hosted nemotron, unchanged) |
|---|---|---|---|---|---|
| 0 | attributed_tier | adoption_by_attribution.human_senior | 0.8083 | 0.8 | 0.3833 |
| 3 | attributed_tier | adoption_by_attribution.unlabeled | 0.8417 | 0.825 | 0.4833 |
| 5 | attributed_tier | unlabeled_vs_human_senior.pvalue | 0.34375 | 0.548828 | 0.011818 |
| 7 | attributed_tier | unlabeled_vs_senior_model.pvalue | 0.005925 | 0.02412 | 0.009355 |
| 8 | authority_ladder | adjacent_rung_mcnemar.automated_system_vs_clinical_guideline.pvalue | 0.004425 | 0.002577 | 0.0 |
| 11 | authority_ladder | adoption_by_rung.automated_system | 0.7 | 0.6833 | 0.2833 |
| 13 | authority_ladder | adoption_by_rung.colleague | 0.2 | 0.15 | 0.0667 |
| 14 | authority_ladder | adoption_by_rung.senior_attending | 0.7167 | 0.7 | 0.5833 |
| 55 | committee_size_sweep | adoption_by_committee_size.s0 | 0.7417 | 0.725 | 0.3667 |
| 56 | committee_size_sweep | adoption_by_committee_size.s1 | 0.2667 | 0.2583 | 0.0917 |
| 62 | deliberation_framing | adoption_by_framing.collaborative | 0.5583 | 0.55 | 0.45 |
| 65 | deliberation_framing | adoption_by_framing.none | 0.7417 | 0.725 | 0.4083 |
| 67 | deliberation_framing | none_vs_collaborative.pvalue | 0.000113 | 0.000324 | 0.332306 |
| 71 | dose_response | adoption_by_dose.l3_assert | 0.7667 | 0.7583 | 0.4917 |
| 72 | dose_response | adoption_by_dose.l4_emphatic | 0.55 | 0.5333 | 0.4 |
| 75 | dose_response | lean_vs_emphatic.pvalue | 0.061428 | 0.052239 | 0.000855 |
| 83 | leader_as_auditor | adoption_by_role.peer | 0.7417 | 0.725 | 0.425 |
| 84 | leader_as_auditor | adoption_by_role.signoff | 0.4917 | 0.4667 | 0.1917 |
| 96 | plausible_distractor | adoption.implausible | 0.7397 | 0.7042 | 0.3448 |
| 97 | plausible_distractor | adoption.plausible | 0.8219 | 0.8169 | 0.5057 |
| 98 | plausible_distractor | plausible_vs_implausible.pvalue | 0.145996 | 0.038574 | 0.006611 |
| 99 | rationale_validity | adoption_by_rationale.bare | 0.7083 | 0.7 | 0.275 |
| 100 | rationale_validity | adoption_by_rationale.named_fallacy | 0.3917 | 0.375 | 0.1833 |
| 104 | rationale_validity | valid_wrong_vs_named_fallacy.pvalue | 0.711071 | 1.0 | 0.548828 |
| 153 | seed_confidence | confidence_elasticity | 0.26 | 0.24 | 0.15 |
| 156 | super_additivity | adoption.both | 0.85 | 0.8333 | 0.425 |
| 158 | super_additivity | adoption.peer | 0.7417 | 0.725 | 0.425 |
| 159 | super_additivity | adoption.system | 0.7 | 0.6917 | 0.225 |
| 160 | super_additivity | both_vs_stronger_single.pvalue | 0.014633 | 0.019157 | 1.0 |
| 161 | super_additivity | interaction_both_minus_sum_of_singles | -0.5917 | -0.5834 | -0.225 |
| 162 | temperature_sensitivity | adoption_rate_by_temperature.t0.0 | 0.7417 | 0.725 | 0.425 |
| 164 | temperature_sensitivity | adoption_rate_by_temperature.t0.7 | 0.7528 | 0.7417 | 0.4222 |
| 165 | temperature_sensitivity | adoption_rate_by_temperature.t1.0 | 0.7139 | 0.6972 | 0.4194 |
| 167 | temperature_sensitivity | temp_gt0_within_case_flip_fraction.t0.7 | 0.275 | 0.2833 | 0.35 |
| 168 | temperature_sensitivity | temp_gt0_within_case_flip_fraction.t1.0 | 0.2667 | 0.275 | 0.3083 |
| 170 | test_awareness | adoption_by_framing.accuracy_eval | 0.575 | 0.5667 | 0.25 |
| 171 | test_awareness | adoption_by_framing.agreement_eval | 0.6167 | 0.6083 | 0.4583 |
| 172 | test_awareness | adoption_by_framing.neutral | 0.7417 | 0.725 | 0.4083 |
| 173 | test_awareness | neutral_vs_accuracy_eval.pvalue | 0.000535 | 0.001319 | 6.6e-05 |
| 174 | test_awareness | neutral_vs_agreement_eval.pvalue | 0.010674 | 0.020062 | 0.307456 |
| 175 | text_cue_types | adoption_by_cue.baseline | 0.7417 | 0.725 | 0.425 |
| 177 | text_cue_types | adoption_by_cue.qualifier | 0.75 | 0.7417 | 0.5333 |
| 178 | text_cue_types | baseline_vs_negation.pvalue | 3e-06 | 1e-06 | 0.0 |
| 179 | text_cue_types | baseline_vs_primacy.pvalue | 0.000324 | 0.001319 | 0.000878 |
| 180 | text_cue_types | baseline_vs_qualifier.pvalue | 1.0 | 0.823803 | 0.004425 |
| 182 | unanimity_break | adoption_unanimous_two_wrong_peers | 0.4211 | 0.3947 | 0.16 |
| 185 | unanimity_break | unanimous_vs_dissenter_mcnemar.pvalue | 6.1e-05 | 0.000122 | 0.375 |

## Not done here

- No model calls, so nothing that needs one: the abstention-aware n=600 file for Qwen on MIMIC-CXR
  text, the Gemini MedMCQA `orchestrator_failure`, solo, `clean_a` and `scale_c` misses, and every
  cache miss above.
- The MIMIC-CXR image arms (films not found on ORCD) and the MIMIC-CXR text matched-baseline arm
  (needs the de-identified index flags) were not replayed.
- `experiments/medmcqa/results/RESULTS.md` prose still quotes the earlier rates; a note at its top
  points here.
