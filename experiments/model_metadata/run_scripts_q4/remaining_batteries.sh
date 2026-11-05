#!/usr/bin/env bash
set -Eeuo pipefail
REPO=/data/PROJECTS/kennedy/model-tests/benchmaxxing-main
MODEL=meta/llama-3.2-11b-vision-instruct
SLUG=meta_llama-3.2-11b-vision-instruct
LOG_DIR=experiments/model_metadata/run_logs/q4_remaining
RUN_LOG=experiments/model_metadata/q4_remaining_run.log
cd "$REPO"
mkdir -p "$LOG_DIR"
export BENCHMAXXING_LOCAL_BASE_URL=http://127.0.0.1:11437/v1

TEXT_PID=$(cat /tmp/run_llama_q4_text_batteries.pid)
printf 'Waiting for text battery PID %s at %s UTC\n' "$TEXT_PID" "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
while kill -0 "$TEXT_PID" 2>/dev/null; do sleep 30; done
if ! grep -q 'TEXT BATTERY COMPLETE' experiments/model_metadata/medqa_medmcqa_q4_run.log; then
  echo 'Text battery did not complete successfully; stopping queued follow-on work.' | tee -a "$RUN_LOG"
  exit 1
fi

run() {
  name=$1
  shift
  printf 'START %s %s UTC\n' "$name" "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
  if "$@" >"$LOG_DIR/$name.log" 2>&1; then
    printf 'DONE %s %s UTC\n' "$name" "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
  else
    rc=$?
    printf 'FAILED %s status=%s %s UTC\n' "$name" "$rc" "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
    tail -80 "$LOG_DIR/$name.log" | tee -a "$RUN_LOG"
    exit "$rc"
  fi
}
I=data/q4_case_cohorts/imaging
IH=experiments/imaging/results/$SLUG
run imaging_nih_solo .venv/bin/python experiments/imaging/imaging_solo.py --manifest "$I/imaging_solo.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cache "$IH/imaging_solo_call_cache_q4_k_m.jsonl"
run imaging_nih_solo_lite .venv/bin/python experiments/imaging/imaging_solo_lite.py --manifest "$I/imaging_solo_lite.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cache "$IH/imaging_solo_lite_call_cache_q4_k_m.jsonl"
run imaging_nih_cascade .venv/bin/python experiments/imaging/imaging_cascade.py --manifest "$I/imaging_cascade.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cue none --cache "$IH/imaging_cascade_call_cache_q4_k_m.jsonl"
run imaging_nih_cascade_cable .venv/bin/python experiments/imaging/imaging_cascade.py --manifest "$I/imaging_cascade_cable.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cue cable --cache "$IH/imaging_cascade_call_cache_q4_k_m.jsonl"
run imaging_nih_cascade_corner_tag .venv/bin/python experiments/imaging/imaging_cascade.py --manifest "$I/imaging_cascade_corner_tag.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cue corner_tag --cache "$IH/imaging_cascade_call_cache_q4_k_m.jsonl"
run imaging_nih_cascade_laterality .venv/bin/python experiments/imaging/imaging_cascade.py --manifest "$I/imaging_cascade_laterality.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cue laterality --cache "$IH/imaging_cascade_call_cache_q4_k_m.jsonl"
run imaging_nih_cascade_scale .venv/bin/python experiments/imaging/imaging_scale.py --manifest "$I/imaging_cascade_scale.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 150 --cue watermark --cache "$IH/imaging_cascade_scale_call_cache_q4_k_m.jsonl"
run imaging_nih_cue_combo .venv/bin/python experiments/imaging/imaging_cue_combo.py --manifest "$I/imaging_cue_combo.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cache "$IH/imaging_cue_combo_call_cache_q4_k_m.jsonl"
run imaging_nih_cue_families .venv/bin/python experiments/imaging/imaging_cue_families.py --manifest "$I/imaging_cue_families.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cache "$IH/imaging_cue_families_call_cache_q4_k_m.jsonl"
# Qwen has both the text-only legacy control and the film-visible judge; retain both under model scope.
run imaging_nih_judge_referee_text_only .venv/bin/python experiments/imaging/imaging_judge_referee.py --cascade-jsonl "$IH/imaging_cascade_none.jsonl" --manifest "$I/imaging_judge_referee.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --text-only --cache "$IH/imaging_judge_referee_text_only_cache_q4_k_m.jsonl"
cp "$IH/imaging_judge_referee.jsonl" "$IH/imaging_judge_referee_text_only.jsonl"
cp "$IH/imaging_judge_referee_summary.json" "$IH/imaging_judge_referee_summary_text_only.json"
run imaging_nih_judge_referee_film .venv/bin/python experiments/imaging/imaging_judge_referee.py --cascade-jsonl "$IH/imaging_cascade_none.jsonl" --manifest "$I/imaging_judge_referee_film.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --cache "$IH/imaging_judge_referee_film_cache_q4_k_m.jsonl"
cp "$IH/imaging_judge_referee.jsonl" "$IH/imaging_judge_referee_film.jsonl"
cp "$IH/imaging_judge_referee_summary.json" "$IH/imaging_judge_referee_summary_film.json"
cp "$IH/imaging_judge_referee_text_only.jsonl" "$IH/imaging_judge_referee.jsonl"
cp "$IH/imaging_judge_referee_summary_text_only.json" "$IH/imaging_judge_referee_summary.json"
run imaging_nih_majority_pressure .venv/bin/python experiments/imaging/imaging_majority_pressure.py --manifest "$I/imaging_majority_pressure.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cue watermark --cache "$IH/imaging_majority_pressure_call_cache_q4_k_m.jsonl"
run imaging_nih_matched_temp .venv/bin/python experiments/imaging/imaging_matched_temp_floor.py --manifest "$I/imaging_matched_temp.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cue watermark
run imaging_nih_multi_round .venv/bin/python experiments/imaging/imaging_multi_round.py --manifest "$I/imaging_multi_round.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --rounds 3 --cache "$IH/imaging_multi_round_call_cache_q4_k_m.jsonl"
run imaging_nih_peer_size_curve .venv/bin/python experiments/imaging/imaging_peer_size_curve.py --manifest "$I/imaging_peer_size_curve.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cue watermark --cache "$IH/imaging_peer_size_curve_call_cache_q4_k_m.jsonl"
run imaging_nih_polarity .venv/bin/python experiments/imaging/imaging_polarity.py --manifest "$I/imaging_polarity.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cache "$IH/imaging_polarity_call_cache_q4_k_m.jsonl"
run imaging_nih_referee .venv/bin/python experiments/imaging/imaging_referee.py --manifest "$I/imaging_referee.csv" --image-root data/nih_cxr14 --cascade-jsonl "$IH/imaging_cascade_none.jsonl" --out experiments/imaging/results --model "$MODEL" --cache "$IH/imaging_referee_call_cache_q4_k_m.jsonl"
run imaging_nih_strength_cascade .venv/bin/python experiments/imaging/imaging_strength_cascade.py --manifest "$I/imaging_strength_cascade.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cache "$IH/imaging_strength_cascade_call_cache_q4_k_m.jsonl"
run imaging_nih_system_flag .venv/bin/python experiments/imaging/imaging_system_flag.py --manifest "$I/imaging_system_flag.csv" --image-root data/nih_cxr14 --out experiments/imaging/results --model "$MODEL" --n 35 --cue none --cache "$IH/imaging_system_flag_call_cache_q4_k_m.jsonl"

C=data/q4_case_cohorts/chexpert
CH=experiments/imaging_chexpert/results/$SLUG
run imaging_chexpert_solo .venv/bin/python experiments/imaging/imaging_solo.py --manifest "$C/imaging_solo.csv" --image-root data/chexpert --out experiments/imaging_chexpert/results --model "$MODEL" --n 35 --cache "$CH/imaging_solo_call_cache_q4_k_m.jsonl"
run imaging_chexpert_cascade .venv/bin/python experiments/imaging/imaging_cascade.py --manifest "$C/imaging_cascade.csv" --image-root data/chexpert --out experiments/imaging_chexpert/results --model "$MODEL" --n 35 --cue none --cache "$CH/imaging_cascade_call_cache_q4_k_m.jsonl"
cp "$CH/imaging_cascade_none.jsonl" "$CH/imaging_cascade.jsonl"
cp "$CH/imaging_cascade_none_summary.json" "$CH/imaging_cascade_summary.json"
run imaging_chexpert_matched_temp .venv/bin/python experiments/imaging/imaging_matched_temp_floor.py --manifest "$C/imaging_matched_temp.csv" --image-root data/chexpert --out experiments/imaging_chexpert/results --model "$MODEL" --n 35 --cue watermark
run imaging_chexpert_device_absent .venv/bin/python experiments/imaging/imaging_cascade.py --manifest "$C/device_absent_cascade.csv" --image-root data/chexpert --out experiments/imaging_chexpert/results --model "$MODEL" --n 150 --cue none --cache "$CH/imaging_cascade_device_absent_call_cache_q4_k_m.jsonl"
cp "$CH/imaging_cascade_none.jsonl" "$CH/imaging_cascade_device_absent.jsonl"
cp "$CH/imaging_cascade_none_summary.json" "$CH/imaging_cascade_device_absent_summary.json"
run imaging_chexpert_natural_cues .venv/bin/python experiments/imaging/imaging_cascade.py --manifest "$C/natural_cues_cascade.csv" --image-root data/chexpert --out experiments/imaging_chexpert/results --model "$MODEL" --n 150 --cue none --cache "$CH/imaging_cascade_natural_cues_call_cache_q4_k_m.jsonl"
cp "$CH/imaging_cascade_none.jsonl" "$CH/imaging_cascade_natural_cues.jsonl"
cp "$CH/imaging_cascade_none_summary.json" "$CH/imaging_cascade_natural_cues_summary.json"
run imaging_chexpert_natural_independent .venv/bin/python experiments/imaging/imaging_cascade.py --manifest "$C/natural_independent_cascade.csv" --image-root data/chexpert --out experiments/imaging_chexpert/results --model "$MODEL" --n 150 --cue none --cache "$CH/imaging_cascade_natural_independent_call_cache_q4_k_m.jsonl"
cp "$CH/imaging_cascade_none.jsonl" "$CH/imaging_cascade_natural_independent.jsonl"
cp "$CH/imaging_cascade_none_summary.json" "$CH/imaging_cascade_natural_independent_summary.json"
run imaging_chexpert_system_flag .venv/bin/python experiments/imaging/imaging_system_flag.py --manifest "$C/system_flag.csv" --image-root data/chexpert --out experiments/imaging_chexpert/results --model "$MODEL" --n 150 --cue none --cache "$CH/imaging_system_flag_call_cache_q4_k_m.jsonl"

S=data/q4_case_cohorts/support2/support2.csv
SH=experiments/support2/results/$SLUG
SC="$SH/support2_call_cache_q4_k_m.jsonl"
SN="$SH/support2_noise_resamples_q4_k_m.jsonl"
run support2_solo .venv/bin/python -m experiments.support2.support2_solo --manifest "$S" --cache "$SC" --noise-log "$SN" --out experiments/support2/results --model "$MODEL" --n 120
run support2_cascade .venv/bin/python -m experiments.support2.support2_cascade --manifest "$S" --cache "$SC" --out experiments/support2/results --model "$MODEL" --n 120
run support2_cascade_strength .venv/bin/python -m experiments.support2.support2_cascade_strength --manifest "$S" --cache "$SC" --out experiments/support2/results --model "$MODEL" --n 120
run support2_referee .venv/bin/python -m experiments.support2.support2_referee --manifest "$S" --cache "$SC" --out experiments/support2/results --model "$MODEL" --n 120
run support2_referee_judge .venv/bin/python -m experiments.support2.support2_referee_judge --manifest "$S" --cache "$SC" --out experiments/support2/results --model "$MODEL" --n 120

R=data/q4_case_cohorts/referee
RH=experiments/referee/results/$SLUG
run referee_deployable .venv/bin/python -m experiments.referee.referee_deployable --manifest "$R/referee_referee_deployable.csv" --cache "$RH/referee_deployable_call_cache_q4_k_m.jsonl" --out experiments/referee/results --model "$MODEL" --n 40
run referee_judge .venv/bin/python -m experiments.referee.referee_judge --manifest "$R/referee_referee_judge.csv" --cache "$RH/referee_judge_call_cache_q4_k_m.jsonl" --out experiments/referee/results --model "$MODEL" --n 40
run referee_requery_design .venv/bin/python -m experiments.referee.referee_requery_design --manifest "$R/referee_referee_requery_design.csv" --board-cache "$RH/referee_requery_board_cache_q4_k_m.jsonl" --requery-cache "$RH/referee_requery_cache_q4_k_m.jsonl" --out experiments/referee/results --model "$MODEL" --n 40
run referee_self_inconsistency .venv/bin/python -m experiments.referee.referee_self_inconsistency --manifest "$R/referee_referee_self_inconsistency.csv" --cache "$RH/referee_self_inconsistency_call_cache_q4_k_m.jsonl" --out experiments/referee/results --model "$MODEL" --n 40
run referee_threshold .venv/bin/python -m experiments.referee.referee_threshold --manifest "$R/referee_referee_threshold.csv" --board-cache "$RH/referee_threshold_board_cache_q4_k_m.jsonl" --requery-cache "$RH/referee_threshold_requery_cache_q4_k_m.jsonl" --out experiments/referee/results --model "$MODEL" --n 40

M=data/q4_case_cohorts/cascade
MR=experiments/cascade/results/$SLUG
run multi_round_n40 .venv/bin/python -m experiments.cascade.multi_round --manifest "$M/multi_round_n40.csv" --cache "$MR/multi_round_n40_call_cache_q4_k_m.jsonl" --out experiments/cascade/results --model "$MODEL" --n 40 --rounds 5
cp "$MR/multi_round.jsonl" "$MR/multi_round_n40.jsonl"
run multi_round_n120 .venv/bin/python -m experiments.cascade.multi_round --manifest "$M/multi_round_n120.csv" --cache "$MR/multi_round_n120_call_cache_q4_k_m.jsonl" --out experiments/cascade/results --model "$MODEL" --n 120 --rounds 5
cp "$MR/multi_round.jsonl" "$MR/multi_round_n120.jsonl"
cp "$MR/multi_round_summary.json" "$MR/multi_round_summary_n120.json"
cp "$MR/multi_round_n40.jsonl" "$MR/multi_round.jsonl"
cp "$MR/multi_round_summary_n40.json" "$MR/multi_round_summary.json"

CON=data/q4_case_cohorts/contamination/contamination_audit.csv
CR=experiments/medqa/results/$SLUG/solo_records.jsonl
run contamination_audit .venv/bin/python -m experiments.contamination.contamination_audit --manifest "$CON" --solo-records "$CR" --cache "experiments/contamination/results/$SLUG/contamination_audit_call_cache_q4_k_m.jsonl" --out experiments/contamination/results --model "$MODEL"
MD=data/q4_case_cohorts/model_dependence/cascade_C_flash.csv
run cascade_C_flash .venv/bin/python -m experiments.model_dependence.cascade_C_flash --manifest "$MD" --case-ids data/q4_case_cohorts/model_dependence/cascade_C_flash.ids.txt --cache "experiments/model_dependence/results/$SLUG/cascade_C_flash_call_cache_q4_k_m.jsonl" --out experiments/model_dependence/results --model "$MODEL" --target 22 --probe-limit 260 --scale-c-summary "experiments/medqa/results/$SLUG/scale_c_summary.json"
printf 'ALL QUEUED FOLLOW-ON BATTERIES COMPLETE %s UTC\n' "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
