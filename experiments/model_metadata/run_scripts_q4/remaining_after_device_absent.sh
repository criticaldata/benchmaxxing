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
printf 'RESUME_AFTER_DEVICE_ABSENT %s UTC\n' "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
C=data/q4_case_cohorts/chexpert
CH=experiments/imaging_chexpert/results/$SLUG
# Restore the shared 35-case base cascade from its complete cache before 150-case arms reuse the generic output path.
run imaging_chexpert_cascade_base_replay .venv/bin/python experiments/imaging/imaging_cascade.py --manifest "$C/imaging_cascade.csv" --image-root data/chexpert --out experiments/imaging_chexpert/results --model "$MODEL" --n 35 --cue none --cache "$CH/imaging_cascade_call_cache_q4_k_m.jsonl"
cp "$CH/imaging_cascade_none.jsonl" "$CH/imaging_cascade.jsonl"
cp "$CH/imaging_cascade_none_summary.json" "$CH/imaging_cascade_summary.json"
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
