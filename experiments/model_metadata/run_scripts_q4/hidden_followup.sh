#!/usr/bin/env bash
set -euo pipefail
REPO=/data/PROJECTS/kennedy/model-tests/benchmaxxing-main
MODEL=meta/llama-3.2-11b-vision-instruct
SLUG=meta_llama-3.2-11b-vision-instruct
cd "$REPO"
mkdir -p experiments/model_metadata
exec > >(tee -a experiments/model_metadata/hidden_followup_q4.log) 2>&1
printf 'Waiting for NIH matched_conditions PID to finish at %s UTC\n' "$(date -u +%FT%TZ)"
PID=$(cat /tmp/llama_q4_nih_matched_conditions_n35.pid)
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
test -s experiments/imaging/results/matched/"$SLUG"/matched_conditions.jsonl
printf 'NIH matched_conditions finished; deriving calibrated rows\n'
.venv/bin/python experiments/blind_metric/matched_conditions_report.py \
  --cohort "nih=experiments/imaging/results/matched/$SLUG" \
  --out "experiments/imaging/results/matched/$SLUG"
export BENCHMAXXING_LOCAL_BASE_URL=http://127.0.0.1:11437/v1
printf 'Adding MedQA matched_baseline_reason repeats=5\n'
.venv/bin/python experiments/matched_baseline/matched_baseline_reason.py \
  --lane text --manifest data/medqa/staged/test/medqa_manifest.csv \
  --cache "experiments/blind_metric/results/$SLUG/call_cache_q4_k_m.jsonl" \
  --out "experiments/blind_metric/results/$SLUG" --model "$MODEL" --n 40 --repeats 5 \
  > "experiments/blind_metric/results/$SLUG/matched_baseline_reason_repeats_q4_k_m_n40.log" 2>&1
printf 'Adding NIH matched_baseline_reason repeats=5\n'
.venv/bin/python experiments/matched_baseline/matched_baseline_reason.py \
  --lane imaging --manifest experiments/imaging/results/nih_manifest.csv --image-root data/nih_cxr14 \
  --cache "experiments/imaging/results/$SLUG/imaging_blind_metric_call_cache_q4_k_m.jsonl" \
  --out "experiments/imaging/results/$SLUG" --model "$MODEL" --n 35 --repeats 5 \
  > "experiments/imaging/results/$SLUG/matched_baseline_reason_repeats_q4_k_m_n35.log" 2>&1
printf 'Starting CheXpert blind_metric n=35\n'
mkdir -p "experiments/chexpert/results/$SLUG" "experiments/chexpert/results/matched/$SLUG"
.venv/bin/python experiments/imaging_chexpert/imaging_blind_metric.py \
  --manifest experiments/imaging_chexpert/results/nih_match_35.csv \
  --image-root data/chexpert --model "$MODEL" --n 35 \
  --cache "experiments/chexpert/results/$SLUG/imaging_blind_metric_call_cache_q4_k_m.jsonl" \
  --out experiments/chexpert/results \
  > "experiments/chexpert/results/$SLUG/imaging_blind_metric_q4_k_m_n35.log" 2>&1
printf 'Starting CheXpert matched_baseline_reason n=35, repeats=5\n'
.venv/bin/python experiments/matched_baseline/matched_baseline_reason.py \
  --lane imaging --manifest experiments/imaging_chexpert/results/nih_match_35.csv --image-root data/chexpert \
  --cache "experiments/chexpert/results/$SLUG/imaging_blind_metric_call_cache_q4_k_m.jsonl" \
  --out "experiments/chexpert/results/$SLUG" --model "$MODEL" --n 35 --repeats 5 \
  > "experiments/chexpert/results/$SLUG/matched_baseline_reason_q4_k_m_n35.log" 2>&1
printf 'Starting CheXpert matched_conditions n=35, repeats=5\n'
.venv/bin/python experiments/blind_metric/matched_conditions.py \
  --lane imaging --manifest experiments/imaging_chexpert/results/nih_match_35.csv \
  --image-root data/chexpert --case-ids data/chexpert_qwen35_case_ids.txt \
  --out experiments/chexpert/results/matched \
  --cache "experiments/chexpert/results/$SLUG/imaging_blind_metric_call_cache_q4_k_m.jsonl" \
  --model "$MODEL" --n 35 --repeats 5 --seed 0 \
  --conditions reason,note,note_after,note_salient,note_flagged,note_salient_flagged,note_noimage --session lin \
  > "experiments/chexpert/results/matched/$SLUG/matched_conditions_q4_k_m_n35.log" 2>&1
printf 'Deriving CheXpert calibrated rows at %s UTC\n' "$(date -u +%FT%TZ)"
.venv/bin/python experiments/blind_metric/matched_conditions_report.py \
  --cohort "chexpert=experiments/chexpert/results/matched/$SLUG" \
  --out "experiments/chexpert/results/matched/$SLUG"
printf 'Hidden-note follow-up complete at %s UTC\n' "$(date -u +%FT%TZ)"
