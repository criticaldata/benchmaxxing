#!/usr/bin/env bash
set -euo pipefail
REPO=/data/PROJECTS/kennedy/model-tests/benchmaxxing-main
MODEL=meta/llama-3.2-11b-vision-instruct
SLUG=meta_llama-3.2-11b-vision-instruct
cd "$REPO"
export BENCHMAXXING_LOCAL_BASE_URL=http://127.0.0.1:11437/v1
printf "Resume CheXpert blind_metric n=35 from cache (%s lines) at %s UTC\\n" "$(wc -l < experiments/chexpert/results/$SLUG/imaging_blind_metric_call_cache_q4_k_m.jsonl)" "$(date -u +%FT%TZ)"
.venv/bin/python experiments/imaging_chexpert/imaging_blind_metric.py --manifest experiments/imaging_chexpert/results/nih_match_35.csv --image-root data/chexpert --model "$MODEL" --n 35 --cache "experiments/chexpert/results/$SLUG/imaging_blind_metric_call_cache_q4_k_m.jsonl" --out experiments/chexpert/results > "experiments/chexpert/results/$SLUG/imaging_blind_metric_q4_k_m_n35.log" 2>&1
printf "Starting CheXpert matched_baseline_reason at %s UTC\\n" "$(date -u +%FT%TZ)"
.venv/bin/python experiments/matched_baseline/matched_baseline_reason.py --lane imaging --manifest experiments/imaging_chexpert/results/nih_match_35.csv --image-root data/chexpert --cache "experiments/chexpert/results/$SLUG/imaging_blind_metric_call_cache_q4_k_m.jsonl" --out "experiments/chexpert/results/$SLUG" --model "$MODEL" --n 35 --repeats 5 > "experiments/chexpert/results/$SLUG/matched_baseline_reason_q4_k_m_n35.log" 2>&1
printf "Starting CheXpert matched_conditions at %s UTC\\n" "$(date -u +%FT%TZ)"
.venv/bin/python experiments/blind_metric/matched_conditions.py --lane imaging --manifest experiments/imaging_chexpert/results/nih_match_35.csv --image-root data/chexpert --case-ids data/chexpert_qwen35_case_ids.txt --out experiments/chexpert/results/matched --cache "experiments/chexpert/results/$SLUG/imaging_blind_metric_call_cache_q4_k_m.jsonl" --model "$MODEL" --n 35 --repeats 5 --seed 0 --conditions reason,note,note_after,note_salient,note_flagged,note_salient_flagged,note_noimage --session lin > "experiments/chexpert/results/matched/$SLUG/matched_conditions_q4_k_m_n35.log" 2>&1
.venv/bin/python experiments/blind_metric/matched_conditions_report.py --cohort "chexpert=experiments/chexpert/results/matched/$SLUG" --out "experiments/chexpert/results/matched/$SLUG"
printf "CheXpert hidden-note complete at %s UTC\\n" "$(date -u +%FT%TZ)"
