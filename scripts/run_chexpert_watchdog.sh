#!/usr/bin/env bash
set -euo pipefail

# Resume CheXpert only after a real inference call to the requested NVIDIA model succeeds.
# This script never reports a phase as complete unless the runner exits with status 0.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

PYTHON=/Users/yehu/Documents/MIT/benchmaxxing/.venv/bin/python
MODEL=meta/llama-3.2-90b-vision-instruct
MANIFEST=experiments/imaging_chexpert/results/solo_600.csv
IMAGE_ROOT=/Users/yehu/.cache/kagglehub/datasets/ashery/chexpert/versions/1
CACHE=${CHEXPERT_CACHE:-experiments/chexpert/results/600_runs/img_cache_llama.jsonl}
STATUS_LOG=${CHEXPERT_STATUS_LOG:-experiments/chexpert/results/api_watchdog.log}
N100_OUT=${CHEXPERT_N100_OUT:-experiments/chexpert/results/phase1_n100}
N300_OUT=${CHEXPERT_N300_OUT:-experiments/chexpert/results/phase1_n300}
N600_OUT=${CHEXPERT_N600_OUT:-experiments/chexpert/results/phase1_n600}

mkdir -p "$(dirname "$STATUS_LOG")" "$N100_OUT" "$N300_OUT" "$N600_OUT"

log_status() {
  local cache_rows=0
  if [ -f "$CACHE" ]; then
    cache_rows=$(wc -l < "$CACHE")
  fi
  printf '%s model=%s phase=%s event=%s http=%s cache_rows=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODEL" "$1" "$2" "$3" \
    "$cache_rows" >> "$STATUS_LOG"
}

load_api_key() {
  "$PYTHON" - <<'PY'
from dotenv import dotenv_values
values = dotenv_values('.env')
key = values.get('N_2') or values.get('NVIDIA_API_KEY')
if key:
    print(key)
PY
}

probe_api() {
  local body code api_key
  api_key=$(load_api_key)
  body=$(mktemp)
  code=$(curl -sS --max-time 30 -o "$body" -w '%{http_code}' \
    https://integrate.api.nvidia.com/v1/chat/completions \
    -H "Authorization: Bearer ${api_key:-}" \
    -H 'Content-Type: application/json' \
    --data-raw '{"model":"meta/llama-3.2-90b-vision-instruct","messages":[{"role":"user","content":"Reply with only OK."}],"max_tokens":1,"temperature":0}' \
    2>/dev/null || true)
  if [ "$code" = 200 ] && grep -q '"choices"' "$body"; then
    rm -f "$body"
    printf '%s\n' "$code"
    return 0
  fi
  rm -f "$body"
  printf '%s\n' "$code"
  return 1
}

run_phase() {
  local n=$1 out=$2 api_key
  api_key=$(load_api_key)
  NVIDIA_API_KEY="$api_key" "$PYTHON" experiments/imaging_chexpert/imaging_blind_metric.py \
    --manifest "$MANIFEST" \
    --image-root "$IMAGE_ROOT" \
    --model "$MODEL" \
    --cache "$CACHE" \
    --out "$out" \
    --n "$n" --rpm 40 --request-retries 2 --max-retries 0 \
    2>&1 | tee -a "$out/run.log"
}

while true; do
  if api_code=$(probe_api); then
    log_status n100 probe_ok "$api_code"
    if run_phase 100 "$N100_OUT"; then
      log_status n100 runner_complete "$api_code"
      if run_phase 300 "$N300_OUT"; then
        log_status n300 runner_complete "$api_code"
        if run_phase 600 "$N600_OUT"; then
          log_status n600 runner_complete "$api_code"
          exit 0
        fi
        log_status n600 runner_failed "$api_code"
      else
        log_status n300 runner_failed "$api_code"
      fi
    else
      log_status n100 runner_failed "$api_code"
    fi
  else
    log_status waiting api_unavailable "$api_code"
  fi
  sleep 60
done
