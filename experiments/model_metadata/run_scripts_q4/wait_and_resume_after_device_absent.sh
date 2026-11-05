#!/usr/bin/env bash
set -Eeuo pipefail
REPO=/data/PROJECTS/kennedy/model-tests/benchmaxxing-main
LEASE=/data/PROJECTS/bohdi/scripts/gpu_lease.sh
WAIT_LOG=/tmp/run_llama_q4_wait_for_gpu1.log
cd "$REPO"
printf 'Waiting for GPU 1 to be free at %s UTC\n' "$(date -u +%FT%TZ)" >> "$WAIT_LOG"
while true; do
  state=$(awk '$1==1 {print $2}' /home/nvidia/GPUS.txt 2>/dev/null || true)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null || echo 99999)
  if [[ "$state" == "free" && "$mem" =~ ^[0-9]+$ ]] && (( mem < 2000 )); then
    if CLAIM_PID=$$ "$LEASE" claim 1 kennedy llama32-vision-q4km-battery; then
      break
    fi
  fi
  printf 'Still waiting %s UTC state=%s memory=%sMiB\n' "$(date -u +%FT%TZ)" "${state:-unknown}" "${mem:-unknown}" >> "$WAIT_LOG"
  sleep 60
done
export BENCHMAXXING_LOCAL_BASE_URL=http://127.0.0.1:11437/v1
trap '"$LEASE" release 1' EXIT
printf 'Resuming after device_absent at %s UTC\n' "$(date -u +%FT%TZ)" >> "$WAIT_LOG"
bash experiments/model_metadata/run_scripts_q4/remaining_after_device_absent.sh
