#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data/PROJECTS/kennedy/model-tests/benchmaxxing-main
LEASE=/data/PROJECTS/bohdi/scripts/gpu_lease.sh

cd "$REPO"
export BENCHMAXXING_LOCAL_BASE_URL=http://127.0.0.1:11437/v1
CLAIM_PID=$$ "$LEASE" claim 1 kennedy llama32-vision-q4km-battery
trap '"$LEASE" release 1' EXIT
bash experiments/model_metadata/run_scripts_q4/remaining_after_nih_judge.sh
