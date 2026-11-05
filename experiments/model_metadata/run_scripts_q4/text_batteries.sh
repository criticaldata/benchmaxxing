#!/usr/bin/env bash
set -uo pipefail
REPO=/data/PROJECTS/kennedy/model-tests/benchmaxxing-main
MODEL=meta/llama-3.2-11b-vision-instruct
SLUG=meta_llama-3.2-11b-vision-instruct
cd "$REPO"
export BENCHMAXXING_LOCAL_BASE_URL=http://127.0.0.1:11437/v1
LOG_DIR=experiments/model_metadata/run_logs/q4
mkdir -p "$LOG_DIR"
RUN_LOG=experiments/model_metadata/medqa_medmcqa_q4_run.log
PRE_PID=$(cat /tmp/run_llama_q4_chexpert_retry.pid)
printf "Waiting for CheXpert hidden-note PID %s at %s UTC\n" "$PRE_PID" "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
while kill -0 "$PRE_PID" 2>/dev/null; do sleep 30; done
if ! grep -q "CheXpert hidden-note complete" /tmp/run_llama_q4_chexpert_retry.wrapper.log; then
  echo "CheXpert hidden-note did not report completion; stopping before the text battery." | tee -a "$RUN_LOG"
  exit 1
fi
run() {
  name=$1
  shift
  printf "START %s %s UTC\n" "$name" "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
  if "$@" >"$LOG_DIR/$name.log" 2>&1; then
    printf "DONE %s %s UTC\n" "$name" "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
  else
    status=$?
    printf "FAILED %s status=%s %s UTC\n" "$name" "$status" "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
    tail -60 "$LOG_DIR/$name.log" | tee -a "$RUN_LOG"
    exit "$status"
  fi
}
M=data/q4_case_cohorts/medqa
FULL=data/medqa/staged/test/medqa_manifest.csv
SHARED_SOLO=experiments/medqa/results/solo_records.jsonl
run medqa_reproduce_solo .venv/bin/python experiments/medqa/reproduce.py --manifest "$M/reproduce_solo.csv" --out experiments/medqa/results --model "$MODEL" --stage solo --solo-n 100 --seed 0
run medqa_reproduce_cascade .venv/bin/python experiments/medqa/reproduce.py --manifest "$M/reproduce_cascade.csv" --out experiments/medqa/results --model "$MODEL" --stage cascade --cascade-n 20 --seed 0
run medqa_attributed_tier .venv/bin/python experiments/medqa/attributed_tier.py --manifest "$M/attributed_tier.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_authority_ladder .venv/bin/python experiments/medqa/authority_ladder.py --manifest "$M/authority_ladder.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_break_it .venv/bin/python experiments/medqa/break_it.py --manifest "$FULL" --solo-records "$SHARED_SOLO" --out experiments/medqa/results --model "$MODEL" --n 20
run medqa_clean_a .venv/bin/python experiments/medqa/clean_a.py --manifest "$FULL" --solo-records "$SHARED_SOLO" --out experiments/medqa/results --model "$MODEL" --n 23
run medqa_push_c .venv/bin/python experiments/medqa/push_c.py --manifest "$FULL" --solo-records "$SHARED_SOLO" --out experiments/medqa/results --model "$MODEL" --n 23
run medqa_scale_c .venv/bin/python experiments/medqa/scale_c.py --manifest "$FULL" --out experiments/medqa/results --model "$MODEL" --target 44 --case-ids "$M/scale_c.ids.txt"
run medqa_committee_size_sweep .venv/bin/python experiments/medqa/committee_size_sweep.py --manifest "$M/committee_size_sweep.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_contamination_cascade .venv/bin/python experiments/medqa/contamination_cascade.py --manifest "$M/contamination_cascade.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_deliberation_channel .venv/bin/python experiments/medqa/deliberation_channel.py --manifest "$M/deliberation_channel.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_deliberation_framing .venv/bin/python experiments/medqa/deliberation_framing.py --manifest "$M/deliberation_framing.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_dose_response .venv/bin/python experiments/medqa/dose_response.py --manifest "$M/dose_response.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_hierarchy_dominance .venv/bin/python experiments/medqa/hierarchy_dominance.py --manifest "$M/hierarchy_dominance.csv" --out experiments/medqa/results --model "$MODEL" --n 40
run medqa_hierarchy_temp .venv/bin/python experiments/medqa/hierarchy_temp.py --manifest "$M/hierarchy_temp.csv" --out experiments/medqa/results --model "$MODEL" --n 40
run medqa_leader_as_auditor .venv/bin/python experiments/medqa/leader_as_auditor.py --manifest "$M/leader_as_auditor.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_live_peer_organic .venv/bin/python experiments/medqa/live_peer_organic.py --manifest "$M/live_peer_organic.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_majority_pressure .venv/bin/python experiments/medqa/majority_pressure.py --manifest "$M/majority_pressure.csv" --out experiments/medqa/results --model "$MODEL" --n 40
run medqa_orchestrator_failure .venv/bin/python experiments/medqa/orchestrator_failure.py --manifest "$FULL" --out experiments/medqa/results --model "$MODEL" --n 95 --case-ids "$M/orchestrator_failure.ids.txt"
run medqa_paraphrase_robustness .venv/bin/python experiments/medqa/paraphrase_robustness.py --manifest "$M/paraphrase_robustness.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_plausible_distractor .venv/bin/python experiments/medqa/plausible_distractor.py --manifest "$M/plausible_distractor.csv" --out experiments/medqa/results --model "$MODEL" --n 106
run medqa_pre_emptive_referee .venv/bin/python experiments/medqa/pre_emptive_referee.py --manifest "$M/pre_emptive_referee.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_rationale_validity .venv/bin/python experiments/medqa/rationale_validity.py --manifest "$M/rationale_validity.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_seed_confidence .venv/bin/python experiments/medqa/seed_confidence.py --manifest "$M/seed_confidence.csv" --out experiments/medqa/results --model "$MODEL" --n 100
run medqa_seed_timing .venv/bin/python experiments/medqa/seed_timing.py --manifest "$M/seed_timing.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_super_additivity .venv/bin/python experiments/medqa/super_additivity.py --manifest "$M/super_additivity.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_temperature_sensitivity .venv/bin/python experiments/medqa/temperature_sensitivity.py --manifest "$M/temperature_sensitivity.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_test_awareness .venv/bin/python experiments/medqa/test_awareness.py --manifest "$M/test_awareness.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_text_cue_types .venv/bin/python experiments/medqa/text_cue_types.py --manifest "$M/text_cue_types.csv" --out experiments/medqa/results --model "$MODEL" --n 120
run medqa_true_peer_control .venv/bin/python experiments/medqa/true_peer_control.py --manifest "$FULL" --out experiments/medqa/results --model "$MODEL" --n 5 --case-ids "$M/true_peer_control.ids.txt"
run medqa_unanimity_break .venv/bin/python experiments/medqa/unanimity_break.py --manifest "$FULL" --out experiments/medqa/results --model "$MODEL" --n 12 --case-ids "$M/unanimity_break.ids.txt"
M=data/q4_case_cohorts/medmcqa
FULL=data/medmcqa_dev.csv
SHARED_SOLO=experiments/medmcqa/results/solo_records.jsonl
run medmcqa_reproduce_solo .venv/bin/python experiments/medqa/reproduce.py --manifest "$M/reproduce_solo.csv" --out experiments/medmcqa/results --model "$MODEL" --stage solo --solo-n 100 --seed 0
run medmcqa_reproduce_cascade .venv/bin/python experiments/medqa/reproduce.py --manifest "$M/reproduce_cascade.csv" --out experiments/medmcqa/results --model "$MODEL" --stage cascade --cascade-n 20 --seed 0
run medmcqa_attributed_tier .venv/bin/python experiments/medqa/attributed_tier.py --manifest "$M/attributed_tier.csv" --out experiments/medmcqa/results --model "$MODEL" --n 120
run medmcqa_authority_ladder .venv/bin/python experiments/medqa/authority_ladder.py --manifest "$M/authority_ladder.csv" --out experiments/medmcqa/results --model "$MODEL" --n 60
run medmcqa_break_it .venv/bin/python experiments/medqa/break_it.py --manifest "$FULL" --solo-records "$SHARED_SOLO" --out experiments/medmcqa/results --model "$MODEL" --n 20
run medmcqa_clean_a .venv/bin/python experiments/medqa/clean_a.py --manifest "$FULL" --solo-records "$SHARED_SOLO" --out experiments/medmcqa/results --model "$MODEL" --n 30
run medmcqa_scale_c .venv/bin/python experiments/medqa/scale_c.py --manifest "$FULL" --out experiments/medmcqa/results --model "$MODEL" --target 95 --case-ids "$M/scale_c.ids.txt"
run medmcqa_committee_size_sweep .venv/bin/python experiments/medqa/committee_size_sweep.py --manifest "$M/committee_size_sweep.csv" --out experiments/medmcqa/results --model "$MODEL" --n 120
run medmcqa_deliberation_framing .venv/bin/python experiments/medqa/deliberation_framing.py --manifest "$M/deliberation_framing.csv" --out experiments/medmcqa/results --model "$MODEL" --n 120
run medmcqa_dose_response .venv/bin/python experiments/medqa/dose_response.py --manifest "$M/dose_response.csv" --out experiments/medmcqa/results --model "$MODEL" --n 120
run medmcqa_hierarchy_dominance .venv/bin/python experiments/medqa/hierarchy_dominance.py --manifest "$M/hierarchy_dominance.csv" --out experiments/medmcqa/results --model "$MODEL" --n 40
run medmcqa_leader_as_auditor .venv/bin/python experiments/medqa/leader_as_auditor.py --manifest "$M/leader_as_auditor.csv" --out experiments/medmcqa/results --model "$MODEL" --n 120
run medmcqa_majority_pressure .venv/bin/python experiments/medqa/majority_pressure.py --manifest "$M/majority_pressure.csv" --out experiments/medmcqa/results --model "$MODEL" --n 25
run medmcqa_orchestrator_failure .venv/bin/python experiments/medqa/orchestrator_failure.py --manifest "$FULL" --out experiments/medmcqa/results --model "$MODEL" --n 61 --case-ids "$M/orchestrator_failure.ids.txt"
run medmcqa_plausible_distractor .venv/bin/python experiments/medqa/plausible_distractor.py --manifest "$M/plausible_distractor.csv" --out experiments/medmcqa/results --model "$MODEL" --n 87
run medmcqa_rationale_validity .venv/bin/python experiments/medqa/rationale_validity.py --manifest "$M/rationale_validity.csv" --out experiments/medmcqa/results --model "$MODEL" --n 120
run medmcqa_referee_deployable .venv/bin/python experiments/referee/referee_deployable.py --manifest "$M/referee_deployable.csv" --out experiments/medmcqa/results --model "$MODEL" --n 40
run medmcqa_referee_judge .venv/bin/python experiments/referee/referee_judge.py --manifest "$M/referee_judge.csv" --out experiments/medmcqa/results --model "$MODEL" --n 40
run medmcqa_seed_confidence .venv/bin/python experiments/medqa/seed_confidence.py --manifest "$M/seed_confidence.csv" --out experiments/medmcqa/results --model "$MODEL" --n 100
run medmcqa_super_additivity .venv/bin/python experiments/medqa/super_additivity.py --manifest "$M/super_additivity.csv" --out experiments/medmcqa/results --model "$MODEL" --n 120
run medmcqa_temperature_sensitivity .venv/bin/python experiments/medqa/temperature_sensitivity.py --manifest "$M/temperature_sensitivity.csv" --out experiments/medmcqa/results --model "$MODEL" --n 120
run medmcqa_test_awareness .venv/bin/python experiments/medqa/test_awareness.py --manifest "$M/test_awareness.csv" --out experiments/medmcqa/results --model "$MODEL" --n 120
run medmcqa_text_cue_types .venv/bin/python experiments/medqa/text_cue_types.py --manifest "$M/text_cue_types.csv" --out experiments/medmcqa/results --model "$MODEL" --n 120
run medmcqa_true_peer_control .venv/bin/python experiments/medqa/true_peer_control.py --manifest "$FULL" --out experiments/medmcqa/results --model "$MODEL" --n 16 --case-ids "$M/true_peer_control.ids.txt"
run medmcqa_unanimity_break .venv/bin/python experiments/medqa/unanimity_break.py --manifest "$FULL" --out experiments/medmcqa/results --model "$MODEL" --n 35 --case-ids "$M/unanimity_break.ids.txt"
printf "TEXT BATTERY COMPLETE at %s UTC\n" "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
