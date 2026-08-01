# MedMCQA battery (Lane B) — how to reproduce

The MedMCQA committee battery has no runners of its own: each arm reuses the dataset-agnostic
script of the same name under `experiments/` (the ones the MedQA capstone uses), pointed at a
MedMCQA manifest. Every arm's model-call cache is committed under [`results/`](results/), so a
re-run replays from cache at **zero API calls** and reproduces the numbers in
[`results/RESULTS.md`](results/RESULTS.md) to the digit. MedMCQA is fully public (no DUA), which is
why the caches live in-tree — unlike the MIMIC lanes, whose transcripts stay gitignored.

## 1. Build the manifest

MedMCQA (Pal et al. 2022) ships as JSON/JSONL splits. Stage the raw `dev.json` into a validated
manifest:

    benchmaxxing datasets stage medmcqa --raw /path/to/medmcqa --out data/medmcqa_dev.csv

`data/` is gitignored: the manifest is rebuilt from the public release, not committed.

## 2. Run an arm

Every arm is the same shape — point its runner at the manifest and redirect the cache/output into
this lane's `results/`:

    python -m experiments.medqa.super_additivity \
        --manifest data/medmcqa_dev.csv \
        --cache experiments/medmcqa/results/super_additivity_cache.jsonl \
        --out experiments/medmcqa/results

With the committed cache in place it makes no API calls. Delete the cache (or push `--n` past the
cases it covers) to spend real ones. The `--n` each arm was run at is recorded as the `n` field in
its `results/<arm>_summary.json`.

## Arms

Runner ↔ committed cache ↔ where the result is written up. Grouped by `RESULTS.md` section.

| arm (cache stem) | runner module | RESULTS.md |
|---|---|---|
| plausible_distractor | `experiments.medqa.plausible_distractor` | §7 seed shape (#269) |
| rationale_validity | `experiments.medqa.rationale_validity` | §7 seed shape (#270) |
| seed_confidence | `experiments.medqa.seed_confidence` | §7 seed shape (#271) |
| attributed_tier | `experiments.medqa.attributed_tier` | §7 seed shape (#272) |
| dose_response | `experiments.medqa.dose_response` | §7 seed shape (#273) |
| text_cue_types | `experiments.medqa.text_cue_types` | §7 seed shape (#274) |
| majority_pressure | `experiments.medqa.majority_pressure` | §8 committee size & pressure (#275) |
| committee_size_sweep | `experiments.medqa.committee_size_sweep` | §8 committee size & pressure (#276) |
| unanimity_break | `experiments.medqa.unanimity_break` | §8 committee size & pressure (#277) |
| deliberation_framing | `experiments.medqa.deliberation_framing` | §8 committee size & pressure (#278) |
| super_additivity | `experiments.medqa.super_additivity` | §8 committee size & pressure (#279) |
| hierarchy_dominance | `experiments.medqa.hierarchy_dominance` | §9 hierarchy & orchestrator (#280) |
| orchestrator_failure | `experiments.medqa.orchestrator_failure` | §9 hierarchy & orchestrator (#281) |
| authority_ladder | `experiments.medqa.authority_ladder` | §9 hierarchy & orchestrator (#282) |
| leader_as_auditor | `experiments.medqa.leader_as_auditor` | §9 hierarchy & orchestrator (#283) |
| true_peer_control | `experiments.medqa.true_peer_control` | §10 controls & robustness (#284) |
| test_awareness | `experiments.medqa.test_awareness` | §10 controls & robustness (#285) |
| temperature_sensitivity | `experiments.medqa.temperature_sensitivity` | §10 controls & robustness (#286) |
| referee_deployable | `experiments.referee.referee_deployable` | §11 referee (#322) |
| referee_judge | `experiments.referee.referee_judge` | §11 referee (#322) |

The foundational arms (solo, cascade, break-it A/C/D — §1–§6, issues #267/#268/#288/#289/#290) share
`results/call_cache.jsonl` rather than a per-arm cache; their runners are the corresponding
`experiments/medqa/*.py` (`clean_a.py`, `push_c.py`, `scale_c.py`, …) driven the same way.
