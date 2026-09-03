# Referee self-inconsistency floor

- dataset: medqa
- model: gemini-2.5-flash-lite
- cases: 40
- temperature: 0

## Headline

| metric | value |
| --- | --- |
| declared pairs | 39 |
| undeclared pairs | 1 |
| undeclared draws | 2 |
| stable cases | 39 |
| unstable cases | 0 |
| temp0 self inconsistency rate | 0.0 |

## Canonical artifacts

- rows: `experiments/referee/results/referee_self_inconsistency.jsonl`
- summary: `experiments/referee/results/referee_self_inconsistency_summary.json`
- cache: `experiments/referee/results/referee_self_inconsistency_cache.jsonl`

## Reproduction

Run provenance is recorded in:

- `config.json`
- `versions.json`
- `run_manifest.json`

The canonical per-case results remain in the referee results directory listed above.
