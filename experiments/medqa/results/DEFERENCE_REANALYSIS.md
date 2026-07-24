# Deference re-analysis on committed MedQA cascade transcripts

Offline follow-up to #118 / #156. Replays every committed turn-level cascade transcript under `results/transcripts/` with `benchmaxxing.transcript_dynamics`.

## Headline

| Family | Shared deference | Isolated deference | Shared − isolated | Seed-sourced (shared) | Organic-peer (shared) |
|---|---:|---:|---:|---:|---:|
| v1 first-distractor (invalid seed) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| v2 baseline-relative | 0.030 | 0.000 | 0.030 | 0.010 | 0.020 |
| repro baseline-relative | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

v2 shared: **3 / 100** eligible follow-up turns deferred (1 seed-sourced, 2 organic-peer). Isolated is exactly 0 in every family.

## Read

- These are still the **near-null seeded** cascade boards, so the rates are a floor, not the anchored/plausible cascade effect.
- Within the only non-zero arm (v2 shared), most of the tiny signal is **organic-peer** adoption (2/3 events), not direct copying of the planted seed turn (1/3).
- A fully organic committee comparison is not yet possible at turn level: `live_peer_organic.jsonl` stores per-case summaries only, with no transcript dump.

## Reproduce

```bash
python experiments/medqa/deference_reanalysis.py \
  --transcript-dir experiments/medqa/results/transcripts \
  --out experiments/medqa/results/deference_reanalysis.json
```

No API key; reads committed transcripts only.
