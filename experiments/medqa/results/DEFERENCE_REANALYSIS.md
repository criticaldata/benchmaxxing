# Deference re-analysis on committed MedQA cascade transcripts

Offline follow-up to #118 / #156. Replays every committed turn-level cascade transcript under `results/transcripts/` with `benchmaxxing.transcript_dynamics`.

## Headline

| Family | Shared deference | Isolated deference | Shared − isolated | Seed-answer match (shared) | Organic-novel (shared) |
|---|---:|---:|---:|---:|---:|
| v1 first-distractor (invalid seed) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| v2 baseline-relative | 0.030 | 0.000 | 0.030 | 0.020 | 0.010 |
| repro baseline-relative | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

v2 shared: **3 / 100** eligible follow-up turns deferred (**2 seed-answer match**, **1 organic-novel**). Isolated is exactly 0 in every family.

The seed/organic split keys on **adopted-answer identity** (does the deferred answer equal the planted seed?), so a seed relayed through an intermediate unplanted peer still counts as seed-sourced. Secondary peer-turn provenance on the same 3 events: 1 last-matching peer was planted, 2 were unplanted relays.

## Read

- These are still the **near-null seeded** cascade boards, so the rates are a floor, not the anchored/plausible cascade effect.
- Within the only non-zero arm (v2 shared), most of the tiny signal is the **seed answer winning** (2/3 events), including one relayed through an unplanted peer before a later agent adopted it.
- A fully organic committee comparison is not yet possible at turn level: `live_peer_organic.jsonl` stores per-case summaries only, with no transcript dump.

## Reproduce

```bash
PYTHONPATH=. python3 experiments/medqa/deference_reanalysis.py \
  --transcript-dir experiments/medqa/results/transcripts \
  --out experiments/medqa/results/deference_reanalysis.json
```

No API key; reads committed transcripts only.
