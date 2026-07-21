# Multi-round cascade dynamics (issue #130)

Does shortcut adoption **build over deliberation rounds**, or is it single-shot? Under the
plausible (case-anchored) cascade, we run the committee for K rounds and record the holdout's
answer at each round, shared vs isolated.

## Result (MedQA, 40 cases, K=5)

| round | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| shared adoption | 0.175 | 0.15 | 0.175 | 0.15 | 0.15 |
| isolated adoption | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Null: adoption does not compound over rounds.** The shared curve is flat (~0.15-0.18 every
round); round 1 vs round 5 is not significant (McNemar p = 1.0), and the curve is not monotone
increasing. The isolated curve is 0 throughout. So the holdout that adopts the plausible shortcut
does so **immediately**, and repeated exposure over rounds recruits no additional agents.

**Read.** This sharpens the story rather than weakening it: the cascade is driven by the
**plausibility** of the shortcut (a single-shot decision on first exposure), not by accumulating
social pressure across rounds. Contagion here is not a slow build; it is decided at first contact.

## Reproduce

```bash
python -m experiments.cascade.multi_round \
    --manifest <medqa_manifest.csv> \
    --cache experiments/cascade/results/call_cache.jsonl \
    --out experiments/cascade/results --n 40 --rounds 5
```

A fully cached run reproduces `results/multi_round_summary.json` with **zero API calls and no key**
(verified: `new_api_calls_this_run = 0`); a key is only needed to fill a cache miss.

The per-round analysis (`per_round_holdout_answers`, `adoption_curve`,
`is_monotone_nondecreasing`) is pure and has no key/network dependency; it is unit-tested on a
hand-built transcript where adoption is known to grow, in `tests/test_multi_round.py`.

## Files

- `multi_round.py`, the runner + the pure per-round analysis functions.
- `results/multi_round_summary.json`, the per-round curves + the round-1-vs-round-K test.
- `results/multi_round.jsonl`, per-case per-round adoption flags (shared and isolated).
- `results/call_cache.jsonl`, raw model calls, so every number reproduces offline.
