# Multi-round cascade dynamics (issue #130)

Does shortcut adoption **build over deliberation rounds**, or is it single-shot? Under the
plausible (case-anchored) cascade, we run the committee for K rounds and record the holdout's
answer at each round, shared vs isolated.

## Result (MedQA, 40 cases, K=5, corrected parser)

| round | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| shared adoption | 0.275 | 0.3 | 0.275 | 0.325 | 0.325 |
| isolated adoption | 0.0 | 0.025 | 0.0 | 0.025 | 0.0 |

**No significant compounding over rounds.** Round 1 vs round 5 (shared): gained 3, lost 1, McNemar
p = 0.625, not significant, and the curve is not monotone increasing. Isolated adoption stays near
the floor throughout (at most 1 of 40 cases per round). So the holdout that adopts the plausible
shortcut mostly does so on **first exposure**, and repeated exposure over rounds does not reliably
recruit more.

**Read.** This sharpens the story rather than weakening it: the cascade is driven by the
**plausibility** of the shortcut on first contact, not by accumulating social pressure across
rounds. Contagion here is not a slow build.

## Isolated-arm fix (addressing review on #155)

An earlier version of this script gave the isolated holdout the exact same prompt every round: the
board text was built only from peer votes (`t.agent_id in ("peer1", "peer2")`), and peer turns are
never visible in isolated mode (`shared=False` hides only peer utterances, not the shared
workspace), so the board was always empty and the prompt was byte-identical round to round. At
temperature 0 this deterministically reproduced the round-1 answer every round, so isolated
adoption was **0 by construction, not by measurement**.

Fixed: the isolated holdout is now reminded of its own previous-round answer (`"In the previous
round you answered ..."`), which **is** visible to it in isolated mode (blackboard.py's
`_visible_turns` hides only *other* agents, not an agent's own prior turns), so the prompt
genuinely differs round to round. The shared arm's board construction is unchanged, so the shared
numbers above are identical to the previously-reviewed values; only the isolated arm's numbers are
new.

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
