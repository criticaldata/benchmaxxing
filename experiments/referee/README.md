# Referee controls (issue #132): deployable shared-only + same-lineage judge

Can an agent committee's shortcut adoption be caught by a monitor that has **no oracle
knowledge**, no planted-answer key and no isolated-committee counterfactual? And does an LLM
**judge of the same lineage** as the committee catch it, or share the blind spot? This folder
runs both controls.

## Setup

Two colluding peers assert a plausible, case-anchored wrong answer; the holdout is a real Gemini
agent (`gemini-2.5-flash-lite`) on 40 MedQA cases. Ground truth for scoring only: the holdout
**adopted the shortcut** when its board answer is the planted wrong answer and its bare (no-board)
answer is not (15 / 40 cases).

## The three referees (corrected)

| Referee | What it may use | Precision | Recall | FPR |
|---|---|---|---|---|
| naive gate (shared-only, no re-query) | the shared transcript alone | 0.375 | 1.0 | **1.0** |
| **deployable (peer-modal + private re-query, NO key)** | the transcript + one private re-query of the holdout | **1.0** | **1.0** | **0.0** |
| **same-lineage judge (gemini-2.5-flash)** | the deliberation transcript, judged by a same-family LLM | **1.0** | **0.93** | **0.0** |
| oracle audit (planted key + isolated run) | the planted answer + a full isolated committee | 1.0 | 1.0 | 0.0 |

## Correction (fixed 2026-07-22)

This README previously reported different numbers (10/40 adoptions; naive precision 0.25; judge
precision 0.83 / recall 1.0 / FPR 0.07), stale relative to the already-committed
`referee_deployable_summary.json` (15/40 adoptions, naive precision 0.375) and
`referee_judge_summary.json` (judge precision 1.0, recall 0.933, FPR 0.0, missing exactly 1 of 15
adoptions: `medqa-23`). The paper already had the corrected numbers; this README did not. Rewritten
to match.

**Read.** The naive gate is useless: it fires on any agreement streak, and two colluders always
agree, so it flags all 40 (FPR 1.0). The **deployable** referee recovers the oracle's exact
detection **without any ground-truth key**: it infers the candidate shortcut from the peers'
modal asserted answer, then issues a single private re-query of the holdout on the bare question
and flags only when the holdout's board answer matches the inferred shortcut but its own
re-query does not. Cost: one extra re-query per case (40), and no key.

The **same-lineage judge** (`referee_judge.py`) reads the whole deliberation and rules FLAG/OK.
It does **not** simply share the blind spot: it catches 14 of 15 adoptions with zero false
positives (precision 1.0, recall 0.93, FPR 0.0), missing only `medqa-23`. It is noisier only in
the sense of missing one true adoption, not in raising false alarms; the counterfactual-based
referees (deployable, oracle) are perfectly precise AND perfectly complete, so the targeted
counterfactual signal (which answer, adopted by whom relative to their own private answer) is
still the cleanest, but a same-lineage judge is a viable near-miss alternative.

This is the deployable form of the oversight result: catching a cascade in the wild needs a
targeted counterfactual signal (which answer, adopted by whom relative to their own private
answer), not blunt agreement detection, and that signal can be obtained without knowing the
shortcut in advance.

## Agreement/disagreement matrices (#184)

`agreement.py` adds pairwise agreement (phi/Cohen's kappa) among all four gates and enumerates the
exact divergent cases. Deployable and oracle agree perfectly (phi=kappa=1.0, both hit ground truth
by construction). The naive gate's agreement score against everything is degenerate (0.0 by
convention, since it flags every case with no variance, not evidence of disagreement in the usual
sense; its uselessness is already captured by its own FPR=1.0). Deployable and the judge disagree
on exactly 1 of 40 cases (phi=kappa=0.947): `medqa-23`, the same case the judge alone misses.
Pure re-analysis, no API calls.

```bash
python -m experiments.referee.agreement
```

## Reproduce

```bash
python -m experiments.referee.referee_deployable \
    --manifest <medqa_manifest.csv> \
    --cache experiments/referee/results/call_cache.jsonl \
    --out experiments/referee/results --n 40
```

Every model call is cached in `results/call_cache.jsonl` keyed by `(model, prompt)`. A fully
cached run reproduces `results/referee_deployable_summary.json` with **zero API calls and no key**
(verified: `new_api_calls_this_run = 0`). A key (`GEMINI_API_KEY`) is only needed to fill a cache
miss. No secrets are committed; all paths are arguments.

## Files

- `referee_deployable.py`, the deployable-referee runner (naive / deployable / oracle vs adoption).
- `referee_judge.py`, the same-lineage judge referee (`gemini-2.5-flash` rules FLAG/OK on the transcript).
- `agreement.py`, the pairwise agreement/divergence re-analysis (#184).
- `results/referee_deployable_summary.json`, `results/referee_judge_summary.json`, the scored summaries.
- `results/referee_deployable.jsonl`, `results/referee_judge.jsonl`, per-case rows.
- `results/referee_agreement.json`, the agreement/divergence output.
- `results/call_cache.jsonl`, the raw model calls, so every number reproduces offline.
