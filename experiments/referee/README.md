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

## Cross-tier referee independence (#183)

`cross_tier.py` probes `referee_independence_note`: does a referee from a DIFFERENT model tier
than the holdout catch adoption as well as a same-tier referee (the in-scope Gemini proxy for the
blocked cross-lineage arm)? Every committed referee so far re-queries the same tier as the
holdout. Both directions turn out zero-cost: reconstructing the identical bare-prompt hash per
case against the shared `call_cache.jsonl` confirms both tiers' answers are already cached for
both case sets (40/40 and 28/28 hits) - no new API calls, verified keyless with no key set at all.

```bash
python -m experiments.referee.cross_tier --manifest <medqa_manifest.csv>
```

| Direction | n | true adoptions | Cross-tier precision | Cross-tier recall |
|---|---|---|---|---|
| flash referee audits flash-lite committee | 40 | 15 | 1.0 | 0.933 |
| flash-lite referee audits flash committee | 28 | 19 | 1.0 | 0.789 |

**Read.** Same-tier is trivially identical to ground truth in both directions by construction
(perfect 1.0/1.0), so the cross-tier numbers are the real finding: independence does **not**
require cross-lineage in either direction - a different Gemini tier catches the large majority of
adoptions with perfect precision, but recall degrades (misses 1 of 15 in direction 1, ~4 of 19 in
direction 2), more so when the referee tier is weaker than the committee it audits. Uses
`experiments/model_dependence/cascade_C_flash.jsonl` for direction 2 (a real, non-trivial cross-
experiment reuse).

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

## Net-harm sign, text lane (#227, the text half of #177)

`net_harm.py` re-analyzes the already-committed `referee_deployable.jsonl` (this folder) cross-
referenced against the MedQA manifest for ground truth, zero new API calls - #227 was opened
assuming a fresh run was needed (the only other per-case artifact with ground truth,
`cascade_v2_per_case.jsonl`, predates the parser fix and is confirmed tainted), but this dataset
was built AFTER the fix with the same robust parser and already logs everything #227 asked a
fresh run to produce (baseline answer, seed target, board outcome); only "is baseline correct"
needed recovering, via the manifest.

```bash
python -m experiments.referee.net_harm --manifest <medqa_manifest.csv>
```

| | n | rate | Wilson 95% |
|---|---|---|---|
| Harm (correct to wrong) | 31 | 0.387 | [0.237, 0.562] |
| Spurious rescue (wrong to correct) | 9 | 0.111 | [0.020, 0.435] |

Fisher exact (harm vs rescue): p = 0.226, OR = 5.05.

**Read.** Unlike the imaging lane (near-total conformity, harm and rescue rates both 0.95-1.0),
the text lane shows only partial conformity: a substantial minority of correct holdouts (19 of
31) resist the seed even under peer pressure, and spurious rescue is rare (1 of 9). Consistent
with this session's broader finding that same-lineage Gemini committees hold their independent
answer against a confident wrong peer far more often in text than in imaging.

## Referee panel majority vote (#178)

`panel_vote.py` activates `benchmaxxing.referee_ablations.referee_panel_vote`/`single_vs_panel`
(previously unit-tested only with mock detectors) on real, reconstructed transcripts built from
the already-committed `referee_deployable.jsonl`/`referee_judge.jsonl` rows. Majority-votes three
genuinely independent, already-computed, no-key detectors - deployable (board-vs-bare
disagreement), naive (peer-vs-peer agreement, blind to the holdout), same-lineage judge - and
compares the panel to the single deployable referee alone. A fourth "conformity-streak" detector,
as the issue's illustrative example names, is not built: this is a single-round dataset (one
board turn, one private re-query per case), so a streak concept has no real signal to ground it
here without reaching into the separate multi-round dataset (#130), a different case set;
imaging-lane half is `experiments/imaging/panel_vote.py`.

```bash
python -m experiments.referee.panel_vote
```

| | Precision | Recall | F1 |
|---|---|---|---|
| Single (deployable alone) | 1.0 | 1.0 | 1.0 |
| Panel (majority of 3) | 1.0 | 1.0 | 1.0 |

**Read.** The single deployable referee already reaches perfect precision and recall on this
dataset, so a majority-vote panel cannot improve on it - it can only match or dilute it, and here
it exactly matches: the judge's flags are a strict subset of the true-adoption cases (precision
1.0), so the panel's majority (naive always votes yes; 2-of-3 needs only one more) never adds a
false positive beyond what deployable alone already catches. The private re-query the deployable
referee performs is the irreplaceable signal: no combination of the other two, which read only the
shared transcript, recovers what an independent counterfactual answer reveals. Pure zero-API
re-analysis; a parity check verifies the reconstructed transcripts reproduce the committed
`naive`/`deployable` fields exactly before any scoring happens.

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
- `cross_tier.py`, `results/cross_tier.json` - #183's cross-tier referee independence.
- `net_harm.py`, `results/net_harm.json` - #227's text-lane net-harm re-analysis.
- `panel_vote.py`, `results/panel_vote.json` - #178's text-lane referee panel majority vote.
