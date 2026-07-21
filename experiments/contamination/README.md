# MedQA contamination / memorization audit (issue #108)

Is the solo shortcut susceptibility (flip rate 0.79 / 0.89) riding on **memorization or dataset
artifacts**, or on genuine uncertainty? Three probes per case (the 100-case solo set) plus a
correlation with baseline correctness.

## Results

| model | full acc | q-only acc | options-only acc | chance | options-only above chance |
|---|---|---|---|---|---|
| gemini-2.5-flash | 0.29 | 0.26 | 0.24 | 0.20 | **+0.04** |
| gemini-2.5-flash-lite | 0.14 | 0.18 | 0.28 | 0.20 | **+0.08** |

| model | flip when baseline-correct | flip when baseline-wrong | Fisher p |
|---|---|---|---|
| gemini-2.5-flash | 0.76 | 1.00 | **9.8e-5** |
| gemini-2.5-flash-lite | 1.00 | 0.99 | 1.0 (flips on ~everything) |

## Read

- **No answer-leakage artifact.** With the question removed, options-only accuracy is at most 8
  points above chance (0.24 / 0.28 vs 0.20). The options do not encode the answer, so the
  susceptibility is not an options-only dataset artifact.
- **No memorization boost.** Question-only accuracy (0.26 / 0.18) is about the same as full
  accuracy (0.29 / 0.14); having the options does not reveal a memorized answer.
- **Susceptibility is uncertainty-driven.** For the stronger model (flash), the answer-preserving
  cue flips the answer significantly more often on cases it gets wrong at baseline (1.00) than on
  cases it gets right (0.76), Fisher p < 1e-4. When flash actually knows the answer it resists the
  cue; the weak flash-lite flips on nearly everything regardless.

So the shortcut susceptibility is **not** explained by memorization or an option artifact; it rides
on genuine uncertainty. Note the baseline accuracy on this set is low (0.29 / 0.14), consistent
with a hard / uncertain regime, which is exactly where answer-preserving cues bite hardest.

## Reproduce

```bash
python -m experiments.contamination.contamination_audit \
    --manifest <medqa_manifest.csv> \
    --solo-records experiments/contamination/results/solo_records.jsonl \
    --cache experiments/contamination/results/call_cache.jsonl \
    --out experiments/contamination/results
```

A fully cached run reproduces `results/contamination_audit_summary.json` with **zero API calls and
no key** (verified: `new_api_calls_this_run = 0`).

## Files

- `contamination_audit.py`, the runner (q-only, options-only, and the flip-vs-correctness Fisher test).
- `results/contamination_audit_summary.json`, the scored summary.
- `results/contamination_audit.jsonl`, per-case rows (q-only and options-only correctness).
- `results/solo_records.jsonl`, the committed solo baseline (flip + correctness) this audits.
- `results/call_cache.jsonl`, raw model calls, so every number reproduces offline.
