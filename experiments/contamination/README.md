# MedQA contamination / memorization audit (issue #108)

Is the solo shortcut susceptibility (per-record flip rate 0.063 / 0.117) riding on **memorization
or dataset artifacts**, or on genuine uncertainty? Three probes per case (the 100-case solo set,
corrected parser), plus a correlation with baseline correctness.

## Results (corrected parser)

| model | full acc | q-only acc | options-only acc | chance | options-only above chance |
|---|---|---|---|---|---|
| gemini-2.5-flash | 0.89 | 0.26 | 0.23 | 0.20 | +0.03 |
| gemini-2.5-flash-lite | 0.78 | 0.18 | 0.28 | 0.20 | +0.08 |

| model | per-record flip rate | ever-flipped rate | flip when baseline-correct | flip when baseline-wrong | Fisher p |
|---|---|---|---|---|---|
| gemini-2.5-flash | 0.063 | 0.12 | 0.079 | 0.455 | 3.3e-3 |
| gemini-2.5-flash-lite | 0.117 | 0.21 | 0.090 | 0.636 | 4.4e-7 |

*Per-record flip rate* is the headline solo susceptibility quantity (mean over all cue x case
combinations, matching the reported 0.063 / 0.117). *Ever-flipped rate* is a coarser, per-case
quantity (did at least one of the three cues flip this case), used only to build the
correct-vs-wrong stratification below; the two are not interchangeable and are reported separately
so neither is mistaken for the other.

## Read

- **No answer-leakage artifact.** With the question removed, options-only accuracy is at most 8
  points above chance (0.23 / 0.28 vs 0.20). The options do not encode the answer, so the
  susceptibility is not an options-only dataset artifact.
- **No memorization boost.** Question-only accuracy (0.26 / 0.18) is well below full accuracy
  (0.89 / 0.78); having the options does not merely surface a memorized answer, the model is
  genuinely using them.
- **Susceptibility is uncertainty-driven, for both tiers.** The answer-preserving cue flips the
  answer far more often on cases the model gets wrong at baseline than on cases it gets right:
  flash 0.455 vs 0.079 (Fisher p = 3.3e-3), flash-lite 0.636 vs 0.090 (Fisher p = 4.4e-7). Both
  models are competent on this set (accuracy 0.89 / 0.78) and both resist the cue when they
  actually know the answer; flash-lite's overall susceptibility is higher only because it is wrong
  more often at baseline, not because it flips more readily conditional on being wrong.

So the shortcut susceptibility is **not** explained by memorization or an option artifact; it rides
on genuine uncertainty, and this holds for both model tiers.

## Grading and reproducibility notes (addressing review on #158)

- **The `full`/clean condition is re-graded here** by exact match against ground truth, the same
  standard `options_only_correct` already used, instead of trusting the imported `clean_correct`
  flag from the prior solo-baseline pipeline as-is. The re-graded values are identical to the
  imported ones (0.89 / 0.78), so there was no actual discrepancy, but the grading standard is now
  transparent and consistent across all three conditions rather than assumed.
- **Cache race fixed.** `_Cache.complete` previously checked the store under a lock but made the
  API call outside it, so two threads could both miss the same key and both append a duplicate
  entry; the committed cache carried one such duplicate. The lock now spans the whole
  miss-check-fetch-store sequence.
- **Cache pruned.** The committed `call_cache.jsonl` is now written containing only the 400 keys
  this script actually requests (100 cases x 2 models x 2 probes), not the full shared cache it was
  copied from.

## Misaligned-proxy arm (#176, text lane)

`misaligned_proxy.py` activates `benchmaxxing.blind_arms.misaligned_proxy_run` (built and
unit-tested with mocks only, invoked by no data runner) on the committed longest_option cue rows
of `solo_records.jsonl` (n=200). Proxy = length of the seeded/contaminated answer text (a
plausible-but-wrong surrogate: a longer option reads as more authoritative, but length has
nothing to do with correctness); truth = `contaminated_correct`; decision = whether the case
flipped toward the seeded option at all. Imaging-lane half is
`experiments/imaging/misaligned_proxy.py`.

```bash
python -m experiments.contamination.misaligned_proxy
```

| corr(flip, length) | corr(flip, correctness) | uptake_delta |
|---|---|---|
| 0.093 | -0.405 | 0.498 |

**Read.** The large positive uptake_delta is driven mostly by the strong negative truth
correlation, not a strong positive proxy correlation: flipping toward the seeded option is, by
design, usually wrong, so it necessarily anti-correlates with correctness. The proxy correlation
itself is weak (r=0.093) - length only weakly predicts which specific cases flip, even though the
cue succeeds overall (158 of 200 cases flip). Read honestly, this does not show that answer
length specifically is what drives which cases flip; the positive uptake_delta here is largely an
artifact of truth's strong negative pull, not evidence that length is doing the work the cue's
name suggests. Pure zero-API re-analysis of the already-committed `solo_records.jsonl`.

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
- `results/call_cache.jsonl`, raw model calls (pruned to the 400 this script requests), so every
  number reproduces offline.
- `misaligned_proxy.py`, `results/misaligned_proxy.json` - #176's text-lane misaligned-proxy arm.
