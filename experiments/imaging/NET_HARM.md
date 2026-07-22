# Net-harm sign of the imaging cascade (#177)

Pure re-analysis of the already-committed imaging cascade transcripts. No API calls, no key
needed. Every case in the manifest is a real finding-positive (filtered to real findings, never
"no finding"), so ground truth for "does this X-ray show {finding}?" is always **yes**. The
cascade scripts define `wrong` as the opposite of the holdout's own uncued `clean` read, so which
direction "adopting wrong" moves the holdout depends on whether its own baseline read was itself
correct:

- `clean == "yes"` (baseline correct): adopting `wrong` (`"no"`) moves correct to incorrect,
  **harm**.
- `clean == "no"` (baseline incorrect): adopting `wrong` (`"yes"`) moves incorrect to correct
  (ground truth), **spurious rescue** (driven by peer pressure, not genuine clinical reasoning).

## Result (35 cases per cue: 22 baseline-correct, 13 baseline-incorrect)

| cue | harm rate (correct to wrong) | spurious rescue rate (wrong to correct) |
|---|---|---|
| cable | 1.00 [0.85, 1.0] | 1.00 [0.77, 1.0] |
| corner tag | 0.95 [0.78, 0.99] | 1.00 [0.77, 1.0] |
| watermark | 0.95 [0.78, 0.99] | 1.00 [0.77, 1.0] |
| laterality | 0.95 [0.78, 0.99] | 1.00 [0.77, 1.0] |

(Wilson 95% CI shown; bootstrap CIs are also in the committed JSON.) A Fisher exact test comparing
the harm and rescue rates finds no significant difference for any cue (p = 1.0 throughout, driven
by both rates sitting near the ceiling; the odds ratio is undefined where a rate is exactly 1.0,
noted explicitly in the output rather than silently emitting `NaN`).

**Read.** The imaging cascade is not selectively harmful. It moves the holdout toward whatever the
peer board asserts almost regardless of whether the holdout was right or wrong on its own: a
correct holdout is flipped to wrong about 95-100% of the time, and a wrong holdout is "corrected"
to the ground-truth answer about 100% of the time. This is closer to near-total conformity to
the shared board than to a shortcut that is selectively exploited only when the model itself is
uncertain. The pooled contagion numbers reported elsewhere (+0.63 to +0.80) undersell how uniform
this is: harm and rescue both sit near the ceiling.

## MedQA text lane: not computed here

The only per-case MedQA artifact with ground truth (`cascade_v2_per_case.jsonl`, from the earlier
baseline-relative cascade design) predates the answer-parser fix and is confirmed tainted by it:
its `medqa-82` row reports `"baseline_majority": "Desmoplastic"`, the exact stray-leading-article
mis-parse documented in the parser-bug correction (the model actually answered "Acral
lentiginous"). The stored transcripts for that run retain only a truncated completion (roughly
120-160 characters, cut well before any `\boxed{}` final answer), so they cannot be re-parsed with
the fixed parser either. A valid text-lane net-harm decomposition needs a fresh run with the
corrected parser and untruncated transcripts, which is not zero-cost re-analysis; tracked
separately so #177 is not silently left half-done.

## Reproduce

```bash
python -m experiments.imaging.net_harm
```

Reads only committed files under `experiments/imaging/results/`; writes `results/net_harm.json`.
No `GEMINI_API_KEY` needed at all.

## Files

- `net_harm.py`, the analysis (harm/rescue classification, Wilson + bootstrap CIs, Fisher exact).
- `results/net_harm.json`, the output.
