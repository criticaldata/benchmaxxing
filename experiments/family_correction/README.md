# Benchmark-wide multiple-comparison correction (#191)

Every correction reported elsewhere in this project is within one arm (e.g. scale_c's own
Benjamini-Hochberg pass). This applies a single correction across a curated family of every
headline p-value this session has independently verified as current (post-parser-fix), so it is
clear which arms survive when tested against the full set rather than in isolation. Pure
computation, no API calls.

## Scope: what's included and what's deliberately excluded

Only p-values from a source independently re-verified for parser-fix correctness in this session
are included (13 tests: 4 text, 4 imaging null comparisons, 5 real-effect tests). See
`family_correction.py`'s `EXCLUDED` list for the full reasons; in short:

- `break_it_summary.json` / `push_c_summary.json` (PR #140's branch): not independently
  re-verified for parser-fix currency in this pass.
- `cascade_v2_per_case.jsonl` (PR #135's branch): confirmed tainted by the pre-fix stray-article
  parser bug.
- The blind-metric and referee arms report precision/recall/FPR, not a p-value comparable in this
  family.

Mixing verified and unverified p-values into one family would make the correction itself
misleading, so the family is deliberately smaller than "every p-value in the project" in exchange
for being honest about what it actually covers.

## Result (13 tests, alpha=0.05)

| test | p (raw) | p (BH) | survives BH | p (Holm) | survives Holm |
|---|---|---|---|---|---|
| contamination, flash-lite Fisher | 4.4e-7 | 6e-6 | yes | 6e-6 | yes |
| contamination, flash Fisher | 0.0033 | 0.0214 | yes | 0.0396 | yes |
| clean-A, flash-lite Fisher | 0.0092 | 0.0397 | **yes** | 0.1008 | no |
| clean-A, flash Fisher | 0.0216 | 0.0701 | no | 0.2155 | no |
| scale_c anchored vs generic | 0.0414 | 0.1076 | no | 0.3725 | no |
| scale_c strongly-anchored | 0.0963 | 0.2085 | no | 0.7700 | no |
| model-dependence (flash) | 1.0 | 1.0 | no | 1.0 | no |
| multi-round, text | 0.625 | 1.0 | no | 1.0 | no |
| multi-round, imaging | 0.25 | 0.4643 | no | 1.0 | no |
| net-harm, 4 cues (imaging) | 1.0 each | 1.0 | no | 1.0 | no |

**3 of 13 survive BH; 2 of 13 survive the stricter Holm.** Only the contamination audit's
correct-vs-wrong Fisher tests (both tiers) are robust to a family-wide correction. Clean-A's
flash-lite result survives BH but not Holm; flash's clean-A and both scale_c plausibility
comparisons do not survive family-wide correction at all, consistent with scale_c's own arm
already flagging the anchored increment as failing a local BH. This is a real honesty check: the
plausibility-cascade headline (claim 3's "+0.12 at p=0.041") is weaker once tested against the
full family, not just its own small local family.

## Reproduce

```bash
python -m experiments.family_correction.family_correction
```

No inputs needed beyond the module itself; the p-values are a curated, cited constant list (see
`family_correction.py`), since the source files live across several different git branches that
do not currently coexist in one checkout.

## Files

- `family_correction.py`, the curated p-value family + BH/Holm correction.
- `results/family_correction.json`, the output.
