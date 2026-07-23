# Imaging cascade re-analysis (#185, #214)

Pure re-analysis of the already-committed imaging cascade transcripts. No API calls, no key
needed; every number here is derived from `results/*.jsonl` and `results/*_summary.json` that the
imaging pipeline (#162-165/#167-170) already produced.

## Claim 4 quantification (#185)

Claim 4 (contagion rides on case plausibility, not the cue's own solo potency) was asserted
qualitatively. Two computations were never done:

**Solo potency vs cascade contagion, across the four cues (n=4, descriptive):**

| cue | solo flip-above-noise | cascade contagion |
|---|---|---|
| cable | -0.029 | +0.80 |
| corner tag | 0.000 | +0.74 |
| watermark | +0.114 | +0.63 |
| laterality | +0.029 | +0.71 |

Spearman rho = **-1.0** (n=4, indicative only, not a powered hypothesis test). If anything,
contagion runs the *opposite* direction from solo potency: the weakest solo cue (cable) has the
*strongest* cascade. This directly supports claim 4: contagion does not track the injected
artifact's own strength.

**Per-case cross-cue agreement (n=35 cases shared across all four cues):** Cochran's Q = 3.0,
p = 0.392 (not significant: no evidence the four cues differ in adoption rate at the per-case
level). Pairwise phi/Jaccard: corner-tag, watermark, and laterality show **perfect agreement**
(phi = 1.0, Jaccard = 1.0): the exact same cases adopt regardless of which of these three cues is
present. Cable is constant (100% adoption on every case, so phi is undefined there by
construction, reported as 0.0 by convention; Jaccard against cable is still meaningful at 0.97).

**Read.** The sharper test here is the cross-cue agreement, not the n=4 correlation: the same
cases cascade almost irrespective of which visual cue triggers the peer assertion, which is
case-driven (not cue-driven) contagion, exactly what claim 4 asserts.

## Finding-type subgroup (#214)

Exploratory only, n=35, several findings with fewer than 5 cases:

| finding | n | shared adopt | Wilson 95% |
|---|---|---|---|
| cardiomegaly | 9 | 1.00 | [0.70, 1.0] |
| hernia | 8 | 1.00 | [0.68, 1.0] |
| infiltration | 6 | 1.00 | [0.61, 1.0] |
| effusion | 3 | 0.67 | [0.21, 0.94] |
| emphysema | 3 | 1.00 | [0.44, 1.0] |
| mass, atelectasis | 2 each | 1.00 | [0.34, 1.0] |
| nodule, pneumothorax | 1 each | 1.00 | [0.21, 1.0] |

No paired test is applied across findings (each finding involves a different set of cases, not
the same subjects under different conditions, so a Cochran's Q or McNemar comparison across
findings would be invalid). No finding stands out as categorically immune to or uniquely
susceptible to the cascade at this sample size; the intervals are wide and overlapping.

## Reproduce

```bash
python -m experiments.imaging.reanalysis
```

Reads only committed files under `experiments/imaging/results/`; writes
`claim4_quantification.json` and `finding_subgroup.json`. No `GEMINI_API_KEY` needed at all.

## Files

- `reanalysis.py`, the analysis (Spearman correlation, cross-cue Cochran's Q + phi/Jaccard,
  per-finding Wilson CIs).
- `results/claim4_quantification.json`, `results/finding_subgroup.json`, the outputs.
