# Imaging experiments (Lane A)

`imaging_solo.py`: single-model shortcut susceptibility on real chest X-rays via multimodal
Gemini. A diagnosis-neutral cue (cable / corner-tag / watermark / laterality) is injected onto
the image and we measure whether the model's yes/no read of the finding flips.

Runs anywhere: the compute is API-side (multimodal Gemini), no local GPU. Reads
`GEMINI_API_KEY` from the env; a fully cached run reproduces the committed numbers with no key.

```bash
python -m experiments.imaging.imaging_solo --manifest <nih_manifest.csv> \
    --image-root <images/> --out experiments/imaging/results --n 35
```

## Result (NIH ChestX-ray14, 35 cases, gemini-2.5-flash)

| cue | flip rate | flip-above-noise |
|---|---|---|
| watermark | 0.34 | +0.21 |
| laterality | 0.26 | +0.13 |
| corner tag | 0.23 | +0.10 |
| cable | 0.20 | +0.07 |

Noise floor (same clean image twice) = 0.13. Modest but real; flips concentrate on uncertain
findings (hernia, mass, nodule). The imaging analog of the strong text-lane solo result.
