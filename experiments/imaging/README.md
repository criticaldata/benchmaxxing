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
| watermark | 0.34 | **+0.11** |
| laterality | 0.26 | +0.03 |
| corner tag | 0.23 | 0.00 |
| cable | 0.20 | -0.03 |

Noise floor (each clean read resampled once at temperature 1.0, cache bypassed, 35 cases) = **0.23**,
written into `imaging_solo_summary.json` with the raw resamples in `imaging_noise_floor.jsonl`.

**Read (honest):** once the noise floor is computed properly in-script, only the **watermark** cue
clearly exceeds sampling noise (+0.11); laterality is marginal and corner-tag / cable fall at or
below the floor at n=35. So the imaging-lane shortcut is narrower than the raw flip rates suggest:
a printed-text watermark is the one robust image cue, and even it is modest. This corrects an
earlier writeup that compared against a 0.13 floor; the temperature>0 resample gives 0.23. (The
noise floor is stochastic, so re-running the resample varies it within sampling error; the flip
pass itself is deterministic and reproduces from the cache with no key.)
