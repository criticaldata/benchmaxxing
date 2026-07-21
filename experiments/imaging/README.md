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

## Imaging-lane cascade (watermark) — a confident wrong peer

Beyond the solo cue, does a multimodal *committee* adopt a wrong finding-read when peers assert it?
On a watermarked X-ray, when two peer radiologists both read the film as the wrong answer, the
holdout adopts that wrong read far more under the shared board than reading alone:

| setting | shared adopt | isolated adopt | contagion |
|---|---|---|---|
| watermark (NIH, 35 cases) | **0.97** | **0.34** | **+0.63** |

So the cascade transfers to the imaging lane and is *stronger* there than in text: a confident
plausible peer moves an imaging committee almost completely. Run:

```bash
python -m experiments.imaging.imaging_cascade \
    --manifest <nih_manifest.csv> --image-root <images/> \
    --cache experiments/imaging/results/img_cache.jsonl --out experiments/imaging/results --n 35
```

Reproduces keyless from the committed `img_cache.jsonl` (`new_api_calls_this_run = 0`). Uses a
yes/no parser, so it was unaffected by the MCQ answer-parser bug.
