# Imaging experiments (Lane A): NIH ChestX-ray14

Manifest build (#162), single-model shortcut susceptibility (#163), an imaging-lane cascade
(#164), and a referee-as-gate evaluation (#165), on real chest X-rays via multimodal Gemini. Runs
anywhere: the compute is API-side, no local GPU. Reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) from
the env; every deterministic (temperature 0) step below reproduces the committed numbers with
`new_api_calls_this_run: 0` and no key.

## Manifest build (`build_manifest.py`, #162)

Parses the real ChestX-ray14 release via the registered `nih_cxr14` adapter
(`benchmaxxing/datasets/nih_cxr14.py`), keeps only cases whose image resolves on disk and that
carry a real finding label, and records a sha256 checksum per kept image
(`results/manifest_provenance.json`) so the exact pixels used are auditable.

```bash
python -m experiments.imaging.build_manifest \
    --raw-root <Data_Entry_2017.csv or its directory> --image-root <images/> \
    --case-ids-file experiments/imaging/results/case_ids_used.txt \
    --out experiments/imaging/results/nih_manifest.csv \
    --provenance experiments/imaging/results/manifest_provenance.json
```

`results/case_ids_used.txt` pins the exact 35-case set the results below were computed on, so a
fresh manifest build reproduces the identical cases (omit `--case-ids-file` to instead sample the
first `--n` eligible cases from the release for a new experiment).

## Solo shortcut susceptibility (`imaging_solo.py`, #163)

A diagnosis-neutral cue (cable / corner-tag / watermark / laterality) is injected onto the image
and we measure whether the model's yes/no read of the finding flips.

```bash
python -m experiments.imaging.imaging_solo --manifest results/nih_manifest.csv \
    --image-root <images/> --cache experiments/imaging/results/img_cache.jsonl \
    --out experiments/imaging/results --n 35
```

### Result (NIH ChestX-ray14, 35 cases, gemini-2.5-flash)

| cue | flip rate | flip-above-noise |
|---|---|---|
| watermark | 0.34 | **+0.11** |
| laterality | 0.26 | +0.03 |
| corner tag | 0.23 | 0.00 |
| cable | 0.20 | -0.03 |

Noise floor (each clean read resampled once at temperature 1.0, cache bypassed, 35 cases) = **0.23**,
in `imaging_solo_summary.json` with the raw resamples in `imaging_noise_floor.jsonl`. Only the
**watermark** cue clearly exceeds sampling noise; laterality is marginal and corner-tag / cable fall
at or below the floor at n=35. The noise floor is the one uncached step (a genuine temperature>0
resample) and needs a key; the deterministic flip pass itself reproduces from the cache with none.

## Imaging-lane cascade (`imaging_cascade.py`, #164) — a confident wrong peer

Beyond the solo cue, does a multimodal *committee* adopt a wrong finding-read when peers assert it?
On a watermarked X-ray, when two peer radiologists both read the film as the wrong answer, the
holdout adopts that wrong read far more under the shared board than reading alone:

```bash
python -m experiments.imaging.imaging_cascade \
    --manifest results/nih_manifest.csv --image-root <images/> \
    --cache experiments/imaging/results/img_cache.jsonl --out experiments/imaging/results --n 35
```

| setting | shared adopt | isolated adopt | contagion |
|---|---|---|---|
| watermark (NIH, 35 cases) | **0.97** | **0.34** | **+0.63** |

The cascade transfers to the imaging lane and is *stronger* there than in text: a confident
plausible peer moves an imaging committee almost completely. Uses a yes/no parser, so it is
unaffected by the MCQ answer-parser bug.

## Referee-as-gate (`imaging_referee.py`, #165)

Does a deployable referee with no privileged knowledge catch the cascade's peer-driven adoptions?
For each case it sees only the transcript (the peers' wrong read plus the holdout's committee
answer) and issues one private re-query: an independent re-read of the same cued film with the peer
discussion withheld. It flags adoption when the committee answer matches the peers but the re-read
does not.

```bash
python -m experiments.imaging.imaging_referee \
    --manifest results/nih_manifest.csv --image-root <images/> \
    --cascade-jsonl experiments/imaging/results/imaging_cascade.jsonl \
    --cache experiments/imaging/results/img_cache.jsonl --out experiments/imaging/results
```

Ground truth (evaluation only): a peer-driven adoption is a case adopted with peers but not alone
on the same cued image (22 of 35). Scored against that:

| Monitor | Precision | Recall | False-positive rate |
|---|---|---|---|
| Deployable referee (transcript + one re-read) | **0.86** | **0.86** | **0.23** |
| Naive conformity gate (flag any agreement) | 0.65 | 1.0 | 0.92 |

The referee cuts false alarms from 0.92 to 0.23 while still catching most adoptions. It does not
reach the text lane's perfect separation (P/R 1.0): on images the cue itself can move the holdout
even without peers, so peer-driven and cue-driven flips are harder to tell apart.

## Files

- `build_manifest.py`, `imaging_solo.py`, `imaging_cascade.py`, `imaging_referee.py` — the four runners.
- `results/nih_manifest.csv` — the 35-case manifest (built via the `nih_cxr14` adapter).
- `results/manifest_provenance.json` — a sha256 checksum per used image.
- `results/case_ids_used.txt` — the exact case_id set the committed results were computed on.
- `results/img_cache.jsonl` — the shared committed model-response cache for all three runners.
- `results/imaging_solo.jsonl` / `_summary.json`, `results/imaging_noise_floor.jsonl` — solo susceptibility.
- `results/imaging_cascade.jsonl` / `_summary.json` — the cascade.
- `results/imaging_referee.jsonl` / `_summary.json` — the referee-as-gate evaluation.
