# MIMIC-CXR imaging subset (Lane A): download + battery (#92, #295)

Replicates the NIH ChestX-ray14 imaging standard battery on MIMIC-CXR so imaging results are
apples-to-apples across datasets. The full release is ~600 GB; we never download it. We build the
manifest from the small metadata CSVs, select a deterministic nested study subset, and download
only the images those studies reference (~800 images for the 600-study superset).

## 0. Data prerequisites (credentialed, one time)

You need PhysioNet credentialed access to MIMIC-CXR-JPG v2.1.0 (images/labels; v2.0.0 is not
accessible) and MIMIC-CXR v2.0.0 (reports).
Download only the small metadata and reports first (tens of MB, no images):

```bash
mkdir -p ~/mimic-cxr-raw && cd ~/mimic-cxr-raw

wget -N -c --user YOUR_PHYSIONET_USER --ask-password \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz
wget -N -c --user YOUR_PHYSIONET_USER --ask-password \
  https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip

gunzip -f mimic-cxr-2.0.0-*.csv.gz
unzip -q mimic-cxr-reports.zip     # unpacks files/pXX/pYYYY/sZZZZ.txt (the adapter's report path)
```

The adapter (`benchmaxxing/datasets/mimic_cxr.py`) builds the manifest from these CSVs alone; no
image needs to be on disk to select the subset.

## 1. Select the nested study subset (`build_subset.py select`)

Ranks studies by a stable hash of the seed and takes the top N per arm. The arm sizes nest, so
every arm is a strict subset of the 600-study superset, and a 35-study set nests inside the 600
for the matched NIH comparison. One seed fixes the whole set; the same seed reproduces the identical
`download_list.txt` byte for byte.

```bash
python -m experiments.mimic_cxr_image.build_subset select \
  --raw-root ~/mimic-cxr-raw \
  --out-dir experiments/mimic_cxr_image/manifests \
  --seed 20240724
```

Writes per-arm manifests (`solo_600.csv`, `cross_modal_600.csv`, `referee_300.csv`,
`cascade_150.csv`, `blind_metric_100.csv`, `nih_match_35.csv`), a `download_list.txt` of the image
URLs for the 600-study superset (the union of every arm), and `selection_summary.json`.

Defaults (from the tracker): solo/cross-modal 600 studies, referee 300, cascade 150, blind-metric
100, NIH-match 35. Only finding-positive (pneumothorax) studies are selected, matching the imaging
pipeline; pass `--include-no-finding` to keep negatives too. A study contributes all of its images
(frontal + lateral), so image counts run a little above the study counts.

Building the full manifest reads ~227k report files (~30 s). To iterate faster, build it once and
reuse it with `--full-manifest path/to/full_manifest.csv`.

## 2. Download only the selected images

The list is plain URLs, so the fetch is a standard-tool one-liner with your own credentials (no
secrets in the repo). This is the reproducible download: same seed gives the same list.

```bash
cd ~/mimic-cxr-raw
wget -N -c --user YOUR_PHYSIONET_USER --ask-password \
  -i /path/to/experiments/mimic_cxr_image/manifests/download_list.txt \
  -x -nH --cut-dirs=3
# -x -nH --cut-dirs=3 recreates files/pXX/... under the current dir, matching image_ref.
```

(The requester-pays AWS mirror `s3://mimic-cxr-jpg-2.1.0.physionet.org/` works too; read the same
list with `aws s3 cp`.)

## 3. Verify + provenance (`build_subset.py verify`)

After the download, confirm every referenced image resolves and pin a sha256 per image (same
auditable-pixels guarantee as the NIH manifest):

```bash
python -m experiments.mimic_cxr_image.build_subset verify \
  --manifest experiments/mimic_cxr_image/manifests/solo_600.csv \
  --image-root ~/mimic-cxr-raw \
  --provenance experiments/mimic_cxr_image/manifests/provenance.json
```

## 4. Run the battery

One-time, install the imaging + model deps (full setup in `CONTRIBUTING.md`). The imaging runners
need the `image` extra (Pillow/OpenCV cue injection) and the Gemini client; `.[models]` (torch +
transformers) is only needed for the open-weights cross-lineage arm.

```bash
pip install -e ".[image,changepoint,stats]" && pip install "google-genai>=0.3"
export GEMINI_API_KEY=...   # multimodal Gemini; compute is API-side, no local GPU
```

`run_battery.py` is the committed runner (#343). It points the dataset-agnostic imaging runners in
`experiments/imaging/` at the per-arm manifests, in the order the arms depend on each other, and
pins the flags that are easy to get wrong by hand:

```bash
python -m experiments.mimic_cxr_image.run_battery --image-root ~/mimic-cxr-raw
```

One arm at a time (`--arm` is repeatable, names are in the table below), and `--dry-run` prints the
commands without calling anything, so the recipe is checkable with no key and no images:

```bash
python -m experiments.mimic_cxr_image.run_battery --image-root ~/mimic-cxr-raw --arm cascade
python -m experiments.mimic_cxr_image.run_battery --image-root ~/mimic-cxr-raw --dry-run
```

| arm | runner | manifest | results dir | tracker |
|---|---|---|---|---|
| `solo` | `imaging_solo` | `solo_600.csv` | `results/` | #310 |
| `nih_match_solo` | `imaging_solo` | `nih_match_35.csv` | `results/nih_match_35/` | #295 |
| `cascade` | `imaging_cascade` | `cascade_150.csv` | `results/cascade_150/` | #311 |
| `system_flag` | `imaging_system_flag` | `solo_600.csv` | `results/` | #312 |
| `strength_cascade` | `imaging_strength_cascade` | `solo_600.csv` | `results/` | #313 |
| `blind_metric` | `imaging_blind_metric` | `blind_metric_100.csv` | `results/` | #314 |
| `referee_cascade` | `imaging_cascade` | `referee_300.csv` | `results/referee_300/` | #315 |
| `referee` | `imaging_referee` | `referee_300.csv` | `results/referee_300/` | #315 |
| `judge` | `imaging_judge_referee` | `referee_300.csv` (transcript only) | `results/referee_300/` | #393 |

What the runner pins that the older hand-written commands did not:

- **The cache stays inside this lane.** Every runner except `imaging_solo` defaults `--cache` into
  `experiments/imaging/results/` (`img_cache.jsonl`, or `img_strength_cache.jsonl` for the strength
  sweep), which are the NIH lane's *committed* caches, so an unpinned MIMIC run appends
  credentialed-report-derived reads into a tracked file. All arms share one `results/img_cache.jsonl`
  instead, which also lets the nested arms (blind 100 < cascade 150 < referee 300 < solo 600) reuse
  each other's identical clean reads rather than pay twice.
- **No silent truncation.** `--n` defaults to the NIH arm sizes (35 cascade, 40 solo), so an
  unpinned run measures the first 35 images of an 834-image arm. The runner passes a bound no arm
  reaches.
- **Ordering and staging.** `imaging_cascade` writes a fixed `imaging_cascade.jsonl` for the
  watermark cue, so the 150-study and 300-study cascades get separate `--out` dirs. `imaging_referee`
  then reads the `referee_300` transcript, and `imaging_system_flag` contrasts its automated-flag
  board against the peer-assertion cascade by reading `imaging_cascade.jsonl` from *its own*
  `--out`. The runner stages the `cascade_150` transcript there first, without which
  `vs_peer_assertion_cascade` silently drops out of `imaging_system_flag_summary.json`.
- **Ground-truth plant.** The runner refuses to launch any cascade-family arm whose runner still
  plants `wrong = flip(clean_read)` instead of against ground truth (#332/#333/#338), so a stale
  checkout cannot quietly regenerate pre-fix numbers.
- **The transcript the replay arms need.** `referee` and `judge` both score
  `referee_300/imaging_cascade.jsonl`, which is gitignored and therefore never present in a fresh
  checkout. Running either without it now fails immediately naming the arm that writes it, instead
  of surfacing as a `FileNotFoundError` from inside the runner.

### The judge arm and the detector table's cohort (#393)

The same-lineage judge is the third detector in the cross-dataset table, beside the deployable
referee and the naive conformity gate. It had no MIMIC-CXR number for one reason: no arm here
invoked it. It is a text pass over the transcript `referee_cascade` already writes, so it adds no
cascade re-execution, and ordering it into the battery closes the window where that gitignored
transcript is produced, consumed once by `referee`, and cleaned up before the judge sees it.

**The cell to read out of the summary is `clean_correct_only`, not the top-level block.** The
MIMIC-CXR referee and gate cells in the table are counted on the 91 cases the model read correctly
on the clean image, not on all 417 — that restriction is where `0.77/0.75/0.21` and
`0.54/1.00/0.81` come from, and both reproduce from `results/deid/` (join `referee.csv` to
`referee_cascade.csv` on `case_index`, keep `clean_correct == 1`). Scored on all 417 instead, the
same two detectors read `0.48/0.64/0.16` and `0.19/1.00/0.96`. `imaging_judge_referee.py` scores
every row it is given, so it now reports both and the comparable one is named.

Results land under `experiments/mimic_cxr_image/results/`, same format as
`experiments/imaging/results/`, and feed issue #295's cross-dataset table. Each run caches per
call, so an interrupted run resumes cheaply.

## 5. Status of the committed summaries

The `*_summary.json` files in `results/` were generated **before** the ground-truth-plant fix
(#333/#338) reached the cascade family, so they are not what `run_battery.py` now produces. Under
the old design the planted wrong read was the opposite of the model's own clean read, which on a
finding-present case *is* the truth whenever the clean read was already wrong. On MIMIC that is the
common case, not the edge case: `plant_direction_check.py` measured **~80%** of cases planting the
truth, because gemini-2.5-flash reads pneumothorax poorly.

So the committed contagion numbers are inflated, and the `genuinely_false` subgroup in
`results/plant_direction_summary.json` is the closest honest figure the committed data supports:

| arm | committed (ALL, pre-fix) | genuinely-false subgroup | % that planted the truth |
|---|---|---|---|
| `cascade` (150) | +0.819 | **+0.488** (n=43) | 80.0% |
| `referee_cascade` (300) | +0.808 | **+0.506** (n=87) | 79.1% |
| `system_flag` | +0.200 | **+0.225** (n=169) | 79.7% |
| `strength_cascade` 0.15 / 0.30 / 0.45 | +0.797 / +0.808 / +0.795 | **+0.548 / +0.531 / +0.553** | 78.5% |

Treat that column as a floor on the honest effect, not a prediction of the rerun. It is measured
only on the ~20% of cases whose clean read was already correct, whereas the corrected design plants
`"no"` on every case. On the other ~80% the holdout's own contaminated read tends to agree with the
planted `"no"` already, which raises isolated adoption and therefore pushes contagion (shared minus
isolated) down. The rerun is the only way to settle where the full arm lands.

Five summaries are affected: the four cascade-family arms in the table plus
`referee_300/imaging_referee_summary.json`, which scores a replay of the cascade transcript.
`imaging_solo_summary.json`, `nih_match_35/imaging_solo_summary.json` and
`imaging_blind_metric_summary.json` plant nothing and reproduce as committed. Regenerating the five
needs a real Gemini run against the credentialed images. After that rerun `plant_direction_check.py`
becomes a no-op by construction (the corrected design plants the truth on 0% of cases) and is kept
only to re-derive the split from an archived pre-fix transcript.

## Reproducibility

Three artifacts pin a run, mirroring the NIH lane:

- `download_list.txt` + the seed: which images (deterministic from the seed).
- the per-arm manifests: which cases each arm ran on (pinned case_ids).
- `provenance.json`: the exact pixels (sha256 per image).
