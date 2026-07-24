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

Point the existing imaging runners at the per-arm manifests (they are dataset-agnostic):

```bash
export GEMINI_API_KEY=...   # multimodal Gemini; compute is API-side, no local GPU

# #310 solo susceptibility + noise floor
python -m experiments.imaging.imaging_solo \
  --manifest experiments/mimic_cxr_image/manifests/solo_600.csv \
  --image-root ~/mimic-cxr-raw \
  --out experiments/mimic_cxr_image/results --n 600

# #311 cascade -> cascade_150.csv ; #315 referee -> referee_300.csv ;
# #314 blind-metric -> blind_metric_100.csv ; #312 system-flag, #313 cue-strength -> solo_600.csv
```

Results land under `experiments/mimic_cxr_image/results/`, same format as
`experiments/imaging/results/`, and feed issue #295's cross-dataset table.

## Reproducibility

Three artifacts pin a run, mirroring the NIH lane:

- `download_list.txt` + the seed: which images (deterministic from the seed).
- the per-arm manifests: which cases each arm ran on (pinned case_ids).
- `provenance.json`: the exact pixels (sha256 per image).
