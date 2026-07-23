# Datasets: where they come from and how to stage them

Every result in this repo is downstream of a manifest, and a manifest is only as trustworthy as
the record of where it came from. This page is the acquisition step: what each dataset is, what
it costs to get, and the one command that turns a downloaded release into a validated manifest
with a provenance record.

No raw data and no credentials belong in this repository. Only code, docs and checksums.

## Where raw data lives

Raw releases go under a dataset root, one directory per dataset:

```bash
export BENCHMAXXING_DATASET_ROOT=/path/to/data     # defaults to ./data, which is gitignored
mkdir -p "$BENCHMAXXING_DATASET_ROOT/nih_cxr14"
```

## The one command

```bash
# uses $BENCHMAXXING_DATASET_ROOT/<name> unless --raw-root says otherwise
benchmaxxing datasets stage nih_cxr14 --check-images

# a subset while you are still setting up
benchmaxxing datasets stage medqa --raw-root /data/medqa/test.jsonl --limit 200
```

It runs the registered adapter, validates the manifest it produced (unique ids, required fields
per modality, and with `--check-images` that every `image_ref` resolves), checksums it, and
writes two files next to it:

- `<name>_manifest.csv`: the manifest everything downstream consumes
- `<name>_provenance.json` and `<name>_SOURCE.txt`: source, access level, licence, staging date,
  benchmaxxing version and git SHA, manifest sha256, row counts by modality and label

A manifest that fails validation is reported as an error rather than written off as usable, so a
broken staging step cannot quietly become the input to a run.

## The datasets

| Dataset | Lane | Access | What you have to do |
| --- | --- | --- | --- |
| `medqa` | text | open | Clone [jind11/MedQA](https://github.com/jind11/MedQA), use `data_clean/questions/US/{train,dev,test}.jsonl`. Staged and verified already: train 10178 / dev 1272 / test 1273. |
| `medmcqa` | text | open | Download from [medmcqa.github.io](https://medmcqa.github.io/); one JSON object per line. |
| `pubmedqa` | text | open | Download `ori_pqal.json` (the labelled subset) from [pubmedqa.github.io](https://pubmedqa.github.io/). |
| `nih_cxr14` | imaging | open | Download from [the NIH box share](https://nihcc.app.box.com/v/ChestXray-NIHCC): `images_XXX/images/*.png` batches plus `Data_Entry_2017.csv`. One batch (2 to 4 GB) is enough to unblock Lane A locally. |
| `chexpert` | imaging | registration | Register at [the CheXpert page](https://stanfordmlgroup.github.io/competitions/chexpert/) and accept the research use agreement. CheXpert-small is about 11 GB. Needed for the natural Support-Devices cue arm. |
| `mimic_cxr` | imaging | credentialed | PhysioNet credentialing plus CITI training, then [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/). Stage a small subset before any full run. |
| `ehr` | context | credentialed | A [MIMIC-IV](https://physionet.org/content/mimiciv/) derived CSV of resource-constraint contexts. Feeds the stage-5 scrutiny panel, not a case manifest. |

Priority order, which follows the machine constraints rather than the science: text and API work
first (`medqa` is done), then the open imaging set (`nih_cxr14`), then the registration-gated one
(`chexpert`), then the credentialed ones (`mimic_cxr`, `ehr`).

## Checking what you staged

```bash
benchmaxxing datasets stats "$BENCHMAXXING_DATASET_ROOT/nih_cxr14_manifest.csv" \
    --image-root "$BENCHMAXXING_DATASET_ROOT/nih_cxr14"
```

Row and modality counts, the MCQ shape check, the label distribution, and how many rows carry
`meta`. The per-case `meta` matters for the imaging lanes: the CheXpert Support-Devices flag and
the view live there, and they survive the round trip to disk.

## Adding a dataset

Write the adapter first ([docs/first-adapter.md](first-adapter.md)), register it, then add its
entry to `SOURCES` in `benchmaxxing/datasets/staging.py` with the canonical URL, the access
level, the licence, and the raw layout. That entry is what a provenance record quotes, so it is
worth getting exactly right.
