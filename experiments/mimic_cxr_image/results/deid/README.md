# De-identified per-case rows (MIMIC-CXR imaging battery)

**Regenerate with `python3 ../../export_deid.py`; verify with `--check`.** These files used to be
produced by hand, which is how they came to sit a run behind the paper: recomputing from them
reproduced superseded numbers rather than the published ones. `--check` recomputes every MIMIC-CXR
cell the paper prints from these CSVs alone and exits non-zero if any has drifted, so the failure
is loud rather than silent.

DUA-safe per-case outcomes for every arm of the corrected rerun (#372), so the published rates are
independently recomputable without redistributing anything under the PhysioNet terms. Committed
because the raw transcripts (`results/**/*.jsonl`) key model reads to MIMIC `dicom_id`s and stay
gitignored; these files carry **no `dicom_id`, no patient/study id, no `image_ref`, no report text,
no raw reads** — only outcome flags keyed by an anonymous `case_index`.

## `case_index`

Rows are sorted on `case_id` and numbered `0..n-1`; `case_id` is then dropped. The manifest is
seed-deterministic (`build_subset.py --seed 20240724`), so the ordering — and therefore every
file here — is reproducible. Note this is **DUA-safe but not unlinkable**: someone who already
holds PhysioNet access can rebuild the same manifest ordering and re-link `case_index` to a study.
It protects against redistribution to people *without* access, which is what the DUA requires.

Every rate in the committed `*_summary.json` is the mean of the corresponding flag column here
(verified to reproduce them exactly). Contagion = `mean(shared_adopt) - mean(iso_adopt)`.

## `clean_correct` is a headroom control, not a plant-direction cut

After the #333/#338 fix the plant is a constant `"no"` on an all-finding-present cohort, so **every
case is genuinely false** — there is no "genuinely-false subgroup" to select. `clean_correct == 1`
marks cases the model read correctly on the *clean* image, i.e. cases with headroom for a peer to
flip; restricting to it is a robustness line, not the honest-vs-contaminated split (that split only
existed pre-fix; see `plant_direction_check.py`).

## Column dictionary

All flag columns are `0`/`1`.

| file | rows | columns |
|---|---|---|
| `solo.csv` | 834 | `case_index`, `clean_correct`, `{cable,corner_tag,watermark,laterality}_flip`, `noise_flip` |
| `nih_match_solo.csv` | 35 | `case_index`, `solo_case_index` (its row in `solo.csv`), then as `solo.csv` |
| `cascade.csv` | 215 | `case_index`, `clean_correct`, `iso_adopt`, `shared_adopt` |
| `referee_cascade.csv` | 417 | `case_index`, `clean_correct`, `iso_adopt`, `shared_adopt` |
| `system_flag.csv` | 834 | `case_index`, `clean_correct`, `iso_adopt`, `shared_adopt` |
| `strength_cascade.csv` | 834 | `case_index`, `clean_correct`, `op{0.15,0.3,0.45}_{iso_adopt,shared_adopt,solo_flip}` |
| `referee.csv` | 417 | `case_index`, `clean_correct`, `gt`, `ref_flag`, `naive_flag` |
| `judge.csv` | 417 | `case_index`, `clean_correct`, `gt`, `judge_flag`, `naive_flag`. Transcript-only judge |
| `judge_with_image.csv` | 417 | same columns. The judge given the film but performing no private re-query, the arm that isolates image access from the re-query |
| `blind_metric.csv` | 141 | `case_index`, `base_is_decoy`, `blind_is_decoy`, `aware_is_decoy`, `named_rubric_when_drifted` |
| `provenance.csv` | 834 | `case_index` (matches `solo.csv`), `sha256`, `bytes` — per-image checksums of the superset, so a holder of the credentialed images can confirm identical pixels |

Field meanings: `iso_adopt` = the model alone adopts the planted `"no"` on the cued image;
`shared_adopt` = it adopts with the committee board asserting `"no"`; `*_flip` = the cued read
differs from the clean read; `gt` = a peer-driven adoption (`shared_adopt==1 & iso_adopt==0`);
`ref_flag`/`naive_flag` = the referee's private-requery gate vs flag-every-adoption gate;
`judge_flag` = the same-lineage judge's FLAG on the transcript alone;
`*_is_decoy` = the model took the decoy option under the baseline / blind / test-aware framing.

## `judge.csv` is the one row-level file still missing (#393)

The `judge` arm of `run_battery.py` was added after the #372 rerun, so the same-lineage judge cell
has no per-case rows here yet and cannot be recomputed without credentialed access — the gap this
file otherwise exists to close. Whoever runs that arm should export it from the runner's
`referee_300/imaging_judge_referee.jsonl` the same way #372 did the rest: sort on `case_id`, number
`0..n-1`, drop `case_id`, and keep `clean_correct`, `gt`, `judge_flag`, `naive_flag`.

`clean_correct` is not optional here. The detector table reports this lane on the 91 cases with
`clean_correct == 1`, so without that column the committed rows cannot reproduce the published
cell. `case_index` is the same seed-deterministic ordering as `referee.csv`, so the two join
directly and the judge can be checked against the referee and the gate case by case.

## A note on `plant_direction_summary.json`

That file in the parent directory is **superseded** and describes the pre-fix transcripts only. It reports
contagion values 1.6x to 5.7x higher than the corrected per-arm summaries beside it (5.68x on
`cascade_150`, 4.38x on `referee_300`, 3.5x to 3.8x across the three opacity arms, 1.58x on the system
flag), and a
genuinely-false / planted-truth split that cannot exist post-fix, because the plant is now a constant `"no"`
on an all-finding-present cohort. It is kept as a record of the pre-fix state and carries a `_superseded`
header saying so. Do not cite it. For a headroom-controlled cut, restrict these rows on `clean_correct == 1`.
