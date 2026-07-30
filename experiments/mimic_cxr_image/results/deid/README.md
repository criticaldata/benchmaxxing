# De-identified per-case rows (MIMIC-CXR imaging battery)

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
| `referee.csv` | 417 | `case_index`, `gt`, `ref_flag`, `naive_flag` |
| `blind_metric.csv` | 141 | `case_index`, `base_is_decoy`, `blind_is_decoy`, `aware_is_decoy`, `named_rubric_when_drifted` |
| `provenance.csv` | 834 | `case_index` (matches `solo.csv`), `sha256`, `bytes` — per-image checksums of the superset, so a holder of the credentialed images can confirm identical pixels |

Field meanings: `iso_adopt` = the model alone adopts the planted `"no"` on the cued image;
`shared_adopt` = it adopts with the committee board asserting `"no"`; `*_flip` = the cued read
differs from the clean read; `gt` = a peer-driven adoption (`shared_adopt==1 & iso_adopt==0`);
`ref_flag`/`naive_flag` = the referee's private-requery gate vs flag-every-adoption gate;
`*_is_decoy` = the model took the decoy option under the baseline / blind / test-aware framing.
