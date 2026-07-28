-- Derives one row per HOSPITAL ADMISSION (hadm_id) with resource-constraint
-- proxies for benchmaxxing.datasets.ehr.load_resource_contexts (#49).
--
-- IMPORTANT: budget_pressure is a FABRICATED PROXY. MIMIC-IV has no real
-- cost tables. This methodology needs sign-off (see #297's framing: "it
-- must support a logical spurious feature") before being treated as the
-- real cascade shortcut feature -- flagged for review, not settled here.
--
-- icu_stay_count is NOT fabricated: the number of distinct ICU stays
-- recorded under the admission, a real MIMIC-native resource-load proxy
-- (an admission requiring multiple ICU stays/transfers plausibly
-- indicates higher acuity than a single uninterrupted stay). It does not
-- measure real bed occupancy or staffing ratios.
--
-- GRAIN BUG (caught in review by @Agastya191 / @sebasmos, fixed here): an
-- earlier version of this query emitted one row per icu.stay_id, but
-- icu_stay_count and budget_pressure are both hadm_id-level constants,
-- so every stay within a multi-stay admission produced an exact payload
-- duplicate. Confirmed impact on the real export: 9,214 of 94,382 rows
-- (9.8%) were exact duplicates, size-biasing the file toward its own load
-- variable -- admissions with 3+ ICU stays were 4.07% of rows but only
-- 1.36% of admissions, and mean icu_stay_count read 1.24 instead of the
-- true 1.11. FIX: emit one row per hadm_id, the actual grain both numeric
-- fields describe. scenario_id is now hadm_id, not stay_id.
-- careunit is taken from the chronologically first ICU stay in the
-- admission (by intime) as a representative value; admissions with
-- multiple stays across different careunits will not have that variation
-- captured -- a known simplification of this fix, not a hidden one.
--
-- CROSS-PATIENT TIME COMPARABILITY (retraction of an earlier, incorrect
-- fix, kept here for history): MIMIC-IV shifts dates independently per
-- subject_id into the 2100-2200 range. The very first version of this
-- query computed a "beds"/"staffing" pair via a cross-patient
-- intime/outtime overlap join; a first attempted fix bucketed that join
-- on patients.anchor_year_group, which does NOT work, since patients
-- sharing a bucket still sit at arbitrary independent offsets from each
-- other. Caught in review; the actual fix (below) derives everything
-- within a single hadm_id, needing no cross-patient comparison at all.
--
-- budget_pressure (hospital LOS = dischtime - admittime) is a
-- within-patient, within-admission interval, unaffected by the shift
-- issue above. A small fraction of admissions have dischtime < admittime
-- (documented MIMIC-IV data-entry artifact); excluded via the WHERE
-- clause since a negative value is not a valid "low pressure" reading.
--
-- Run in the BigQuery console or via `bq query --use_legacy_sql=false`.
-- Do NOT commit the query output (CSV) to this repository: MIMIC-IV's DUA
-- prohibits redistributing patient-level data, even de-identified derived
-- rows. Export locally only.
--
-- Dataset names below are the versioned MIMIC-IV v3.1 BigQuery datasets
-- (mimiciv_3_1_hosp / mimiciv_3_1_icu), matching what this query was
-- actually run and validated against. If your BigQuery access instead
-- grants the unversioned default (mimiciv_hosp / mimiciv_icu, currently
-- v2.2), substitute accordingly.

WITH icu_stay_counts AS (
  SELECT
    hadm_id,
    COUNT(*) AS icu_stay_count
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
  GROUP BY hadm_id
),
first_stay AS (
  SELECT
    hadm_id,
    first_careunit,
    ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) AS rn
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
),
admission_info AS (
  SELECT
    hadm_id,
    admission_type,
    insurance,
    DATETIME_DIFF(dischtime, admittime, HOUR) / 24.0 AS hosp_los_days
  FROM `physionet-data.mimiciv_3_1_hosp.admissions`
)
SELECT
  a.hadm_id AS scenario_id,
  c.icu_stay_count AS icu_stay_count,
  ROUND(a.hosp_los_days, 2) AS budget_pressure,
  fs.first_careunit AS careunit,
  a.admission_type,
  a.insurance
FROM admission_info a
JOIN icu_stay_counts c ON a.hadm_id = c.hadm_id
JOIN first_stay fs ON a.hadm_id = fs.hadm_id AND fs.rn = 1
WHERE a.hosp_los_days >= 0
ORDER BY a.hadm_id;