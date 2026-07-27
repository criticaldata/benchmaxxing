-- Derives one row per ICU stay with resource-constraint proxies for
-- benchmaxxing.datasets.ehr.load_resource_contexts (#49).
--
-- IMPORTANT: staffing and budget_pressure are FABRICATED PROXIES.
-- MIMIC-IV has no real staffing or cost tables. Neither reflects real
-- hospital operations data. This methodology needs sign-off (see #297's
-- framing: "it must support a logical spurious feature") before being
-- treated as the real cascade shortcut feature -- flagged for review, not
-- settled by this script.
--
-- CROSS-PATIENT TIME COMPARABILITY (retraction of an earlier, incorrect
-- fix): MIMIC-IV shifts dates independently per subject_id into the
-- 2100-2200 range. An earlier version of this query attempted to fix a
-- cross-patient concurrency computation by bucketing on
-- patients.anchor_year_group. That does NOT work: two patients sharing an
-- anchor_year_group still sit at arbitrary independent offsets from each
-- other within that window, so an intime/outtime overlap between them is
-- still coincidental, not evidence of real concurrency (caught in review
-- by @Agastya191 and @sebasmos -- thank you). The earlier 102-to-37 drop
-- in the beds range was the eligible comparison pool shrinking to one of
-- five buckets, not the number becoming real.
--
-- FIX: staffing and beds are now derived entirely WITHIN a single
-- admission (hadm_id), which needs no cross-patient comparison at all and
-- is therefore immune to the shift problem. beds = the number of distinct
-- ICU stays recorded under the same hospital admission (icu_stay_count):
-- a shift-invariant, MIMIC-native signal that a sicker or more unstable
-- admission required multiple ICU stays/transfers. staffing = 1 /
-- icu_stay_count, the same inverse-load framing as before but now
-- grounded in a real within-admission count instead of a fabricated
-- cross-patient collision.
--
-- budget_pressure (hospital LOS = dischtime - admittime) was never
-- affected by the shift issue: it's a within-patient, within-admission
-- interval, which MIMIC-IV's per-patient shift preserves exactly. A small
-- fraction of admissions have dischtime < admittime, a documented
-- MIMIC-IV data-entry artifact; excluded below since a negative value is
-- not a valid "low pressure" reading for this proxy.
--
-- Run in the BigQuery console or via `bq query --use_legacy_sql=false`.
-- Do NOT commit the query output (CSV) to this repository: MIMIC-IV's DUA
-- prohibits redistributing patient-level data, even de-identified derived
-- rows. Export locally only.
--
-- Dataset names below are PhysioNet's current default (mimiciv_hosp /
-- mimiciv_icu = v2.2 as of this writing). If your BigQuery access grants a
-- versioned dataset instead (e.g. mimiciv_v3_1_hosp / mimiciv_v3_1_icu),
-- substitute accordingly.

WITH icu_stay_counts AS (
  SELECT
    hadm_id,
    COUNT(*) AS icu_stay_count
  FROM `physionet-data.mimiciv_3_1_icu.icustays`
  GROUP BY hadm_id
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
  icu.stay_id AS scenario_id,
  ROUND(1.0 / c.icu_stay_count, 4) AS staffing,
  c.icu_stay_count AS beds,
  ROUND(a.hosp_los_days, 2) AS budget_pressure,
  icu.first_careunit AS careunit,
  a.admission_type,
  a.insurance
FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
JOIN icu_stay_counts c ON icu.hadm_id = c.hadm_id
INNER JOIN admission_info a ON icu.hadm_id = a.hadm_id
WHERE a.hosp_los_days >= 0
ORDER BY icu.stay_id;