-- Derives one row per ICU stay with resource-constraint proxies for
-- benchmaxxing.datasets.ehr.load_resource_contexts (#49).
--
-- IMPORTANT: staffing and budget_pressure are FABRICATED PROXIES.
-- MIMIC-IV has no real staffing or cost tables. staffing is the inverse of
-- concurrent-patient load in the same careunit; budget_pressure is hospital
-- length of stay in days as a rough cost signal. Neither reflects real
-- hospital operations data. This methodology needs sign-off (see #297's
-- framing: "it must support a logical spurious feature") before being
-- treated as the real cascade shortcut feature -- flagged for review, not
-- settled by this script.
--
-- CROSS-PATIENT TIME COMPARABILITY: MIMIC-IV shifts dates independently
-- per subject_id into the 2100-2200 range; distinct patients' timestamps
-- are NOT directly comparable (PhysioNet docs: "two patients admitted in
-- [year] were not necessarily admitted in the same year"). A naive
-- cross-patient overlap join on intime/outtime is therefore statistically
-- meaningless. This query buckets concurrency comparisons by
-- patients.anchor_year_group -- a deliberately coarse (3-year) REAL
-- calendar period MIMIC-IV provides for exactly this purpose -- so
-- concurrency is only computed between patients confirmed to fall in the
-- same real-world period. Side effect: a stay whose patient has a NULL
-- anchor_year_group is excluded entirely (NULL = NULL is not true in SQL,
-- so even the self-match fails), rather than being silently assigned a
-- meaningless concurrency count. Confirmed on a real export: row count
-- drops from 94,444 to 87,290 after this fix.
--
-- budget_pressure (hospital LOS = dischtime - admittime) is unaffected by
-- the cross-patient issue above: it's a within-patient, within-admission
-- interval, which MIMIC-IV's per-patient shift preserves exactly. A small
-- fraction of admissions (71 of 87,290 in the real export) have
-- dischtime < admittime, a documented MIMIC-IV data-entry artifact, not a
-- query bug. A negative value is not a valid "low pressure" reading for
-- this proxy, so those rows are excluded below.
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

WITH stay_window AS (
  SELECT
    icu.stay_id,
    icu.subject_id,
    icu.hadm_id,
    icu.first_careunit,
    icu.intime,
    icu.outtime,
    pat.anchor_year_group
  FROM `physionet-data.mimiciv_icu.icustays` icu
  JOIN `physionet-data.mimiciv_hosp.patients` pat
    ON icu.subject_id = pat.subject_id
),
concurrent_counts AS (
  SELECT
    a.stay_id,
    a.first_careunit,
    COUNT(DISTINCT b.stay_id) AS concurrent_patients
  FROM stay_window a
  JOIN stay_window b
    ON a.first_careunit = b.first_careunit
    AND a.anchor_year_group = b.anchor_year_group
    AND a.intime < b.outtime
    AND a.outtime > b.intime
  GROUP BY a.stay_id, a.first_careunit
),
admission_info AS (
  SELECT
    hadm_id,
    admission_type,
    insurance,
    DATETIME_DIFF(dischtime, admittime, HOUR) / 24.0 AS hosp_los_days
  FROM `physionet-data.mimiciv_hosp.admissions`
)
SELECT
  s.stay_id AS scenario_id,
  ROUND(1.0 / c.concurrent_patients, 4) AS staffing,
  c.concurrent_patients AS beds,
  ROUND(a.hosp_los_days, 2) AS budget_pressure,
  s.first_careunit AS careunit,
  s.anchor_year_group,
  a.admission_type,
  a.insurance
FROM stay_window s
JOIN concurrent_counts c ON s.stay_id = c.stay_id
INNER JOIN admission_info a ON s.hadm_id = a.hadm_id
WHERE a.hosp_los_days >= 0
ORDER BY s.stay_id;
