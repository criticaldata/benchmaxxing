-- Derives one row per ICU stay with resource-constraint proxies for
-- benchmaxxing.datasets.ehr.load_resource_contexts (#49).
--
-- IMPORTANT: budget_pressure is a FABRICATED PROXY. MIMIC-IV has no real
-- cost tables, so hospital length-of-stay stands in for cost pressure with
-- no ground truth behind it. This methodology needs sign-off (see #297's
-- framing: "it must support a logical spurious feature") before being
-- treated as the real cascade shortcut feature -- flagged for review, not
-- settled by this script.
--
-- icu_stay_count is NOT fabricated: it is a real, MIMIC-native count (the
-- number of distinct ICU stays recorded under the same hospital
-- admission). It stands in as a resource-load proxy on the reasoning that
-- an admission requiring multiple ICU stays/transfers indicates higher
-- acuity or instability than a single, uninterrupted stay -- but it does
-- NOT measure bed occupancy or staffing ratios, and earlier versions of
-- this script and the ehr.py schema incorrectly labeled it as both under
-- two separate column names ("beds" and "staffing", with staffing defined
-- as 1/beds -- purely algebraically dependent on beds, not an independent
-- signal). Caught in review by @Agastya191 / @sebasmos (see below); fixed
-- by shipping one honestly-named column instead of two names for one
-- number.
--
-- CROSS-PATIENT TIME COMPARABILITY (retraction of an earlier, incorrect
-- fix, kept here for history): MIMIC-IV shifts dates independently per
-- subject_id into the 2100-2200 range. The very first version of this
-- query computed a "beds"/"staffing" pair via a cross-patient
-- intime/outtime overlap join. A first attempted fix bucketed that join on
-- patients.anchor_year_group -- this does NOT work, since two patients
-- sharing an anchor_year_group still sit at arbitrary independent offsets
-- from each other within that window, so an overlap between them is still
-- coincidental, not evidence of real concurrency. That fix's 102-to-37
-- drop in the "beds" range was the eligible comparison pool shrinking to
-- one of five buckets, not the number becoming real (caught in review by
-- @Agastya191 / @sebasmos).
--
-- ACTUAL FIX: icu_stay_count is derived entirely WITHIN a single admission
-- (hadm_id), which needs no cross-patient comparison at all and is
-- therefore immune to the shift problem by construction, not
-- approximation.
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
  c.icu_stay_count AS icu_stay_count,
  ROUND(a.hosp_los_days, 2) AS budget_pressure,
  icu.first_careunit AS careunit,
  a.admission_type,
  a.insurance
FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
JOIN icu_stay_counts c ON icu.hadm_id = c.hadm_id
INNER JOIN admission_info a ON icu.hadm_id = a.hadm_id
WHERE a.hosp_los_days >= 0
ORDER BY icu.stay_id;