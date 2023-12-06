SELECT
  stay.subject_id
  , stay.hadm_id
  , stay.stay_id
  , stay.gender
  , stay.admission_age
  , stay.admittime
  , stay.dischtime
  , stay.icu_intime
  , stay.icu_outtime
  , stay.hospital_expire_flag
  , ecg.study_id
  , ecg.ecg_time
  , ROW_NUMBER() OVER (PARTITION BY stay.stay_id ORDER BY ecg.ecg_time ASC) AS ecg_order
FROM `mimic-cloud.mimic_ecg.record_list` ecg
INNER JOIN `physionet-data.mimic_derived.icustay_detail` stay ON
  stay.subject_id = ecg.subject_id
  AND ecg.ecg_time BETWEEN DATE_ADD(stay.icu_intime, INTERVAL -8 HOUR) AND DATE_ADD(stay.icu_intime, INTERVAL 24 HOUR)
  