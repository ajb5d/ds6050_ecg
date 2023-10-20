SELECT
  le.subject_id
  , le.charttime
  , rec.study_id
  , rec.ecg_time
  , lab.label
  , le.itemid
  , le.valuenum
  , le.valueuom
FROM
  `physionet-data.mimic_hosp.labevents` le
INNER JOIN
  `mimic-cloud.mimic_ecg.record_list` rec ON
    le.subject_id = rec.subject_id AND
    rec.ecg_time BETWEEN DATE_ADD(le.charttime, INTERVAL -4 HOUR) AND DATE_ADD(le.charttime, INTERVAL 4 HOUR)
LEFT JOIN `physionet-data.mimic_hosp.d_labitems` lab ON
  le.itemid = lab.itemid
WHERE
  le.itemid IN (51002, 52642, 50971, 52610, 50983, 52623)
  