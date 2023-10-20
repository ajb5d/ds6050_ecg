#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from datetime import timedelta

BASE_ECG_PATH = Path("/scratch/ajb5d/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/")
BASE_DATA_PATH = Path("./data/")

lab_data = pd.read_csv(BASE_DATA_PATH / "lab-results.csv")
for x in ['charttime', 'ecg_time']:
    lab_data[x] = pd.to_datetime(lab_data[x])

lab_data['offset'] = (lab_data.charttime - lab_data.ecg_time) / timedelta(minutes=1)
lab_data['abs_offset'] = lab_data.offset.abs()

results = []
for target in lab_data.label.unique():
    results.append(
        lab_data[lab_data.label == target]
            .sort_values("abs_offset", ascending = True)
            .groupby("study_id")
            .first()
            [["valuenum", "offset"]]
            .rename(columns={
                "valuenum": target.lower(),
                "offset": f"{target.lower()}_offset"
            })
    )
    
result_df = results.pop(0)
while len(results) > 0:
    result_df = result_df.join(results.pop(0), how = "outer")

result_df.reset_index().to_csv(BASE_DATA_PATH / "ecg_lab_results.csv", index=False)

outcomes_data = pd.read_csv(BASE_DATA_PATH / "outcome-results.csv")
for x in ['admittime', 'dischtime', 'icu_intime', 'icu_outtime', 'ecg_time']:
    outcomes_data[x] = pd.to_datetime(outcomes_data[x])

outcomes_data['icu_expire_flag'] = (
    (outcomes_data['icu_outtime'] >= outcomes_data['dischtime'])
        .map(lambda x: 1 if x else 0)
)
outcomes_data['ecg_offset'] = (
    (outcomes_data['ecg_time'] - outcomes_data['icu_intime']) / timedelta(minutes=1)
)

outcomes_data = outcomes_data[(outcomes_data.ecg_order == 1)]

study_counts = outcomes_data.study_id.value_counts()
good_studys = list(study_counts[study_counts == 1].index) 
outcomes_data = outcomes_data[outcomes_data.study_id.isin(good_studys)]

outcomes_data[["study_id", "hadm_id", "hospital_expire_flag",
    "icu_expire_flag", "ecg_offset"]].to_csv(BASE_DATA_PATH / "ecg_hosp_results.csv", index=False)