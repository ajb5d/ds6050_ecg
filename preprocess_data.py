#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import wfdb
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser(
    prog = "summarize_data.py",
    description = "Summarize and report data quality"
)

parser.add_argument("--dataset", action = "store", required = True, choices = ['test', 'train', 'val'])
parser.add_argument("--output", action = "store", required = True)
parser.add_argument("--first", action = "store", type=int)
args = parser.parse_args()

BASE_ECG_PATH = Path("/scratch/ajb5d/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/")
BASE_DATA_PATH = Path("./data/")
OUTPUT_DATA_PATH = Path(args.output)

SHARD_SIZE = 512

def _tfr_int(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

def _tfr_float(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[v]))

def wfdb_to_example(rec, labs, hosp):
    r = wfdb.rdrecord(BASE_ECG_PATH / rec.path)
    dat = r.p_signal
    min = np.min(dat, axis=0)
    max = np.max(dat, axis=0)

    if np.any(min == max) or np.any(np.isnan(dat)):
        return None

    dat = (dat - np.mean(dat, axis = 0)) / np.std(dat, axis = 0)
    gender_value = 1 if rec.gender == "M" else 0

    feature = {
        'ecg/data': tf.train.Feature(float_list=tf.train.FloatList(value=dat.T.flatten())),
        'age': _tfr_float(rec.ecg_age),
        'gender': _tfr_int(gender_value),
        'file_name': _tfr_int(rec.file_name),
    }

    lab_hits = labs[labs.study_id == rec.study_id]

    if len(lab_hits) > 0:
        lab_rec = lab_hits.iloc[0]
        for label in ["troponin i", "potassium", "sodium"]:
            feature[label] = _tfr_float(lab_rec[label])
            feature[f"{label}_offset"] = _tfr_float(lab_rec[f"{label}_offset"])
    else:
        for label in ["troponin i", "potassium", "sodium"]:
            feature[label] = _tfr_float(np.nan)
            feature[f"{label}_offset"] = _tfr_float(np.nan)

    hosp_hits = hosp[hosp.study_id == rec.study_id]

    if len(hosp_hits) > 0:
        hosp_rec = hosp_hits.iloc[0]

        feature['hospital_expire_flag'] = _tfr_float(hosp_rec['hospital_expire_flag'])
        feature['icu_expire_flag'] = _tfr_float(hosp_rec['icu_expire_flag'])
        feature['ecg_offset'] = _tfr_float(hosp_rec['ecg_offset'])
    else:
        feature['hospital_expire_flag'] = _tfr_float(np.nan)
        feature['icu_expire_flag'] = _tfr_float(np.nan)
        feature['ecg_offset'] = _tfr_float(np.nan)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString() 

def output_file(target, file_count):
    return str(OUTPUT_DATA_PATH / f"{target}-{file_count:04}.tfrecords")


ecg_data = pd.read_csv(BASE_DATA_PATH / f"{args.dataset}_ecgs.csv")
lab_data = pd.read_csv(BASE_DATA_PATH / "ecg_lab_results.csv")
hosp_data = pd.read_csv(BASE_DATA_PATH / "ecg_hosp_results.csv")

record_count = 0
file_count = 0

writer = tf.io.TFRecordWriter(output_file(args.dataset, file_count))
for r in tqdm(ecg_data.itertuples(), total = args.first or len(ecg_data)):
    ex = wfdb_to_example(r, lab_data, hosp_data)
    if ex is not None:
        writer.write(ex)
        record_count += 1

    if record_count > SHARD_SIZE:
        record_count = 0
        file_count += 1
        writer.close()
        writer = tf.io.TFRecordWriter(output_file(args.dataset, file_count))

    if args.first is not None and (file_count * SHARD_SIZE + record_count) > args.first:
        break

writer.close()