#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import wfdb
import tensorflow as tf
import numpy as np
from tqdm import tqdm

BASE_ECG_PATH = Path("/scratch/ajb5d/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/")
BASE_DATA_PATH = Path("./data/")

def wfdb_to_example(rec):
    r = wfdb.rdrecord(BASE_ECG_PATH / rec.path)
    dat = r.p_signal
    min = np.min(dat, axis=0)
    max = np.max(dat, axis=0)
    if np.any(min == max):
        print(f"Skipping {rec.path} for zero signal")
        return None
    
    dat = (dat - min) / (max - min)

    gender_value = 1 if rec.gender == "M" else 0
    
    feature = {
        'ecg/data': tf.train.Feature(float_list=tf.train.FloatList(value=dat.T.flatten())),
        'age': tf.train.Feature(float_list=tf.train.FloatList(value=[rec.ecg_age])),
        'gender': tf.train.Feature(int64_list=tf.train.Int64List(value=[gender_value]))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString() 

for target in ['test', 'train', 'val']:
    ecg_data = pd.read_csv(BASE_DATA_PATH / f"{target}_ecgs.csv")
    with tf.io.TFRecordWriter(str(BASE_DATA_PATH / f"{target}.tfrecords")) as writer:
        for r in tqdm(ecg_data.itertuples(), total = len(ecg_data)):
            ex = wfdb_to_example(r)
            if ex is not None:
                writer.write(ex)