#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import wfdb

import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

BASE_ECG_PATH = Path("/scratch/ajb5d/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/")
BASE_DATA_PATH = Path("./data/")
OUTPUT_DATA_PATH = Path('/scratch/ajb5d/ecg/tfrecords')

parser = ArgumentParser(
    prog = "summarize_data.py",
    description = "Summarize and report data quality"
)

parser.add_argument("--dataset", action = "store", required = True, choices = ['test', 'train', 'val'])
parser.add_argument("--head", action = "store_true")

args = parser.parse_args()

import tensorflow as tf

CANONICAL_SIG_ORDER = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def wfdb_to_summary(rec):

    r = wfdb.rdrecord(BASE_ECG_PATH / rec.path)
    dat = r.p_signal
    min_sig = np.min(dat, axis=0)
    max_sig = np.max(dat, axis=0)
    
    ret_val =  {
        'filename': rec.path,
        'fs': r.fs,
        'mean': np.mean(dat),
        'sd': np.std(dat),
        'all_zeros': np.any(min_sig == max_sig),
        'any_nan': np.any(np.isnan(dat))
    }

results = []
ecg_data = pd.read_csv(BASE_DATA_PATH / f"{args.dataset}_ecgs.csv")
record_count = 0

for r in tqdm(ecg_data.itertuples(), total = len(ecg_data)):
    ex = wfdb_to_summary(r)
    results.append(ex)
    record_count += 1
    if args.head and record_count >= 10:
        break

pd.DataFrame(results).to_csv(f"data/{args.dataset}_data_quality.csv", index = False)