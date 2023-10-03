#!/usr/bin/env python3
from pathlib import Path

def fix_name(stem):
    target, seq = stem.split("-")
    seq = int(seq)

    return f"{target}-{seq:04}"

OUTPUT_DATA_PATH = Path('/scratch/ajb5d/ecg/tfrecords')
for file in OUTPUT_DATA_PATH.glob("*.tfrecords"):
    source = file
    target = f"{file.parent}/{fix_name(file.stem)}.tfrecords"

    print(f"{source} -> {target}")
    file.rename(target)
