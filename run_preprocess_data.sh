#!/usr/bin/env bash

sbatch preprocess_data.slurm --output /scratch/ajb5d/ecg/tfrecords/ --dataset train 
sbatch preprocess_data.slurm --output /scratch/ajb5d/ecg/tfrecords/ --dataset test
sbatch preprocess_data.slurm --output /scratch/ajb5d/ecg/tfrecords/ --dataset val
