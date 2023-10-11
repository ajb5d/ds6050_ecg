#!/usr/bin/env bash

sbatch preprocess_data.slurm --dataset train
sbatch preprocess_data.slurm --dataset test
sbatch preprocess_data.slurm --dataset val