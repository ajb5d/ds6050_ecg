#!/usr/bin/env bash

sbatch summarize_data.slurm --dataset train
sbatch summarize_data.slurm --dataset test
sbatch summarize_data.slurm --dataset val
