#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH -A ds6050
#SBATCH --output output/slurm-%j.out

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate ds6050-ecg

python preprocess_data.py