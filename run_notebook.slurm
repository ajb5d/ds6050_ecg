#!/usr/bin/env bash
#SBATCH --account ds6050
#SBATCH --partition=gpu
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output output/slurm-%j.out
#SBATCH --gres=gpu:1 

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate ds6050-ecg

module load singularity tensorflow/2.10.0

python -m nbconvert \
    --to script \
    --stdout "$@" | 
singularity run --nv $CONTAINERDIR/tensorflow-2.10.0.sif - 
