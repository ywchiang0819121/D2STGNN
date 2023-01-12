#!/bin/bash

#SBATCH --job-name=D2STGNN2021
#SBATCH -o /storage/internal/home/y-chiang/logs/output-%j.out
#SBATCH -e /storage/internal/home/y-chiang/logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=8
#SBATCH --exclude=cn[1-21,32-55]
#squeue -h -o "%A %N" -u y-chiang
#### JOB LOGIC ###
#### JOB LOGIC ###

#export OPENBLAS_NUM_THREADS=1
#export OMP_NUM_THREADS=1
#export USE_SIMPLE_THREADED_LEVEL3=1
#export MKL_NUM_THREADS=1

/storage/internal/home/y-chiang/miniconda3/bin/python main_stream.py
