#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --job-name=hpc_ass_3_par_1
#SBATCH --gpus=1
#SBATCH --output=out/grayscott_par_1.log

module load CUDA
nvcc  -diag-suppress 550 -O2 -lm grayscott_par_1.cu -o grayscott_par_1

srun  ./grayscott_par_1 4096 5000 1 0.16 0.08 0.060 0.062
srun  ./grayscott_par_1 2048 5000 1 0.16 0.08 0.060 0.062
srun  ./grayscott_par_1 1024 5000 1 0.16 0.08 0.060 0.062
srun  ./grayscott_par_1 512 5000 1 0.16 0.08 0.060 0.062
srun  ./grayscott_par_1 256 5000 1 0.16 0.08 0.060 0.062