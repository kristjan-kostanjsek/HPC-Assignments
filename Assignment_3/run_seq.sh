#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=hpc_ass_3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=out/grayscott_seq.log
#SBATCH --time=01:30:00

gcc -O2 -lm -fopenmp grayscott_seq.c -o grayscott_seq

#srun ./grayscott_seq 256 5000 1 0.16 0.08 0.060 0.062
#srun ./grayscott_seq 512 5000 1 0.16 0.08 0.060 0.062
#srun ./grayscott_seq 1024 5000 1 0.16 0.08 0.060 0.062
#srun ./grayscott_seq 2048 5000 1 0.16 0.08 0.060 0.062
echo "Gray-Scott seq 4096"
srun ./grayscott_seq 4096 5000 1 0.16 0.08 0.060 0.062
