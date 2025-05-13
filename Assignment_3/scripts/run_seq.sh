#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=hpc_ass_3
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=out/grayscott_seq2.log

gcc src/grayscott_seq.c -o exe/grayscott_seq -lm -fopenmp -O2
#echo "256x256"
#srun ./exe/grayscott_seq 256 5000 1 0.16 0.08 0.060 0.062
#srun ./exe/grayscott_seq 256 5000 1 0.16 0.08 0.060 0.062
#srun ./exe/grayscott_seq 256 5000 1 0.16 0.08 0.060 0.062

#echo "512x512"
#srun ./exe/grayscott_seq 512 5000 1 0.16 0.08 0.060 0.062
#srun ./exe/grayscott_seq 512 5000 1 0.16 0.08 0.060 0.062
#srun ./exe/grayscott_seq 512 5000 1 0.16 0.08 0.060 0.062

#echo "1024x1024"
#srun ./exe/grayscott_seq 1024 5000 1 0.16 0.08 0.060 0.062
#srun ./exe/grayscott_seq 1024 5000 1 0.16 0.08 0.060 0.062
#srun ./exe/grayscott_seq 1024 5000 1 0.16 0.08 0.060 0.062

echo "2048x2048"
srun ./exe/grayscott_seq 2048 5000 1 0.16 0.08 0.060 0.062
srun ./exe/grayscott_seq 2048 5000 1 0.16 0.08 0.060 0.062
srun ./exe/grayscott_seq 2048 5000 1 0.16 0.08 0.060 0.062

echo "4096x4096"
srun ./exe/grayscott_seq 4096 5000 1 0.16 0.08 0.060 0.062
srun ./exe/grayscott_seq 4096 5000 1 0.16 0.08 0.060 0.062
srun ./exe/grayscott_seq 4096 5000 1 0.16 0.08 0.060 0.062