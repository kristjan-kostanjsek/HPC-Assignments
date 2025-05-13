#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --constraint=v100s
##SBATCH --reservation=fri
#SBATCH --job-name=hpc_ass_3
#SBATCH --gpus=1
#SBATCH --output=out/grayscott_par_basic.log

NUM_RUNS=11

module load CUDA

nvcc  -diag-suppress 550 -O2 -lm src/grayscott_par_basic.cu -o exe/grayscott_par_basic
srun  ./exe/grayscott_par_basic 8192 5000 1 0.16 0.08 0.060 0.062
srun  ./exe/grayscott_par_basic 8192 5000 1 0.16 0.08 0.060 0.062
srun  ./exe/grayscott_par_basic 8192 5000 1 0.16 0.08 0.060 0.062

#echo "256"
#for ((run=1; run<=NUM_RUNS; run++)); do
#    srun  ./exe/grayscott_par_basic 256 5000 1 0.16 0.08 0.060 0.062
#done

#echo "512"
#for ((run=1; run<=NUM_RUNS; run++)); do
#    srun  ./exe/grayscott_par_basic 512 5000 1 0.16 0.08 0.060 0.062
#done

#echo "1024"
#for ((run=1; run<=NUM_RUNS; run++)); do
#    srun  ./exe/grayscott_par_basic 1024 5000 1 0.16 0.08 0.060 0.062
#done

#echo "2048"
#for ((run=1; run<=NUM_RUNS; run++)); do
#    srun  ./exe/grayscott_par_basic 2048 5000 1 0.16 0.08 0.060 0.062
#done

#echo "4096"
#for ((run=1; run<=NUM_RUNS; run++)); do
#    srun  ./exe/grayscott_par_basic 4096 5000 1 0.16 0.08 0.060 0.062
#done