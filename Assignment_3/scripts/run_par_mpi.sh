#!/bin/bash

#SBATCH --job-name=gray_scott
##SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --constraint=v100s
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=per_task:1
#SBATCH --output=out/grayscott_par_mpi.log

#SBATCH --gpus-per-node=2
#SBATCH --tasks-per-node=2 
#SBATCH --nodes=2

NUM_RUNS=11

#LOAD MODULES 
module load OpenMPI
module load CUDA

#BUILD
make

#RUN
#echo "256"
#for ((run=1; run<=NUM_RUNS; run++)); do
#    mpirun -np 2 ./exe/grayscott_par_mpi 256 5000 1 0.16 0.08 0.060 0.062
#done

#echo "512"
#for ((run=1; run<=NUM_RUNS; run++)); do
#    mpirun -np 2 ./exe/grayscott_par_mpi 512 5000 1 0.16 0.08 0.060 0.062
#done

#echo "1024"
#for ((run=1; run<=NUM_RUNS; run++)); do
#    mpirun -np 2 ./exe/grayscott_par_mpi 1024 5000 1 0.16 0.08 0.060 0.062
#done

#echo "2048"
#for ((run=1; run<=NUM_RUNS; run++)); do
#    mpirun -np 2 ./exe/grayscott_par_mpi 2048 5000 1 0.16 0.08 0.060 0.062
#done

#echo "4096"
#for ((run=1; run<=NUM_RUNS; run++)); do
#    mpirun -np 2 ./exe/grayscott_par_mpi 4096 5000 1 0.16 0.08 0.060 0.062
#done

echo "8196"
for ((run=1; run<=NUM_RUNS; run++)); do
    mpirun -np 2 ./exe/grayscott_par_mpi 8196 5000 1 0.16 0.08 0.060 0.062
done

echo "16384"
for ((run=1; run<=NUM_RUNS; run++)); do
    mpirun -np 2 ./exe/grayscott_par_mpi 16384 5000 1 0.16 0.08 0.060 0.062
done

echo "32768"
for ((run=1; run<=NUM_RUNS; run++)); do
    mpirun -np 2 ./exe/grayscott_par_mpi 32768 5000 1 0.16 0.08 0.060 0.062
done
