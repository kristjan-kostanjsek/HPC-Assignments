#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=hpc_hw4_mpi
#SBATCH --output=out/grayscott_mpi_1.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=0

# Load MPI module
module load OpenMPI

# Print some info
echo "Running Gray-Scott MPI simulation on $SLURM_NTASKS tasks"

# Compile the program
echo "Compiling Gray-Scott MPI program..."
mpicc -o grayscott_mpi_1 grayscott_mpi_1.c -lm -O2

# Run the program with mpirun
mpirun -np $SLURM_NTASKS ./grayscott_mpi_1 256 5000 1 0.16 0.08 0.060 0.062