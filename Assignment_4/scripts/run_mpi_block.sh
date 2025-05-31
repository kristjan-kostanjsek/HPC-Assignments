#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=hpc_hw4_mpi
#SBATCH --output=out/grayscott_mpi_block_2_nodes.log
#SBATCH --ntasks=64
#SBATCH --tasks-per-node=32
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=0

# Load MPI module
module load OpenMPI

# Print some info
echo "Running Gray-Scott MPI simulation on $SLURM_NTASKS tasks"

# Compile the program
echo "Compiling Gray-Scott MPI program..."
mpicc -o exe/grayscott_mpi_block src/grayscott_mpi_block.c -lm -O3

# Run the program with mpirun
mpirun -np $SLURM_NTASKS ./exe/grayscott_mpi_block 256 5000 1 0.16 0.08 0.060 0.062
mpirun -np $SLURM_NTASKS ./exe/grayscott_mpi_block 512 5000 1 0.16 0.08 0.060 0.062
mpirun -np $SLURM_NTASKS ./exe/grayscott_mpi_block 1024 5000 1 0.16 0.08 0.060 0.062
mpirun -np $SLURM_NTASKS ./exe/grayscott_mpi_block 2048 5000 1 0.16 0.08 0.060 0.062
mpirun -np $SLURM_NTASKS ./exe/grayscott_mpi_block 4096 5000 1 0.16 0.08 0.060 0.062