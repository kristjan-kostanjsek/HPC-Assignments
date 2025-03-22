#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=hpc_assignment_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:00:05
#SBATCH --output=sample_out.log


export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -------------------- SEQUENTIAL RUN (single thread) -----------------------
# forced on 1 thread with --cpus-per-task=1

#gcc -O2 -lm --openmp seam_seq.c -o seam_seq

#srun seam_seq in_images/720x480.png out_images/out_720x480.png 128
#srun seam_seq in_images/1024x768.png out_images/out_1024x768.png 128
#srun seam_seq in_images/1920x1200.png out_images/out_1920x1200.png 128
#srun seam_seq in_images/3840x2160.png out_images/out_3840x2160.png 128
#srun seam_seq in_images/7680x4320.png out_images/out_7680x4320.png 128

# -------------------- PARALLEL ENERGY CALCULATION -----------------------

#gcc -O2 -lm --openmp seam_par_1.c -o seam_par_1

#srun seam_par_1 in_images/720x480.png out_images/out_720x480.png 128
#srun seam_par_1 in_images/1024x768.png out_images/out_1024x768.png 128
#srun seam_par_1 in_images/1920x1200.png out_images/out_1920x1200.png 128
#srun seam_par_1 in_images/3840x2160.png out_images/out_3840x2160.png 128
#srun seam_par_1 in_images/7680x4320.png out_images/out_7680x4320.png 128


# -------------------- PARALLEL CUMU. ENERGY CALCULATION -----------------------

#gcc -O2 -lm --openmp seam_par_2.c -o seam_par_2

#srun seam_par_2 in_images/720x480.png out_images/out_720x480.png 128
#srun seam_par_2 in_images/1024x768.png out_images/out_1024x768.png 128
#srun seam_par_2 in_images/1920x1200.png out_images/out_1920x1200.png 128
#srun seam_par_2 in_images/3840x2160.png out_images/out_3840x2160.png 128
#srun seam_par_2 in_images/7680x4320.png out_images/out_7680x4320.png 128


# -------------------- PARALLEL SEAM FIND & CUT -----------------------

gcc -O2 -lm --openmp seam_par_3.c -o seam_par_3

srun seam_par_3 in_images/720x480.png out_images/out_720x480.png 128
#srun seam_par_3 in_images/1024x768.png out_images/out_1024x768.png 512
#srun seam_par_3 in_images/1920x1200.png out_images/out_1920x1200.png 960
#srun seam_par_3 in_images/3840x2160.png out_images/out_3840x2160.png 128
#srun seam_par_3 in_images/7680x4320.png out_images/out_7680x4320.png 128