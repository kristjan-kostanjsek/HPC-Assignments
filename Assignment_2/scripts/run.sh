#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --reservation=fri
#SBATCH --job-name=hw2
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=output.log

# ===== USER CONFIGURATION =====
INPUT_DIR="in_images"
OUTPUT_DIR="out_images"
INPUT_FILES=("7680x4320.png")  # Add more files as needed
NUM_RUNS=100                     # Runs per input file
# =============================

module load CUDA

# Compile (only once)
nvcc -diag-suppress 550 -Wno-deprecated-gpu-targets -O2 -lm histeq_par_2.cu -o histeq_par_2
#gcc -o histeq_seq histeq_seq.c -lm -O2 -fopenmp

# Create output directory if missing
mkdir -p "$OUTPUT_DIR"

# Run benchmark for each input file
for input_file in "${INPUT_FILES[@]}"; do
  input_path="$INPUT_DIR/$input_file"
  output_path="$OUTPUT_DIR/out_${input_file}"

  echo "=== Processing $input_file ==="
  for ((run=1; run<=NUM_RUNS; run++)); do
    echo "Run $run:"
    #srun ./histeq_seq "$input_path" "$output_path"
    srun ./histeq_par_2 "$input_path" "$output_path"
  done
done