#!/bin/bash
#
#SBATCH --job-name=opencl_basic_profiling
#SBATCH --output=results/opencl_basic_profiling_result.txt
#SBATCH --partition=batch
# --- resources ------------------------------------------------------
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=0-1:00:00             # walltime D‑HH:MM:SS
# --- environment ----------------------------------------------------

# --- run ------------------------------------------------------------
./execs/opencl_basic_bench