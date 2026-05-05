#!/bin/bash
#
#SBATCH --job-name=opencl_profiling
#SBATCH --output=results/opencl_profiling_result.txt
#SBATCH --partition=batch
# --- resources ------------------------------------------------------
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=0-1:00:00             # walltime D‑HH:MM:SS
# --- environment ----------------------------------------------------
module load CUDA/12.8.0
# --- run ------------------------------------------------------------
./run_opencl_bench.sh 