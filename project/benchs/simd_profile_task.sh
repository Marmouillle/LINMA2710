#!/bin/bash
#
#SBATCH --job-name=simd_matrix_profiling
#SBATCH --output=simd_profiling_result.txt
#
#SBATCH --time=01:00:00
#SBATCH --mem=6G

./execs/simd_bench