#!/bin/bash
#
#SBATCH --job-name=simd_param_profiling
#SBATCH --output=results/simd_param_profiling_result.txt
#
#SBATCH --time=01:00:00
#SBATCH --mem=6G

./execs/simd_param_bench