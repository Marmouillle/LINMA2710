#!/bin/bash
#
#SBATCH --job-name=basic_block_profiling
#SBATCH --output=results/block_profiling_result.txt
#
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=2G

./execs/basic_bench