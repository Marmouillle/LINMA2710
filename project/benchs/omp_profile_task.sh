#!/bin/bash
#
#SBATCH --job-name=omp_matrix_profiling
#SBATCH --output=omp_profiling_result.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./execs/omp_bench 8192 16