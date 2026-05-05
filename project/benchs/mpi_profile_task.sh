#!/bin/bash
#
#SBATCH --job-name=mpi_matrix_profiling
#SBATCH --output=results/mpi_profiling_result.txt
#
#SBATCH --ntasks=16
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=6G

module load OpenMPI
bash ./run_mpi_bench.sh