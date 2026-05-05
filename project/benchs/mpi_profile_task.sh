#!/bin/bash
#
#SBATCH --job-name=mpi_matrix_profiling
#SBATCH --output=mpi_profiling_result.txt
#
#SBATCH --ntasks=16
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1G

module load OpenMPI
./run_mpi_bench.sh