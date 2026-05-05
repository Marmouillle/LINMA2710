#!/bin/bash

# Ensure the executable exists
chmod +x ./execs/mpi_bench

for x in 1 2 4 8 16; do
    echo "Running with $x processes..."
    # -n $x tells Slurm to use only 'x' tasks for this specific run
    srun -n $x --oversubscribe ./execs/mpi_bench 8192
done