#!/bin/bash

for x in 8 16; do
    mpirun -np $x ./mpi_bench 5000
done