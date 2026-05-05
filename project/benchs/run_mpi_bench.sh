#!/bin/bash

for x in 1 2 4 8 16; do
    mpirun -np $x ./execs/mpi_bench 8192
done