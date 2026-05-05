#!/bin/bash

# Iterate over multiple tile sizes and sub-tile sizes
for TILE_SIZE in 8 16 32 64; do
    for SUB_TILE_SIZE in 1 2 4 8 16; do
        # TILE_SIZE must be divisible by SUB_TILE_SIZE and SUB_TILE_SIZE must be less than or equal to TILE_SIZE
        if (( TILE_SIZE % SUB_TILE_SIZE != 0 )) || (( SUB_TILE_SIZE > TILE_SIZE )); then
            echo "Skipping TILE_SIZE $TILE_SIZE and SUB_TILE_SIZE $SUB_TILE_SIZE because they are not compatible."
            continue
        fi
        ./execs/opencl_bench $TILE_SIZE $SUB_TILE_SIZE 20000
    done
done