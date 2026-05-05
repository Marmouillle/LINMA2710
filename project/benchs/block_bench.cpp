#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <omp.h>

#include "matrix.hpp"

int main(int argc, char *argv[])
{
    std::fstream bench_file;
    bench_file.open("block_bench.csv", std::ios::out);
    bench_file << "Size,Duration,Block_Size\n";

    int size = 2048;
    int * block_sizes = new int[7]{16, 32, 48, 64, 80, 128, 256};
    for (int i = 0; i < 7; ++i)
    {
        int block_size = block_sizes[i];
        Matrix A(size, size, block_size);
        Matrix B(size, size, block_size);
        A.fill(1.0);
        B.fill(1.0);

        const int runs = 5;
        double total_duration = 0.0;

        for (int run = 0; run < runs; ++run) {
            auto start_time = std::chrono::high_resolution_clock::now();
            Matrix C = A * B;
            auto end_time = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end_time - start_time;
            total_duration += duration.count();
        }

        double average_duration = total_duration / runs;
        std::cout << "Size: " << size << ", Block Size: " << block_size << ", Average Duration: " << average_duration << " seconds\n";
        bench_file << size << "," << average_duration << "," << block_size << "\n";
    }
    bench_file.close();
    return 0;
    
}