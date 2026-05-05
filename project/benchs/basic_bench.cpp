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
    int max_size = 8192; // Maximum matrix size to test
    std::fstream bench_file;
    bench_file.open("basic_bench.csv", std::ios::out);
    bench_file << "Size,Duration\n";
    if (argc == 2)
    {
        max_size = std::stoi(argv[1]);
    }

    for (int size = 64; size <= max_size; size *= 2)
    {
        std::cout << "Testing matrix multiplication with size: " << size << "x" << size << std::endl;
        Matrix A(size, size);
        Matrix B(size, size);
        A.fill(1.0);
        B.fill(2.0);
        const int runs = 5;
        double total_duration = 0.0;
        for (int r = 0; r < runs; ++r) {
            auto start_time = std::chrono::high_resolution_clock::now();
            Matrix C = A * B;
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            total_duration += duration.count();
            std::cout << " Run " << (r+1) << ": " << duration.count() << " s" << std::endl;
        }
        double avg_duration = total_duration / runs;
        std::cout << "Average duration over " << runs << " runs: " << avg_duration << " seconds" << std::endl;
        bench_file << size << "," << avg_duration << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }
    bench_file.close();
    return 0;
    
}