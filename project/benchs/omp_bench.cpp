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
    bench_file.open("omp_bench.csv", std::ios::out);
    bench_file << "Size,Duration,NumThreads\n";
    std::cout << "Max number of threads available: " << omp_get_max_threads() << std::endl;
    int max_threads = omp_get_max_threads();
    if (argc == 2)
    {
        max_size = std::stoi(argv[1]);
    }
    if (argc == 3)
    {
        max_threads = std::stoi(argv[2]);
        if (max_threads > omp_get_max_threads())
        {
            std::cerr << "Requested number of threads exceeds maximum available. Using " << omp_get_max_threads() << " threads instead." << std::endl;
            max_threads = omp_get_max_threads();
        }
    }
    for (int num_threads = 1; num_threads <= max_threads; num_threads *= 2)
    {
        for (int size = 64; size <= max_size; size *= 2)
        {
            omp_set_num_threads(num_threads);
            std::cout << "Testing matrix multiplication with size: " << size << "x" << size << " and " << num_threads << " threads" << std::endl;
            Matrix A(size, size);
            Matrix B(size, size);
            A.fill(1.0);
            B.fill(2.0);
            constexpr int num_runs = 5;
            double total_duration = 0.0;
            for (int run = 0; run < num_runs; ++run)
            {
                auto start_time
                    = std::chrono::high_resolution_clock::now();
                Matrix C = A * B;
                auto end_time
                    = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration
                    = end_time - start_time;
                total_duration += duration.count();
            }
            double avg_duration = total_duration / num_runs;
            std::cout << "Big multiplication test duration: "
                                  << avg_duration << " seconds (avg over " << num_runs << " runs)" << std::endl;
                        bench_file << size << "," << avg_duration << "," << num_threads << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        }
    }
    bench_file.close();
    return 0;
    
}