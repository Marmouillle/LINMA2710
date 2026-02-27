#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <omp.h>

#include "matrix copy.hpp"

int main(int argc, char *argv[])
{
    int max_size = 5000; // Maximum matrix size to test
    std::fstream bench_file;
    bench_file.open("bench.csv", std::ios::out);
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
        for (int size = 500; size <= max_size; size += 500)
        {
            omp_set_num_threads(num_threads);
            std::cout << "Testing matrix multiplication with size: " << size << "x" << size << " and " << num_threads << " threads" << std::endl;
            Matrix A(size, size);
            Matrix B(size, size);
            A.fill(1.0);
            B.fill(2.0);
            auto start_time
                = std::chrono::high_resolution_clock::now();
            Matrix C = A * B;
            auto end_time
                = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration
                = end_time - start_time;
            std::cout << "Big multiplication test duration: "
                      << duration.count() << " seconds" << std::endl;
            bench_file << size << "," << duration.count() << "," << num_threads << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        }
    }
    bench_file.close();
    return 0;
    
}