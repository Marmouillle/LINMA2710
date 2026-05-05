#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <omp.h>

#include "matrix_opti.hpp"

int main(int argc, char *argv[])
{
    std::fstream bench_file;
    bench_file.open("simd_param_bench.csv", std::ios::out);
    bench_file << "Size,Duration,MC,KC,NC\n";

    int size = 2048;
    for (int mc : {8, 16, 32, 48, 64, 80})
    {
        for (int kc : {48, 64, 128, 192, 256, 384, 512})
        {
            for (int nc : {64, 128, 256, 512, 1024, 2048})
            {
                Matrix A(size, size, mc, kc, nc);
                Matrix B(size, size, mc, kc, nc);
                A.fill(1.0);
                B.fill(1.0);

                double total_duration = 0.0;
                for (int run = 0; run < 5; ++run)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    Matrix C = A * B;
                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> duration = end - start;
                    total_duration += duration.count();
                }
                double avg_duration = total_duration / 5.0;
                
                bench_file << size << "," << avg_duration << "," << mc << "," << kc << "," << nc << "\n";
                std::cout << "Size: " << size << ", Duration: " << avg_duration << " seconds (MC=" << mc << ", KC=" << kc << ", NC=" << nc << ")\n";
            }
        }
    }
    
    bench_file.close();
    return 0;
    
}