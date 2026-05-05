#include "distributed_matrix.hpp"
#include "matrix.hpp"
#include <mpi.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <functional>
#include <chrono>
#include <fstream>

int main(int argc, char** argv) {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized)
        MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    std::cout << "Max number of processes available: " << num_processes << std::endl;

    int max_size = 4096; // Maximum matrix size to test
    if (argc == 2)
    {
        max_size = std::stoi(argv[1]);
    }
    std::fstream bench_file;
    if (rank == 0) {
        bench_file.open("mpi_bench.csv", std::ios::app);
        if (bench_file.is_open() && bench_file.tellp() == 0) {
            bench_file << "Size,DurationTotal,DurationComm,NumProcesses\n";
        }
    }
    
    for (int size = 64; size <= max_size; size *= 2)
    {
        Matrix A(size, size);
        Matrix B(size, size);
        A.fill(1.0);
        B.fill(1.0);

        const int runs = 5;
        double total_duration = 0.0;
        double times_sum[2] = {0.0, 0.0};

        for (int run = 0; run < runs; ++run) {
            double times[2] = {0.0, 0.0};
            DistributedMatrix distA(A, num_processes);
            DistributedMatrix distB(B, num_processes);

            auto start_time = std::chrono::high_resolution_clock::now();
            Matrix distC = distA.multiplyTransposed(distB, times);
            auto end_time = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end_time - start_time;
            total_duration += duration.count();
            times_sum[0] += times[0];
            times_sum[1] += times[1];
        }

        double avg_duration = total_duration / runs;
        double avg_times[2] = { times_sum[0] / runs, times_sum[1] / runs };

        if (rank == 0) {
            std::cout << "Distributed matrix multiplication average duration over " << runs << " runs: " << avg_duration << " seconds" << std::endl;
            bench_file << size << "," << avg_times[0] << "," << avg_times[1] << "," << num_processes << "\n";
            std::cout << "Benchmark completed." << std::endl;
        }
    }
    if (rank == 0) {
        if (bench_file.is_open())
            bench_file.close();
        std::cout << "Benchmark completed succesfully" << std::endl;
    }

    MPI_Finalize();
    return 0;
}