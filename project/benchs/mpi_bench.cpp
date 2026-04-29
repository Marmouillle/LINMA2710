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

    int max_size = 5000; // Maximum matrix size to test
    if (argc == 2)
    {
        max_size = std::stoi(argv[1]);
    }
    std::fstream bench_file;
    if (rank == 0) {
        bench_file.open("mpi_bench.csv", std::ios::app);
        if (bench_file.is_open() && bench_file.tellp() == 0) {
            bench_file << "Size,Duration,NumProcesses\n";
        }
    }
    
    for (int size = 500; size <= max_size; size += 500)
    {
        Matrix A(size, size);
        Matrix B(size, size);
        A.fill(1.0);
        B.fill(1.0);

        DistributedMatrix distA(A, num_processes); // Create distributed matrices with 1 process (no actual distribution)
        DistributedMatrix distB(B, num_processes);


        auto start_time = std::chrono::high_resolution_clock::now();
        Matrix distC = distA.multiplyTransposed(distB);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (rank == 0) {
            std::chrono::duration<double> duration = end_time - start_time;
            std::cout << "Distributed matrix multiplication duration: " << duration.count() << " seconds" << std::endl;
            bench_file << size << "," << duration.count() << "," << num_processes << "\n";
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