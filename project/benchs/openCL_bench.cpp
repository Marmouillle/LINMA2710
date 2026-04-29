#include "matrix_opencl.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <functional>
#include <chrono>
#include <fstream>

cl::Context context;
cl::CommandQueue queue;

void setupOpenCL() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(!platforms.empty());

    cl::Platform platform = platforms.front();
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
    assert(!devices.empty());

    cl::Device device = devices.front();
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    MatrixCL::initializeKernels(context, {device});

    std::cout << "setupOpenCL passed." << std::endl;
}

int main(int argc, char** argv) {

    try {
        setupOpenCL();
    } catch (const cl::BuildError& err) {
        std::cerr << "OpenCL Build Error: " << err.what() << " (" << err.err() << ")" << std::endl;
        for (const auto& pair : err.getBuildLog())
            std::cerr << "Build Log (" << pair.first.getInfo<CL_DEVICE_NAME>() << "):\n" << pair.second << std::endl;
        return 1;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error: " << err.what() << " (" << err.err() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    int max_size = 5000; // Maximum matrix size to test
    if (argc == 2)
    {
        max_size = std::stoi(argv[1]);
    }
    std::fstream bench_file;
    if (rank == 0) {
        bench_file.open("openCL_bench.csv", std::ios::app);
        if (bench_file.is_open() && bench_file.tellp() == 0) {
            bench_file << "Size,Duration,NumProcesses\n";
        }
    }
    
    for (int size = 500; size <= max_size; size += 500)
    {
        MatrixCL A(size, size);
        Matrix B(size, size);
        A.fill(1.0);
        B.fill(1.0);

        DistributedMatrix distA(A, num_processes);
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