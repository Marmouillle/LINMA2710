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

void setupOpenCL(int TILE_SIZE = 64, int SUB_TILE_SIZE = 4) {
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

    MatrixCL::initializeKernels(context, {device}, TILE_SIZE, SUB_TILE_SIZE);

    std::cout << "setupOpenCL passed." << std::endl;

    for (size_t p = 0; p < platforms.size(); ++p) {
        std::cout << "=== Platform " << p << " ===" << std::endl;
        std::cout << "  Name:    " << platforms[p].getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "  Vendor:  " << platforms[p].getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        std::cout << "  Version: " << platforms[p].getInfo<CL_PLATFORM_VERSION>() << std::endl;

        std::vector<cl::Device> devices;
        platforms[p].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (size_t d = 0; d < devices.size(); ++d) {
            const auto& dev = devices[d];
            std::cout << "\n  --- Device " << d << " ---" << std::endl;
            std::cout << "    Name:              " << dev.getInfo<CL_DEVICE_NAME>() << std::endl;

            cl_device_type type = dev.getInfo<CL_DEVICE_TYPE>();
            std::string type_str = (type == CL_DEVICE_TYPE_GPU) ? "GPU" :
                                   (type == CL_DEVICE_TYPE_CPU) ? "CPU" : "OTHER";
            std::cout << "    Type:              " << type_str << std::endl;

            std::cout << "    Compute units:     " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "    Max work-group:    " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
            std::cout << "    Clock frequency:   " << dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz" << std::endl;
            std::cout << "    Global memory:     " << dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024*1024) << " MB" << std::endl;
            std::cout << "    Local memory:      " << dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KB" << std::endl;
            std::cout << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    // parse TILE and SUB_TILE from command line arguments if provided
    int TILE_SIZE = 64;
    int SUB_TILE_SIZE = 4;
    if (argc >= 3) {
        TILE_SIZE = std::stoi(argv[1]);
        SUB_TILE_SIZE = std::stoi(argv[2]);
    }
    try {
        setupOpenCL(TILE_SIZE, SUB_TILE_SIZE);
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

    int max_size = 16384; // 2^14
    if (argc == 4)
    {
        max_size = std::stoi(argv[3]);
    }
    std::fstream bench_file;
    bench_file.open("opencl_param_bench.csv", std::ios::out | std::ios::app);
    if (bench_file.is_open()) {
        if (bench_file.tellp() == 0) {
            bench_file << "Size,Duration,TILE_SIZE,SUB_TILE_SIZE\n";
        }
    } else {
        std::cerr << "Failed to open benchmark output file." << std::endl;
        return 1;
    }
    
    for (int size = max_size; size <= max_size; size *= 2)
    {
        printf("Running benchmark for size %d...\n", size);
        MatrixCL A(size, size, context, queue);
        MatrixCL B(size, size, context, queue);
        A.fill(1.0);
        B.fill(2.0);
        
        double total_duration = 0.0;
        // average over multiple runs to reduce noise
        for (int run = 0; run < 5; ++run) {
            auto start_time = std::chrono::high_resolution_clock::now();
            MatrixCL C = A*B;
            auto end_time = std::chrono::high_resolution_clock::now();
        
            std::chrono::duration<double> duration = end_time - start_time;
            total_duration += duration.count();
        }
        total_duration /= 5.0; // average duration
        
        std::cout << "Distributed matrix multiplication average duration: " << total_duration << " seconds" << std::endl;
        bench_file << size << "," << total_duration << "," << TILE_SIZE << "," << SUB_TILE_SIZE << "\n";
        std::cout << "Benchmark of size" << size << "completed." << std::endl;
        }
    if (bench_file.is_open())
        bench_file.close();
    std::cout << "Benchmark completed succesfully" << std::endl;

    return 0;
}