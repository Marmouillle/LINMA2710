#include "matrix_opencl_basic.hpp"
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

    int max_size = 16384; // 2^14
    if (argc == 2) {
        max_size = std::stoi(argv[1]);
    }

    for (int size = 64; size <= max_size; size *= 2)
    {
        printf("Running benchmark for size %d...\n", size);
        MatrixCL A(size, size, context, queue);
        MatrixCL B(size, size, context, queue);
        A.fill(1.0);
        B.fill(2.0);
        
        // average over multiple runs to reduce noise
        for (int run = 0; run < 5; ++run) {
            MatrixCL C = A*B;
        }
    }

    return 0;
}