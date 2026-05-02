#include "matrix_opencl.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

bool verifyMatrix(const MatrixCL& mat, const std::vector<float>& expected, float epsilon = 1e-5f) {
    if (static_cast<size_t>(mat.numRows() * mat.numCols()) != expected.size()){
        printf("Size mismatch: matrix has %d elements but expected has %zu elements\n", mat.numRows() * mat.numCols(), expected.size());
        return false;
    }
    std::vector<float> actual = mat.copyToHost();
    for (size_t i = 0; i < actual.size(); ++i){
        //printf("actual[%zu] = %f, expected[%zu] = %f\n", i, actual[i], i, expected[i]);
        if (!approxEqual(actual[i], expected[i], epsilon))
            return false;
    }
    return true;
}

cl::Context context;
cl::CommandQueue queue;

void setupOpenCL(int TILE = 64, int SUB_TILE = 8) {
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

    MatrixCL::initializeKernels(context, {device}, TILE, SUB_TILE);

    std::cout << "setupOpenCL passed." << std::endl;
}

void testMatrixMultiplication() {
    MatrixCL matA(16384, 16384, context, queue);
    MatrixCL matC(16384, 16384, context, queue);
    matA.fill(1.0f);
    matC.fill(2.0f);

    // start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    MatrixCL result = matA * matC;
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
    
    std::cout << "Matrix multiplication took " << duration_ms.count() << " ms." << std::endl;
    assert(verifyMatrix(result, std::vector<float>(16384 * 16384, 2*16384.0f)));

    std::cout << "testMatrixMultiplication passed." << std::endl;
}


int main() {
    try {
        setupOpenCL();
        testMatrixMultiplication();
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

    std::cout << "All OpenCL matrix tests passed." << std::endl;
    return 0;
}
// marie aime maxim et maxim aime marie