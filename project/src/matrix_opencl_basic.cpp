#include "matrix_opencl_basic.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <fstream>

std::shared_ptr<KernelCache> MatrixCL::kernels_ = nullptr;

cl::Program loadAndBuildProgram(cl::Context context,
                                const std::vector<cl::Device>& devices,
                                const std::string& sourceCode,
                                const std::string& kernel_name_for_error)
{
    cl::Program program(context, sourceCode);
    try {
        program.build(devices);
    } catch (const cl::BuildError& err) {
        std::cerr << "OpenCL Build Error for kernel source '" << kernel_name_for_error << "':\n"
                  << err.what() << "(" << err.err() << ")" << std::endl;
        for (const auto& pair : err.getBuildLog()) {
            std::cerr << "Device " << pair.first.getInfo<CL_DEVICE_NAME>() << ":" << std::endl;
            std::cerr << pair.second << std::endl;
        }
        throw;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error during program build for '" << kernel_name_for_error << "': "
                  << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    }
    return program;
}

// --- OpenCL Kernel Source Code ---

const std::string kernel_source_fill = R"(
    __kernel void fill(__global float* matrix, float value, int rows, int cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < rows &&  j < cols){
            matrix[i*cols + j] = value;
        }
    }
)";

const std::string kernel_source_add = R"(
    __kernel void add(__global const float* A,
                      __global const float* B,
                      __global float* C,
                      int rows, int cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < rows &&  j < cols){
            C[i*cols + j] = A[i*cols + j] + B[i*cols + j];
        }
    }
)";

const std::string kernel_source_sub = R"(
    __kernel void sub(__global const float* A,
                      __global const float* B,
                      __global float* C,
                      int rows, int cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < rows &&  j < cols){
            C[i*cols + j] = A[i*cols + j] - B[i*cols + j];
        }
    }
)";

const std::string kernel_source_sub_mul = R"(
    __kernel void sub_mul(__global float* A,
                          __global const float* B,
                          float scalar,
                          int rows, int cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < rows &&  j < cols){
            A[i*cols + j] = A[i*cols + j] - scalar * B[i*cols + j];
        }
    }
)";

const std::string kernel_source_transpose = R"(
    __kernel void transpose(__global const float* A,
                            __global float* B,
                            int A_rows, int A_cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < A_rows &&  j < A_cols){
            B[j*A_rows + i] = A[i*A_cols + j];
        }
    }
)";

const std::string kernel_source_matrix_mul = R"(
    __kernel void matrix_mul(__global const float* A,
                             __global const float* B,
                             __global float* C,
                             int A_rows, int A_cols, int B_cols) {
        int k;
        int i = get_global_id(0);
        int j = get_global_id(1);
        float temp = 0.0f;
        if (i < A_rows &&  j < B_cols){
            for(k = 0; k < A_cols; k++){
                temp += A[i * A_cols + k] * B[k * B_cols + j];
            }
            C[i * B_cols + j] = temp;
        }
        
    }
)";

const std::string kernel_source_mult = R"(
    __kernel void mult(__global const float* A, __global float* B,
                     float scalar,
                      int rows, int cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < rows &&  j < cols){
            B[i*cols + j] = A[i*cols + j] * scalar;
        }
    }
)";

// --- KernelCache ---

void KernelCache::compileKernels(cl::Context context, const std::vector<cl::Device>& devices) {
    if (initialized) return;

    std::cout << "Compiling OpenCL kernels..." << std::endl;
    try {
        cl::Program prog_fill = loadAndBuildProgram(context, devices, kernel_source_fill, "fill");
        kernel_fill = cl::Kernel(prog_fill, "fill");

        cl::Program prog_add = loadAndBuildProgram(context, devices, kernel_source_add, "add");
        kernel_add = cl::Kernel(prog_add, "add");

        cl::Program prog_sub = loadAndBuildProgram(context, devices, kernel_source_sub, "sub");
        kernel_sub = cl::Kernel(prog_sub, "sub");

        cl::Program prog_sub_mul = loadAndBuildProgram(context, devices, kernel_source_sub_mul, "sub_mul");
        kernel_sub_mul = cl::Kernel(prog_sub_mul, "sub_mul");

        cl::Program prog_transpose = loadAndBuildProgram(context, devices, kernel_source_transpose, "transpose");
        kernel_transpose = cl::Kernel(prog_transpose, "transpose");

        cl::Program prog_matrix_mul = loadAndBuildProgram(context, devices, kernel_source_matrix_mul, "matrix_mul");
        kernel_matrix_mul = cl::Kernel(prog_matrix_mul, "matrix_mul");

        cl::Program prog_mult = loadAndBuildProgram(context, devices, kernel_source_mult, "mult");
        kernel_mult = cl::Kernel(prog_mult, "mult");

        initialized = true;
        std::cout << "OpenCL kernels compiled successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to compile one or more OpenCL kernels. Aborting." << std::endl;
        throw;
    }
}

// --- MatrixCL Static Methods ---

void MatrixCL::initializeKernels(cl::Context context, const std::vector<cl::Device>& devices) {
    try {
        if (!kernels_ || !kernels_->initialized) {
            std::cout << "Creating and compiling kernels..." << std::endl;
            kernels_ = std::make_shared<KernelCache>();
            kernels_->compileKernels(context, devices);
        }
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in kernel initialization: "
                  << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Exception in kernel initialization: " << e.what() << std::endl;
        throw;
    }
}

// --- MatrixCL Implementation ---

size_t MatrixCL::buffer_size_bytes() const {
    return static_cast<size_t>(rows_) * cols_ * sizeof(float);
}

MatrixCL::MatrixCL(int rows, int cols, cl::Context context, cl::CommandQueue queue, const std::vector<float>* initial_data)
    : rows_(rows), cols_(cols), context_(context), queue_(queue)
{
    // TODO
    // Store data vector in the Device buffer
    if (initial_data == nullptr){
        try {
        buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(float) * rows_ * cols_);
        } catch (const cl::Error& e) {
            throw std::runtime_error(std::string("Buffer creation failed: ")  + e.what() + " (" + std::to_string(e.err()) + ")");
        }
        return;
    }
    try {
        buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * rows * cols, (void*)initial_data->data());
    } catch (const cl::Error& e) {
        throw std::runtime_error(std::string("Buffer creation failed: ")  + e.what() + " (" + std::to_string(e.err()) + ")");
    }
}

MatrixCL::MatrixCL(const MatrixCL& other)
    : rows_(other.rows_), cols_(other.cols_),
      context_(other.context_), queue_(other.queue_)
{
    // TODO
    // Just initialize an empty buffer 
    try {
        std::vector<float> empty_data(static_cast<size_t>(rows_) * cols_, 0.0f);
        empty_data = other.copyToHost();
        buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * rows_ * cols_, empty_data.data());
    } catch (const cl::Error& e) {
        throw std::runtime_error(std::string("Buffer creation failed: ")  + e.what() + " (" + std::to_string(e.err()) + ")");
    }
}

MatrixCL& MatrixCL::operator=(const MatrixCL& other)
{
    if (this == &other) return *this;

    // TODO
    rows_ = other.rows_;
    cols_ = other.cols_;
    context_ = other.context_;
    queue_ = other.queue_;
    try {
        std::vector<float> empty_data(static_cast<size_t>(rows_) * cols_, 0.0f);
        empty_data = other.copyToHost();
        buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * rows_ * cols_, empty_data.data());
    } catch (const cl::Error& e) {
        throw std::runtime_error(std::string("Buffer creation failed: ")  + e.what() + " (" + std::to_string(e.err()) + ")");
    }

    return *this;
}

int MatrixCL::numRows() const { return rows_; }
int MatrixCL::numCols() const { return cols_; }
cl::Context MatrixCL::getContext() const { return context_; }
cl::CommandQueue MatrixCL::getQueue() const { return queue_; }
const cl::Buffer& MatrixCL::getBuffer() const { return buffer_; }

std::vector<float> MatrixCL::copyToHost() const
{
    std::vector<float> host_data(static_cast<size_t>(rows_) * cols_);
    size_t size = buffer_size_bytes();
    if (size == 0) return host_data;

    // TODO
    try {
        queue_.enqueueReadBuffer(buffer_, CL_TRUE, 0, size, host_data.data());
    }catch (const cl::Error& e) {
        throw std::runtime_error(std::string("Copy to Host failed: ")  + e.what() + " (" + std::to_string(e.err()) + ")");
    }

    return host_data;
}

void MatrixCL::fill(float value)
{
    if (rows_ * cols_ == 0) return;

    // TODO
    cl::Kernel current_fill = kernels_->kernel_fill;
    current_fill.setArg(0, buffer_);
    current_fill.setArg(1, value);
    current_fill.setArg(2, rows_);
    current_fill.setArg(3, cols_);

    cl::NDRange h_range(rows_, cols_);
    queue_.enqueueNDRangeKernel(current_fill, cl::NullRange, h_range, cl::NullRange);

    queue_.finish();
}

MatrixCL MatrixCL::operator+(const MatrixCL& other) const
{
    MatrixCL result(rows_, cols_, context_, queue_);
    if (rows_ * cols_ == 0) return result;

    // TODO
    cl::Kernel current_add = kernels_->kernel_add;
    current_add.setArg(0, buffer_);
    current_add.setArg(1, other.buffer_);
    current_add.setArg(2, result.buffer_);
    current_add.setArg(3, rows_);
    current_add.setArg(4, cols_);

    cl::NDRange h_range(rows_, cols_);
    queue_.enqueueNDRangeKernel(current_add, cl::NullRange, h_range, cl::NullRange);

    queue_.finish();
    return result;
}

MatrixCL MatrixCL::operator-(const MatrixCL& other) const
{
    MatrixCL result(rows_, cols_, context_, queue_);
    if (rows_ * cols_ == 0) return result;

    // TODO
    cl::Kernel current_sub = kernels_->kernel_sub;
    current_sub.setArg(0, buffer_);
    current_sub.setArg(1, other.buffer_);
    current_sub.setArg(2, result.buffer_);
    current_sub.setArg(3, rows_);
    current_sub.setArg(4, cols_);

    cl::NDRange h_range(rows_, cols_);
    queue_.enqueueNDRangeKernel(current_sub, cl::NullRange, h_range, cl::NullRange);

    queue_.finish();
    return result;
}

MatrixCL MatrixCL::operator*(float scalar) const
{
    MatrixCL result(rows_, cols_, context_, queue_);
    if (rows_ * cols_ == 0) return result;

    // TODO
    cl::Kernel current_mult = kernels_->kernel_mult;
    current_mult.setArg(0, buffer_);
    current_mult.setArg(1, result.buffer_);
    current_mult.setArg(2, scalar);
    current_mult.setArg(3, rows_);
    current_mult.setArg(4, cols_);

    cl::NDRange h_range(rows_, cols_);
    queue_.enqueueNDRangeKernel(current_mult, cl::NullRange, h_range, cl::NullRange);

    queue_.finish();
    return result;
}

void save_profiling_info(const std::string& filename, size_t N, size_t flops, double exec_time, double comm_time) {
    std::fstream bench_file;
    bench_file.open(filename, std::ios::out | std::ios::app);
    if (bench_file.is_open()) {
        if (bench_file.tellp() == 0) {
            bench_file << "Size,Duration,Comm_time, Gflops\n";
        }
        bench_file << N << "," << exec_time << "," << comm_time << "," << (flops / (exec_time * 1e6)) << "\n";
        bench_file.close();
    } else {
        std::cerr << "Failed to open profiling output file." << std::endl;
    }
}

MatrixCL MatrixCL::operator*(const MatrixCL& other) const
{
    int C_rows = this->rows_;
    int C_cols = other.cols_;
    MatrixCL result(C_rows, C_cols, context_, queue_);
    if (C_rows * C_cols == 0) return result;

    // TODO
    cl::Kernel current_mat_mul = kernels_->kernel_matrix_mul;
    current_mat_mul.setArg(0, buffer_);
    current_mat_mul.setArg(1, other.buffer_);
    current_mat_mul.setArg(2, result.buffer_);
    current_mat_mul.setArg(3, rows_);
    current_mat_mul.setArg(4, cols_);
    current_mat_mul.setArg(5, other.cols_);

    cl::NDRange h_range(rows_, cols_);
    cl::Event event;
    queue_.enqueueNDRangeKernel(current_mat_mul, cl::NullRange, h_range, cl::NullRange, nullptr, &event);

    queue_.finish();
    // Get profiling info
    cl_ulong time_queued = event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    cl_ulong time_submit = event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    cl_ulong time_start  = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong time_end    = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    double queued_to_submit = (time_submit - time_queued) * 1e-6;
    double submit_to_start  = (time_start - time_submit) * 1e-6;
    double exec_time        = (time_end   - time_start) * 1e-6;
    double total_time       = (time_end   - time_queued) * 1e-6;
    double comm_time = total_time - exec_time;

    std::cout << "=== OpenCL Profiling ===\n";
    std::cout << "Kernel execution time   : " << exec_time << " ms\n";
    std::cout << "Queue -> Submit latency : " << queued_to_submit << " ms\n";
    std::cout << "Submit -> Start latency : " << submit_to_start << " ms\n";
    std::cout << "Total event time        : " << total_time << " ms\n";
    std::cout << "Communication time      : " << comm_time << " ms\n";
    size_t N = rows_;
    size_t M = other.cols_;
    size_t K = cols_;
    double flops = 2.0 * N * M * K;
    std::fstream bench_file;
    save_profiling_info("opencl_basic_bench.csv", N, flops, total_time, comm_time);

    return result;
}

MatrixCL MatrixCL::transpose() const
{
    MatrixCL result(cols_, rows_, context_, queue_);
    if (rows_ * cols_ == 0) return result;

    // TODO
    cl::Kernel current_transpose = kernels_->kernel_transpose;
    current_transpose.setArg(0, buffer_);
    current_transpose.setArg(1, result.buffer_);
    current_transpose.setArg(2, rows_);
    current_transpose.setArg(3, cols_);

    cl::NDRange h_range(rows_, cols_);
    queue_.enqueueNDRangeKernel(current_transpose, cl::NullRange, h_range, cl::NullRange);

    queue_.finish();

    return result;
}

void MatrixCL::sub_mul(float scalar, const MatrixCL& other)
{
    if (rows_ * cols_ == 0) return;

    // TODO
    cl::Kernel current_sub_mul = kernels_->kernel_sub_mul;
    current_sub_mul.setArg(0, buffer_);
    current_sub_mul.setArg(1, other.buffer_);
    current_sub_mul.setArg(2, scalar);
    current_sub_mul.setArg(3, rows_);
    current_sub_mul.setArg(4, cols_);

    cl::NDRange h_range(rows_, cols_);
    queue_.enqueueNDRangeKernel(current_sub_mul, cl::NullRange, h_range, cl::NullRange);

    queue_.finish();
}
