#include "matrix_opencl.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <fstream>

int MatrixCL::TILE = 64; // default tile size
int MatrixCL::SUB_TILE = 8; // default subtile size (TILE/SUB_TILE must be an integer)
std::shared_ptr<KernelCache> MatrixCL::kernels_ = nullptr;

cl::Program loadAndBuildProgram(cl::Context context, 
                                    const std::vector<cl::Device>& devices,
                                    const std::string& sourceCode,
                                    const std::string& kernel_name_for_error,
                                    const std::string& options = "")
{
    cl::Program program(context, sourceCode);
    try {
        program.build(devices, options.c_str());
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
    __kernel void fill(__global float* matrix, float value, int rows, int cols, int padded_cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < rows &&  j < cols){
            matrix[i*padded_cols + j] = value;
        }
    }
)";

const std::string kernel_source_add = R"(
    __kernel void add(__global const float* A,
                      __global const float* B,
                      __global float* C,
                      int rows, int cols, int padded_cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < rows &&  j < cols){
            C[i*padded_cols + j] = A[i*padded_cols + j] + B[i*padded_cols + j];
        }
    }
)";

const std::string kernel_source_sub = R"(
    __kernel void sub(__global const float* A,
                      __global const float* B,
                      __global float* C,
                      int rows, int cols, int padded_cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < rows &&  j < cols){
            C[i*padded_cols + j] = A[i*padded_cols + j] - B[i*padded_cols + j];
        }
    }
)";

const std::string kernel_source_sub_mul = R"(
    __kernel void sub_mul(__global float* A,
                          __global const float* B,
                          float scalar,
                          int rows, int cols, int padded_cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < rows &&  j < cols){
            A[i*padded_cols + j] = A[i*padded_cols + j] - scalar * B[i*padded_cols + j];
        }
    }
)";

const std::string kernel_source_transpose = R"(
    __kernel void transpose(__global const float* A,
                            __global float* B,
                            int A_rows, int A_cols, int padded_cols, int padded_rows) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < A_rows &&  j < A_cols){
            B[j*padded_rows + i] = A[i*padded_cols + j];
        }
    }
)";

const std::string kernel_source_matrix_mul = R"(
    __kernel void matrix_mul(__global const float* A,
                             __global const float* B,
                             __global float* C,
                             __local float* Aloc,
                             __local float* Bloc,
                             int A_padded_rows, int A_padded_cols, int B_padded_cols) {
        int k;
        int lrow = get_group_id(0); // local row of workgroup
        int lcol = get_group_id(1); // local col of workgroup
        int prow = get_local_id(0); // private row of work unit
        int pcol = get_local_id(1); // private col of work unit

        int local_id = get_local_id(0) * get_local_size(1) + get_local_id(1);

        const int TILE_loc = TILE;
        const int SUB_TILE_loc = SUB_TILE;
        float Apriv[SUB_TILE_loc];
        float Bpriv[SUB_TILE_loc];
        float Cpriv[SUB_TILE_loc * SUB_TILE_loc];
        for (int i = 0; i < SUB_TILE_loc * SUB_TILE_loc; i++) Cpriv[i] = 0.0f;
        
        int row;
        int col;

        // Iterate over part of A rows and B cols
        for (int t = 0; t < A_padded_cols / TILE_loc; t++){

            // Inititialize Local blocks (common to work group) Each work item contributes
            for (int j = 0; j < SUB_TILE_loc; j++){
                for (int k = 0; k < SUB_TILE_loc; k++){
                    row = lrow * TILE_loc + prow * SUB_TILE_loc + j;
                    col = t * TILE_loc + pcol * SUB_TILE_loc + k;
                    Aloc[(prow * SUB_TILE_loc + j) * TILE_loc + pcol * SUB_TILE_loc + k] = A[row * A_padded_cols + col];

                    row = t * TILE_loc + prow * SUB_TILE_loc + j;
                    col = lcol * TILE_loc + pcol * SUB_TILE_loc + k;
                    Bloc[(prow * SUB_TILE_loc + j) * TILE_loc + pcol * SUB_TILE_loc + k] = B[row * B_padded_cols + col];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Compute private tile contributions, iterating over the shared dimension in tiles
            for (k = 0; k < TILE_loc; k++){
                // load one element from each row of A local and one element from each column of B local into private memory
                for (int j = 0; j < SUB_TILE_loc; j++){
                    Apriv[j] = Aloc[(prow * SUB_TILE_loc + j) * TILE_loc + k];
                    Bpriv[j] = Bloc[(k) * TILE_loc + pcol * SUB_TILE_loc + j];
                }
                // compute contribution of column k/ row k of A/ B to private tile
                for (int j = 0; j < SUB_TILE_loc; j++){
                    for (int m = 0; m < SUB_TILE_loc; m++){
                        Cpriv[j * SUB_TILE_loc + m] += Apriv[j] * Bpriv[m];
                    }
                }

            }
            
            barrier(CLK_LOCAL_MEM_FENCE);    

        }
        // Write private Tile back to C global
        for (int j = 0; j < SUB_TILE_loc; j++){
            for (int k = 0; k < SUB_TILE_loc; k++){
                C[(lrow * TILE_loc + prow * SUB_TILE_loc + j) * B_padded_cols + lcol * TILE_loc + pcol * SUB_TILE_loc + k] = Cpriv[j * SUB_TILE_loc + k];
            }
        }
    }
)";

const std::string kernel_source_mult = R"(
    __kernel void mult(__global const float* A, __global float* B,
                     float scalar,
                      int rows, int cols, int padded_cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < rows &&  j < cols){
            B[i*padded_cols + j] = A[i*padded_cols + j] * scalar;
        }
    }
)";

// --- KernelCache ---

void KernelCache::compileKernels(cl::Context context, const std::vector<cl::Device>& devices, int TILE, int SUB_TILE) {
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

        std::string options = "-DTILE=" + std::to_string(TILE) + " -DSUB_TILE=" + std::to_string(SUB_TILE);
        cl::Program prog_matrix_mul = loadAndBuildProgram(context, devices, kernel_source_matrix_mul, "matrix_mul", options);
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

void MatrixCL::initializeKernels(cl::Context context, const std::vector<cl::Device>& devices, int TILE_SIZE, int SUB_TILE_SIZE) {
    TILE = TILE_SIZE;
    SUB_TILE = SUB_TILE_SIZE;
    try {
        if (!kernels_ || !kernels_->initialized) {
            std::cout << "Creating and compiling kernels..." << std::endl;
            kernels_ = std::make_shared<KernelCache>();
            kernels_->compileKernels(context, devices, TILE, SUB_TILE);
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
    return static_cast<size_t>(padded_rows_) * padded_cols_ * sizeof(float);
}

MatrixCL::MatrixCL(int rows, int cols, cl::Context context, cl::CommandQueue queue, const std::vector<float>* initial_data)
    : rows_(rows), cols_(cols), context_(context), queue_(queue)
{
    // Pad rows and cols to be a multiple of TILE
    padded_rows_ = ((rows_ + TILE - 1) / TILE) * TILE;
    padded_cols_ = ((cols_ + TILE - 1) / TILE) * TILE;
    // Copy initial data to a padded vector if provided, otherwise create an empty padded vector
    std::vector<float> padded_data;
    if (initial_data != nullptr) {
        padded_data.resize(static_cast<size_t>(padded_rows_) * padded_cols_, 0.0f);
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                padded_data[i * padded_cols_ + j] = (*initial_data)[i * cols_ + j];
            }
        }
    } else {
        padded_data.assign(static_cast<size_t>(padded_rows_) * padded_cols_, 0.0f);
    }

    // Store data vector in the Device buffer
    if (initial_data == nullptr){
        try {
        buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(float) * padded_rows_ * padded_cols_);
        } catch (const cl::Error& e) {
            throw std::runtime_error(std::string("Buffer creation failed: ")  + e.what() + " (" + std::to_string(e.err()) + ")");
        }
        return;
    }
    try {
        buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * padded_rows_ * padded_cols_, (void*)padded_data.data());
    } catch (const cl::Error& e) {
        throw std::runtime_error(std::string("Buffer creation failed: ")  + e.what() + " (" + std::to_string(e.err()) + ")");
    }
}

MatrixCL::MatrixCL(const MatrixCL& other)
    : rows_(other.rows_), cols_(other.cols_), padded_rows_(other.padded_rows_), padded_cols_(other.padded_cols_),
      context_(other.context_), queue_(other.queue_)
{
    try {
        std::vector<float> empty_data(static_cast<size_t>(rows_) * cols_, 0.0f);
        empty_data = other.copyToHost();
        std::vector<float> padded_data(static_cast<size_t>(padded_rows_) * padded_cols_, 0.0f);
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                padded_data[i * padded_cols_ + j] = empty_data[i * cols_ + j];
            }
        }

        buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * padded_rows_ * padded_cols_, padded_data.data());
    } catch (const cl::Error& e) {
        throw std::runtime_error(std::string("Buffer creation failed: ")  + e.what() + " (" + std::to_string(e.err()) + ")");
    }
}

MatrixCL& MatrixCL::operator=(const MatrixCL& other)
{
    if (this == &other) return *this;

    rows_ = other.rows_;
    cols_ = other.cols_;
    padded_cols_ = other.padded_cols_;
    padded_rows_ = other.padded_rows_;
    context_ = other.context_;
    queue_ = other.queue_;
    try {
        std::vector<float> empty_data(static_cast<size_t>(rows_) * cols_, 0.0f);
        empty_data = other.copyToHost();
        std::vector<float> padded_data(static_cast<size_t>(padded_rows_) * padded_cols_, 0.0f);
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                padded_data[i * padded_cols_ + j] = empty_data[i * cols_ + j];
            }
        }

        buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * padded_rows_ * padded_cols_, padded_data.data());
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
    std::vector<float> host_data(static_cast<size_t>(padded_rows_) * padded_cols_);
    size_t size = buffer_size_bytes();
    if (size == 0) return host_data;

    // TODO
    try {
        queue_.enqueueReadBuffer(buffer_, CL_TRUE, 0, size, host_data.data());
    }catch (const cl::Error& e) {
        throw std::runtime_error(std::string("Copy to Host failed: ")  + e.what() + " (" + std::to_string(e.err()) + ")");
    }

    // Remove padding zeros :
    std::vector<float> real_data(static_cast<size_t>(rows_) * cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            // printf("Copying element : %f,  (%d, %d) from padded index (%d, %d)\n", host_data[i * padded_cols_ + j], i, j, i, j);
            real_data[i * cols_ + j] = host_data[i * padded_cols_ + j];
        }
    }

    return real_data;
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
    current_fill.setArg(4, padded_cols_);

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
    current_add.setArg(5, padded_cols_);

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
    current_sub.setArg(5, padded_cols_);

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
    current_mult.setArg(5, padded_cols_);

    cl::NDRange h_range(rows_, cols_);
    queue_.enqueueNDRangeKernel(current_mult, cl::NullRange, h_range, cl::NullRange);

    queue_.finish();
    return result;
}

void save_profiling_info(const std::string& filename, size_t N, size_t flops, double exec_time, double comm_time, int TILE_SIZE, int SUB_TILE_SIZE) {
    std::fstream bench_file;
    bench_file.open(filename, std::ios::out | std::ios::app);
    if (bench_file.is_open()) {
        if (bench_file.tellp() == 0) {
            bench_file << "Size,Duration,Comm_time, Gflops,TILE_SIZE,SUB_TILE_SIZE\n";
        }
        bench_file << N << "," << exec_time << "," << comm_time << "," << (flops / (exec_time * 1e6)) << "," << TILE_SIZE << "," << SUB_TILE_SIZE << "\n";
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

    size_t local_mem_size = TILE * TILE * sizeof(float);

    cl::Kernel current_mat_mul = kernels_->kernel_matrix_mul;
    current_mat_mul.setArg(0, buffer_);
    current_mat_mul.setArg(1, other.buffer_);
    current_mat_mul.setArg(2, result.buffer_);
    current_mat_mul.setArg(3, cl::Local(local_mem_size)); // Aloc
    current_mat_mul.setArg(4, cl::Local(local_mem_size)); // Bloc
    current_mat_mul.setArg(5, padded_rows_);
    current_mat_mul.setArg(6, padded_cols_);
    current_mat_mul.setArg(7, other.padded_cols_);
    
    cl::NDRange global(padded_rows_ / SUB_TILE, other.padded_cols_ / SUB_TILE);  // one work item per SUB_TILE×SUB_TILE output
    cl::NDRange local(TILE / SUB_TILE, TILE / SUB_TILE);        // 16×16 = 256 work items per group
    // create event for profiling
    cl::Event event;
    queue_.enqueueNDRangeKernel(current_mat_mul, cl::NullRange, global, local, nullptr, &event);

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
    size_t N = padded_rows_;
    size_t M = other.padded_cols_;
    size_t K = padded_cols_;
    double flops = 2.0 * N * M * K;
    std::fstream bench_file;
    save_profiling_info("opencl_bench.csv", N, flops, total_time, comm_time, MatrixCL::TILE, MatrixCL::SUB_TILE);

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
    current_transpose.setArg(4, padded_cols_);
    current_transpose.setArg(5, padded_rows_);

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
    current_sub_mul.setArg(5, padded_cols_);

    cl::NDRange h_range(rows_, cols_);
    queue_.enqueueNDRangeKernel(current_sub_mul, cl::NullRange, h_range, cl::NullRange);

    queue_.finish();
}
