#pragma GCC optimize("O3,unroll-loops")
#pragma GCC optimize("fast-math")
#pragma GCC target("avx2,fma")

#include <cstring>
#include <algorithm>
#include <thread>
#include <vector>
#include <cstdlib>
#include <immintrin.h>

#include "matrix.hpp"
#include <stdexcept>
#include <iostream>

Matrix::Matrix(int rows, int cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0)
{
    if (rows <= 0 || cols <= 0)
        throw std::invalid_argument("Matrix dimensions must be positive");
}

Matrix::Matrix(const Matrix &other)
    : rows(other.rows), cols(other.cols), data(other.data)
{
}

int Matrix::numRows() const
{
    return rows;
}

int Matrix::numCols() const
{
    return cols;
}

double Matrix::get(int i, int j) const
{
    return data[i*cols + j];
}

void Matrix::set(int i, int j, double value)
{
    data[i*cols + j] = value;
}

void Matrix::fill(double value)
{
    std::fill(data.begin(), data.end(), value);
}

Matrix Matrix::operator+(const Matrix &other) const
{

    Matrix result(rows, cols);
    for (int k = 0; k < cols*rows; ++k) {
        result.data[k] = data[k] + other.data[k];
    }

    return result;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    Matrix result(rows, cols);
    for (int k = 0; k < cols*rows; ++k) {
        result.data[k] = data[k] - other.data[k];
    }

    return result;
}
// Matrix multiplication using AVX2 vectorization
// FMA for fast operation (C = A*B + C)
// Operates on blocks of 8×8 to fit in the registers

// Blocking parameters (tuned for L1/L2 cache sizes)
static constexpr int MC = 48;
static constexpr int KC = 64;
static constexpr int NC = 512;

// 8×8 AVX2/FMA micro-kernel
__attribute__((always_inline))
static inline void micro_8x8(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double*       __restrict__ C,
    int klen, int n)
{
    __m256d c[8][2];
    // First load C into registers
    for (int r = 0; r < 8; r++) {
        c[r][0] = _mm256_loadu_pd(C + r*n);
        c[r][1] = _mm256_loadu_pd(C + r*n + 4);
    }

    const double* Brow = B;
    // Then loop over k dimension, broadcasting A elements and multiplying with B
    for (int k = 0; k < klen; k++, Brow += n) {
        __m256d b0 = _mm256_loadu_pd(Brow);
        __m256d b1 = _mm256_loadu_pd(Brow + 4);
        // prefetch is used to bring the next row of B into cache while we are computing with the current one
        _mm_prefetch(reinterpret_cast<const char*>(Brow + 8*n), _MM_HINT_T0); 

    #define MUL_ROW(r) { \
    __m256d a = _mm256_broadcast_sd(A + (r)*n + k); \
    c[r][0] = _mm256_fmadd_pd(a, b0, c[r][0]); \
    c[r][1] = _mm256_fmadd_pd(a, b1, c[r][1]); }
        MUL_ROW(0) MUL_ROW(1) MUL_ROW(2) MUL_ROW(3)
        MUL_ROW(4) MUL_ROW(5) MUL_ROW(6) MUL_ROW(7)
    #undef MUL_ROW
    }

    // Finally store the results back to C
    for (int r = 0; r < 8; r++) {
        _mm256_storeu_pd(C + r*n,     c[r][0]);
        _mm256_storeu_pd(C + r*n + 4, c[r][1]);
    }
}

Matrix Matrix::operator*(const Matrix& other) const
{
    // Square only in the tests ?
    const int n  = rows;
    const int n2 = other.cols;

    Matrix result(n, n2);

    const double* A = this->data.data();
    const double* B = other.data.data();
    double* C = result.data.data();

    std::memset(C, 0, (size_t)n * n2 * sizeof(double));

    for (int kk = 0; kk < cols; kk += KC) {
        const int kkend = std::min(kk + KC, cols);
        const int klen  = kkend - kk;

        for (int ii = 0; ii < n; ii += MC) {
            const int iiend = std::min(ii + MC, n);

            for (int jj = 0; jj < n2; jj += NC) {
                const int jjend = std::min(jj + NC, n2);

                for (int i = ii; i < iiend; i += 8) {
                    const int ir = std::min(8, iiend - i);
                    for (int j = jj; j < jjend; j += 8) {
                        const int jr = std::min(8, jjend - j);

                        if (ir == 8 && jr == 8) {
                            micro_8x8(A + i*cols + kk, B + kk*n2 + j, C + i*n2  + j, klen, n2);
                        } else {
                            for (int i2 = i;  i2 < i  + ir;   i2++)
                            for (int k  = kk; k  < kkend;     k++)
                            for (int j2 = j;  j2 < j  + jr;   j2++)
                                C[i2*n2 + j2] += A[i2*cols + k] * B[k*n2 + j2];
                        }
                    }
                }
            }
        }
    }

    return result;
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix result(rows, cols);
    for (int i = 0; i< rows*cols; ++i){
        result.data[i] = scalar*data[i];
    }
    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(cols, rows);
    for (int i = 0; i<cols; ++i){
        double *result_row = &result.data[i*rows];
        for (int j = 0; j< rows; ++j){
            result_row[j] = data[j*cols + i];
        }
    }
    return result;
}

Matrix Matrix::apply(const std::function<double(double)> &func) const
{
    Matrix result(rows, cols);
    for (int i = 0; i< rows*cols; ++i){
        result.data[i] = func(data[i]);
    }
    return result;
}

void Matrix::sub_mul(double scalar, const Matrix &other)
{
    //if (rows != other.rows || cols != other.cols)
    //    throw std::invalid_argument("Matrix dimensions must match for operation");
    
    for(int i = 0; i<cols*rows; ++i){
        data[i] = data[i] - scalar*other.data[i];
    }
}