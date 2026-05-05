#pragma GCC optimize("O3,unroll-loops")
#pragma GCC optimize("fast-math")
#pragma GCC target("avx2,fma")

#include <cstring>
#include <algorithm>
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
static constexpr int MC = 80;
static constexpr int KC = 48;
static constexpr int NC = 256;

// 8×8 AVX2/FMA micro-kernel
__attribute__((always_inline))
static inline void micro_8x8(
    const double* __restrict__ Apacked,
    const double* __restrict__ Bpacked,
    double*       __restrict__ C,
    int klen, int jlen, int n2)
{
    __m256d c[8][2];
    // First load C into registers
    for (int r = 0; r < 8; r++) {
        c[r][0] = _mm256_loadu_pd(C + r*n2);
        c[r][1] = _mm256_loadu_pd(C + r*n2 + 4);
    }

    // Then loop over the rows of B (should be kc rows)
    for (int k = 0; k < klen; k++) {
        __m256d b0 = _mm256_loadu_pd(Bpacked + k * jlen);
        __m256d b1 = _mm256_loadu_pd(Bpacked + k * jlen + 4);
        // prefetch : bring the next row of B into cache while we are computing with the current one
        _mm_prefetch(reinterpret_cast<const char*>(Bpacked + (k + 4) * jlen), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(Apacked + (k + 8)), _MM_HINT_T0);

        // broadcast A[r][k] and perform FMA for each row of C
        #define MUL_ROW(r) { \
            __m256d a = _mm256_broadcast_sd(Apacked + (r)*klen + k); \
            c[r][0] = _mm256_fmadd_pd(a, b0, c[r][0]); \
            c[r][1] = _mm256_fmadd_pd(a, b1, c[r][1]); }
        MUL_ROW(0) MUL_ROW(1) MUL_ROW(2) MUL_ROW(3)
        MUL_ROW(4) MUL_ROW(5) MUL_ROW(6) MUL_ROW(7)
        #undef MUL_ROW
    }

    // Finally store the results back to C
    for (int r = 0; r < 8; r++) {
        _mm256_storeu_pd(C + r*n2, c[r][0]);
        _mm256_storeu_pd(C + r*n2 + 4, c[r][1]);
    }
}

// NC x KC
void pack_B(int klen, int jlen, int n2, const double* B_src, double* B_packed) {
    for (int k = 0; k < klen; ++k) {
        const double* src_row = B_src + k * n2;
        double* dest_row = B_packed + k * jlen;

        _mm_prefetch(reinterpret_cast<const char*>(B_src + (k + 4) * n2), _MM_HINT_T0);
        int j = 0;
        // copy 4 doubles at a time
        for (; j <= jlen - 4; j += 4) {
            _mm256_storeu_pd(dest_row + j, _mm256_loadu_pd(src_row + j));
        }
        // copy remaining elements if jlen is not a multiple of 4
        for (; j < jlen; ++j) {
            dest_row[j] = src_row[j];
        }
    }
}

void pack_A(int klen, int ilen, int n, const double* A_src, double* A_packed) {
    for (int i = 0; i < ilen; ++i) {
        const double* src_row = A_src + i * n;
        double* dest_row = A_packed + i * klen;

        _mm_prefetch(reinterpret_cast<const char*>(A_src + (i + 4) * n), _MM_HINT_T0);
        int k = 0;
        // copy 4 doubles at a time
        for (; k <= klen - 4; k += 4) {
            _mm256_storeu_pd(dest_row + k, _mm256_loadu_pd(src_row + k));
        }
        // copy remaining elements if klen is not a multiple of 4
        for (; k < klen; ++k) {
            dest_row[k] = src_row[k];
        }
    }
}

Matrix Matrix::operator*(const Matrix& other) const
{
    // Square only in the tests ?
    const int n  = rows;
    const int n2 = other.cols;

    Matrix result(n, n2);

    alignas(64) double packedA[MC * KC];
    alignas(64) double packedB[KC * NC];
    double* C = result.data.data();

    // best cache blocking for L1/L2 : fast access but lower cache size
    for (int kk = 0; kk < cols; kk += KC) {
        int klen  = std::min(KC, cols - kk);
        // best cache blocking for L2 : good access and cache size
        for (int jj = 0; jj < n2; jj += NC) {
            int jlen = std::min(NC, n2 - jj);

            pack_B(klen, jlen, n2, other.data.data() + kk * n2 + jj, packedB);

            // best cache blocking for L3 : lower access but larger cache size
            for (int ii = 0; ii < n; ii += MC) {
                int ilen = std::min(MC, n - ii);

                pack_A(klen, ilen, n, data.data() + ii * n + kk, packedA);

                for (int i = ii; i < ilen; i += 8) {
                    for (int j = jj; j < jlen; j += 8) {

                        // Computes (in general) for blocks A : 8xklen, B : klenx8, C : 8x8
                        if (std::min(8, ilen - i) == 8 && std::min(8, jlen - j) == 8) {
                            micro_8x8(packedA + i*klen, packedB + j, C + (ii + i)*n2 + (jj + j), klen, jlen, n2);
                        } else {
                            for (int i2 = i;  i2 < i  + std::min(8, ilen - i);   i2++)
                            for (int k  = kk; k  < kk + klen;     k++)
                            for (int j2 = j;  j2 < j  + std::min(8, jlen - j);   j2++)
                                C[i2*n2 + j2] += packedA[i2*klen + k] * packedB[k*n2 + j2];
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