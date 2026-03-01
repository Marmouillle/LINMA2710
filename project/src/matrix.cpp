#include "matrix.hpp"
#include <stdexcept>
#include <iostream>
#include <omp.h>

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
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix dimensions must match for addition");

    Matrix result(rows, cols);
    #pragma omp parallel for
    for (int k = 0; k < cols*rows; ++k) {
        result.data[k] = data[k] + other.data[k];
    }

    return result;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix dimensions must match for substraction");

    Matrix result(rows, cols);
    #pragma omp parallel for
    for (int k = 0; k < cols*rows; ++k) {
        result.data[k] = data[k] - other.data[k];
    }

    return result;
}

Matrix Matrix::operator*(const Matrix &other) const
{
    if (cols != other.rows)
        throw std::invalid_argument("Matrix dimensions must match for multiplication");

    int newcols = other.cols;
    Matrix result(rows, newcols);
    int block = 32;
    #pragma omp parallel
    {
        #pragma omp single
        //std::cout << "Threads used: " << omp_get_num_threads() << std::endl;
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < rows; ii += block){
        int iimax = std::min(ii + block, rows);
        for (int jj = 0; jj < other.cols; jj += block)
        {
            int jjmax = std::min(jj + block, other.cols);
            for (int kk = 0; kk < cols; kk += block)
            {   
                int kkmax = std::min(kk + block, cols);

                for (int i = ii; i < iimax; ++i)
                {
                    double *result_row = &result.data[i*other.cols];
                    const double *Arow = &data[i*cols];

                    for (int k = kk; k < kkmax; ++k)
                    {
                        double a = Arow[k];
                        const double *Brow = &other.data[k*other.cols];

                        for (int j = jj; j < jjmax; ++j)
                            result_row[j] += a * Brow[j];
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
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix dimensions must match for operation");
    
    for(int i = 0; i<cols*rows; ++i){
        data[i] = data[i] - scalar*other.data[i];
    }
}
