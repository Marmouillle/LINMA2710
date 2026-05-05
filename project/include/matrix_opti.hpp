#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <functional>
#include <immintrin.h>

class Matrix
{
private:
    int rows, cols, mc, kc, nc;
    std::vector<double> data;

public:
    // --- Constructors & Assignment ---
    Matrix(int rows, int cols, int mc = 256, int kc = 256, int nc = 256);
    Matrix(const Matrix &other);
    Matrix &operator=(const Matrix &other)
    {
        if (this != &other)
        {
            rows = other.rows;
            cols = other.cols;
            mc = other.mc;
            kc = other.kc;
            nc = other.nc;
            data = other.data;
        }
        return *this;
    }

    // --- Common API (shared with DistributedMatrix and MatrixCL) ---

    int numRows() const;
    int numCols() const;

    void fill(double value);

    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(const Matrix &other) const; // Matrix multiplication
    Matrix operator*(double scalar) const;       // Scalar multiplication

    Matrix transpose() const;

    // this = this - scalar * other
    void sub_mul(double scalar, const Matrix &other);

    // --- Matrix-specific operations ---

    double get(int i, int j) const;
    void set(int i, int j, double value);

    // Apply a function element-wise
    Matrix apply(const std::function<double(double)> &func) const;
};

#endif // MATRIX_H
