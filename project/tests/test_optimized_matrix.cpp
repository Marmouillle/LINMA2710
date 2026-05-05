#include <cassert>
#include <cmath>
#include <iostream>
#include <chrono>

#include "matrix.hpp"

bool approxEqual(double a, double b, double epsilon = 1e-6)
{
    return std::fabs(a - b) < epsilon;
}

bool matricesEqual(const Matrix &a, const Matrix &b, double epsilon = 1e-6)
{
    if (a.numRows() != b.numRows() || a.numCols() != b.numCols())
        return false;
    for (int i = 0; i < a.numRows(); ++i)
        for (int j = 0; j < a.numCols(); ++j)
            if (!approxEqual(a.get(i, j), b.get(i, j), epsilon))
                return false;
    return true;
}

void testBigMultiplication()
{
    int size = 1000;
    Matrix A(size, size);
    Matrix B(size, size);

    // Fill with deterministic pseudo-random values
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
            A.set(i, j, std::sin(i * 0.1 + j * 0.3));
            B.set(i, j, std::cos(i * 0.2 + j * 0.1));
        }

    Matrix C = A * B;

    // Verify a sample of elements against a naive scalar computation
    // (checking every element would be O(n^3) — too slow for a unit test)
    auto naive_element = [&](int row, int col) {
        double sum = 0.0;
        for (int k = 0; k < size; ++k)
            sum += A.get(row, k) * B.get(k, col);
        return sum;
    };

    // Check diagonal
    for (int i = 0; i < size; ++i)
        assert(approxEqual(C.get(i, i), naive_element(i, i)));

    // Check a strided sample across the full matrix
    for (int i = 0; i < size; i += 97)
        for (int j = 0; j < size; j += 97)
            assert(approxEqual(C.get(i, j), naive_element(i, j)));

    // Check known hard positions (corners and center)
    for (auto [r, c] : std::vector<std::pair<int,int>>{
            {0, 0}, {0, size-1}, {size-1, 0}, {size-1, size-1}, {size/2, size/2}})
        assert(approxEqual(C.get(r, c), naive_element(r, c)));

    std::cout << "testBigMultiplication passed." << std::endl;
}
int main()
{
    
    auto start_time
        = std::chrono::high_resolution_clock::now();
    testBigMultiplication();    
    auto end_time
        = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration
        = end_time - start_time;
    std::cout << "Big multiplication test duration: "
              << duration.count() << " seconds" << std::endl;
    std::cout << "All matrix tests passed." << std::endl;
    return 0;
}
