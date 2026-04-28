#include <cassert>
#include <cmath>
#include <iostream>
#include <chrono>

#include "new_matrix.hpp"

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
    int size = 2000; 
    Matrix A(size, size);
    Matrix B(size, size);
    A.fill(1.0);
    B.fill(2.0);

    Matrix C = A * B;
    for (int i = 0; i < C.numRows(); ++i)
        for (int j = 0; j < C.numCols(); ++j)
            assert(approxEqual(C.get(i, j), 2.0 * size));
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
