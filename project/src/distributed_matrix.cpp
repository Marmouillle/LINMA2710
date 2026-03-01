#include "distributed_matrix.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
//OpenMPI/4.1.5-GCC-12.3.0

// The matrix is split by columns across MPI processes.
// Each process stores a local Matrix with a subset of columns.
// Columns are distributed as evenly as possible.

DistributedMatrix::DistributedMatrix(const Matrix& matrix, int numProcs)
    : globalRows(matrix.numRows()),
      globalCols(matrix.numCols()),
      localCols(0),
      startCol(0),
      numProcesses(numProcs),
      rank(0),
      localData(matrix.numRows(), 1)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    localCols = globalCols / numProcesses;
    int remainder = globalCols % numProcesses;
    // Distribute remaining columns in first processes
    if (rank < remainder)
    {        localCols++;
        startCol = rank * localCols;
    }
    else
    {
        startCol = rank * localCols + remainder;
    }
    localData = Matrix(globalRows, localCols);
    for (int i = 0; i < globalRows; i++)
    {
        for (int j = 0; j < localCols; j++)
        {            localData.set(i, j, matrix.get(i, startCol + j));
        }
    }

}

DistributedMatrix::DistributedMatrix(const DistributedMatrix& other)
    : globalRows(other.globalRows),
      globalCols(other.globalCols),
      localCols(other.localCols),
      startCol(other.startCol),
      numProcesses(other.numProcesses),
      rank(other.rank),
      localData(other.localData)
{
}

int DistributedMatrix::numRows() const { return globalRows; }
int DistributedMatrix::numCols() const { return globalCols; }
const Matrix& DistributedMatrix::getLocalData() const { return localData; }

double DistributedMatrix::get(int i, int j) const
{
    int p = ownerProcess(j);
    if (p != rank)
    {
        throw std::out_of_range("Column index not owned by this process");
    }
    return localData.get(i, localColIndex(j));
}

void DistributedMatrix::set(int i, int j, double value)
{
    int p = ownerProcess(j);
    if (p != rank)
    {
        throw std::out_of_range("Column index not owned by this process");
    }
    localData.set(i, localColIndex(j), value);
}

int DistributedMatrix::globalColIndex(int localColIdx) const
{
    if (localColIdx < 0 || localColIdx >= localCols)
    {
        throw std::out_of_range("Local column index out of range for this process");
    }
    return startCol+localColIdx;
}

int DistributedMatrix::localColIndex(int globalColIdx) const
{
    if (globalColIdx < startCol || globalColIdx >= startCol + localCols)
    {
        throw std::out_of_range("Global column index not owned by this process");
    }
    return globalColIdx-startCol;
}

int DistributedMatrix::ownerProcess(int globalColIdx) const
{
    int colPerProcess = globalCols / numProcesses;
    int remainder = globalCols % numProcesses;
    if (globalColIdx < 0 || globalColIdx >= globalCols)
    {
        throw std::out_of_range("Global column index out of range");
    }
    if (globalColIdx < (colPerProcess + 1) * remainder)
    {        return globalColIdx / (colPerProcess + 1);
    }
    else
    {        return (globalColIdx - remainder) / colPerProcess;
    }
}

void DistributedMatrix::fill(double value)
{
    localData.fill(value);
}

DistributedMatrix DistributedMatrix::operator+(const DistributedMatrix& other) const
{
    // TODO
    if (other.globalRows != globalRows || other.globalCols != globalCols)
    {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    else if (other.startCol != startCol){
        return DistributedMatrix(*this);
    }
    else{
        DistributedMatrix result = DistributedMatrix(*this);
        result.localData = localData + other.localData;
        return result;
    }
    
}

DistributedMatrix DistributedMatrix::operator-(const DistributedMatrix& other) const
{
    if (other.globalRows != globalRows || other.globalCols != globalCols)
    {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    else if (other.startCol != startCol){
        return DistributedMatrix(*this);
    }
    else{
        DistributedMatrix result = DistributedMatrix(*this);
        result.localData = localData - other.localData;
        return result;
    }
}

DistributedMatrix DistributedMatrix::operator*(double scalar) const
{
    DistributedMatrix result = DistributedMatrix(*this);
    result.localData = localData*scalar;
    return result;
}

Matrix DistributedMatrix::transpose() const
{   
    Matrix global = gather();
    return global.transpose();
}

void DistributedMatrix::sub_mul(double scalar, const DistributedMatrix& other)
{
    if (other.globalRows != globalRows || other.globalCols != globalCols)
    {
        throw std::invalid_argument("Matrix dimensions must match for operation");
    }
    else if (other.startCol != startCol){
        return;
    }
    else{
        localData.sub_mul(scalar, other.localData);
    }
}

DistributedMatrix DistributedMatrix::apply(const std::function<double(double)>& func) const
{
    DistributedMatrix result = DistributedMatrix(*this);
    result.localData = localData.apply(func);
    return result;
}

DistributedMatrix DistributedMatrix::applyBinary(
    const DistributedMatrix& a,
    const DistributedMatrix& b,
    const std::function<double(double, double)>& func)
{
    if (a.globalRows != b.globalRows || a.globalCols != b.globalCols)
    {
        throw std::invalid_argument("Matrix dimensions must match for operation");
    }
    else if (a.startCol != b.startCol){
        return DistributedMatrix(a);
    }
    DistributedMatrix result = DistributedMatrix(a);
    for (int i = 0; i < a.localData.numRows(); i++){
        for (int j = 0; j < a.localData.numCols(); j++){
            double valA = a.localData.get(i, j);
            double valB = b.localData.get(i, j);
            result.localData.set(i, j, func(valA, valB));
        }
    }
    return result;
}

DistributedMatrix multiply(const Matrix& left, const DistributedMatrix& right){
    DistributedMatrix result = DistributedMatrix(right);
    result.globalRows = left.numRows();
    result.localData = Matrix(result.globalRows, result.localCols);
    // Compute resulting matrice on cols corresponding to local cols of right
    for (int i = 0; i < left.numRows(); i++){
        for (int j = 0; j < right.localCols; j++){
            double sum = 0.0;
            for (int k = 0; k < right.globalRows; k++){
                sum += left.get(i, k) * right.localData.get(k, j);
            }
            result.localData.set(i, j, sum);
        }
    }
    return result;
}

Matrix DistributedMatrix::multiplyTransposed(const DistributedMatrix& other) const
{
    // TODO
    std::vector<double> sendBuffer(other.globalRows * globalRows);
    if (globalCols != other.globalCols){
        throw std::invalid_argument("Matrix dimensions must match for operation");
    }

    
    if (startCol == other.startCol){
        Matrix result = localData * other.localData.transpose();
        for (int i = 0; i < globalRows; i++){
            for (int j = 0; j < other.globalRows; j++){
                sendBuffer[i * other.globalRows + j] = result.get(i, j);
            }
        }
    }
    else{
        std::fill(sendBuffer.begin(), sendBuffer.end(), 0.0);
        std::cout << "Process " << rank << " has no local columns, sending zeros." << std::endl;
    }
    //Assemble data
    std::vector<double> recvBuffer(globalRows * other.globalRows);
    MPI_Allreduce(sendBuffer.data(), recvBuffer.data(), globalRows * other.globalRows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Matrix result(globalRows, other.globalRows);
    for (int i = 0; i < globalRows; i++){
        for (int j = 0; j < other.globalRows; j++){
            result.set(i, j, recvBuffer[i * other.globalRows + j]);
        }
    }
    return result;
}

double DistributedMatrix::sum() const
{
    double localSum = 0.0;
    for (int i = 0; i < localData.numRows(); i++){
        for (int j = 0; j < localData.numCols(); j++){
            localSum += localData.get(i, j);
        }
    }
    double globalSum = 0.0;
    MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return globalSum;
}

Matrix DistributedMatrix::gather() const
{
    Matrix result_gathered(globalRows, globalCols);
    // 1) assemble local data
    std::vector<double> sendBuffer(localCols * globalRows);
    for (int i = 0; i < globalRows; i++) {
        for (int j = 0; j < localCols; j++) {
            sendBuffer[i * localCols + j] = localData.get(i, j);
        }
    }

    // 2) Initialize receive buffers and counts for gathering
    std::vector<double> recvBuffer(globalRows * globalCols);
    std::vector<int> recvCounts(numProcesses);
    std::vector<int> displs(numProcesses);
    int remainder = globalCols % numProcesses;
    int colPerProcess = globalCols / numProcesses;
    int startCol = 0;
    for (int p = 0; p < numProcesses; p++) {
        if (p < remainder) {
            recvCounts[p] = globalRows * (colPerProcess + 1);
            displs[p] = startCol * globalRows;
            startCol += colPerProcess + 1;
        }
        else {
            recvCounts[p] = globalRows * colPerProcess;
            displs[p] = startCol * globalRows;
            startCol += colPerProcess;
        }
    }
    // 3) gather data from all processes (every process gets the full matrix)
    MPI_Allgatherv(sendBuffer.data(), localCols * globalRows, MPI_DOUBLE, recvBuffer.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // 4) Assemble received data
    int colsFromP, startColP;
    int offset;
    for (int p = 0; p < numProcesses; p++) {
        colsFromP = recvCounts[p] / globalRows;
        startColP = displs[p] / globalRows;
        offset = displs[p];
        for (int i = 0; i < globalRows; i++) {
            for (int j = 0; j < colsFromP; j++) {
                result_gathered.set(i, startColP + j, recvBuffer[offset + i * colsFromP + j]);
            }
        }

    }
    return result_gathered;
}

void sync_matrix(Matrix *matrix, int rank, int src)
{
    std::vector<double> buffer(matrix->numRows() * matrix->numCols());
    // broadcast from 'src' to all processes
    if (rank == src) {
        for (int i = 0; i < matrix->numRows(); i++) {
            for (int j = 0; j < matrix->numCols(); j++) {
                buffer[i * matrix->numCols() + j] = matrix->get(i, j);
            }
        }
    }
    MPI_Bcast(buffer.data(), matrix->numRows() * matrix->numCols(), MPI_DOUBLE, src, MPI_COMM_WORLD);
    // broadcast to 'src' from all processes
    if (rank != src) {
        for (int i = 0; i < matrix->numRows(); i++) {
            for (int j = 0; j < matrix->numCols(); j++) {
                matrix->set(i, j, buffer[i * matrix->numCols() + j]);
            }
        }
    }
}
