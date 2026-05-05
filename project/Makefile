CXX = g++
MPICXX = mpic++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
CVFLAGS = -O3 -march=native -ffast-math -funroll-loops
INCLUDE = -Iinclude

# OpenMP flag: set to empty to disable (e.g., on macOS without OpenMP)
OPENMP_FLAGS ?= -fopenmp

SRC_DIR ?= src

test_matrix: tests/test_matrix.cpp $(SRC_DIR)/matrix.cpp include/matrix.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o test_matrix tests/test_matrix.cpp $(SRC_DIR)/matrix.cpp
# --- Part 1 & 2: Basic + OpenMP Matrix ---
# test_matrix: tests/test_matrix.cpp $(SRC_DIR)/matrix.cpp include/matrix.hpp
# 	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(INCLUDE) -o test_matrix tests/test_matrix.cpp $(SRC_DIR)/matrix.cpp

run_matrix: test_matrix
	./test_matrix
#
test_optimized: tests/test_optimized_matrix.cpp $(SRC_DIR)/matrix.cpp include/matrix.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o test_optimized tests/test_optimized_matrix.cpp $(SRC_DIR)/matrix.cpp

run_optimized: test_optimized
	./test_optimized

# --- Part 3: Distributed Matrix (MPI) ---
test_distributed: tests/test_distributed.cpp $(SRC_DIR)/distributed_matrix.cpp $(SRC_DIR)/matrix.cpp include/distributed_matrix.hpp include/matrix.hpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) -o test_distributed tests/test_distributed.cpp $(SRC_DIR)/distributed_matrix.cpp $(SRC_DIR)/matrix.cpp

run_distributed: test_distributed
	mpirun -np 4 ./test_distributed

# --- Part 4: OpenCL Matrix ---
test_opencl: tests/test_opencl.cpp $(SRC_DIR)/matrix_opencl.cpp include/matrix_opencl_blocked.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o test_opencl tests/test_opencl.cpp $(SRC_DIR)/matrix_opencl.cpp -lOpenCL

run_opencl: test_opencl
	./test_opencl

test_opencl_new : tests/test_opencl_new.cpp $(SRC_DIR)/matrix_opencl.cpp include/matrix_opencl.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o test_opencl_new tests/test_opencl_new.cpp $(SRC_DIR)/matrix_opencl.cpp -lOpenCL


# --- Part 5: Additional Benchmarks ---
basic_bench: benchs/basic_bench.cpp $(SRC_DIR)/matrix.cpp include/matrix.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o basic_bench benchs/basic_bench.cpp $(SRC_DIR)/matrix.cpp

simd_bench: benchs/simd_bench.cpp $(SRC_DIR)/matrix_opti.cpp include/matrix.hpp
	$(CXX) $(CVFLAGS) $(INCLUDE) -o simd_bench benchs/simd_bench.cpp $(SRC_DIR)/matrix_opti.cpp

mpi_bench: benchs/mpi_bench.cpp $(SRC_DIR)/distributed_matrix.cpp $(SRC_DIR)/matrix.cpp include/distributed_matrix.hpp include/matrix.hpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) -o mpi_bench benchs/mpi_bench.cpp $(SRC_DIR)/distributed_matrix.cpp $(SRC_DIR)/matrix.cpp

omp_bench: benchs/omp_bench.cpp $(SRC_DIR)/matrix.cpp include/matrix.hpp
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(INCLUDE) -o omp_bench benchs/omp_bench.cpp $(SRC_DIR)/matrix.cpp

opencl_bench: benchs/opencl_bench.cpp $(SRC_DIR)/matrix_opencl.cpp include/matrix_opencl.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o opencl_bench benchs/opencl_bench.cpp $(SRC_DIR)/matrix_opencl.cpp -lOpenCL

opencl_basic_bench: benchs/opencl_basic_bench.cpp $(SRC_DIR)/matrix_opencl_basic.cpp include/matrix_opencl_basic.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o opencl_basic_bench benchs/opencl_basic_bench.cpp $(SRC_DIR)/matrix_opencl_basic.cpp -lOpenCL

run_opencl_bench: opencl_bench
	./opencl_bench


# --- Utilities ---
all: test_matrix test_distributed test_opencl

clean:
	rm -f test_matrix test_distributed test_opencl mpi_bench omp_bench test_optimized opencl_bench simd_bench basic_bench test_opencl_new

.PHONY: all clean run_matrix run_distributed run_opencl
