import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test_bench_plot():
    df = pd.read_csv("./project/openMP_bench.csv")
    plt.figure(figsize=(10, 6))
    for threads in df['NumThreads'].unique():
        subset = df[df['NumThreads'] == threads]
        fitted_curve = np.polyfit(np.log(subset['Size']), np.log(subset['Duration']), 1)
        plt.loglog(subset['Size'], subset['Duration'], marker='o', label=f'Threads: {threads}')
        plt.loglog(subset['Size'], np.exp(fitted_curve[1]) * subset['Size'] ** fitted_curve[0], linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('OpenMP Benchmark')
    plt.xlabel('Size')
    plt.ylabel('Duration (seconds)')
    plt.grid(True, which="both", ls="--")
    plt.savefig('openMP_bench_plot.png')
    plt.close()

def speed_up():
    df = pd.read_csv("./project/openMP_bench.csv")
    plt.figure(figsize=(10, 6))
    no_threads = df[df['NumThreads'] == 1]
    for threads in df['NumThreads'].unique():
        subset = df[df['NumThreads'] == threads]
        speedup = no_threads['Duration'].values / subset['Duration'].values
        plt.plot(subset['Size'], speedup/threads, marker='o', label=f'Threads: {threads}')
    plt.legend()
    plt.title('Speedup vs Size')
    plt.xlabel('Size')
    plt.ylabel('Speedup')
    plt.grid(True, which="both", ls="--")
    plt.savefig('openMP_speedup_plot.png')
    plt.close()

def mpi_bench_plot():
    df = pd.read_csv("./project/mpi_bench.csv")
    plt.figure(figsize=(10, 6))
    for processes in df['NumProcesses'].unique():
        subset = df[df['NumProcesses'] == processes]
        fitted_curve = np.polyfit(np.log(subset['Size']), np.log(subset['Duration']), 1)
        plt.loglog(subset['Size'], subset['Duration'], marker='o', label=f'Processes: {processes}')
        plt.loglog(subset['Size'], np.exp(fitted_curve[1]) * subset['Size'] ** fitted_curve[0], linestyle='--', alpha=0.7)
    plt.title('MPI Benchmark')
    plt.xlabel('Size')
    plt.ylabel('Duration (seconds)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('mpi_bench_plot.png')
    plt.close()

def mpi_vs_openmp():
    df_omp = pd.read_csv("./project/openMP_bench.csv")
    df_mpi = pd.read_csv("./project/mpi_bench.csv")
    plt.figure(figsize=(10, 6))
    for threads in df_omp['NumThreads'].unique():
        subset_omp = df_omp[df_omp['NumThreads'] == threads]
        subset_mpi = df_mpi[df_mpi['NumProcesses'] == threads]
        plt.loglog(subset_omp['Size'], subset_omp['Duration'], marker='o', label=f'OpenMP Threads: {threads}', alpha=0.6)
        plt.loglog(subset_mpi['Size'], subset_mpi['Duration'], marker='x', linestyle=':', label=f'MPI Processes: {threads}', color=plt.gca().lines[-1].get_color())
    plt.title('OpenMP vs MPI Benchmark')
    plt.xlabel('Size')
    plt.ylabel('Duration (seconds)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('openMP_vs_MPI_plot.png')
    plt.close()

if __name__ == "__main__":
    test_bench_plot()
    speed_up()
    mpi_bench_plot()
    mpi_vs_openmp()