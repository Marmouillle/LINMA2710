import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test_bench_plot():
    df = pd.read_csv("./project/omp_bench.csv")
    plt.figure(figsize=(10, 6))
    for threads in df['NumThreads'].unique():
        subset = df[df['NumThreads'] == threads]
        fitted_curve = np.polyfit(np.log(subset['Size']), np.log(subset['Duration']), 1)
        plt.loglog(subset['Size'], subset['Duration'], marker='o', label=f'Threads: {threads}')
        plt.loglog(subset['Size'], np.exp(fitted_curve[1]) * subset['Size'] ** fitted_curve[0], linestyle='--', alpha=0.7, label=f'fit {threads} with slope {fitted_curve[0]:.2f}')
    plt.legend()
    plt.title('OpenMP Benchmark')
    plt.xlabel('Size')
    plt.ylabel('Duration (seconds)')
    plt.grid(True, which="both", ls="--")
    plt.savefig('openMP_bench_plot.png')
    plt.close()

def speed_up():
    df = pd.read_csv("./project/omp_bench.csv")
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
        plt.loglog(subset['Size'], np.exp(fitted_curve[1]) * subset['Size'] ** fitted_curve[0], linestyle='--', alpha=0.7, label=f'fit {processes} with slope {fitted_curve[0]:.2f}')
    plt.title('MPI Benchmark')
    plt.xlabel('Size')
    plt.ylabel('Duration (seconds)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('mpi_bench_plot.png')
    plt.close()

def opencl_bench_plot():
    df = pd.read_csv("openCL_bench.csv")
    plt.figure(figsize=(10, 6))
    fitted_curve = np.polyfit(np.log(df['Size']), np.log(df['Duration']), 1)
    plt.loglog(df['Size'], df['Duration'], marker='o', label=f'OpenCL')
    plt.loglog(df['Size'], np.exp(fitted_curve[1]) * df['Size'] ** fitted_curve[0], linestyle='--', alpha=0.7, label=f'fit with slope {fitted_curve[0]:.2f}')
    plt.title('OpenCL Benchmark')
    plt.xlabel('Size')
    plt.ylabel('Duration (seconds)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('openCL_bench_plot.png')
    plt.close()

def mpi_vs_openmp_vs_opencl():
    df_omp = pd.read_csv("omp_bench.csv")
    df_mpi = pd.read_csv("mpi_bench.csv")
    df = pd.read_csv("openCL_bench.csv")
    plt.figure(figsize=(10, 6))
    plt.loglog(df['Size'], df['Duration'], marker='d', linestyle='--', label=f'OpenCL')
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

def opencl_param_bench_plot(tile_size=64):
    df = pd.read_csv("opencl_param_bench.csv")
    plt.figure(figsize=(10, 6))
    for sub_tile_size in df[df['TILE_SIZE'] == tile_size]['SUB_TILE_SIZE'].unique():
        subset = df[(df['TILE_SIZE'] == tile_size) & (df['SUB_TILE_SIZE'] == sub_tile_size)]
        plt.plot(subset['Size'], subset['Duration'], marker='o', label=f'SUB_TILE_SIZE: {sub_tile_size}')
    plt.title(f'OpenCL Parameter Benchmark for TILE_SIZE = {tile_size}')
    plt.xlabel('SUB_TILE_SIZE')
    plt.ylabel('Duration (seconds)')
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('openCL_param_bench_plot.png')
    plt.close()

def opencl_optimal_param_plot(size=16384):
    # Plot a grid of TILE_SIZE vs SUB_TILE_SIZE with color representing the duration for size 16384
    # Red box for the optimal parameters
    df = pd.read_csv("opencl_param_bench.csv")
    df_16384 = df[df['Size'] == size]
    pivot_table = df_16384.pivot(index='TILE_SIZE', columns='SUB_TILE_SIZE', values='Duration')
    optimal_params = df_16384.loc[df_16384['Duration'].idxmin()]
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_table, cmap='viridis', aspect='auto')
    plt.colorbar(label='Duration (seconds)')
    plt.title(f'OpenCL Optimal Parameters for Size {size}')
    plt.xlabel('SUB_TILE_SIZE')
    plt.ylabel('TILE_SIZE')
    plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=pivot_table.columns)
    plt.yticks(ticks=np.arange(len(pivot_table.index)), labels=pivot_table.index)

    optimal_x = list(pivot_table.columns).index(optimal_params['SUB_TILE_SIZE'])
    optimal_y = list(pivot_table.index).index(optimal_params['TILE_SIZE'])
    plt.gca().add_patch(
        plt.Rectangle(
            (optimal_x - 0.5, optimal_y - 0.5),
            1, 1,
            fill=False, edgecolor='red', linewidth=2,
        )
    )

    # mark the boundary TILE = 32 * SUB_TILE
    # for each SUB_TILE_SIZE column, find where TILE crosses 32*SUB_TILE
    sub_tile_values = list(pivot_table.columns)
    tile_values     = list(pivot_table.index)
    boundary_x = []
    boundary_y = []
    for xi, sub in enumerate(sub_tile_values):
        threshold = 32 * sub   # TILE <= 32*SUB_TILE is valid below this
        for yi, tile in enumerate(tile_values):
            if tile > threshold:
                # boundary is between yi-1 and yi
                boundary_x.append(xi)
                boundary_y.append(yi - 0.5)
                break
    if boundary_x:
        plt.plot(boundary_x, boundary_y, color='white', linewidth=2,
                 linestyle='--', label='TILE = 32 × SUB_TILE')
        plt.legend(loc='upper left')

    plt.grid(False)
    plt.savefig(f'openCL_optimal_param_plot_{size}.png')
    plt.close()

# Modules to load
# module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
# module load Python/3.11.3-GCCcore-12.3.0
if __name__ == "__main__":
    #test_bench_plot()
    #speed_up()
    #opencl_bench_plot()
    #mpi_bench_plot()
    #mpi_vs_openmp_vs_opencl()
    #opencl_param_bench_plot()
    opencl_optimal_param_plot(16384)