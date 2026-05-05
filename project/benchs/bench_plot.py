import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_tuning_results(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)

    # 1. Find the optimum (minimum duration)
    optimal_row = df.loc[df['Duration'].idxmin()]
    print("--- Optimal Parameter Set ---")
    print(f"MC: {optimal_row['MC']}")
    print(f"KC: {optimal_row['KC']}")
    print(f"NC: {optimal_row['NC']}")
    print(f"Min Duration: {optimal_row['Duration']:.4f}s")

    # 2. Slice visualization: Heatmap of MC vs KC at the optimal NC
    best_nc = optimal_row['NC']
    slice_df = df[df['NC'] == best_nc]
    
    # Pivot for heatmap
    pivot_table = slice_df.pivot(index='MC', columns='KC', values='Duration')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu_r")
    plt.title(f"Matrix Multiplication Performance (Duration in s)\n$MC$ vs $KC$ (fixed $NC={int(best_nc)}$)")
    plt.xlabel("$KC$")
    plt.ylabel("$MC$")
    plt.savefig('optimization_heatmap.png')
    
    # 3. Trends across all NC values (Boxplot)
    plt.clf()
    sns.boxplot(x='NC', y='Duration', data=df)
    plt.title("Duration Distribution by $NC$")
    plt.xlabel("$NC$ Value")
    plt.ylabel("Duration (s)")
    plt.savefig('nc_distribution.png')

# ---------------------------------------------------------- #
# ----------- FINAL PLOTS ---------------------------------- #
# ---------------------------------------------------------- #

# OMP BENCHS
def omp_bench_plot():
    df = pd.read_csv("omp_bench.csv")
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

def omp_efficiency():
    df = pd.read_csv("omp_bench.csv")
    plt.figure(figsize=(10, 6))
    no_threads = df[df['NumThreads'] == 1]
    for threads in df['NumThreads'].unique():
        subset = df[df['NumThreads'] == threads]
        speedup = no_threads['Duration'].values / subset['Duration'].values
        plt.plot(subset['Size'], speedup/threads, marker='o', label=f'Threads: {threads}')
    plt.legend()
    plt.title('Efficiency vs Size')
    plt.xlabel('Size')
    plt.ylabel('Efficiency (Speedup / NumThreads)')
    plt.grid(True, which="both", ls="--")
    plt.savefig('omp_efficiency_plot.png')
    plt.close()

# MPI BENCHS
def mpi_overhead():
    df = pd.read_csv("mpi_bench.csv")
    plt.figure(figsize=(10, 6))
    for processes in df['NumProcesses'].unique():
        subset = df[df['NumProcesses'] == processes]
        comm_time = subset['Comm_time']
        total_time = subset['Duration']
        overhead = comm_time / total_time
        plt.plot(subset['Size'], overhead, marker='o', label=f'Processes: {processes}')
    plt.legend()
    plt.title('MPI Communication Overhead vs Size')
    plt.xlabel('Size')
    plt.ylabel('Communication Overhead (Comm_time / Duration)')
    plt.grid(True, which="both", ls="--")
    plt.savefig('mpi_overhead_plot.png')
    plt.close()

def mpi_vs_expected():
    # load omp benchs with 1 thread as expected time
    df_omp = pd.read_csv("omp_bench.csv")
    df_mpi = pd.read_csv("mpi_bench.csv")
    plt.figure(figsize=(10, 6))
    omp_1_thread = df_omp[df_omp['NumThreads'] == 1]
    for processes in df_mpi['NumProcesses'].unique():
        subset = df_mpi[df_mpi['NumProcesses'] == processes]
        expected_time = omp_1_thread['Duration'].values / processes
        plt.loglog(subset['Size'], subset['Duration'], marker='o', label=f'MPI Processes: {processes}')
        plt.loglog(subset['Size'], expected_time, linestyle='--', alpha=0.7, label=f'Expected with {processes} processes')
    plt.legend()
    plt.title('MPI Benchmark vs Expected Time from OpenMP 1 Thread')
    plt.xlabel('Size')
    plt.ylabel('Duration (seconds)')
    plt.grid(True, which="both", ls="--")
    plt.savefig('mpi_vs_expected_plot.png')
    plt.close()

# OPENCL BENCHS
def opencl_overhead():
    df = pd.read_csv("openCL_bench.csv")
    plt.figure(figsize=(10, 6))
    comm_time = df['Comm_time']
    total_time = df['Duration']
    overhead = comm_time / total_time
    plt.plot(df['Size'], overhead, marker='o', label=f'OpenCL')
    plt.legend()
    plt.title('OpenCL Communication Overhead vs Size')
    plt.xlabel('Size')
    plt.ylabel('Communication Overhead (Comm_time / Duration)')
    plt.grid(True, which="both", ls="--")
    plt.savefig('openCL_overhead_plot.png')
    plt.close()

def code_carbon_plot():
    # each df contains with one entry: timestamp,project_name,run_id,experiment_id,duration,emissions,emissions_rate,cpu_power,gpu_power,ram_power,cpu_energy,gpu_energy,ram_energy,energy_consumed,water_consumed,country_name,country_iso_code,region,cloud_provider,cloud_region,os,python_version,codecarbon_version,cpu_count,cpu_model,gpu_count,gpu_model,longitude,latitude,ram_total_size,tracking_mode,cpu_utilization_percent,gpu_utilization_percent,ram_utilization_percent,ram_used_gb,on_cloud,pue,wue
    df_basic = pd.read_csv("csv/opencl_basic_power.csv")
    df_complex = pd.read_csv("csv/opencl_complex_power.csv")
    plt.figure(figsize=(10, 6))
    plt.bar(['OpenCL Basic Kernel', 'OpenCL Complex Kernel'], [df_basic['emissions'].values[0], df_complex['emissions'].values[0]], color=['blue', 'orange'])
    plt.title('CO2 Emissions for OpenCL Benchmarks')
    plt.ylabel('Emissions (kg CO2)')
    plt.savefig('code_carbon_plot.png')
    plt.close()

# ALL BENCHS
def mpi_vs_openmp_vs_opencl_vs_simd():
    df_omp = pd.read_csv("omp_bench.csv")
    df_mpi = pd.read_csv("mpi_bench.csv")
    df_opencl = pd.read_csv("opencl_bench.csv")
    df_simd = pd.read_csv("simd_bench.csv")
    plt.figure(figsize=(10, 6))
    plt.loglog(df_opencl['Size'], df_opencl['Duration'], marker='d', linestyle='--', label=f'OpenCL')
    plt.loglog(df_simd['Size'], df_simd['Duration'], marker='s', linestyle='-.', label=f'SIMD')
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


# Modules to load
# module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
# module load Python/3.11.3-GCCcore-12.3.0

if __name__ == "__main__":
    visualize_tuning_results("csv/simd_param_bench.csv")