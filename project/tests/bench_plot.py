import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test_bench_plot():
    df = pd.read_csv("openMP_bench.csv")
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

def speed_up():
    df = pd.read_csv("openMP_bench.csv")
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

if __name__ == "__main__":
    test_bench_plot()
    speed_up()