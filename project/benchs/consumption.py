import subprocess
from codecarbon import OfflineEmissionsTracker

def run_benchmark(binary_path, project_name, output_file):
    print(f"--- Starting: {project_name} ---")
    
    # Initialize the tracker
    tracker = OfflineEmissionsTracker(
        country_iso_code="BEL",
        project_name=project_name,
        output_file=output_file
    )

    tracker.start()
    try:
        # Run your OpenCL binary
        subprocess.run([binary_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {binary_path}: {e}")
    finally:
        # Guaranteed to stop and save even if the binary crashes
        tracker.stop()
    print(f"--- Finished: {project_name} ---\n")

if __name__ == "__main__":
    # Run your two benchmarks
    run_benchmark("./execs/opencl_basic_bench", "OpenCL basic Kernel", "opencl_basic.csv")
    run_benchmark("./execs/opencl_bench", "OpenCL complex Kernel", "opencl_complex.csv")