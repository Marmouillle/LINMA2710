#!/bin/bash
#
#SBATCH --job-name=power_consumption
#SBATCH --output=results/power_consumption_result.txt
#SBATCH --partition=batch
# --- resources ------------------------------------------------------
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=0-00:20:00             # walltime D‑HH:MM:SS
# --- environment ----------------------------------------------------
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.8.0
# --- run ------------------------------------------------------------
python consumption.py