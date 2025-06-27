#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=gpu-job_%j.out

set -x # Print commands and their arguments as they are executed.
set -e # Exit immediately if a command exits with a non-zero status.

# Load modules
module purge            
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

# Change directory
cd "$HOME/master-thesis/"

# Activate .venv_ssm
source .venv_ssm/bin/activate

# Install repository as executable
pip install -e .

# Train
python -m spec_mamba.exp.run "$@"