#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=staging
#SBATCH --time=10:00:00
#SBATCH --output=cpu-job_%j.out

set -x # Print commands and their arguments as they are executed.
set -e # Exit immediately if a command exits with a non-zero status.

# Load modules
module purge            
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Change directory
cd "$HOME/master-thesis/"

# Activate .venv_ssm
source .venv_ssm/bin/activate

# Copy data
python scripts/copy_data.py "$@"