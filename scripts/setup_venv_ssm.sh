#!/bin/bash
# Create a Python 3.11 virtual environment including the causal-conv1d, mamba-ssm dependencies in a .venv_ssm directory

set -x # Print commands and their arguments as they are executed.
set -e # Exit immediately if a command exits with a non-zero status.

PYTHON_V=311
CUDA_V=12
CUDA_VERSION=124
TORCH_V=2.5
TORCH_VERSION=2.5.1
TORCHVISION_VERSION=0.20.1
TORCHAUDIO_VERSION=2.5.1
CXX11ABI=FALSE
CAUSAL_CONV1D_V=1.5.0.post8
MAMBA_V=2.2.4

# Deactivate any existing environments
if [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate
fi

# Create and activate environment
rm -rf .venv_ssm
python3.11 -m venv .venv_ssm
source .venv_ssm/bin/activate

# Install basic tools and torch
pip install --upgrade pip setuptools
pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
mkdir -p .temp && cd .temp

# Install causal-conv1d
wget "https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_CONV1D_V}/causal_conv1d-${CAUSAL_CONV1D_V}+cu${CUDA_V}torch${TORCH_V}cxx11abi${CXX11ABI}-cp${PYTHON_V}-cp${PYTHON_V}-linux_x86_64.whl"
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install "causal_conv1d-${CAUSAL_CONV1D_V}+cu${CUDA_V}torch${TORCH_V}cxx11abi${CXX11ABI}-cp${PYTHON_V}-cp${PYTHON_V}-linux_x86_64.whl"
rm "causal_conv1d-${CAUSAL_CONV1D_V}+cu${CUDA_V}torch${TORCH_V}cxx11abi${CXX11ABI}-cp${PYTHON_V}-cp${PYTHON_V}-linux_x86_64.whl"

# Install mamba-ssm
wget "https://github.com/state-spaces/mamba/releases/download/v${MAMBA_V}/mamba_ssm-${MAMBA_V}+cu${CUDA_V}torch${TORCH_V}cxx11abi${CXX11ABI}-cp${PYTHON_V}-cp${PYTHON_V}-linux_x86_64.whl"
MAMBA_FORCE_BUILD=TRUE pip install "mamba_ssm-${MAMBA_V}+cu${CUDA_V}torch${TORCH_V}cxx11abi${CXX11ABI}-cp${PYTHON_V}-cp${PYTHON_V}-linux_x86_64.whl"
rm "mamba_ssm-${MAMBA_V}+cu${CUDA_V}torch${TORCH_V}cxx11abi${CXX11ABI}-cp${PYTHON_V}-cp${PYTHON_V}-linux_x86_64.whl"

# Install editable module and other requirements
cd .. && rm -r .temp
pip install -e .

set +x