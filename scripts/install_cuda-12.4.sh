#!/bin/bash
# Install CUDA Toolkit 12.4 for all users (/usr/local) or locally (~) based on the --local flag

set -x # Print commands and their arguments as they are executed.
set -e # Exit immediately if a command exits with a non-zero status.

INSTALLER=cuda_12.4.0_550.54.14_linux.run
LOCAL=FALSE

if [ "$1" == "--local" ]; then
    LOCAL=TRUE
fi

# Get installer
mkdir -p .temp && cd .temp
wget -c "https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/$INSTALLER"

# Install
if [ "$LOCAL" = "TRUE" ]; then
    CUDA_HOME="$HOME/cuda-12.4"
    bash "$INSTALLER" --silent --override --toolkit --toolkitpath="$CUDA_HOME"
else
    CUDA_HOME=/usr/local/cuda-12.4 # Default location
    sudo bash "$INSTALLER" --silent --override --toolkit
fi

cd .. && rm -r .temp

# Add to ~/.bashrc
cat >> "$HOME/.bashrc" << EOF

# CUDA 12.4
export CUDA_HOME="$CUDA_HOME"
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
EOF

source "$HOME/.bashrc"
set +x