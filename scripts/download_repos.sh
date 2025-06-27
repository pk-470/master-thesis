#!/bin/bash
# Download related repositories in a .repos directory.

set -x # Print commands and their arguments as they are executed.
set -e # Exit immediately if a command exits with a non-zero status.

mkdir .repos
cd .repos

git clone https://github.com/Dao-AILab/causal-conv1d.git
git clone https://github.com/state-spaces/mamba.git
git clone https://github.com/kaistmm/Audio-Mamba-AuM.git
git clone https://github.com/YuanGongND/ssast.git
git clone https://github.com/SarthakYadav/audio-mamba-official.git
git clone https://github.com/johnma2006/mamba-minimal.git