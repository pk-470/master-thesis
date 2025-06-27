# Environment setup

## OS

Ubuntu 24.04

## GCC

```bash
sudo apt install build-essential
gcc --version
```

## Ninja

```bash
sudo apt install ninja-build
ninja --version
```

## Zip/unzip

```bash
sudo apt install zip unzip
```

## Install CUDA Toolkit 12.4

### Add universe repo to sources

```bash
sudo nano /etc/apt/sources.list.d/ubuntu.sources
```

Append to file:

```text
Types: deb
URIs: http://old-releases.ubuntu.com/ubuntu/
Suites: lunar
Components: universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
```

### Install CUDA Toolkit 12.4

Use the `--local` flag to install in the user home directory.

```bash
source scripts/install_cuda-12.4.sh [--local]
```

Verify CUDA version:

```bash
which nvcc
nvcc --version
```

## Install Python 3.11 with venv and dev

Use the `--local` flag to install in the user home directory.

```bash
source scripts/install_python-3.11.sh [--local]
```

Verify Python installation:

```bash
which python3.11
python3.11 --version
```

## Python 3.11 virtual environment

Sets up and activates a virtual environment named .venv_ssm with all the requirements installed and the `spec_mamba` package installed as an editable module.

```bash
source scripts/setup_venv_ssm.sh
```
