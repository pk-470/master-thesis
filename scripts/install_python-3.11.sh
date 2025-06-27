#!/bin/bash
# Install python3.11-venv, python3.11-dev for all users (/usr/local) or locally (~) based on the --local flag

set -x # Print commands and their arguments as they are executed.
set -e # Exit immediately if a command exits with a non-zero status.

PYTHON_VERSION=3.11.11
LOCAL=FALSE

if [ "$1" == "--local" ]; then
    LOCAL=TRUE
fi

if [ "$LOCAL" = "TRUE" ]; then
    PYTHON_HOME="$HOME/python3.11"
    
    # Download and build from source
    mkdir -p .temp && cd .temp
    wget "https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz"
    tar -xzf "Python-$PYTHON_VERSION.tgz"
    (
        cd "Python-$PYTHON_VERSION"
        ./configure --prefix="$PYTHON_HOME" --with-ensurepip=install
        make -j "$(nproc)"
        make install
    )
    cd .. && rm -r .temp
else
    PYTHON_HOME=/usr/bin/python3.11 # Default location
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.11-venv python3.11-dev
fi

# Add to ~/.bashrc
cat >> "$HOME/.bashrc" << EOF

# Python 3.11
export PATH="$PYTHON_HOME/bin:\$PATH"
EOF

source "$HOME/.bashrc"
set +x
