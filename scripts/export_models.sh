#!/bin/bash
# Export models and dependencies.

set -x # Print commands and their arguments as they are executed.
set -e # Exit immediately if a command exits with a non-zero status.

EXPORT_DIR=.export_models
PACKAGE_DIR=src/spec_mamba

# Make directories and add __init__.py
rm -rf "$EXPORT_DIR"
mkdir -p "$EXPORT_DIR/$PACKAGE_DIR" "$EXPORT_DIR/scripts"
touch "$EXPORT_DIR/$PACKAGE_DIR/__init__.py"

# Copy files
cp -R src/bi_mamba_ssm "$EXPORT_DIR/src/"
cp -R "$PACKAGE_DIR/models" "$EXPORT_DIR/$PACKAGE_DIR/"
cp -R export_models/* "$EXPORT_DIR"
cp scripts/install_cuda-12.4.sh scripts/install_python-3.11.sh scripts/setup_venv_ssm.sh "$EXPORT_DIR/scripts"
cp docs/SETUP.md "$EXPORT_DIR/README.md"

# Remove __pycache__
find "$EXPORT_DIR" -type d -name "__pycache__" -exec rm -rf {} +

# Zip
rm -f spec_mamba.zip
(cd "$EXPORT_DIR" && zip -r ../spec_mamba.zip *)
