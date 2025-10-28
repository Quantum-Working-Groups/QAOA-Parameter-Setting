#!/bin/bash

# ==============================================================================
# Initialization script for the QAOA-Parameter-Setting project on an HPC system (It definitely works on Perlmutter at NERSC)
#
# This script loads the correct Python module, sets up a virtual environment,
# installs dependencies in editable mode, and creates a necessary symlink.
#
# USAGE:
#   1. Clean up old environments: rm -rf venv qaoa_training_pipeline qopt-best-practices
#   2. Source this script:       source initialization.sh
# ==============================================================================

echo "Setting up the environment for QAOA-Parameter-Setting..."

# --- 1. Load HPC Modules ---
echo "Loading Python 3.11 module..."
module load python/3.11

# --- 2. Set up Python Virtual Environment ---
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
else
    echo "Virtual environment 'venv' already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
pip install --upgrade pip

# --- 3. Install Dependencies in Editable Mode ---
echo "Cloning and installing dependencies in editable mode..."

if [ ! -d "qaoa_training_pipeline" ]; then
    git clone https://github.com/qiskit-community/qaoa_training_pipeline.git
fi
pip install --pre -e ./qaoa_training_pipeline

if [ ! -d "qopt-best-practices" ]; then
    git clone https://github.com/qiskit-community/qopt-best-practices.git
fi
pip install -e ./qopt-best-practices

pip install "qiskit-optimization[cplex]"

# --- 4. Create Symlink for 'train' module ---
# The run_methods.sh script calls `python -m train`, but the installed module is
# `qaoa_training_pipeline.train`. This symlink makes `train.py` discoverable
# at the top-level of the site-packages directory, allowing `python -m train` to work.
VENV_PATH="$PWD/venv"
PYTHON_VERSION=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES_PATH="$VENV_PATH/lib/$PYTHON_VERSION/site-packages"

echo "Creating a symbolic link for the 'train' module..."
# The source path for an editable install points to the project directory structure
SOURCE_PATH="$PWD/qaoa_training_pipeline/qaoa_training_pipeline/train.py"
LINK_PATH="${SITE_PACKAGES_PATH}/train.py"

if [ -f "$SOURCE_PATH" ]; then
    ln -sf "$SOURCE_PATH" "$LINK_PATH"
    echo "Symlink created successfully."
else
    echo "Warning: Could not find source for symlink ($SOURCE_PATH). The 'train' module might not be found."
fi

# --- 5. Confirmation ---
echo ""
echo "Environment setup complete."
echo "You can now run the experiments, for example:"
echo "bash scripts/run_methods.sh --method_dir methods/ --instance_dir instances/erdos_renyi/ --save_dir data/simulations/ --depths 1"
echo "To deactivate the virtual environment when you are done, run: deactivate"

 