#!/bin/bash

# ==============================================================================
# Initialization script for the QAOA-Parameter-Setting project on an HPC system (It definitely works on Perlmutter at NERSC)
#
# This script assumes that the hpc has pre-installed python 3.11 and julia 1.10.0. 
# It loads the Python and Julia modules, sets up a virtual environment,
# installs dependencies in editable mode, and creates a necessary symlink.
# Personally, I was getting an error regarding julia_env/lock.pid. As a workaround, I created a lock bypass symlink to /dev/null. 
# You should probably remove this bypass if you are not getting the same error.
#
# USAGE:
#   1. Clean up old environments: rm -rf venv qaoa_training_pipeline qopt-best-practices (if you want to start fresh)
#   2. Source this script:       source scripts/hpc_initialization.sh
# ==============================================================================

echo "Setting up the environment for QAOA-Parameter-Setting..."

# --- 0. Ensure script is run from project root ---
# Get the directory where the script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Assume the project root is one level above the script directory
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Change to the project root directory
cd "$PROJECT_ROOT"
echo "Running initialization from project root: $PWD"

# --- 0. Set up Scratch Directory for Julia ---
# HPC systems often have limited home directory space. We'll try to use a scratch
# space for Julia packages to avoid quota issues.
#
# If your HPC uses a different environment variable for scratch space (e.g., $MYSCRATCH),
# you can manually set SCRATCH_DIR here before running the script:
# SCRATCH_DIR="$MYSCRATCH"

if [ -z "$SCRATCH_DIR" ]; then
    if [ -n "$SCRATCH" ]; then
        SCRATCH_DIR="$SCRATCH"
        echo "Using SCRATCH environment variable for scratch space: $SCRATCH_DIR"
    elif [ -n "$PSCRATCH" ]; then
        SCRATCH_DIR="$PSCRATCH"
        echo "Using PSCRATCH environment variable for scratch space: $SCRATCH_DIR"
    else
        SCRATCH_DIR="$PWD"
        echo "WARNING: Could not find \$SCRATCH or \$PSCRATCH environment variables."
        echo "Julia packages will be installed in a local directory: ${SCRATCH_DIR}/.julia"
        echo "If you are on an HPC, consider setting the SCRATCH_DIR variable at the top of this script."
    fi
fi

# Define the full path for the Julia depot
JULIA_DEPOT_LOCATION="${SCRATCH_DIR}/.julia"

# --- 1. Load HPC Modules ---
echo "Loading Python 3.11 module..."
module load python/3.11

echo "Loading Julia module..."
module load julia

# Set Julia to use scratch space instead of home directory (avoids quota issues)
export JULIA_DEPOT_PATH="${JULIA_DEPOT_LOCATION}:${JULIA_DEPOT_PATH}"
export JULIA_PKG_PRECOMPILE_AUTO=0  # Disable auto-precompilation for faster installs
echo "Julia depot set to: $JULIA_DEPOT_PATH"

# --- 2. Set up Python Virtual Environment ---
# Check if a valid virtual environment exists by looking for the activate script.
# If it doesn't exist, remove the potentially corrupted venv directory and create a new one.
if [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found or is invalid. Creating a new one..."
    rm -rf venv
    python3 -m venv venv
else
    echo "Valid virtual environment 'venv' already exists."
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

echo "Installing juliacall for Python-Julia interoperability..."
pip install juliacall

echo "Installing PauliPropagation.jl Julia package..."
julia -e 'using Pkg; Pkg.add("PauliPropagation")'

echo "Installing PythonCall.jl for Julia interoperability..."
julia -e 'using Pkg; Pkg.add("PythonCall")'

# --- 4. Setup Julia Environment Variables and Lock Bypass ---
echo "Setting up Julia environment variables for Python..."
# Export these so they're available in the current shell session
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
export PYTHON_JULIACALL_THREADS=auto
export PYTHON_JULIAPKG_OFFLINE=yes

# Find system Julia and point juliapkg to it
JULIA_PATH=$(which julia)
if [ -n "$JULIA_PATH" ]; then
    export PYTHON_JULIAPKG_EXE="$JULIA_PATH"
    echo "Julia executable set to: $JULIA_PATH"
fi

# Create lock bypass to prevent hanging during juliacall initialization
LOCK_FILE="$PWD/venv/julia_env/lock.pid"
# Ensure the directory for the lock file exists before creating the symlink
mkdir -p "$(dirname "$LOCK_FILE")"
if [ -e "$LOCK_FILE" ] || [ -L "$LOCK_FILE" ]; then
    rm -f "$LOCK_FILE"
fi
# Symlink to /dev/null prevents file-based locking from blocking
ln -sf /dev/null "$LOCK_FILE"
echo "Lock bypass created at: $LOCK_FILE -> /dev/null"

# Add these exports to a convenience file users can source
cat > venv/julia_env_vars.sh << 'EOF'
# Source this file to set Julia environment variables for this session
export JULIA_DEPOT_PATH="${JULIA_DEPOT_LOCATION}:\${JULIA_DEPOT_PATH}"
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
export PYTHON_JULIACALL_THREADS=auto
export PYTHON_JULIAPKG_OFFLINE=yes
JULIA_PATH=$(which julia 2>/dev/null)
if [ -n "$JULIA_PATH" ]; then
    export PYTHON_JULIAPKG_EXE="$JULIA_PATH"
fi
EOF
echo "Created venv/julia_env_vars.sh for convenience"

# --- 5. Create Symlink for 'train' module ---
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

# --- 6. Confirmation ---
echo ""
echo "Environment setup complete."
echo ""
echo "NOTE: Julia environment variables are already set for this session."
echo "If you start a new shell, run: source venv/julia_env_vars.sh"
echo ""
echo "You can now run the experiments, for example:"
echo "bash scripts/run_methods.sh --method_dir methods/ --instance_dir instances/erdos_renyi/ --save_dir data/simulations/ --depths 1"
echo "To deactivate the virtual environment when you are done, run: deactivate"

 