#!/bin/bash
# ==============================================================================
# Build CUDA Kernels for TurboQuant
# ==============================================================================
# This script builds CUDA extensions and automatically copies them to cuda_kernels/
#
# Usage:
#   ./build_cuda_kernels.sh          # Build for current environment
#   ./build_cuda_kernels.sh --clean  # Clean build
#
# Environment:
#   - Requires py310 conda environment (or adjust CONDA_ENV below)
#   - Requires CUDA 12.x installed
#   - Requires system g++ available at /usr/bin/g++
# ==============================================================================

set -e

# Configuration
CONDA_ENV="py310"
CONDA_ROOT="/data/anaconda3"
CUDA_PATH="/usr/local/cuda"
TORCH_CUDA_ARCH_LIST="8.6"
SYSTEM_CC="/usr/bin/gcc"
SYSTEM_CXX="/usr/bin/g++"

# Directories (script is in csrc/, so parent is turboquant/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TURBOQUANT_DIR="$(dirname "${SCRIPT_DIR}")"
CSRC_DIR="${SCRIPT_DIR}"
BUILD_DIR="${CSRC_DIR}/build"
CUDA_KERNELS_DIR="${TURBOQUANT_DIR}/cuda_kernels"

echo "=============================================================================="
echo "TurboQuant CUDA Kernel Build"
echo "=============================================================================="
echo "Build directory: ${BUILD_DIR}"
echo "Target directory: ${CUDA_KERNELS_DIR}"
echo ""

# Clean previous build if requested
if [ "$1" == "--clean" ]; then
    echo "Cleaning previous build..."
    rm -rf "${BUILD_DIR}"
    echo "Cleaned!"
    echo ""
fi

# Ensure target directory exists
mkdir -p "${CUDA_KERNELS_DIR}"

# Activate conda environment
echo "Activating ${CONDA_ENV} environment..."
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Python version: $(python --version)"
echo ""

# Build CUDA extensions
echo "Building CUDA extensions..."
echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "  CUDA_PATH=${CUDA_PATH}"
echo "  CC=${SYSTEM_CC}"
echo "  CXX=${SYSTEM_CXX}"
echo ""

cd "${CSRC_DIR}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
export CUDA_PATH="${CUDA_PATH}"
export CC="${SYSTEM_CC}"
export CXX="${SYSTEM_CXX}"
export CUDA_HOME="${CUDA_PATH}"

python setup.py build_ext --inplace

echo ""
echo "=============================================================================="
echo "Build Complete!"
echo "=============================================================================="
echo ""
echo "So files location:"
echo "  Build output: ${BUILD_DIR}/"
echo "  Target: ${CUDA_KERNELS_DIR}/"
echo ""
echo "To verify:"
echo "  ls -la ${CUDA_KERNELS_DIR}/*.so"
echo ""
