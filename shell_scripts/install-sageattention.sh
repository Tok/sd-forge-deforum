#!/bin/bash
# install-sageattention.sh - Install SageAttention optimization library
#
# Requires:
#   - PyTorch installed in Forge venv
#   - CUDA toolkit (nvcc, CUDA_HOME) for compilation
#
# Usage:
#   ./install-sageattention.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTENSION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FORGE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color

echo -e "${BB0_ZENITH}========================================"
echo "SageAttention Installation"
echo "========================================${NC}"
echo ""

# Check if already installed
if "$FORGE_DIR/venv/bin/python" -c "import sageattention" 2>/dev/null; then
    echo -e "${BB0_ZENITH}✓ SageAttention already installed${NC}"
    exit 0
fi

echo -e "${BB0_MIDNIGHT}Prerequisites check:${NC}"

# Set CUDA_HOME if it exists but not exported yet
if [ -z "$CUDA_HOME" ] && [ -d "/usr/local/cuda-12.6" ]; then
    export CUDA_HOME=/usr/local/cuda-12.6
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

# Check PyTorch
if ! "$FORGE_DIR/venv/bin/python" -c "import torch" 2>/dev/null; then
    echo -e "${BB0_GLITCH}✗ PyTorch not installed${NC}"
    echo -e "${BB0_GLITCH}Run: ./setup.sh --prepare${NC}"
    exit 1
fi
TORCH_VERSION=$("$FORGE_DIR/venv/bin/python" -c "import torch; print(torch.__version__)")
echo -e "${BB0_ZENITH}✓ PyTorch $TORCH_VERSION${NC}"

# Check CUDA toolkit
if ! command -v nvcc &> /dev/null; then
    echo -e "${BB0_GLITCH}✗ CUDA toolkit (nvcc) not found${NC}"
    echo -e "${BB0_GLITCH}Run: ./install-cuda-toolkit.sh${NC}"
    exit 1
fi
NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo -e "${BB0_ZENITH}✓ CUDA toolkit $NVCC_VERSION${NC}"

# Check CUDA_HOME
if [ -z "$CUDA_HOME" ]; then
    echo -e "${BB0_GLITCH}⚠ CUDA_HOME not set, attempting auto-detect...${NC}"
    if [ -d "/usr/local/cuda-12.6" ]; then
        export CUDA_HOME=/usr/local/cuda-12.6
        echo -e "${BB0_ZENITH}✓ Set CUDA_HOME=$CUDA_HOME${NC}"
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo -e "${BB0_ZENITH}✓ Set CUDA_HOME=$CUDA_HOME${NC}"
    else
        echo -e "${BB0_GLITCH}✗ Could not find CUDA installation${NC}"
        exit 1
    fi
else
    echo -e "${BB0_ZENITH}✓ CUDA_HOME=$CUDA_HOME${NC}"
fi

echo ""
echo -e "${BB0_ZENITH}Installing SageAttention...${NC}"

# Step 1: Install build dependencies
echo -e "${BB0_MIDNIGHT}Step 1/2: Installing build dependencies (wheel, ninja)...${NC}"
"$FORGE_DIR/venv/bin/pip" install wheel ninja -q

# Step 2: Compile SageAttention
echo -e "${BB0_MIDNIGHT}Step 2/2: Compiling SageAttention...${NC}"
echo -e "${BB0_MIDNIGHT}Using --no-build-isolation to access torch during build${NC}"
echo -e "${BB0_GLITCH}This may take 5-10 minutes to compile...${NC}"
echo ""

cd "$FORGE_DIR"

# Install with --no-build-isolation to access torch
if "$FORGE_DIR/venv/bin/pip" install --no-build-isolation sageattention; then
    echo ""
    echo -e "${BB0_ZENITH}✓ SageAttention installed successfully!${NC}"
    echo ""
    echo -e "${BB0_MIDNIGHT}You can now use --sage flag when launching Forge:${NC}"
    echo -e "${BB0_ZENITH}../../start-forge.sh${NC}"
else
    echo ""
    echo -e "${BB0_GLITCH}✗ Installation failed${NC}"
    echo ""
    echo -e "${BB0_GLITCH}Common issues:${NC}"
    echo "  1. CUDA_HOME not set: export CUDA_HOME=/usr/local/cuda-12.6"
    echo "  2. nvcc not in PATH: export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "  3. Missing CUDA libraries: sudo apt-get install cuda-toolkit-12-6"
    echo ""
    echo -e "${BB0_MIDNIGHT}Check build log above for specific error${NC}"
    exit 1
fi
