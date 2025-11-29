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
FORGE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================"
echo "SageAttention Installation"
echo "========================================${NC}"
echo ""

# Check if already installed
if "$FORGE_DIR/venv/bin/python" -c "import sageattention" 2>/dev/null; then
    SAGE_VERSION=$("$FORGE_DIR/venv/bin/python" -c "import sageattention; print(sageattention.__version__)" 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓ SageAttention already installed (version $SAGE_VERSION)${NC}"
    exit 0
fi

echo -e "${BLUE}Prerequisites check:${NC}"

# Check PyTorch
if ! "$FORGE_DIR/venv/bin/python" -c "import torch" 2>/dev/null; then
    echo -e "${RED}✗ PyTorch not installed${NC}"
    echo -e "${YELLOW}Run: ./setup.sh --prepare${NC}"
    exit 1
fi
TORCH_VERSION=$("$FORGE_DIR/venv/bin/python" -c "import torch; print(torch.__version__)")
echo -e "${GREEN}✓ PyTorch $TORCH_VERSION${NC}"

# Check CUDA toolkit
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}✗ CUDA toolkit (nvcc) not found${NC}"
    echo -e "${YELLOW}Run: ./install-cuda-toolkit.sh${NC}"
    exit 1
fi
NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo -e "${GREEN}✓ CUDA toolkit $NVCC_VERSION${NC}"

# Check CUDA_HOME
if [ -z "$CUDA_HOME" ]; then
    echo -e "${YELLOW}⚠ CUDA_HOME not set, attempting auto-detect...${NC}"
    if [ -d "/usr/local/cuda-12.6" ]; then
        export CUDA_HOME=/usr/local/cuda-12.6
        echo -e "${GREEN}✓ Set CUDA_HOME=$CUDA_HOME${NC}"
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo -e "${GREEN}✓ Set CUDA_HOME=$CUDA_HOME${NC}"
    else
        echo -e "${RED}✗ Could not find CUDA installation${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ CUDA_HOME=$CUDA_HOME${NC}"
fi

echo ""
echo -e "${CYAN}Installing SageAttention...${NC}"
echo -e "${BLUE}Using --no-build-isolation to access torch during build${NC}"
echo -e "${YELLOW}This may take 5-10 minutes to compile...${NC}"
echo ""

cd "$FORGE_DIR"

# Install with --no-build-isolation to access torch
if "$FORGE_DIR/venv/bin/pip" install --no-build-isolation sageattention; then
    echo ""
    echo -e "${GREEN}✓ SageAttention installed successfully!${NC}"

    # Verify
    SAGE_VERSION=$("$FORGE_DIR/venv/bin/python" -c "import sageattention; print(sageattention.__version__)")
    echo -e "${GREEN}Version: $SAGE_VERSION${NC}"
    echo ""
    echo -e "${BLUE}You can now use --sage flag when launching Forge:${NC}"
    echo -e "${GREEN}./start-forge.sh${NC}"
else
    echo ""
    echo -e "${RED}✗ Installation failed${NC}"
    echo ""
    echo -e "${YELLOW}Common issues:${NC}"
    echo "  1. CUDA_HOME not set: export CUDA_HOME=/usr/local/cuda-12.6"
    echo "  2. nvcc not in PATH: export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "  3. Missing CUDA libraries: sudo apt-get install cuda-toolkit-12-6"
    echo ""
    echo -e "${BLUE}Check build log above for specific error${NC}"
    exit 1
fi
