#!/bin/bash
# setup-deforum.sh - Setup Deforum extension with Python 3.11.9 check
#
# This script:
# - Checks Python version compatibility
# - Installs Deforum dependencies
# - Provides guidance for Forge optimizations
#
# Usage:
#   ./setup-deforum.sh              # Interactive setup
#   ./setup-deforum.sh --check      # Just check status
#   ./setup-deforum.sh --install    # Install deps only

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

FORGE_DIR="$(cd ../../ && pwd)"
EXTENSION_DIR="$(pwd)"

# Parse args
CHECK_ONLY=false
INSTALL_ONLY=false

for arg in "$@"; do
    case $arg in
        --check) CHECK_ONLY=true ;;
        --install) INSTALL_ONLY=true ;;
        --help)
            echo "Usage: $0 [--check|--install|--help]"
            echo ""
            echo "  --check    Check current status only"
            echo "  --install  Install dependencies only"
            echo "  --help     Show this help"
            exit 0
            ;;
    esac
done

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Deforum Extension Setup${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
PYTHON_VER=$(python --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}Current Python: $PYTHON_VER${NC}"

if [[ "$PYTHON_VER" == "3.11."* ]]; then
    echo -e "${GREEN}✓ Using Python 3.11 (recommended)${NC}"
    PYTHON_OK=true
elif [[ "$PYTHON_VER" == "3.12."* ]]; then
    echo -e "${YELLOW}⚠ Using Python 3.12 (Forge Neo recommends 3.11.9)${NC}"
    echo -e "${YELLOW}  This may cause compatibility issues.${NC}"
    PYTHON_OK=false
else
    echo -e "${RED}✗ Python version $PYTHON_VER not recommended${NC}"
    PYTHON_OK=false
fi
echo ""

# Check PyTorch
echo -e "${BLUE}Checking PyTorch...${NC}"
TORCH_INFO=$(python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" 2>/dev/null || echo "PyTorch: NOT FOUND")
echo "$TORCH_INFO"
echo ""

# Check Deforum dependencies
echo -e "${BLUE}Checking Deforum dependencies...${NC}"
MISSING=()
DEPS=("pandas" "rich" "librosa" "soundfile" "plotly" "numexpr" "av" "pims" "gdown" "easydict" "transformers" "accelerate")

for dep in "${DEPS[@]}"; do
    if ! python -c "import $dep" 2>/dev/null; then
        MISSING+=("$dep")
    fi
done

if [ ${#MISSING[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All dependencies installed${NC}"
    DEPS_OK=true
else
    echo -e "${YELLOW}⚠ Missing: ${MISSING[*]}${NC}"
    DEPS_OK=false
fi
echo ""

# Check optimizations
echo -e "${BLUE}Checking Forge optimizations...${NC}"
SAGE_INSTALLED=$(python -c "import sageattention; print('yes')" 2>/dev/null || echo "no")
FLASH_INSTALLED=$(python -c "import flash_attn; print('yes')" 2>/dev/null || echo "no")

if [ "$SAGE_INSTALLED" = "yes" ]; then
    echo -e "${GREEN}✓ SageAttention installed${NC}"
else
    echo -e "${YELLOW}⚠ SageAttention not installed${NC}"
    echo -e "  ${BLUE}Install with: python webui.py --sage${NC}"
fi

if [ "$FLASH_INSTALLED" = "yes" ]; then
    echo -e "${GREEN}✓ FlashAttention installed${NC}"
else
    echo -e "${YELLOW}⚠ FlashAttention not installed (optional)${NC}"
fi
echo ""

# If check-only mode, exit here
if [ "$CHECK_ONLY" = true ]; then
    echo -e "${CYAN}========================================${NC}"
    exit 0
fi

# Show recommendations
if [ "$PYTHON_OK" = false ] || [ "$DEPS_OK" = false ]; then
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Recommendations:${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    if [ "$PYTHON_OK" = false ]; then
        echo -e "${YELLOW}Python Version:${NC}"
        echo -e "  Forge Neo recommends Python 3.11.9"
        echo -e "  To recreate venv with Python 3.11:"
        echo -e "    ${CYAN}cd $FORGE_DIR${NC}"
        echo -e "    ${CYAN}rm -rf venv${NC}"
        echo -e "    ${CYAN}python3.11 -m venv venv${NC}"
        echo -e "    ${CYAN}source venv/bin/activate${NC}"
        echo -e "    ${CYAN}python launch.py --skip-torch-cuda-test --exit${NC}"
        echo ""
    fi

    if [ "$DEPS_OK" = false ]; then
        echo -e "${YELLOW}Deforum Dependencies:${NC}"
        echo -e "  Install with:"
        echo -e "    ${CYAN}pip install -r requirements.txt${NC}"
        echo ""
    fi
fi

# Install dependencies if requested or if missing
if [ "$INSTALL_ONLY" = true ] || ( [ "$DEPS_OK" = false ] && [ "$CHECK_ONLY" = false ] ); then
    echo -e "${BLUE}Installing Deforum dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
    echo ""
fi

# Show optimization flags
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Forge Optimization Flags:${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "Launch Forge with these flags for best performance:"
echo ""
echo -e "${CYAN}cd $FORGE_DIR${NC}"
echo -e "${CYAN}python webui.py \\${NC}"
echo -e "${CYAN}  --sage              ${BLUE}# SageAttention (auto-installs)${NC}"
echo -e "${CYAN}  --fast-fp16         ${BLUE}# Fast FP16 (requires PyTorch 2.7+)${NC}"
echo -e "${CYAN}  --cuda-malloc       ${BLUE}# CUDA malloc optimization${NC}"
echo -e "${CYAN}  --cuda-stream       ${BLUE}# CUDA stream optimization${NC}"
echo ""
echo -e "${YELLOW}Optional (may cause OOM on some systems):${NC}"
echo -e "${CYAN}  --pin-shared-memory ${BLUE}# Pin shared memory${NC}"
echo ""
echo -e "${YELLOW}Alternative to --sage:${NC}"
echo -e "${CYAN}  --flash             ${BLUE}# FlashAttention (manual install: pip install flash-attn)${NC}"
echo ""

echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${CYAN}========================================${NC}"
