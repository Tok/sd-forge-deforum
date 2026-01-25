#!/bin/bash

# rebuild-forge-venv.sh
# Rebuild Forge Neo venv with correct PyTorch/CUDA versions
# Fixes CUDA crashes caused by version mismatches

set -e  # Exit on error

# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color

# Get Forge root (three levels up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo ""
echo -e "${BB0_VOID}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BB0_VOID}  Forge Neo venv Rebuild${NC}"
echo -e "${BB0_VOID}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BB0_MIDNIGHT}This will:${NC}"
echo -e "  ${BB0_ZENITH}1.${NC} Backup current venv → venv.old"
echo -e "  ${BB0_ZENITH}2.${NC} Create fresh venv with Python 3.11"
echo -e "  ${BB0_ZENITH}3.${NC} Install PyTorch 2.9.1+cu130 (Forge Neo default)"
echo -e "  ${BB0_ZENITH}4.${NC} Install all requirements"
echo ""

# Check if we're in the right directory
cd "$FORGE_ROOT" || exit 1
if [ ! -f "webui.py" ]; then
    echo -e "${BB0_GLITCH}❌ Error: Must be in Forge Neo root directory${NC}"
    exit 1
fi
echo -e "${BB0_MIDNIGHT}Forge root: ${NC}$FORGE_ROOT"
echo ""

# Backup old venv
if [ -d "venv" ]; then
    echo -e "${BB0_MIDNIGHT}📦 Backing up current venv...${NC}"
    rm -rf venv.old
    mv venv venv.old
    echo -e "${BB0_ZENITH}✓ Backup created: venv.old${NC}"
else
    echo -e "${BB0_MIDNIGHT}⚠️  No existing venv found, creating fresh${NC}"
fi

# Check Python 3.11
echo ""
echo -e "${BB0_MIDNIGHT}🔍 Checking Python version...${NC}"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo -e "${BB0_ZENITH}✓ Found: $($PYTHON_CMD --version)${NC}"
elif command -v python3 &> /dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [ "$PY_VER" == "3.11" ]; then
        PYTHON_CMD="python3"
        echo -e "${BB0_ZENITH}✓ Using: python3 (version $PY_VER)${NC}"
    else
        echo -e "${BB0_GLITCH}⚠️  Warning: python3 is version $PY_VER (recommended: 3.11)${NC}"
        echo -e "${BB0_MIDNIGHT}   Continuing with $PY_VER anyway...${NC}"
        PYTHON_CMD="python3"
    fi
else
    echo -e "${BB0_GLITCH}❌ Error: Python 3 not found${NC}"
    exit 1
fi

# Create new venv
echo ""
echo -e "${BB0_MIDNIGHT}🏗️  Creating fresh venv...${NC}"
$PYTHON_CMD -m venv venv
echo -e "${BB0_ZENITH}✓ venv created${NC}"

# Activate venv
echo ""
echo -e "${BB0_MIDNIGHT}🔌 Activating venv...${NC}"
source venv/bin/activate
echo -e "${BB0_ZENITH}✓ venv activated: $(which python)${NC}"

# Upgrade pip
echo ""
echo -e "${BB0_MIDNIGHT}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools --quiet
echo -e "${BB0_ZENITH}✓ pip upgraded${NC}"

# Install PyTorch (Forge Neo default: 2.9.1+cu130)
echo ""
echo -e "${BB0_VOID}🔥 Installing PyTorch 2.9.1+cu130...${NC}"
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
echo -e "${BB0_ZENITH}✓ PyTorch installed${NC}"

# Verify PyTorch
echo ""
echo -e "${BB0_MIDNIGHT}🔍 Verifying PyTorch installation...${NC}"
python -c "
import torch
print('  PyTorch: \033[38;2;23;167;254m{}\033[0m'.format(torch.__version__))
print('  CUDA available: \033[38;2;23;167;254m{}\033[0m'.format(torch.cuda.is_available()))
print('  CUDA version: \033[38;2;23;167;254m{}\033[0m'.format(torch.version.cuda))
if torch.cuda.is_available():
    print('  GPU: \033[38;2;23;167;254m{}\033[0m'.format(torch.cuda.get_device_name(0)))
"

# Install xformers
echo ""
echo -e "${BB0_MIDNIGHT}⚡ Installing xformers...${NC}"
pip install xformers==0.0.33.post2 --extra-index-url https://download.pytorch.org/whl/cu130 --quiet
echo -e "${BB0_ZENITH}✓ xformers installed${NC}"

# Install bitsandbytes
echo ""
echo -e "${BB0_MIDNIGHT}🔢 Installing bitsandbytes...${NC}"
pip install bitsandbytes==0.48.2 --quiet
echo -e "${BB0_ZENITH}✓ bitsandbytes installed${NC}"

# Install Forge requirements
echo ""
echo -e "${BB0_MIDNIGHT}📋 Installing Forge requirements...${NC}"
pip install -r requirements.txt --quiet
echo -e "${BB0_ZENITH}✓ Forge requirements installed${NC}"

# Install Deforum requirements
echo ""
echo -e "${BB0_MIDNIGHT}📋 Installing Deforum requirements...${NC}"
pip install -r extensions/sd-forge-deforum/requirements.txt --quiet
echo -e "${BB0_ZENITH}✓ Deforum requirements installed${NC}"

# Final summary
echo ""
echo -e "${BB0_VOID}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BB0_ZENITH}✅ venv Rebuild Complete!${NC}"
echo -e "${BB0_VOID}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BB0_MIDNIGHT}Summary:${NC}"
python -c "
import sys
import torch
print('  Python: \033[38;2;23;167;254m{}\033[0m'.format(sys.version.split()[0]))
print('  PyTorch: \033[38;2;23;167;254m{}\033[0m'.format(torch.__version__))
print('  CUDA: \033[38;2;23;167;254m{}\033[0m'.format(torch.version.cuda))
print('  CUDA available: \033[38;2;23;167;254m{}\033[0m'.format(torch.cuda.is_available()))
"
echo ""
echo -e "${BB0_MIDNIGHT}Next steps:${NC}"
echo -e "  ${BB0_ZENITH}1.${NC} Deactivate: ${BB0_VOID}deactivate${NC}"
echo -e "  ${BB0_ZENITH}2.${NC} Test: ${BB0_VOID}python webui.py${NC}"
echo ""
echo -e "${BB0_MIDNIGHT}If issues persist:${NC}"
echo -e "  ${BB0_ZENITH}•${NC} Check NVIDIA driver: ${BB0_VOID}nvidia-smi${NC}"
echo -e "  ${BB0_ZENITH}•${NC} Try: ${BB0_VOID}python webui.py --attention-pytorch${NC}"
echo -e "  ${BB0_ZENITH}•${NC} Rollback: ${BB0_VOID}rm -rf venv && mv venv.old venv${NC}"
echo ""