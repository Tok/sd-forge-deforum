#!/bin/bash
# setup.sh - Unified Deforum setup and migration script
#
# This script can:
# 1. Check current status
# 2. Install dependencies only
# 3. Migrate venv to Python 3.11.9
#
# Usage:
#   ./setup.sh              # Interactive menu
#   ./setup.sh --check      # Check status only
#   ./setup.sh --install    # Install deps only
#   ./setup.sh --migrate    # Full venv migration
#   ./setup.sh --help       # Show help

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

# Parse arguments
MODE="interactive"
for arg in "$@"; do
    case $arg in
        --check) MODE="check" ;;
        --install) MODE="install" ;;
        --prepare) MODE="prepare" ;;
        --migrate) MODE="migrate" ;;
        --help)
            cat << 'HELP'
Deforum Setup Script

Usage:
  ./setup.sh              Interactive menu
  ./setup.sh --check      Check current status
  ./setup.sh --install    Install Deforum dependencies
  ./setup.sh --prepare    Prepare for first launch (PyTorch + SageAttention)
  ./setup.sh --migrate    Migrate venv to Python 3.11.9
  ./setup.sh --help       Show this help

Modes:
  check     - Display Python version, dependencies, optimizations
  install   - Install Deforum requirements.txt only
  prepare   - First-time setup: Install PyTorch, SageAttention, and Deforum deps
              (Recommended for new installations)
  migrate   - Full venv recreation with Python 3.11
              (backups current venv, deletes it, recreates with 3.11,
               reinstalls everything)

Examples:
  ./setup.sh --check      # Quick status check
  ./setup.sh --prepare    # First-time setup (recommended)
  ./setup.sh --install    # Install missing deps only
  ./setup.sh --migrate    # Full Python 3.11 migration

Documentation:
  docs/SETUP.md           # Comprehensive setup guide
  CLAUDE.md               # Development documentation
HELP
            exit 0
            ;;
    esac
done

# Function: Check Python version
check_python() {
    PYTHON_VER=$(python --version 2>&1 | awk '{print $2}')
    echo -e "${BLUE}Current Python: $PYTHON_VER${NC}"

    if [[ "$PYTHON_VER" == "3.11."* ]]; then
        echo -e "${GREEN}✓ Using Python 3.11 (recommended)${NC}"
        return 0
    elif [[ "$PYTHON_VER" == "3.12."* ]]; then
        echo -e "${YELLOW}⚠ Using Python 3.12 (Forge Neo recommends 3.11.9)${NC}"
        return 1
    else
        echo -e "${RED}✗ Python $PYTHON_VER not recommended${NC}"
        return 1
    fi
}

# Function: Check dependencies
check_deps() {
    echo -e "${BLUE}Checking Deforum dependencies...${NC}"
    MISSING=()
    DEPS=("pandas" "rich" "librosa" "soundfile" "plotly" "numexpr" "av" "pims" "gdown" "easydict")

    for dep in "${DEPS[@]}"; do
        if ! python -c "import $dep" 2>/dev/null; then
            MISSING+=("$dep")
        fi
    done

    if [ ${#MISSING[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ All dependencies installed${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Missing: ${MISSING[*]}${NC}"
        return 1
    fi
}

# Function: Check optimizations
check_optimizations() {
    echo -e "${BLUE}Checking Forge optimizations...${NC}"

    if python -c "import sageattention" 2>/dev/null; then
        echo -e "${GREEN}✓ SageAttention installed${NC}"
    else
        echo -e "${YELLOW}⚠ SageAttention not installed${NC}"
        echo -e "  ${BLUE}Install: python webui.py --sage${NC}"
    fi

    if python -c "import flash_attn" 2>/dev/null; then
        echo -e "${GREEN}✓ FlashAttention installed${NC}"
    else
        echo -e "${YELLOW}⚠ FlashAttention not installed (optional)${NC}"
    fi
}

# Function: Status check
do_check() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Deforum Status Check${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    check_python
    PYTHON_OK=$?
    echo ""

    echo -e "${BLUE}PyTorch:${NC}"
    python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "  NOT INSTALLED"
    echo ""

    check_deps
    DEPS_OK=$?
    echo ""

    check_optimizations
    echo ""

    return $(( PYTHON_OK + DEPS_OK ))
}

# Function: Install dependencies
do_install() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Installing Deforum Dependencies${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    pip install -r requirements.txt
    echo ""
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Function: Prepare Forge for first launch (install PyTorch + SageAttention)
do_prepare() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Prepare Forge for First Launch${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${BLUE}This will:${NC}"
    echo -e "  1. Ensure Forge venv exists"
    echo -e "  2. Install PyTorch via Forge (quick launch with --exit)"
    echo -e "  3. Install SageAttention"
    echo -e "  4. Install Deforum dependencies"
    echo ""

    # Check we're in extension dir
    if [ ! -f "../../webui.py" ]; then
        echo -e "${RED}Error: Not in Forge extension directory${NC}"
        echo "Please run from extensions/sd-forge-deforum/"
        exit 1
    fi

    cd "$FORGE_DIR"

    # Step 1: Install PyTorch via launch.py (doesn't start UI, just installs deps)
    echo -e "${YELLOW}Step 1/3: Installing PyTorch via Forge...${NC}"
    echo -e "${BLUE}This will install PyTorch and dependencies (may take a few minutes)${NC}"

    # Use launch.py which installs dependencies without starting the UI
    # Pipe 'yes' to automatically answer any prompts
    if [ -f "venv/bin/python" ]; then
        echo "" | ./venv/bin/python launch.py --skip-torch-cuda-test --exit
    elif [ -f "webui.sh" ]; then
        # Create venv if it doesn't exist
        echo "" | ./webui.sh --exit
    else
        echo -e "${RED}Error: No Python found${NC}"
        exit 1
    fi

    # Verify PyTorch is now installed
    if ! ./venv/bin/python -c "import torch" 2>/dev/null; then
        echo -e "${RED}Error: PyTorch not installed after Forge launch${NC}"
        echo -e "${YELLOW}You may need to run Forge manually once to complete setup${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ PyTorch installed${NC}"
    echo ""

    # Step 2: Install SageAttention (now that torch is available)
    echo -e "${YELLOW}Step 2/3: Installing SageAttention...${NC}"
    echo -e "${BLUE}Using --no-build-isolation to access torch during build${NC}"
    if [ -f "venv/bin/pip" ]; then
        ./venv/bin/pip install --no-build-isolation sageattention
    else
        pip install --no-build-isolation sageattention
    fi
    echo -e "${GREEN}✓ SageAttention installed${NC}"
    echo ""

    # Step 3: Install Deforum dependencies
    echo -e "${YELLOW}Step 3/3: Installing Deforum dependencies...${NC}"
    cd "$EXTENSION_DIR"
    if [ -f "$FORGE_DIR/venv/bin/pip" ]; then
        "$FORGE_DIR/venv/bin/pip" install -r requirements.txt
    else
        pip install -r requirements.txt
    fi
    echo -e "${GREEN}✓ Deforum dependencies installed${NC}"
    echo ""

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Preparation Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}You can now launch Forge with full optimizations:${NC}"
    echo -e "  ${GREEN}./start-forge.sh --sage${NC}"
    echo ""
}

# Function: Migrate venv
do_migrate() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Forge venv Migration to Python 3.11.9${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}This will:${NC}"
    echo -e "  1. Install Python 3.11 (requires sudo)"
    echo -e "  2. Backup current venv"
    echo -e "  3. Delete $FORGE_DIR/venv"
    echo -e "  4. Create new venv with Python 3.11"
    echo -e "  5. Reinstall all dependencies (~15 minutes)"
    echo ""
    echo -e "${RED}⚠ WARNING: This will delete your current venv!${NC}"
    echo ""
    read -p "$(echo -e ${YELLOW}Continue? [y/N]: ${NC})" -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Migration cancelled.${NC}"
        exit 0
    fi

    # Step 1: Install Python 3.11
    echo ""
    echo -e "${CYAN}Step 1/6: Install Python 3.11${NC}"
    if command -v python3.11 &> /dev/null; then
        echo -e "${GREEN}✓ Python 3.11 already installed${NC}"
    else
        echo -e "${BLUE}Installing Python 3.11 (requires sudo)...${NC}"
        sudo apt update
        sudo apt install -y python3.11 python3.11-venv python3.11-dev
        echo -e "${GREEN}✓ Python 3.11 installed${NC}"
    fi

    # Step 2: Backup
    echo ""
    echo -e "${CYAN}Step 2/6: Backup Current venv${NC}"
    BACKUP_DIR="$EXTENSION_DIR/venv-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    if [ -d "$FORGE_DIR/venv" ]; then
        "$FORGE_DIR/venv/bin/pip" freeze > "$BACKUP_DIR/requirements.txt"
        "$FORGE_DIR/venv/bin/python" --version > "$BACKUP_DIR/python-version.txt" 2>&1
        echo -e "${GREEN}✓ Backup: $BACKUP_DIR${NC}"
    fi

    # Step 3: Delete venv
    echo ""
    echo -e "${CYAN}Step 3/6: Delete Current venv${NC}"
    if [ -d "$FORGE_DIR/venv" ]; then
        rm -rf "$FORGE_DIR/venv"
        echo -e "${GREEN}✓ venv deleted${NC}"
    fi

    # Step 4: Create venv
    echo ""
    echo -e "${CYAN}Step 4/6: Create venv with Python 3.11${NC}"
    cd "$FORGE_DIR"
    python3.11 -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip -q
    echo -e "${GREEN}✓ venv created${NC}"

    # Step 5: Install Forge
    echo ""
    echo -e "${CYAN}Step 5/6: Install Forge Dependencies${NC}"
    echo -e "${YELLOW}This takes 10-15 minutes...${NC}"
    python launch.py --skip-torch-cuda-test --exit
    echo -e "${GREEN}✓ Forge installed${NC}"

    # Step 6: Install Deforum
    echo ""
    echo -e "${CYAN}Step 6/6: Install Deforum Dependencies${NC}"
    cd "$EXTENSION_DIR"
    pip install -r requirements.txt -q
    echo -e "${GREEN}✓ Deforum installed${NC}"

    # Verify
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Verification${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    python --version
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    echo ""
    echo -e "${GREEN}✓ Migration Complete!${NC}"
    echo ""
    echo -e "${BLUE}Next: python webui.py --sage --fast-fp16 --cuda-malloc --cuda-stream${NC}"
}

# Main logic
if [ "$MODE" = "check" ]; then
    do_check
    exit $?
elif [ "$MODE" = "install" ]; then
    do_install
    exit 0
elif [ "$MODE" = "prepare" ]; then
    do_prepare
    exit 0
elif [ "$MODE" = "migrate" ]; then
    do_migrate
    exit 0
else
    # Interactive menu
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Deforum Setup${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo "What would you like to do?"
    echo ""
    echo "  1) Check status"
    echo "  2) Install Deforum dependencies only"
    echo "  3) Prepare for first launch (PyTorch + SageAttention)"
    echo "  4) Migrate venv to Python 3.11.9"
    echo "  5) Exit"
    echo ""
    read -p "Choose [1-5]: " choice

    case $choice in
        1) do_check ;;
        2) do_install ;;
        3) do_prepare ;;
        4) do_migrate ;;
        5) echo "Goodbye!"; exit 0 ;;
        *) echo -e "${RED}Invalid choice${NC}"; exit 1 ;;
    esac
fi
