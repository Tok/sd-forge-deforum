#!/bin/bash
# Deforum Tuning Lab Launcher
# Starts Forge with the Deforum tuning tab enabled and optimization flags
#
# Usage:
#   ./run-tuning-lab.sh           # Start with optimizations
#   ./run-tuning-lab.sh --no-opt  # Start without optimizations

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
echo -e "Deforum Tuning Lab"
echo -e "========================================${NC}"
echo ""

# Check for existing WebUI instances to prevent duplicate launches
# This prevents VRAM exhaustion from multiple instances running simultaneously
EXISTING_PID=$(pgrep -f "python.*launch.py" | head -1)
if [[ -n "$EXISTING_PID" ]]; then
    echo ""
    echo -e "${BB0_GLITCH}========================================${NC}"
    echo -e "${BB0_GLITCH}ERROR: WebUI is already running!${NC}"
    echo -e "${BB0_GLITCH}========================================${NC}"
    echo ""
    echo -e "Found existing process: ${BB0_VOID}PID $EXISTING_PID${NC}"
    echo ""
    echo -e "${BB0_MIDNIGHT}Running multiple instances causes VRAM exhaustion and OOM errors.${NC}"
    echo ""
    echo -e "${BB0_ZENITH}To stop the existing instance:${NC}"
    echo -e "  ${BB0_VOID}kill $EXISTING_PID${NC}"
    echo ""
    echo -e "${BB0_ZENITH}Or to force kill all WebUI instances:${NC}"
    echo -e "  ${BB0_VOID}pkill -f 'python.*launch.py'${NC}"
    echo ""
    echo -e "${BB0_ZENITH}Then restart with:${NC}"
    echo -e "  ${BB0_VOID}./shell_scripts/run-tuning-lab.sh${NC}"
    echo ""
    exit 1
fi

cd "$FORGE_DIR"

# Clear Python bytecode cache to ensure latest code is loaded
echo -e "${BB0_MIDNIGHT}Clearing Python bytecode cache...${NC}"
find extensions/sd-forge-deforum -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null
find extensions/sd-forge-deforum -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo -e "${BB0_ZENITH}✓ Cache cleared${NC}"
echo ""

# Check for --no-opt flag
USE_OPTIMIZATIONS=true
EXTRA_ARGS=""

for arg in "$@"; do
    if [ "$arg" = "--no-opt" ]; then
        USE_OPTIMIZATIONS=false
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

# Build optimization flags
if [ "$USE_OPTIMIZATIONS" = true ]; then
    echo -e "${BB0_ZENITH}Starting with optimizations:${NC}"
    echo -e "  ${BB0_MIDNIGHT}--sage${NC}              SageAttention (RTX 30/40/50)"
    echo -e "  ${BB0_MIDNIGHT}--fast-fp16${NC}         Fast FP16 accumulation"
    echo -e "  ${BB0_MIDNIGHT}--cuda-malloc${NC}       CUDA malloc optimization"
    echo -e "  ${BB0_MIDNIGHT}--cuda-stream${NC}       CUDA stream optimization"
    echo ""
    echo -e "${BB0_GLITCH}Note: If --sage fails, run: ./setup.sh --prepare${NC}"
    echo ""
    OPT_FLAGS="--sage --fast-fp16 --cuda-malloc --cuda-stream"
else
    echo -e "${BB0_MIDNIGHT}Starting without optimizations${NC}"
    echo ""
    OPT_FLAGS=""
fi

echo -e "${BB0_ZENITH}========================================${NC}"
echo ""

# Cleanup function to kill child processes
cleanup() {
    echo ""
    echo "Shutting down Forge..."
    # Kill all python webui.py processes in this process group
    pkill -P $$ 2>/dev/null || true
    # Also kill by name as fallback
    pkill -f "webui.py.*--deforum" 2>/dev/null || true
    exit 0
}

# Trap Ctrl+C and other termination signals
trap cleanup SIGINT SIGTERM

# Build flags
# Note: --deforum-api and --deforum-run-tuning are required for tuning tab
FLAGS="$OPT_FLAGS --deforum-api --deforum-run-tuning $EXTRA_ARGS"

echo -e "${BB0_ZENITH}Launching Forge...${NC}"
echo ""

# Execute - use webui.sh if available, otherwise venv/bin/python
if [ -f "webui.sh" ]; then
    # webui.sh handles venv activation
    ./webui.sh $FLAGS &
    WEBUI_PID=$!
    wait $WEBUI_PID
elif [ -f "venv/bin/python" ]; then
    # Use venv python directly
    ./venv/bin/python webui.py $FLAGS &
    WEBUI_PID=$!
    wait $WEBUI_PID
else
    echo -e "${RED}Error: Neither webui.sh nor venv/bin/python found!${NC}"
    echo "Please run from Forge root directory with a valid venv."
    exit 1
fi
