#!/bin/bash
# Deforum Tuning Lab Launcher
# Starts Forge with the Deforum tuning tab enabled and optimization flags
#
# Usage:
#   ./run-tuning-lab.sh           # Start with optimizations
#   ./run-tuning-lab.sh --no-opt  # Start without optimizations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================"
echo "Deforum Tuning Lab"
echo "========================================${NC}"

cd "$FORGE_DIR"

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
    echo -e "${GREEN}Starting with optimizations:${NC}"
    echo -e "  ${BLUE}--sage${NC}              SageAttention (RTX 30/40/50)"
    echo -e "  ${BLUE}--fast-fp16${NC}         Fast FP16 accumulation"
    echo -e "  ${BLUE}--cuda-malloc${NC}       CUDA malloc optimization"
    echo -e "  ${BLUE}--cuda-stream${NC}       CUDA stream optimization"
    echo ""
    OPT_FLAGS="--sage --fast-fp16 --cuda-malloc --cuda-stream"
else
    echo -e "${BLUE}Starting without optimizations${NC}"
    echo ""
    OPT_FLAGS=""
fi

echo -e "${CYAN}========================================${NC}"
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

echo -e "${GREEN}Launching Forge...${NC}"
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
