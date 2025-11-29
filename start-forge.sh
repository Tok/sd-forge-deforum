#!/bin/bash
# start-forge.sh - Start Forge with Deforum optimizations
#
# This script launches Forge WebUI with recommended optimization flags:
#   --sage              SageAttention (auto-installs on first run)
#   --fast-fp16         Fast FP16 accumulation (PyTorch 2.7+)
#   --cuda-malloc       CUDA malloc optimization
#   --cuda-stream       CUDA stream optimization
#
# Usage:
#   ./start-forge.sh                # Start with optimizations
#   ./start-forge.sh --listen       # Add custom flags
#   ./start-forge.sh --no-opt       # Start without optimizations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Forge WebUI + Deforum${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

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

# Build command with optimization flags
if [ "$USE_OPTIMIZATIONS" = true ]; then
    echo -e "${GREEN}Starting with optimizations:${NC}"
    echo -e "  ${BLUE}--sage${NC}              SageAttention (RTX 30/40/50)"
    echo -e "  ${BLUE}--fast-fp16${NC}         Fast FP16 accumulation"
    echo -e "  ${BLUE}--cuda-malloc${NC}       CUDA malloc optimization"
    echo -e "  ${BLUE}--cuda-stream${NC}       CUDA stream optimization"
    echo ""

    CMD="python webui.py --sage --fast-fp16 --cuda-malloc --cuda-stream"
else
    echo -e "${BLUE}Starting without optimizations${NC}"
    echo ""
    CMD="python webui.py"
fi

# Add extra args
if [ -n "$EXTRA_ARGS" ]; then
    echo -e "${BLUE}Extra flags:${NC} $EXTRA_ARGS"
    echo ""
    CMD="$CMD $EXTRA_ARGS"
fi

echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}Launching Forge...${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down Forge..."
    pkill -P $$ 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Execute
eval $CMD &
WEBUI_PID=$!
wait $WEBUI_PID
