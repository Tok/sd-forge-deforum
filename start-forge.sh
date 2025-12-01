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

# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color

echo -e "${BB0_ZENITH}========================================${NC}"
echo -e "${BB0_ZENITH}Forge WebUI + Deforum${NC}"
echo -e "${BB0_ZENITH}========================================${NC}"
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
    echo -e "${BB0_ZENITH}Starting with optimizations:${NC}"
    echo -e "  ${BB0_MIDNIGHT}--sage${NC}              SageAttention (RTX 30/40/50)"
    echo -e "  ${BB0_MIDNIGHT}--fast-fp16${NC}         Fast FP16 accumulation"
    echo -e "  ${BB0_MIDNIGHT}--cuda-malloc${NC}       CUDA malloc optimization"
    echo -e "  ${BB0_MIDNIGHT}--cuda-stream${NC}       CUDA stream optimization"
    echo ""
    echo -e "${BB0_GLITCH}Note: If --sage fails, run: ./setup.sh --prepare${NC}"
    echo ""

    FLAGS="--sage --fast-fp16 --cuda-malloc --cuda-stream"
else
    echo -e "${BB0_MIDNIGHT}Starting without optimizations${NC}"
    echo ""
    FLAGS=""
fi

# Add extra args
if [ -n "$EXTRA_ARGS" ]; then
    echo -e "${BB0_MIDNIGHT}Extra flags:${NC} $EXTRA_ARGS"
    echo ""
    FLAGS="$FLAGS $EXTRA_ARGS"
fi

echo -e "${BB0_ZENITH}========================================${NC}"
echo -e "${BB0_ZENITH}Launching Forge...${NC}"
echo -e "${BB0_ZENITH}========================================${NC}"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down Forge..."
    pkill -P $$ 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

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
