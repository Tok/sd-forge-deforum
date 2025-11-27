#!/bin/bash
# Deforum Tuning Lab Launcher
# Starts Forge with the Deforum tuning tab enabled

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "Deforum Tuning Lab"
echo "========================================"
echo "Starting Forge with tuning tab..."
echo "========================================"

cd "$FORGE_DIR"

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

# Use webui.sh if available, otherwise fallback to python webui.py
# Note: --deforum-api is required for tuning tab to work
if [ -f "webui.sh" ]; then
    ./webui.sh --deforum-api --deforum-run-tuning "$@" &
    WEBUI_PID=$!
    wait $WEBUI_PID
else
    python webui.py --deforum-api --deforum-run-tuning "$@" &
    WEBUI_PID=$!
    wait $WEBUI_PID
fi
