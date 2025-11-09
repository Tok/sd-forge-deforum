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

# Use webui.sh if available, otherwise fallback to python webui.py
if [ -f "webui.sh" ]; then
    ./webui.sh --deforum-run-tuning "$@"
else
    python webui.py --deforum-run-tuning "$@"
fi
