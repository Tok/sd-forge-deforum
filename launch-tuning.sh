#!/bin/bash
# Deforum Tuning Mode Launcher
# Starts Forge with the Deforum tuning tab enabled

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "Deforum Tuning Mode"
echo "========================================"
echo "Starting Forge with tuning tab..."
echo "========================================"

cd "$FORGE_DIR"

# Launch with tuning mode flag
python webui.py --deforum-run-tuning "$@"
