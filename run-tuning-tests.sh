#!/bin/bash

# Deforum Parameter Tuning Test Runner
# Runs GPU-required tuning tests against running Forge instance

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORGE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXT_DIR="$SCRIPT_DIR"

echo -e "${GREEN}========================================"
echo -e "Deforum Parameter Tuning Test Runner"
echo -e "========================================${NC}"
echo "Forge directory: $FORGE_DIR"
echo "Extension directory: $EXT_DIR"
echo -e "${GREEN}========================================${NC}"

# Check if server should be started
START_SERVER=false
REUSE_SERVER=false
for arg in "$@"; do
    if [ "$arg" = "--start-server" ]; then
        START_SERVER=true
    fi
    if [ "$arg" = "--reuse-server" ]; then
        REUSE_SERVER=true
    fi
done

# Start server if requested
if [ "$START_SERVER" = true ]; then
    echo -e "${BLUE}Starting Forge server with Deforum API...${NC}"
    echo -e "${BLUE}Server log: $EXT_DIR/test-server.log${NC}"

    cd "$FORGE_DIR"
    "$FORGE_DIR/venv/bin/python" webui.py --skip-prepare-environment --deforum-api --listen > "$EXT_DIR/test-server.log" 2>&1 &
    SERVER_PID=$!

    echo -e "${BLUE}Server started (PID: $SERVER_PID)${NC}"
    echo -e "${BLUE}Waiting for server to start (max 300s)...${NC}"

    # Wait for server
    for i in {1..60}; do
        if curl -s http://localhost:7860/deforum_api/jobs/ > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server is ready!${NC}"
            break
        fi
        if [ $i -eq 60 ]; then
            echo -e "${RED}✗ Server failed to start in 300s${NC}"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
        echo -e "${BLUE}  Still waiting... (${i}s)${NC}"
        sleep 5
    done
fi

# Check dependencies
echo -e "${BLUE}Checking test dependencies...${NC}"
"$FORGE_DIR/venv/bin/python" -c "
import sys
try:
    import cv2
    import PIL
    import skimage
    import numpy
    print('✓ All tuning test dependencies available')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    print('Install with: pip install opencv-python scikit-image')
    sys.exit(1)
"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test dependencies OK${NC}"
else
    echo -e "${RED}✗ Missing dependencies${NC}"
    echo -e "${YELLOW}Install with: $FORGE_DIR/venv/bin/pip install opencv-python scikit-image${NC}"
    [ "$START_SERVER" = true ] && kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Run tuning tests
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running Parameter Tuning Tests${NC}"
echo -e "${YELLOW}⚠️  WARNING: These tests are SLOW and GPU-intensive${NC}"
echo -e "${YELLOW}⚠️  Each parameter combination generates multiple images${NC}"
echo -e "${GREEN}========================================${NC}"

cd "$EXT_DIR"

# Remove --start-server and --reuse-server from args (they're for this script only)
PYTEST_ARGS=""
for arg in "$@"; do
    if [ "$arg" != "--start-server" ] && [ "$arg" != "--reuse-server" ]; then
        PYTEST_ARGS="$PYTEST_ARGS $arg"
    fi
done

# If no specific test specified, run all tuning tests
if [ -z "$PYTEST_ARGS" ]; then
    PYTEST_ARGS="tests/tuning/"
fi

"$FORGE_DIR/venv/bin/python" -m pytest $PYTEST_ARGS -v --tb=short --no-cov

TEST_EXIT_CODE=$?

# Cleanup
if [ "$START_SERVER" = true ]; then
    echo -e "${BLUE}Cleaning up...${NC}"
    echo -e "${BLUE}Stopping Forge server (PID: $SERVER_PID)${NC}"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
fi

# Report results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Tests completed successfully${NC}"
    echo -e "${BLUE}Results saved to: outputs/deforum-tuning/${NC}"
else
    echo -e "${RED}✗ Tests failed or were interrupted${NC}"
    [ "$START_SERVER" = true ] && echo -e "${YELLOW}Server log available at: $EXT_DIR/test-server.log${NC}"
fi

exit $TEST_EXIT_CODE
