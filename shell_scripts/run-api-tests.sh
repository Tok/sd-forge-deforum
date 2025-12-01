#!/bin/bash
# run-api-tests.sh - Start Forge server with Deforum API and run integration tests
#
# Usage:
#   ./run-api-tests.sh                          # Run all tests (force restarts server by default)
#   ./run-api-tests.sh --quick                  # Skip post-processing tests (faster)
#   ./run-api-tests.sh --reuse-server           # Reuse existing server if running
#   ./run-api-tests.sh --force-restart          # Force restart server (default behavior)
#   ./run-api-tests.sh --snapshot-update        # Update test snapshots (after intentional changes)
#   ./run-api-tests.sh tests/integration/api_test.py::test_simple_settings  # Run specific test

set -e  # Exit on error

# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color
# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color
# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color
# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color
# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color
# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTENSION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FORGE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SERVER_URL="http://localhost:7860"
DEFORUM_API_URL="${SERVER_URL}/deforum_api/jobs"
MAX_WAIT=300  # Maximum seconds to wait for server (5 minutes)
SERVER_LOG="${EXTENSION_DIR}/test-server.log"
PID_FILE="${EXTENSION_DIR}/.test-server.pid"

# Parse arguments
QUICK_MODE=false
FORCE_RESTART=true  # Default to always restart for test consistency
REUSE_SERVER=false
SNAPSHOT_UPDATE=false
TEST_ARGS=""
PYTEST_ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--quick" ]; then
        QUICK_MODE=true
    elif [ "$arg" = "--reuse-server" ]; then
        REUSE_SERVER=true
        FORCE_RESTART=false
    elif [ "$arg" = "--force-restart" ]; then
        FORCE_RESTART=true
        REUSE_SERVER=false
    elif [ "$arg" = "--snapshot-update" ]; then
        SNAPSHOT_UPDATE=true
        PYTEST_ARGS="$PYTEST_ARGS --snapshot-update"
    else
        TEST_ARGS="$TEST_ARGS $arg"
    fi
done

# Set default test path if none specified
if [ -z "$TEST_ARGS" ]; then
    if [ "$QUICK_MODE" = true ]; then
        TEST_ARGS="tests/integration/api_test.py"
        echo -e "${BB0_GLITCH}Quick mode: Running only API tests (skipping post-processing tests)${NC}"
    else
        TEST_ARGS="tests/integration/"
        echo -e "${BB0_MIDNIGHT}Running all integration tests (API + post-processing)${NC}"
    fi
fi

# Cleanup function - called on exit
cleanup() {
    local exit_code=$?
    echo -e "\n${BB0_MIDNIGHT}Cleaning up...${NC}"

    # Kill server if PID file exists
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${BB0_MIDNIGHT}Stopping Forge server (PID: $pid)${NC}"
            kill "$pid" 2>/dev/null || true
            # Give it a moment to shut down gracefully
            sleep 2
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                echo -e "${BB0_GLITCH}Force killing server${NC}"
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi

    if [ $exit_code -eq 0 ]; then
        echo -e "${BB0_ZENITH}✓ Tests completed successfully${NC}"
    else
        echo -e "${BB0_GLITCH}✗ Tests failed or were interrupted${NC}"
        echo -e "${BB0_GLITCH}Server log available at: ${SERVER_LOG}${NC}"
    fi

    exit $exit_code
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Check if server is already running
check_server() {
    curl -s -f "$DEFORUM_API_URL" > /dev/null 2>&1
    return $?
}

# Wait for server to be ready
wait_for_server() {
    local waited=0
    local interval=2

    echo -e "${BB0_MIDNIGHT}Waiting for server to start (max ${MAX_WAIT}s)...${NC}"

    while [ $waited -lt $MAX_WAIT ]; do
        if check_server; then
            echo -e "${BB0_ZENITH}✓ Server is ready!${NC}"
            return 0
        fi

        # Show progress every 10 seconds
        if [ $((waited % 10)) -eq 0 ]; then
            echo -e "${BB0_MIDNIGHT}  Still waiting... (${waited}s)${NC}"
        fi

        sleep $interval
        waited=$((waited + interval))
    done

    echo -e "${BB0_GLITCH}✗ Server failed to start after ${MAX_WAIT}s${NC}"
    echo -e "${BB0_GLITCH}Check server log at: ${SERVER_LOG}${NC}"
    return 1
}

# Check if we're in the extension directory
if [ ! -f "scripts/deforum.py" ]; then
    echo -e "${BB0_GLITCH}Error: Must run from sd-forge-deforum extension directory${NC}"
    echo -e "Current directory: $(pwd)"
    exit 1
fi

# Check if venv exists
if [ ! -d "$FORGE_DIR/venv" ]; then
    echo -e "${BB0_GLITCH}Error: Forge venv not found at $FORGE_DIR/venv${NC}"
    echo -e "Please run this from the Forge installation"
    exit 1
fi

echo -e "${BB0_ZENITH}========================================${NC}"
echo -e "${BB0_ZENITH}Deforum Test Runner${NC}"
echo -e "${BB0_ZENITH}========================================${NC}"
echo -e "Forge directory: ${FORGE_DIR}"
echo -e "Extension directory: ${EXTENSION_DIR}"
echo -e "Test arguments: ${TEST_ARGS}"
echo -e "${BB0_ZENITH}========================================${NC}\n"

# Check if server is already running
EXISTING_SERVER=false
if check_server; then
    if [ "$REUSE_SERVER" = true ]; then
        echo -e "${BB0_GLITCH}⚠ Forge server is already running at ${SERVER_URL}${NC}"
        echo -e "${BB0_GLITCH}Reusing existing server. Press Ctrl+C within 5s to cancel...${NC}"
        sleep 5
        EXISTING_SERVER=true
    elif [ "$FORCE_RESTART" = true ]; then
        echo -e "${BB0_GLITCH}⚠ Forge server is already running at ${SERVER_URL}${NC}"
        echo -e "${BB0_MIDNIGHT}Force restart enabled - killing existing server...${NC}"

        # Find and kill existing server processes
        pkill -f "webui.py.*deforum-api" 2>/dev/null || true

        # Wait for server to shut down
        sleep 3

        # Verify server is down
        if check_server; then
            echo -e "${BB0_GLITCH}Failed to kill existing server - please kill manually${NC}"
            exit 1
        fi
        echo -e "${BB0_ZENITH}✓ Existing server killed${NC}"
    fi
fi

if [ "$EXISTING_SERVER" = false ]; then

    # Start Forge server in background
    echo -e "${BB0_MIDNIGHT}Starting Forge server with Deforum API...${NC}"
    echo -e "${BB0_MIDNIGHT}Server log: ${SERVER_LOG}${NC}\n"

    cd "$FORGE_DIR"

    # Start server and capture PID
    # Use venv's Python to ensure dependencies are available
    nohup "$FORGE_DIR/venv/bin/python" webui.py \
        --skip-prepare-environment \
        --api \
        --deforum-api \
        --skip-version-check \
        --no-gradio-queue \
        > "$SERVER_LOG" 2>&1 &

    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"

    echo -e "${BB0_MIDNIGHT}Server started (PID: $SERVER_PID)${NC}"

    cd "$EXTENSION_DIR"

    # Wait for server to be ready
    if ! wait_for_server; then
        echo -e "${BB0_GLITCH}Failed to start server. Last 50 lines of log:${NC}"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
fi

# Check test dependencies
echo -e "\n${BB0_MIDNIGHT}Checking test dependencies...${NC}"
if ! "$FORGE_DIR/venv/bin/python" -c "import pytest, syrupy, moviepy" 2>/dev/null; then
    echo -e "${BB0_GLITCH}Installing test dependencies...${NC}"
    "$FORGE_DIR/venv/bin/pip" install -q -r requirements-dev.txt
    echo -e "${BB0_ZENITH}✓ Test dependencies installed${NC}"
else
    echo -e "${BB0_ZENITH}✓ Test dependencies OK${NC}"
fi

# Run tests
echo -e "\n${BB0_ZENITH}========================================${NC}"
if [ "$SNAPSHOT_UPDATE" = true ]; then
    echo -e "${BB0_GLITCH}⚠️  SNAPSHOT UPDATE MODE ENABLED${NC}"
    echo -e "${BB0_GLITCH}This will update test snapshots with current output${NC}"
    echo -e "${BB0_GLITCH}Only use this after intentional changes!${NC}"
    echo -e "${BB0_ZENITH}========================================${NC}\n"
else
    echo -e "${BB0_ZENITH}Running Integration Tests (Unit tests excluded)${NC}"
    echo -e "${BB0_ZENITH}========================================${NC}\n"
fi

set +e  # Don't exit on test failure
# Disable coverage for integration tests (not useful and slows down tests)
# Explicitly ignore unit tests to keep integration test logs clean
"$FORGE_DIR/venv/bin/python" -m pytest $TEST_ARGS $PYTEST_ARGS -v --tb=short --no-cov --ignore=tests/unit
TEST_EXIT_CODE=$?
set -e

# Show reminder after snapshot update
if [ "$SNAPSHOT_UPDATE" = true ] && [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "\n${BB0_ZENITH}✓ Snapshots updated successfully${NC}"
    echo -e "${BB0_GLITCH}⚠️  IMPORTANT: Review changes before committing!${NC}"
    echo -e "\n${BB0_MIDNIGHT}Next steps:${NC}"
    echo -e "  1. Review: ${BB0_ZENITH}git diff tests/__snapshots__/${NC}"
    echo -e "  2. Commit: ${BB0_ZENITH}git add tests/__snapshots__/ && git commit -m 'test: Update snapshots for [reason]'${NC}"
    echo -e "  3. Or revert: ${BB0_GLITCH}git checkout tests/__snapshots__/${NC}"
fi

# If using existing server, don't clean it up
if [ "$EXISTING_SERVER" = true ]; then
    echo -e "\n${BB0_MIDNIGHT}Leaving existing server running${NC}"
    rm -f "$PID_FILE"  # Don't kill existing server
    trap - EXIT INT TERM  # Remove cleanup trap
fi

exit $TEST_EXIT_CODE
