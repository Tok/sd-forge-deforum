#!/bin/bash
# run-unit-tests.sh - Run unit tests (no server required)
#
# Usage:
#   ./run-unit-tests.sh                    # Run all unit tests
#   ./run-unit-tests.sh --coverage         # Run with coverage report
#   ./run-unit-tests.sh tests/unit/test_keyframes.py  # Run specific test file

set -e  # Exit on error

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
UNIT_TEST_DIR="tests/unit"

# Parse arguments
COVERAGE_MODE=false
TEST_ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--coverage" ]; then
        COVERAGE_MODE=true
    else
        TEST_ARGS="$TEST_ARGS $arg"
    fi
done

# Set default test path if none specified
if [ -z "$TEST_ARGS" ]; then
    if [ -d "$UNIT_TEST_DIR" ]; then
        TEST_ARGS="$UNIT_TEST_DIR/"
        echo -e "${BB0_MIDNIGHT}Running all unit tests from ${UNIT_TEST_DIR}/${NC}"
    else
        echo -e "${BB0_GLITCH}Warning: ${UNIT_TEST_DIR}/ does not exist yet${NC}"
        echo -e "${BB0_GLITCH}Create unit tests in ${UNIT_TEST_DIR}/ directory${NC}"
        echo -e "\n${BB0_MIDNIGHT}Example structure:${NC}"
        echo -e "  tests/unit/"
        echo -e "    ├── test_keyframes.py"
        echo -e "    ├── test_prompts.py"
        echo -e "    ├── test_args.py"
        echo -e "    └── test_wan_integration.py"
        exit 1
    fi
fi

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
echo -e "${BB0_ZENITH}Deforum Unit Test Runner${NC}"
echo -e "${BB0_ZENITH}========================================${NC}"
echo -e "Extension directory: ${EXTENSION_DIR}"
echo -e "Test arguments: ${TEST_ARGS}"
if [ "$COVERAGE_MODE" = true ]; then
    echo -e "Coverage reporting: ${BB0_ZENITH}ENABLED${NC}"
fi
echo -e "${BB0_ZENITH}========================================${NC}\n"

# Check test dependencies
echo -e "${BB0_MIDNIGHT}Checking test dependencies...${NC}"
if ! "$FORGE_DIR/venv/bin/python" -c "import pytest" 2>/dev/null; then
    echo -e "${BB0_GLITCH}Installing test dependencies...${NC}"
    "$FORGE_DIR/venv/bin/pip" install -q pytest pytest-cov
    echo -e "${BB0_ZENITH}✓ Test dependencies installed${NC}"
else
    echo -e "${BB0_ZENITH}✓ Test dependencies OK${NC}"
fi

# Run tests
echo -e "\n${BB0_ZENITH}========================================${NC}"
echo -e "${BB0_ZENITH}Running Unit Tests (Integration tests excluded)${NC}"
echo -e "${BB0_ZENITH}========================================${NC}\n"

# Build pytest command
# Explicitly ignore integration tests to keep unit test suite separate
PYTEST_CMD="$FORGE_DIR/venv/bin/python -m pytest $TEST_ARGS -v --tb=short --ignore=tests/integration"

if [ "$COVERAGE_MODE" = true ]; then
    # Add coverage options
    PYTEST_CMD="$PYTEST_CMD --cov=deforum --cov=scripts --cov-report=term-missing --cov-report=html"
fi

set +e  # Don't exit on test failure
eval "$PYTEST_CMD"
TEST_EXIT_CODE=$?
set -e

# Print results
echo -e "\n${BB0_ZENITH}========================================${NC}"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${BB0_ZENITH}✓ All unit tests passed!${NC}"
    if [ "$COVERAGE_MODE" = true ]; then
        echo -e "${BB0_MIDNIGHT}Coverage report: htmlcov/index.html${NC}"
    fi
else
    echo -e "${BB0_GLITCH}✗ Some unit tests failed${NC}"
fi
echo -e "${BB0_ZENITH}========================================${NC}"

exit $TEST_EXIT_CODE
