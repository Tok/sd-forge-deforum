#!/bin/bash
# run-unit-tests.sh - Run unit tests (no server required)
#
# Usage:
#   ./run-unit-tests.sh                    # Run all unit tests
#   ./run-unit-tests.sh --coverage         # Run with coverage report
#   ./run-unit-tests.sh tests/unit/test_keyframes.py  # Run specific test file

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FORGE_DIR="$(cd "../../" && pwd)"
EXTENSION_DIR="$(pwd)"
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
        echo -e "${BLUE}Running all unit tests from ${UNIT_TEST_DIR}/${NC}"
    else
        echo -e "${YELLOW}Warning: ${UNIT_TEST_DIR}/ does not exist yet${NC}"
        echo -e "${YELLOW}Create unit tests in ${UNIT_TEST_DIR}/ directory${NC}"
        echo -e "\n${BLUE}Example structure:${NC}"
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
    echo -e "${RED}Error: Must run from sd-forge-deforum extension directory${NC}"
    echo -e "Current directory: $(pwd)"
    exit 1
fi

# Check if venv exists
if [ ! -d "$FORGE_DIR/venv" ]; then
    echo -e "${RED}Error: Forge venv not found at $FORGE_DIR/venv${NC}"
    echo -e "Please run this from the Forge installation"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deforum Unit Test Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Extension directory: ${EXTENSION_DIR}"
echo -e "Test arguments: ${TEST_ARGS}"
if [ "$COVERAGE_MODE" = true ]; then
    echo -e "Coverage reporting: ${GREEN}ENABLED${NC}"
fi
echo -e "${GREEN}========================================${NC}\n"

# Check test dependencies
echo -e "${BLUE}Checking test dependencies...${NC}"
if ! "$FORGE_DIR/venv/bin/python" -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    "$FORGE_DIR/venv/bin/pip" install -q pytest pytest-cov
    echo -e "${GREEN}✓ Test dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Test dependencies OK${NC}"
fi

# Run tests
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Running Unit Tests (Integration tests excluded)${NC}"
echo -e "${GREEN}========================================${NC}\n"

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
echo -e "\n${GREEN}========================================${NC}"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All unit tests passed!${NC}"
    if [ "$COVERAGE_MODE" = true ]; then
        echo -e "${BLUE}Coverage report: htmlcov/index.html${NC}"
    fi
else
    echo -e "${RED}✗ Some unit tests failed${NC}"
fi
echo -e "${GREEN}========================================${NC}"

exit $TEST_EXIT_CODE
