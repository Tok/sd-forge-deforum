# Deforum Test Suite

Organized test suite following pytest best practices.

## Directory Structure

```
tests/
├── integration/         # Integration tests
│   ├── api_test.py                  # API integration tests (require server + GPU)
│   ├── test_api_swagger.py          # Swagger/OpenAPI tests (fast, no GPU)
│   ├── test_visual_processing.py    # Visual processing tests (depth, RAFT, keyframes, etc.)
│   ├── test_wan_flf2v.py            # Wan FLF2V tween generation tests
│   ├── test_flux_integration.py     # Flux model integration tests
│   ├── postprocess_test.py          # Post-processing tests (FILM, RIFE, upscaling)
│   ├── test_parseq_adapter.py       # Parseq integration tests
│   ├── utils.py                     # Shared utilities for API tests
│   ├── testdata/                    # Test fixtures (settings files, videos)
│   ├── __snapshots__/               # Snapshot data for regression testing
│   └── conftest.py                  # Integration test fixtures
│
├── unit/                # Fast unit tests (no server, no GPU, no external deps)
│   └── ~1000+ tests across 33 test files (test_*.py)
│
├── functional/          # End-to-end functional tests
│   └── (empty - reserved for future use)
│
├── performance/         # Performance benchmarking tests
│   └── (empty - reserved for future use)
│
├── conftest.py          # Pytest fixtures and configuration
├── manual_api_test.py   # Standalone manual API test script
└── README.md            # This file
```

## Test Types

### Integration Tests (`integration/`)
**What:** Tests that verify Deforum API endpoints work correctly and generate valid videos.

**Requirements:**
- Running Forge server with `--deforum-api` flag
- GPU (for most tests)
- Loaded models (checkpoints, VAE)
- Slow (minutes per test)

**Examples:**
- `api_test.py::test_simple_settings` - Generate animation via API
- `api_test.py::test_3d_mode` - Generate 3D depth-warped animation
- `test_api_swagger.py` - Verify Swagger/OpenAPI documentation (fast, no GPU)
- `test_visual_processing.py::test_depth_map_generation` - Test Depth-Anything V2
- `test_visual_processing.py::test_optical_flow_raft` - Test RAFT optical flow
- `test_visual_processing.py::test_camera_shakify` - Test shakify patterns
- `test_visual_processing.py::test_keyframe_distribution_*` - Test distribution modes
- `test_wan_flf2v.py::test_wan_flf2v_tween_generation` - Test Wan AI tweens
- `test_wan_flf2v.py::test_wan_flf2v_guidance_scale_variations` - Test guidance settings
- `test_flux_integration.py::test_flux_basic_generation` - Test Flux.1 keyframes
- `test_flux_integration.py::test_flux_controlnet_v2` - Test Flux ControlNet
- `test_flux_integration.py::test_flux_wan_hybrid_mode` - Test Flux+Wan hybrid
- `postprocess_test.py::test_post_process_FILM` - Frame interpolation with FILM
- `postprocess_test.py::test_post_process_RIFE` - Frame interpolation with RIFE

**When to run:** Before releases, after major refactoring, in CI

### Unit Tests (`unit/`)
**What:** Tests for individual functions and classes in isolation (~1000+ tests).

**Requirements:**
- No server
- No GPU
- No external dependencies (use mocks)
- Fast (seconds for entire suite)

**Examples:**
- `test_keyframes.py` - Test keyframe distribution algorithms
- `test_args.py` - Test argument parsing and validation
- `test_prompt.py` - Test prompt scheduling logic
- `test_animation.py` - Test animation calculations
- `test_noise.py` - Test noise generation
- `test_schedule_utils.py` - Test schedule manipulation
- ... and 27 more test files

**When to run:** During development, on every commit, in CI

**Coverage:** Optional, enable with `./run-unit-tests.sh --coverage`

### Functional Tests (`functional/`) - Reserved
**What:** End-to-end tests of complete workflows from user perspective.

**Future use:** When we have more complex multi-step workflows to test.

### Performance Tests (`performance/`) - Reserved
**What:** Benchmarking tests to measure speed and resource usage.

**Future use:** To track performance regressions and optimize bottlenecks.

## Running Tests

> **IMPORTANT:** Unit tests and integration tests are now **strictly separated**. Each test runner explicitly ignores the other suite to keep logs clean and enable different workflows.

### Unit Tests (Fast, No Server, With Coverage)
Run all unit tests (no coverage by default):
```bash
./run-unit-tests.sh
```

Run with HTML coverage report:
```bash
./run-unit-tests.sh --coverage
# Opens htmlcov/index.html to view coverage
```

Run specific unit test file:
```bash
./run-unit-tests.sh tests/unit/test_keyframes.py
```

**Characteristics:**
- Runs ~1000+ tests in seconds
- No server required
- No GPU required
- Coverage optional (via `--coverage` flag)
- Suitable for CI/CD pipelines

### Integration Tests (Slow, Requires Server, No Coverage)
Run all integration tests (starts server automatically):
```bash
./run-api-tests.sh
```

Run only fast API tests (skip post-processing):
```bash
./run-api-tests.sh --quick
```

Reuse existing server instead of restarting:
```bash
./run-api-tests.sh --reuse-server
```

Run specific integration test:
```bash
./run-api-tests.sh tests/integration/api_test.py::test_simple_settings
```

**Characteristics:**
- Runs ~17 tests (may take minutes)
- Starts/stops Forge server automatically
- Requires GPU and loaded models
- Coverage disabled (via `--no-cov` flag)
- Local testing only (not suitable for CI)

### Update Test Snapshots (After Intentional Changes)
When you make intentional changes that affect test output (e.g., frame numbering, subtitle format), you need to update snapshots:

```bash
./update-snapshots.sh
```

This script will:
1. Start the Forge server
2. Run tests with `--snapshot-update` flag
3. Show you what changed
4. Prompt you to review and commit

**IMPORTANT:** Only update snapshots for **intentional changes**. If snapshots fail unexpectedly, investigate the root cause first!

**When to update snapshots:**
- ✅ After changing frame indexing (0-based vs 1-based)
- ✅ After changing subtitle format
- ✅ After changing API response structure
- ❌ NOT when tests randomly fail (that's a bug!)
- ❌ NOT automatically in CI (defeats the purpose)

**Manual snapshot update:**
```bash
# Start server
python ../../../webui.py --deforum-api &

# Update snapshots
pytest tests/integration/ --snapshot-update

# Review changes
git diff tests/__snapshots__/

# Commit if correct
git add tests/__snapshots__/
git commit -m "test: Update snapshots for [reason]"
```

### Quick Manual API Test (No pytest required)
For quick verification that the API and Swagger documentation are working:

```bash
# 1. Start server in one terminal
cd ../../..  # Go to webui root
python3 webui.py --deforum-api

# 2. Run manual test in another terminal
cd extensions/sd-forge-deforum
python3 tests/manual_api_test.py
```

This runs fast verification tests without GPU/models:
- ✓ Swagger UI accessible at /docs
- ✓ OpenAPI schema properly generated
- ✓ All endpoints documented
- ✓ Response models defined
- ✓ Basic endpoints work (list jobs/batches)

### Run Tests by Category

**Run only fast Swagger/API tests (no GPU):**
```bash
pytest tests/integration/test_api_swagger.py -v
```

**Run only visual processing tests:**
```bash
pytest tests/integration/test_visual_processing.py -v
```

**Run only Wan FLF2V tests:**
```bash
pytest tests/integration/test_wan_flf2v.py -v
```

**Run only Flux integration tests:**
```bash
pytest tests/integration/test_flux_integration.py -v
```

### Run Tests by Marker
```bash
pytest -m "not slow"             # Fast tests only
pytest -m "flux"                 # Flux-specific tests only
pytest -m "wan"                  # Wan-specific tests only
pytest -m "visual"               # Visual processing tests only
```

Available markers (see `pytest.ini`):
- `slow` - Tests that take a long time (full generation)
- `flux` - Tests requiring Flux model
- `wan` - Tests requiring Wan model
- `visual` - Tests for visual processing features

## Test Organization Principles

### 1. **Separation of Concerns**
- **Integration tests** verify system behavior (does it work?)
- **Unit tests** verify correctness (does this function work?)
- **Performance tests** verify efficiency (is it fast enough?)

### 2. **Fast Feedback Loop**
- Unit tests should run in seconds
- Integration tests can take minutes
- Developers should run unit tests frequently
- CI runs both, but unit tests on every commit

### 3. **Test Pyramid**
```
        /\
       /  \     Few slow integration tests (comprehensive scenarios)
      /----\
     /      \   More unit tests (edge cases, error handling)
    /________\
```

Most tests should be fast unit tests. Integration tests should focus on critical paths.

### 4. **No Test Interdependence**
- Each test should be independent
- Tests should not rely on execution order
- Use fixtures for shared setup

### 5. **Clear Naming**
```python
def test_<what>_<condition>_<expected_result>():
    """Clear docstring explaining what and why"""
```

Example:
```python
def test_distribute_keyframes_evenly_spaced_matches_exactly():
    """When prompts are evenly spaced, keyframes should match frame numbers exactly"""
```

## Writing New Tests

### For Integration Tests
Add to `tests/integration/` when testing:
- Full rendering pipeline
- API endpoints
- Video output validation
- Multi-component interaction

```python
# tests/integration/test_new_feature.py
import requests
from .utils import API_BASE_URL, wait_for_job_to_complete

def test_new_feature():
    """Test that new feature generates valid output"""
    # Given: settings with new feature enabled
    settings = {...}

    # When: job is submitted
    response = requests.post(f"{API_BASE_URL}/batches", json={"deforum_settings": [settings]})
    job_id = response.json()["job_ids"][0]

    # Then: job completes successfully
    status = wait_for_job_to_complete(job_id)
    assert status.status == "SUCCEEDED"
```

### For Unit Tests
Add to `tests/unit/` when testing:
- Individual functions
- Classes and methods
- Edge cases and error handling
- Complex algorithms

```python
# tests/unit/test_keyframe_distribution.py
import pytest
from scripts.deforum_helpers.rendering.data.frame.key_frame_distribution import distribute_keyframes

def test_distribute_keyframes_empty_prompts_returns_empty():
    """Empty prompt dict should return empty keyframe list"""
    result = distribute_keyframes({}, max_frames=100)
    assert result == []

def test_distribute_keyframes_single_prompt_creates_one_keyframe():
    """Single prompt should create exactly one keyframe at frame 0"""
    result = distribute_keyframes({0: "test prompt"}, max_frames=100)
    assert len(result) == 1
    assert result[0].frame_number == 0
```

## Best Practices

### ✅ Do
- Write descriptive test names
- Add docstrings explaining what you're testing
- Use fixtures for common setup
- Mock external dependencies in unit tests
- Test edge cases and error conditions
- Keep tests focused (one concept per test)

### ❌ Don't
- Write tests that depend on other tests
- Use sleep() unless absolutely necessary
- Test implementation details (test behavior, not internals)
- Skip writing tests for "simple" code (simple code can have bugs too)
- Commit tests that are flaky or sometimes fail

## Dependencies

Install test dependencies:
```bash
pip install -r requirements-dev.txt
```

See also: `TEST_STATUS.md` for current test status and issues.
