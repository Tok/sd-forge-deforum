# Test Suite Status Report

**Date:** October 21, 2025
**Branch:** dev
**Status:** ⚠️ **Tests Not Running** (Server Required)

## Summary

The test suite **exists and is structurally valid**, but cannot run without a Forge WebUI server. All tests are API integration tests that require:
1. Running Forge server with `--deforum-api` flag
2. Loaded models (checkpoints, VAE)
3. GPU (for most tests)

## Test Collection Results

### ✅ Successfully Collected (5 tests)

From `tests/deforum_test.py`:
1. `test_simple_settings` - Basic 2D animation with default settings
2. `test_api_cancel_active_job` - Job cancellation via API
3. `test_3d_mode` - 3D depth warping animation
4. `test_with_parseq_inline_without_overrides` - Parseq integration without overrides
5. `test_with_parseq_inline_with_overrides` - Parseq integration with overrides

### ⚠️ Collection Failed (5 tests)

From `tests/deforum_postprocess_test.py`:
- Tests fail during collection because they check for GPU at import time
- Connection refused error (server not running at localhost:7860)
- Tests in this file:
  1. `test_post_process_FILM` - FILM frame interpolation
  2. `test_post_process_RIFE` - RIFE v4.26 interpolation
  3. `test_post_process_UPSCALE` - RealESRGAN upscaling
  4. `test_post_process_UPSCALE_FILM` - Combined upscaling + interpolation
  5. (One more test likely exists)

## Issues Fixed

### MoviePy Version Mismatch
- **Problem:** Tests use `from moviepy.editor import VideoFileClip` (moviepy 1.x syntax)
- **Current:** MoviePy 2.1.2 installed (different import structure)
- **Solution:** Downgraded to moviepy <2.0 to match test expectations
- **Status:** ✅ Fixed

### Missing Dependencies
- **Problem:** Test dependencies not in main venv
- **Solution:** Installed from `requirements-dev.txt`
- **Status:** ✅ Fixed

## How to Run Tests

### ⭐ Option 1: Automated Test Runner (Recommended)
```bash
cd ~/workspace/stable-diffusion-webui-forge/extensions/sd-forge-deforum

# Run all tests (auto-starts server, runs tests, cleans up)
./run-tests.sh

# Quick mode (skip slow post-processing tests)
./run-tests.sh --quick

# Run specific test
./run-tests.sh tests/deforum_test.py::test_simple_settings
```

**Features:**
- ✅ Automatically starts Forge server with `--deforum-api`
- ✅ Waits for server to be ready (health checks)
- ✅ Installs test dependencies if missing
- ✅ Cleans up server on exit (even if tests fail)
- ✅ Detects if server already running (reuses it)
- ✅ Colored output for easy reading
- ✅ Server logs saved to `test-server.log`

### Option 2: Using Pytest's Built-in Server Start
```bash
cd ~/workspace/stable-diffusion-webui-forge
pytest extensions/sd-forge-deforum/tests/ --start-server
```

### Option 3: Manual Server Start
```bash
# Terminal 1: Start server
cd ~/workspace/stable-diffusion-webui-forge
python webui.py --deforum-api

# Terminal 2: Run tests
cd extensions/sd-forge-deforum
pytest tests/
```

## Known Issues

### 1. **No Unit Tests**
- All tests are heavyweight API integration tests
- Require full server + GPU + models
- Slow to run (minutes per test)
- Hard to debug when they fail

**Recommendation:** Build unit test suite (see TESTING.md)

### 2. **Tests Designed for A1111, Not Forge**
- Original test infrastructure assumes A1111 WebUI v1.6.0
- May encounter compatibility issues with Forge
- CI workflow disabled (`.github/workflows/run_tests.yaml.disabled`)

**Recommendation:** Verify tests work on Forge before re-enabling CI

### 3. **Missing Test Coverage**
No tests exist for:
- ❌ Flux/Wan video generation mode
- ❌ Experimental render core (now the only core)
- ❌ Keyframe distribution system
- ❌ Qwen prompt expansion
- ❌ Camera Shakify integration

**Recommendation:** Add integration tests for new features

### 4. **Server Dependency**
- Cannot run tests during local development without server
- Cannot run tests in CI without complex setup
- Tests are "all or nothing" (can't test individual functions)

**Recommendation:** Separate unit tests from integration tests

## Test Dependencies

From `requirements-dev.txt`:
```
coverage         # Code coverage reporting
syrupy          # Snapshot testing for SRT regression
pytest          # Test framework
tenacity        # Retry utilities
pydantic_requests  # API client
moviepy<2.0     # Video processing (pinned to 1.x for compatibility)
```

**Installation:**
```bash
pip install -r requirements-dev.txt
pip install "moviepy<2.0"  # Must pin to 1.x
```

## Next Steps

### Immediate (To Run Existing Tests)
1. Start Forge server with `python webui.py --deforum-api`
2. Run `pytest tests/deforum_test.py -v` to verify basic tests work
3. Document any failures or compatibility issues with Forge

### Short-term (Improve Reliability)
1. Update `requirements-dev.txt` to pin `moviepy<2.0`
2. Add server health check to test utilities
3. Add clear error messages when server isn't running

### Long-term (Build Better Test Infrastructure)
1. Create unit tests that don't require server (see TESTING.md)
2. Keep 3-5 lightweight integration tests for smoke testing
3. Set up CI to run unit tests on every commit
4. Set up nightly CI for integration tests

## Files

- Test suite: `tests/deforum_test.py:1`
- Post-processing tests: `tests/deforum_postprocess_test.py:1`
- Test utilities: `tests/utils.py:1`
- Pytest config: `tests/conftest.py:1`
- Test data: `tests/testdata/`
- Snapshots: `tests/__snapshots__/`
- Dev requirements: `requirements-dev.txt:1`
- Disabled CI: `.github/workflows/run_tests.yaml.disabled:1`
