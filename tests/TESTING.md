# Testing Guide

## Test Organization

### Directory Structure

```
tests/
├── unit/           # Fast unit tests (run in CI)
├── integration/    # Slow integration tests (local only)
└── TESTING.md      # This file
```

### Test Markers

Tests can be marked with pytest markers to control when/where they run:

| Marker | Purpose | CI | Local |
|--------|---------|-----|-------|
| `slow` | Long-running tests that generate outputs | ❌ Skip | ✅ Run with `--run-slow` |
| `skip_ci` | Tests requiring heavy dependencies | ❌ Skip | ✅ Run |
| `flux` | Tests requiring Flux model | ❌ Skip | ✅ Run if model available |
| `wan` | Tests requiring Wan model | ❌ Skip | ✅ Run if model available |
| `visual` | Tests that verify visual processing | ✅ Run | ✅ Run |

## Running Tests

### Local Development

**Run all unit tests (including skip_ci):**
```bash
pytest tests/unit/ -v
```

**Run only CI-compatible tests (no heavy dependencies):**
```bash
pytest tests/unit/ -v -m "not skip_ci"
```

**Run specific test file:**
```bash
pytest tests/unit/test_da3_3dgs.py -v
```

**Run with coverage:**
```bash
pytest tests/unit/ -v --cov=deforum/utils --cov-report=term
```

### CI Environment (GitHub Actions)

CI automatically skips tests marked with `skip_ci`:
```bash
pytest tests/unit/ -v -m "not skip_ci" --junitxml=test-results.xml
```

## Tests Requiring Heavy Dependencies

These tests are marked with `@pytest.mark.skip_ci` and **run locally but skip in CI**:

### DA3-3DGS Tests (`test_da3_3dgs.py`)
**Dependencies:** `transformers`, `gsplat`, `torch`
**Reason:** Not installed in CI (adds ~2GB to dependencies)
**Tests:** 39 tests for 3D Gaussian Splatting functionality

```python
# Entire module marked to skip in CI
pytestmark = pytest.mark.skip_ci
```

### Why Skip in CI?

1. **Heavy Dependencies:**
   - `transformers`: ~500MB (Hugging Face models)
   - `gsplat`: Requires CUDA compilation
   - Not needed for most unit tests

2. **CI Performance:**
   - Faster test runs (skip expensive setup)
   - Smaller Docker images
   - Reduced bandwidth usage

3. **Local Development:**
   - Full test suite runs locally where deps are installed
   - Developers can verify all functionality before pushing

## Adding New Tests

### Unit Tests (tests/unit/)

**Should be:**
- Fast (<1 second per test)
- Isolated (no external dependencies)
- Deterministic (same input = same output)

**Example:**
```python
def test_calculate_interpolation():
    """Test frame interpolation calculation."""
    result = interpolate_frames(start=0, end=10, count=5)
    assert result == [0, 2.5, 5.0, 7.5, 10.0]
```

### Tests with Heavy Dependencies

**Mark with `skip_ci`:**
```python
import pytest

@pytest.mark.skip_ci
def test_depth_estimation_with_da3():
    """Test depth estimation using DA3 model."""
    # This test requires transformers, which isn't in CI
    from transformers import pipeline
    # ... test code
```

**Or mark entire module:**
```python
import pytest

# Skip all tests in this file in CI
pytestmark = pytest.mark.skip_ci

def test_feature_one():
    # Runs locally, skipped in CI
    pass

def test_feature_two():
    # Runs locally, skipped in CI
    pass
```

### Integration Tests (tests/integration/)

**Should be:**
- Comprehensive (test full workflows)
- Generate real outputs (images, videos)
- May require models, network access

**Location:** `tests/integration/` (never run in CI)

## Test Development Workflow

1. **Write unit test** in `tests/unit/`
2. **Add heavy dependency?** Mark with `@pytest.mark.skip_ci`
3. **Run locally:** `pytest tests/unit/test_myfeature.py -v`
4. **Verify CI compatibility:** `pytest tests/unit/ -m "not skip_ci"`
5. **Push:** CI runs lightweight tests, local devs run full suite

## Continuous Integration

### GitHub Actions Workflow

Located in `.github/workflows/unit-tests.yml`

**What runs in CI:**
- All tests in `tests/unit/`
- **Except:** Tests marked with `skip_ci`
- Python 3.10 (to match WebUI Forge)
- Coverage reports uploaded to Codecov

**Dependencies installed in CI:**
- Core: `numpy`, `torch`, `torchvision`, `opencv-python-headless`
- ML: `pandas`, `pillow`, `einops`, `scikit-image`
- Audio: `scipy`, `librosa`, `soundfile`
- UI: `gradio`
- Utils: `numexpr`, `matplotlib`, `av`, `pims`, `rich`

**NOT installed in CI:**
- `transformers` (too large, DA3 tests only)
- `gsplat` (requires CUDA, 3DGS tests only)
- `diffusers` (full Forge environment, integration tests only)

## Best Practices

### ✅ Do

- Write fast, isolated unit tests
- Mock heavy dependencies when possible
- Mark tests requiring large libraries with `skip_ci`
- Run full test suite locally before pushing
- Use descriptive test names and docstrings

### ❌ Don't

- Import `transformers` in tests that don't need it
- Generate large files in unit tests (use integration tests)
- Skip tests just because they're slow (use `@pytest.mark.slow` instead)
- Test implementation details (test behavior, not internals)

## Debugging Failed Tests

### Locally

**Run with verbose traceback:**
```bash
pytest tests/unit/test_failing.py -vv --tb=long
```

**Run specific test:**
```bash
pytest tests/unit/test_failing.py::TestClass::test_method -vv
```

**Drop into debugger on failure:**
```bash
pytest tests/unit/test_failing.py --pdb
```

### In CI

1. Check GitHub Actions logs for test output
2. Look for "FAILED" markers and error messages
3. Reproduce locally with same Python version (3.10)
4. Run CI-compatible tests: `pytest tests/unit/ -m "not skip_ci"`

## Coverage Reports

Coverage is tracked for `deforum/utils/` only (core utilities).

**Generate HTML report:**
```bash
pytest tests/unit/ --cov=deforum/utils --cov-report=html
open htmlcov/index.html
```

**Coverage uploaded to Codecov on every CI run.**

## Questions?

- Check existing tests for examples
- Review pytest documentation: https://docs.pytest.org/
- Ask in GitHub Issues or Discussions
