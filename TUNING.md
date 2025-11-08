# Deforum Parameter Tuning System

**Automated, Empirical Parameter Optimization Through Quality Metrics**

## Overview

This system provides automated testing to find optimal Deforum parameters through scientific measurement of generation quality. Instead of guessing parameter values, we use empirical metrics to measure:

1. **Color Preservation** - How many I2V iterations before colors degrade to grayscale?
2. **Temporal Consistency** - Are frames stable or jittery?
3. **Overall Quality** - Combined weighted score across all metrics

## Quick Start

### Option 1: Run Tuning Tests (Command Line)

```bash
# Start Forge with API
python webui.py --deforum-api

# In another terminal, run tuning tests
cd extensions/sd-forge-deforum
./run-tuning-tests.sh --reuse-server

# Or let the script start the server
./run-tuning-tests.sh --start-server
```

### Option 2: Interactive Tuning Mode (Coming Soon)

```bash
# Launch Forge with tuning UI
python webui.py --deforum-run-tuning

# Opens WebUI with "Tuning" tab for interactive parameter exploration
```

## Current Status

### ✅ Implemented

- [x] CLI flag `--deforum-run-tuning`
- [x] Quality metrics module with 3 core metrics
- [x] Color preservation test (I2V chaining degradation)
- [x] Unit tests for all metrics (10/10 passing)
- [x] Helper scripts (run-tuning-tests.sh/bat)
- [x] Comprehensive documentation

### 🚧 In Progress

- [ ] WebUI tuning tab (interactive parameter exploration)
- [ ] HTML comparison grid generator
- [ ] Results analysis and visualization
- [ ] Run comprehensive parameter sweeps
- [ ] Update defaults based on empirical findings

## How It Works

### 1. Color Preservation Test

**Goal:** Find strength values that prevent color washout during I2V chaining.

**Method:**
1. Generate vibrant rainbow gradient test image
2. Feed it into Flux with specific strength values
3. Use output as init for next iteration
4. Repeat until colors degrade to grayscale (saturation < 20)
5. Measure: iterations until degradation, color score trajectory

**Parameters Tested (Updated with Fractional Precision):**

### Flux Dev (20 steps) - Fine-Grained Sweep

Now using 0.01 (1%) precision instead of 0.05 (5%) thanks to fractional t_enc!

| Model | Steps | Normal Strength | Keyframe Strength | t_enc (actual) | Notes |
|-------|-------|----------------|-------------------|----------------|-------|
| Flux Dev | 20 | 0.85 | 0.20 | 3.0 / 16.0 | Current defaults |
| Flux Dev | 20 | 0.87 | 0.18 | 2.6 / 16.4 | Fractional test +2% |
| Flux Dev | 20 | 0.83 | 0.22 | 3.4 / 15.6 | Fractional test -2% |
| Flux Dev | 20 | 0.90 | 0.15 | 2.0 / 17.0 | High stability baseline |
| Flux Dev | 20 | 0.80 | 0.25 | 4.0 / 15.0 | Low stability baseline |
| Flux Dev | 20 | 0.92 | 0.12 | 1.6 / 17.6 | Very high stability |
| Flux Dev | 20 | 0.88 | 0.16 | 2.4 / 16.8 | Fine-tune high |
| Flux Dev | 20 | 0.82 | 0.24 | 3.6 / 15.2 | Fine-tune low |

### Flux Schnell (4 steps) - Comprehensive I2V Chaining Evaluation

**PRIMARY GOAL:** Determine if Flux Schnell can sustain I2V chaining at 4 steps.

With fractional precision, we can now test fine-grained strength values:

| Model | Steps | Normal Strength | Keyframe Strength | t_enc (actual) | Expected Behavior |
|-------|-------|----------------|-------------------|----------------|-------------------|
| Flux Schnell | 4 | 0.85 | 0.25 | 0.6 / 3.0 | Baseline (likely unstable) |
| Flux Schnell | 4 | 0.90 | 0.20 | 0.4 / 3.2 | Higher stability test |
| Flux Schnell | 4 | 0.92 | 0.18 | 0.32 / 3.28 | **Fractional precision test** |
| Flux Schnell | 4 | 0.88 | 0.22 | 0.48 / 3.12 | **Fractional precision test** |
| Flux Schnell | 4 | 0.93 | 0.15 | 0.28 / 3.4 | Very high stability (may be too rigid) |
| Flux Schnell | 4 | 0.87 | 0.23 | 0.52 / 3.08 | **Fractional precision test** |
| Flux Schnell | 4 | 0.95 | 0.10 | 0.2 / 3.6 | Maximum stability (test limit) |
| Flux Schnell | 4 | 0.80 | 0.30 | 0.8 / 2.8 | Lower stability (viability check) |
| Flux Schnell | 4 | 0.83 | 0.27 | 0.68 / 2.92 | **Fractional precision test** |
| Flux Schnell | 4 | 0.77 | 0.33 | 0.92 / 2.68 | **Fractional precision test** |

**Test Objectives:**
1. Find optimal fractional strength values for stable 4-step I2V chaining
2. Measure color preservation across 20+ iterations
3. Identify at what strength values Schnell becomes viable for production
4. Compare quality/speed tradeoff vs Flux Dev (20 steps)

**Success Criteria:**
- Sustain 10+ I2V iterations without color degradation
- Temporal consistency > 70
- Overall quality score > 65

**Total:** 18 parameter combinations (8 Flux Dev + 10 Flux Schnell)

### 2. Quality Metrics

#### Color Preservation Score (0-100)

Measures HSV saturation to detect grayscale degradation.

```python
def measure_color_preservation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].mean() / 255.0
    return saturation * 100  # 100 = full color, 0 = grayscale
```

#### Temporal Consistency Score (0-100)

Uses SSIM (Structural Similarity Index) for frame-to-frame stability.

```python
from skimage.metrics import structural_similarity as ssim

def measure_temporal_consistency(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    score = ssim(gray1, gray2)
    return score * 100  # 100 = identical, 0 = completely different
```

#### Comprehensive Quality Score

Weighted combination with degradation rate:

```python
overall_score = (
    0.5 * avg_color_score +
    0.5 * avg_temporal_score
)

# Linear regression to find degradation rate
degradation_rate = abs(slope_of_color_scores)
```

### 3. Test Output

Results saved to `outputs/deforum-tuning/`:

```
outputs/deforum-tuning/
└── color_preservation/
    ├── steps20_norm0.85_kf0.15/
    │   ├── iteration_000_color95.png
    │   ├── iteration_001_color92.png
    │   ├── iteration_010_color45.png
    │   ├── iteration_015_color18.png  (grayscale threshold)
    │   └── metrics.json
    ├── steps20_norm0.90_kf0.15/
    │   └── ...
    └── comparison_grid.html  (coming soon)
```

**metrics.json structure:**

```json
{
  "parameters": {
    "steps": 20,
    "normal_strength": 0.85,
    "keyframe_strength": 0.15
  },
  "iterations_completed": 15,
  "iterations_until_grayscale": 15,
  "color_scores": [95.2, 92.1, 87.3, ..., 18.4],
  "metrics": {
    "overall_score": 68.5,
    "avg_color": 62.3,
    "avg_temporal": 74.7,
    "degradation_rate": 4.2
  }
}
```

## Understanding the Dual Strength System

Deforum's "New 3D" mode uses two strength schedules:

### Normal Strength (Tween Frames)

- **Purpose:** Stability between keyframes
- **Default:** 0.85 (85% previous frame influence)
- **Effect:** Higher = more stability, less drift
- **Formula:** `actual_steps = steps - (strength * steps)`
  - 0.85 strength @ 20 steps = 3 diffusion steps (20 - 17)

### Keyframe Strength (Keyframes with Prompts)

- **Purpose:** Dramatic changes at prompt boundaries
- **Default:** 0.15 (15% previous frame influence)
- **Effect:** Lower = more creativity, bigger changes
- **Formula:** Same as above
  - 0.15 strength @ 20 steps = 17 diffusion steps (20 - 3)

### The Paradox

- **Problem:** With only 4 steps (Schnell), even 0.15 strength = 3.4 effective steps
- **Question:** Can Schnell work with optimal strength tuning?
- **This Testing Will Answer:** What strength values make Schnell viable?

## Expected Findings

Based on theoretical understanding:

### Flux Dev (20 steps) - Hypothesis

| Metric | Predicted Optimal | Reasoning |
|--------|-------------------|-----------|
| Normal Strength | 0.80-0.90 | Balance stability vs freshness |
| Keyframe Strength | 0.10-0.20 | Enough steps for changes (18-20) |
| Iterations Until Grayscale | 15-20 | Good color preservation |

### Flux Schnell (4 steps) - Hypothesis

| Metric | Predicted Optimal | Reasoning |
|--------|-------------------|-----------|
| Normal Strength | 0.60-0.70 | Lower to get 1-2 effective steps |
| Keyframe Strength | 0.30-0.40 | Still need some steps (2-3) |
| Iterations Until Grayscale | 5-10 | May degrade faster |
| **Viability** | **Unknown** | This testing will determine |

## Running Custom Tests

### Test a Specific Parameter

```bash
# Run just one parameter combination
./run-tuning-tests.sh tests/tuning/test_color_preservation.py::test_color_preservation_sweep[20-0.85-0.15]
```

### Test Flux Schnell Only

```bash
# Run all Schnell tests (4 step variants)
./run-tuning-tests.sh tests/tuning/test_color_preservation.py -k "steps4"
```

### Add Your Own Parameters

Edit `tests/tuning/test_color_preservation.py`:

```python
@pytest.mark.parametrize("steps,normal_strength,keyframe_strength", [
    # Add your custom parameter here
    (20, 0.87, 0.13),  # Your hypothesis
])
def test_color_preservation_sweep(steps, normal_strength, keyframe_strength):
    ...
```

## Next Steps

### 1. Run Comprehensive Sweep (Estimate: 2-3 hours)

```bash
./run-tuning-tests.sh tests/tuning/test_color_preservation.py --reuse-server
```

This will test all 10 parameter combinations and generate detailed metrics.

### 2. Analyze Results

Compare `outputs/deforum-tuning/color_preservation/*/metrics.json` to find:
- Which parameters preserve color longest?
- What degradation rate is acceptable?
- Is Schnell viable with any strength values?

### 3. Update Defaults

Based on empirical findings, update `deforum/config/args.py`:

```python
"strength_schedule": {
    "value": "0: (0.87)",  # Updated from 0.85 based on testing
    ...
},
"keyframe_strength_schedule": {
    "value": "0: (0.13)",  # Updated from 0.15 based on testing
    ...
},
```

### 4. Build WebUI Tab (Coming Next)

Interactive tool for:
- Running custom parameter sweeps
- Viewing results in real-time
- Comparing parameter combinations side-by-side
- One-click parameter application

## Technical Details

### Dependencies

- `opencv-python` - Color space conversions, image processing
- `scikit-image` - SSIM metric
- `PIL` - Image loading/saving
- `numpy` - Numerical operations

Install: `pip install opencv-python scikit-image`

### Test Architecture

- Uses same API infrastructure as `tests/integration/`
- Shares utilities from `tests/integration/utils.py`
- GPU-required (cannot run in CI)
- Slow (each test generates 5-20 images)

### Metrics Validation

All metrics have unit tests:

```bash
# Run metrics unit tests (fast, no GPU needed)
pytest tests/tuning/test_metrics.py -v

# Results: 10/10 passing
```

## Limitations

1. **GPU Required** - Cannot run in CI/CD
2. **Slow** - Each parameter takes 2-10 minutes
3. **Subjective Elements** - Metrics may not capture all quality aspects
4. **Model-Specific** - Results only valid for Flux Dev/Schnell
5. **Single Test Image** - Rainbow gradient may not represent all use cases

## Contributing

To add new metrics or tests:

1. Add metric function to `tests/tuning/metrics.py`
2. Add unit tests to `tests/tuning/test_metrics.py`
3. Create test file in `tests/tuning/test_<feature>.py`
4. Update this document with findings

## References

- **SSIM Paper:** Wang et al. (2004) "Image Quality Assessment: From Error Visibility to Structural Similarity"
- **Perceptual Hashing:** Zauner (2010) "Implementation and Benchmarking of Perceptual Image Hash Functions"
- **Deforum Docs:** See `/CLAUDE.md` for Deforum architecture details
