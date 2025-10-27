# Deforum Parameter Tuning Tests

**GPU-Required Test Suite for Automated Parameter Optimization**

This test suite uses empirical quality metrics to find optimal Deforum parameters through automated generation and analysis.

## ⚠️ Important Limitations

- **Requires GPU**: Cannot run in CI/CD (no GPU available)
- **Requires Forge API**: Must have WebUI running with `--deforum-api`
- **Slow**: Each test generates multiple images/videos
- **Local only**: Meant for developer tuning, not CI validation

## Running Tuning Tests

### Option 1: With Tuning UI (Recommended)

```bash
# Launch Forge with tuning mode
python webui.py --deforum-run-tuning

# This will:
# - Enable Deforum API automatically
# - Show "Tuning" tab in WebUI
# - Allow interactive parameter exploration
```

### Option 2: Batch Mode (Command Line)

```bash
# Start Forge with API
python webui.py --deforum-api

# In another terminal, run tuning tests
cd extensions/sd-forge-deforum
pytest tests/tuning/ -v --start-server=false

# Or use the helper script
./run-tuning-tests.sh
```

## Test Categories

### 1. Color Preservation Tests (`test_color_preservation.py`)

Measures how many iterations of I2V chaining it takes before colors degrade to grayscale.

**Metrics:**
- Color saturation decay rate
- Grayscale conversion threshold
- Optimal strength to prevent color loss

**Parameters Tested:**
- `strength_schedule` (normal/tween frames): 0.5 → 0.95
- `keyframe_strength_schedule`: 0.0 → 0.5
- `steps`: 4, 8, 12, 16, 20

### 2. Temporal Consistency Tests (`test_temporal_consistency.py`)

Measures frame-to-frame stability and jitter.

**Metrics:**
- SSIM (Structural Similarity Index)
- Perceptual hash distance
- Optical flow variance

**Parameters Tested:**
- Same as color preservation

### 3. Flux-Specific Tests (`test_flux_parameters.py`)

Evaluates Flux Dev vs Schnell with different strength values.

**Focus:**
- Can Flux Schnell (4 steps) be viable with optimal strength?
- What's the ideal strength for Flux Dev (20 steps)?
- Steps vs strength trade-offs

## Test Output

Results are saved to `outputs/deforum-tuning/`:

```
outputs/deforum-tuning/
├── color_preservation/
│   ├── strength_0.50_steps_20/
│   │   ├── iteration_001.png
│   │   ├── iteration_010.png
│   │   └── metrics.json
│   └── comparison_grid.html
├── temporal_consistency/
│   └── ...
└── summary.json
```

## Quality Metrics

### Color Preservation Score (0-100)

```python
def measure_color_preservation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].mean() / 255.0
    return saturation * 100  # 100 = full color, 0 = grayscale
```

### Temporal Consistency Score (0-100)

```python
from skimage.metrics import structural_similarity as ssim

def measure_temporal_consistency(frame1, frame2):
    score = ssim(frame1, frame2, multichannel=True)
    return score * 100  # 100 = identical, 0 = completely different
```

## Expected Results

Based on theoretical understanding:

**Flux Dev (20 steps):**
- Normal strength: 0.80-0.90 (high stability for tweens)
- Keyframe strength: 0.10-0.20 (low for dramatic changes)

**Flux Schnell (4 steps):**
- May require lower strengths due to fewer steps
- Needs empirical validation

## Integration with WebUI

When `--deforum-run-tuning` is active:

1. **Tuning Tab appears** in main UI
2. **Interactive controls** for parameter sweeps
3. **Real-time metrics** displayed as charts
4. **One-click parameter application** to update defaults

## Development Notes

- Tests use the same API infrastructure as `tests/integration/`
- Share utilities with `tests/integration/utils.py`
- Results can inform default values in `deforum/config/args.py`
- HTML comparison grids for visual validation
