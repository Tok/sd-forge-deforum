# Deforum Parameter Tuning Tests

**GPU-Required Test Suite for Automated Parameter Optimization**

This test suite uses empirical quality metrics to find optimal Deforum parameters through automated generation and analysis.

## ⚠️ Important Limitations

- **Requires GPU**: Cannot run in CI/CD (no GPU available)
- **Requires Forge API**: Must have WebUI running with `--deforum-api`
- **Slow**: Each test generates multiple images/videos
- **Local only**: Meant for developer tuning, not CI validation

## Running Tuning Tests

### Option 1: With Tuning UI (Recommended) ✅ IMPLEMENTED

```bash
# Launch Forge with tuning mode
python webui.py --deforum-run-tuning

# This will:
# - Auto-enable Deforum API (no need for --deforum-api flag)
# - Show "Deforum Tuning" tab in WebUI
# - Allow interactive parameter exploration
# - Execute real GPU-based quality tests
```

**Features:**
- **Interactive parameter selection** - Choose steps, strength ranges, test limits
- **Real-time progress tracking** - Auto-refresh every 2 seconds
- **Quality metrics visualization** - Bar charts comparing configurations
- **Parameter heatmaps** - Visual grid showing optimal parameter combinations
- **Results table** - Sortable table of all test configurations
- **Best parameter detection** - Automatically identifies optimal settings
- **One-click apply** - Apply best parameters to Deforum defaults (coming soon)

**Usage:**
1. Start Forge with `--deforum-run-tuning` flag
2. Navigate to the "Tuning" tab in WebUI
3. Select test type (Color Preservation, Temporal Consistency, or Flux Parameter Sweep)
4. Configure parameter ranges (steps, strengths, etc.)
5. Click "🚀 Run Tests" to start
6. Monitor progress in real-time
7. Review results in Summary, Charts, and Image Comparison tabs
8. Apply best settings when satisfied

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

### 4. Depth Warping Orbit Tuning (`test_depth_warping_orbit_tuning.py`) ✅ IMPLEMENTED

Sweeps translation/rotation factors for orbital camera paths with depth warping.

**Goal:** Find optimal rotation factor that keeps subject centered during ~20 I2I depth warp iterations.

**Metrics:**
- Subject position drift (centroid tracking with Otsu thresholding)
- Temporal consistency (frame-to-frame SSIM)
- Depth map consistency
- Color preservation

**Parameters Tested:**
- Aspect ratios: 16:9 (landscape), 9:16 (portrait), 1:1 (square)
- Rotation factors: -3.0, -4.0, -5.0, -6.0, -7.0
- Total: 13 parameter combinations

**Usage:**
```bash
# Run all rotation factor sweeps
./run-tuning-tests.sh tests/tuning/test_depth_warping_orbit_tuning.py

# Run specific aspect ratio
./run-tuning-tests.sh tests/tuning/test_depth_warping_orbit_tuning.py -k "16/9"

# With auto server management
./run-tuning-tests.sh --start-server tests/tuning/test_depth_warping_orbit_tuning.py
```

**Output:** `outputs/deforum-tuning/depth_warping_orbits/aspect{ratio}_{W}x{H}_factor{N}/`

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

## Implementation Details

### Architecture

The tuning system consists of three main components:

1. **UI Layer** (`deforum/ui/ui_tuning.py`)
   - Gradio-based interface with tabs for configuration, results, charts, and logs
   - Auto-refresh functionality polls API every 2 seconds
   - Interactive parameter selection with sliders and dropdowns

2. **API Layer** (`deforum/api/tuning_api.py`)
   - RESTful endpoints for starting, monitoring, and cancelling tests
   - `POST /deforum_api/tuning/start` - Submit new test configuration
   - `GET /deforum_api/tuning/{test_id}` - Get test status and results
   - `POST /deforum_api/tuning/{test_id}/cancel` - Cancel running test
   - Background thread execution for non-blocking test runs

3. **Visualization Layer** (`deforum/ui/tuning_charts.py`)
   - Matplotlib-based chart generation
   - Metrics comparison bar charts
   - Parameter heatmaps for optimal configuration identification
   - Degradation rate analysis plots

### Integration Points

- **CLI Flag:** `--deforum-run-tuning` in `preload.py:143`
- **Tab Registration:** `scripts/deforum.py:75-87` (conditional on flag)
- **API Registration:** `deforum/api/api.py:676-679` (auto-enabled with tuning)
- **Test Infrastructure:** Uses existing metrics from `tests/tuning/metrics.py`

### Current Status

**✅ Implemented:**
- Tuning UI with all tabs and controls
- API endpoints for test management
- Real-time status polling
- Chart visualization (metrics, heatmaps)
- Parameter sweep logic
- **Real test execution** - Calls actual `test_color_preservation.py` functions
- I2V chaining with quality metrics measurement
- Automatic test image generation

**🚧 In Progress:**
- Image comparison gallery
- Apply best parameters functionality

**📋 TODO:**
- Generate HTML comparison grids
- Persist test results to database
- Export results as CSV/JSON
- Add temporal consistency test type
- Add Flux parameter sweep test type
