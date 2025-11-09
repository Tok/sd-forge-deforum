# Depth Warping Orbit Tests

## Overview

Complete implementation of rotation factor optimization for depth warping orbital camera paths. This system empirically measures subject stability and temporal consistency across different rotation factors to find optimal translation/rotation balance.

## Architecture

### 1. Test Infrastructure (`tests/integration/test_depth_warping_orbit_tuning.py`)

**Core Functions:**
- `generate_orbit_schedules()` - Creates circular orbit with counter-rotation
- `measure_subject_position_drift()` - Tracks centroid movement (Otsu thresholding + image moments)
- `test_orbit_rotation_factor_sweep()` - Parametrized pytest test for CI/CD

**Metrics Measured:**
- **max_drift**: Maximum pixel distance from center
- **avg_drift**: Average drift across all frames
- **drift_rate**: Linear drift trend (pixels/frame)
- **temporal_consistency**: Frame-to-frame SSIM score
- **overall_score**: 60% drift reduction + 40% temporal quality

**Test Matrix:**
```python
@pytest.mark.parametrize("aspect_ratio,width,height,rotation_factor", [
    # Landscape (16:9)
    (16/9, 512, 288, -3.0), (16/9, 512, 288, -5.0), (16/9, 512, 288, -7.0),
    # Portrait (9:16)
    (9/16, 288, 512, -3.0), (9/16, 288, 512, -5.0), (9/16, 288, 512, -7.0),
    # Square (1:1)
    (1.0, 512, 512, -5.0),
])
```

### 2. API Backend (`deforum/api/tuning_api.py`)

**Enhancements:**
- Added `TuningTestType.DEPTH_WARPING_ORBIT` enum
- Extended `TuningTestConfig` with orbit parameters:
  - `aspect_ratios`: List of (ratio, width, height) tuples
  - `rotation_factor_min/max/step`: Sweep range
  - `orbit_radius`: Orbit size (20-100px)
  - `orbit_iterations`: I2I depth warp frames (10-40)

**Pipeline:**
1. `_run_test()` routes to `_run_orbit_tests()`
2. `_run_orbit_tests()` sweeps aspect ratios × rotation factors
3. `_run_orbit_single_test()` runs each configuration:
   - Generates orbit schedules
   - Submits 3D animation job via Deforum API
   - Measures drift and temporal metrics
   - Returns comprehensive results dict

**API Endpoints:**
```bash
POST /deforum_api/tuning/start
{
  "test_type": "depth_warping_orbit",
  "aspect_ratios": [[1.777, 512, 288]],  # 16:9
  "rotation_factor_min": -7.0,
  "rotation_factor_max": -3.0,
  "rotation_factor_step": 1.0,
  "orbit_radius": 50.0,
  "orbit_iterations": 20
}

GET /deforum_api/tuning/{test_id}
POST /deforum_api/tuning/{test_id}/cancel
```

### 3. Visualization (`deforum/ui/tuning_charts.py`)

**New Functions:**

**`create_orbit_metrics_plot()`** - Dual line plots
- Top: Drift vs rotation factor (lower is better)
- Bottom: Overall score vs rotation factor (higher is better)
- Color-coded by aspect ratio (slopcore gradient)
- Inverted x-axis (more negative factors on right)

**`create_orbit_heatmap()`** - 2D parameter space
- Rows: Aspect ratios (16:9, 9:16, 1:1)
- Columns: Rotation factors (-3.0 to -7.0)
- Adaptive colormap: inverted for drift, normal for scores
- Text annotations for exact values

**`find_best_orbit_configuration()`** - Optimal parameter selection
- Maximizes overall_score (lowest drift + highest temporal)

**`generate_orbit_summary_stats()`** - Statistical summary
- Best/worst/avg overall scores
- Drift statistics (avg, min, max)
- Standard deviation

### 4. UI Integration (`deforum/ui/ui_tuning.py`)

**Test Type Dropdown:**
```
- Color Preservation (I2V Chaining)
- Temporal Consistency (Frame Stability)
- Flux Parameter Sweep
- Depth Warping Orbit (Translation/Rotation Factor)  ← NEW
```

**Orbit Controls:**
- **Aspect Ratios**: Checkboxes for 16:9, 9:16, 1:1
- **Rotation Factor Range**: Min/max/step sliders (-10.0 to -1.0)
- **Orbit Radius**: 20-100px slider
- **I2I Iterations**: 10-40 frames slider

**Dynamic UI:**
- Orbit controls visible only when "Depth Warping Orbit" selected
- Standard I2V controls hidden for orbit tests
- Automatic chart routing based on result structure

## Usage

### 1. Launch Tuning Lab

```bash
# Linux/Mac
./run-tuning-lab.sh

# Windows
run-tuning-lab.bat
```

### 2. Configure Orbit Test

1. Navigate to **Tuning** tab in WebUI
2. Select **"Depth Warping Orbit"** test type
3. Configure parameters:
   - Check desired aspect ratios (e.g., 16:9 Landscape)
   - Set rotation factor range (-7.0 to -3.0)
   - Set step size (1.0 for quick sweep, 0.5 for detailed)
   - Set orbit radius (50px recommended)
   - Set iterations (20 recommended)

### 3. Run Tests

1. Click **"Run Tests"** button
2. Monitor status box for progress
3. Click **"Refresh Results"** periodically
4. View charts and results table

### 4. Interpret Results

**Line Plots (Metrics tab):**
- **Drift curve**: Look for lowest point (optimal factor)
- **Score curve**: Look for peak (highest quality)
- **Intersection**: Best balance of stability + quality

**Heatmap (Charts tab):**
- **Dark purple**: Poor quality (high drift or low score)
- **Bright cyan**: Excellent quality (low drift or high score)
- **Optimal zones**: Bright regions in heatmap

**Results Table (Summary tab):**
- Sort by `overall_score` descending
- Check `max_drift` for stability
- Verify `temporal_consistency` for smoothness

**Best Parameters (Summary tab JSON):**
```json
{
  "aspect_ratio": 1.78,
  "width": 512,
  "height": 288,
  "rotation_factor": -5.0,
  "orbit_radius": 50.0,
  "max_drift": 12.3,
  "overall_score": 87.4
}
```

## Expected Results

### Hypothesis
**Optimal rotation factor ≈ -5.0** for most aspect ratios

### Reasoning
- Too weak (-3.0): Insufficient counter-rotation, subject drifts outward
- Too strong (-7.0): Excessive counter-rotation, subject drifts inward
- Balanced (-5.0): Translation and rotation cancel, subject stays centered

### Validation
Empirical testing across:
- 3 aspect ratios (16:9, 9:16, 1:1)
- 5 rotation factors (-3.0, -4.0, -5.0, -6.0, -7.0)
- 20 I2I depth warp iterations per test
- = 15 total configurations

## Integration with Camera Path Generator

Once optimal factors identified:

1. Update preset defaults in `deforum/ui/handlers/camera_path_generator.py`
2. Apply to `generate_rotate_around_path()` calls
3. Document optimal values in `CLAUDE.md`

## Related Components

**Tennis Ball Seam Fix** (`deforum/utils/spline_camera_path.py:256-390`)
- Implemented adaptive curve-following look-at
- Eliminates sideways drift in rotate-around paths
- Complements rotation factor optimization

**Frame Overlap Simulator** (`deforum/utils/frame_overlap_canvas.py`)
- Real-time preview of frame overlap during camera movement
- Validates depth warping viability before full render

**Tuning Lab Launcher** (`run-tuning-lab.sh`, `run-tuning-lab.bat`)
- Simple wrapper to start WebUI with `--deforum-run-tuning` flag
- Enables Tuning tab in WebUI

## Files Modified

```
deforum/api/tuning_api.py           +334, -54   API backend implementation
deforum/ui/tuning_charts.py         +195, -0    Orbit visualization functions
deforum/ui/ui_tuning.py             +11, -4     UI routing and auto-detection
tests/integration/test_depth_warping_orbit_tuning.py  (already existed)
```

## Future Enhancements

1. **Adaptive Factor Selection**: Auto-adjust based on aspect ratio
2. **Multi-Objective Optimization**: Pareto frontier analysis
3. **Real-time Preview**: Live drift tracking during generation
4. **Batch Comparison**: Side-by-side video comparison tool
5. **Export to Parseq**: Apply optimal factors to Parseq JSON
