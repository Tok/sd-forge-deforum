# I2V Chaining Test Plan - Color Preservation & Temporal Consistency

## Overview

This test measures **BOTH color preservation AND temporal consistency** in a single comprehensive test. It evaluates how well different strength parameters maintain quality through cascading img2img (I2V) chaining.

## What It Measures

The test runs **20 iterations** of img2img chaining (output → input feedback loop) and measures:

1. **Color Preservation** - Color saturation score (0-100)
   - Measures how long vibrant colors persist before degrading to grayscale
   - Higher = better color retention

2. **Temporal Consistency** - Frame-to-frame SSIM (0-100)
   - Measures structural similarity between consecutive frames
   - Higher = smoother transitions, less jitter

3. **Degradation Rate** - Quality decay slope (%/iteration)
   - Linear regression of color scores over iterations
   - Lower = slower quality degradation

4. **Overall Quality Score** - Weighted average (0-100)
   - Combines color (50%) + temporal (50%)
   - Single metric for comparing configurations

## Test Methodology

**Input:** Colorful rainbow gradient image with text (512×512)

**Process:**
1. Generate 30 frames using img2img
2. Take the LAST frame (frame 29) as output
3. Feed it back as INPUT for next iteration
4. Repeat until 20 iterations OR color score < 20 (grayscale threshold)

**Why 30 frames per iteration?**
- With New 3D mode (REDISTRIBUTED):
  - Frame 0: Keyframe (uses keyframe_strength)
  - Frames 1-28: Mostly cadence frames (use **normal_strength**) ← PRIMARY TEST TARGET
  - Frame 29: Keyframe (uses keyframe_strength)
- ~27/30 frames use normal_strength, so this tests **cadence stability**

## Parameters to Test

### 1. Sampling Steps
- **20 (Flux Dev)** - Recommended for best quality
  - 1/20 = 0.05 strength resolution
  - Fine control over diffusion amount
- **4 (Flux Schnell)** - Fast but coarse
  - 1/4 = 0.25 strength resolution
  - Limited precision, harder to tune

**Recommendation:** Start with 20 steps, test Schnell later for speed comparison

### 2. Normal Strength (Cadence Frame Stability)

**Meaning:** How much to preserve from previous frame (INVERTED from standard img2img)
- **Higher (0.85-0.95):** More preservation, fewer diffusion steps, more stable
- **Lower (0.75-0.80):** Less preservation, more diffusion steps, more variation

**Deforum Inversion:**
```python
forge_strength = 1 - deforum_strength
# deforum=0.85 → forge=0.15 → 3/20 steps (high preservation)
# deforum=0.80 → forge=0.20 → 4/20 steps (moderate preservation)
```

**Recommended Test Range:** 0.80 to 0.95, step 0.01
- Uses fractional strength (0.01 = 1% precision)
- 16 test values: 0.80, 0.81, 0.82, ..., 0.95

### 3. Keyframe Strength (Keyframe Change Amount)

**Meaning:** How much diffusion at keyframes (lower = more change)
- **Lower (0.10-0.15):** More diffusion, more dramatic changes
- **Higher (0.20-0.25):** Less diffusion, more subtle changes

**Recommended Test Range:** 0.10 to 0.25, step 0.01
- 16 test values: 0.10, 0.11, 0.12, ..., 0.25

### 4. Test Limits
- **Max iterations:** 20 (good coverage of degradation curve)
- **Grayscale threshold:** 20 (stops when color < 20/100)

## Recommended Test Configuration

### Phase 1: Normal Strength Sweep (Primary)

**Goal:** Find optimal cadence stability

**Fixed Parameters:**
- Steps: 20 (Flux Dev)
- Keyframe strength: 0.15 (balanced)
- Max iterations: 20

**Sweep Parameters:**
- Normal strength: 0.80 to 0.95, step 0.01
- **Total tests:** 16 configurations

**Expected Results:**
- Higher values (0.90-0.95): Better color preservation, longer until grayscale
- Lower values (0.80-0.85): Faster color degradation
- Optimal likely around 0.88-0.92 (balance of stability and quality)

### Phase 2: Keyframe Strength Sweep (Secondary)

**Goal:** How keyframe strength affects retention

**Fixed Parameters:**
- Steps: 20
- Normal strength: 0.85 (or best from Phase 1)
- Max iterations: 20

**Sweep Parameters:**
- Keyframe strength: 0.10 to 0.25, step 0.05
- **Total tests:** 4 configurations

**Expected Results:**
- Less critical than normal strength (fewer keyframes in 30-frame sequence)
- May see modest differences in overall stability

### Phase 3: Combined Optimization (Optional)

**Goal:** Test if combined tuning improves results

**Test Combinations:**
- (normal=0.90, kf=0.10) - High stability + dramatic keyframes
- (normal=0.92, kf=0.12)
- (normal=0.94, kf=0.10) - Very high stability
- **Total tests:** 3-5 configurations

### Phase 4: Schnell Viability (Future)

**Goal:** Check if Schnell can maintain stability with coarse resolution

**Fixed Parameters:**
- Steps: 4 (Flux Schnell)
- Max iterations: 20

**Sweep Parameters:**
- Normal strength: 0.75 to 1.00, step 0.25
  - 0.75 = 3/4 steps (similar to 0.85 @ 20 steps)
  - 1.00 = 4/4 steps (no diffusion, pure feed-forward)
- **Total tests:** 2-4 configurations

**Note:** Coarse resolution (0.25) limits fine-tuning

## How to Run Tests

### Via Tuning Lab UI (Recommended)

1. **Launch Tuning Lab:**
   ```bash
   # Restart WebUI (to pick up UI changes)
   python webui.py
   ```

2. **Navigate to Tuning tab** in WebUI

3. **Configure Phase 1:**
   - Test type: "Color Preservation (I2V Chaining)"
   - Steps to test: [20 (Dev)]
   - Normal strength: min=0.80, max=0.95, step=0.01
   - Keyframe strength: min=0.15, max=0.15, step=0.01 (fixed)
   - Max iterations: 20
   - Grayscale threshold: 20

4. **Click "Run Tests"**

5. **Monitor Progress:**
   - Status box shows current test
   - Progress bar indicates completion
   - Results appear in real-time

6. **Analyze Results:**
   - Summary tab: Best parameters found
   - Results table: All configurations sorted by quality
   - Charts: Degradation curves (if available)

### Via Integration Tests (Automated)

```bash
cd /home/zirteq/workspace/forge-neo/extensions/sd-forge-deforum
pytest tests/integration/test_color_preservation.py -v --start-server
```

**Limitations:**
- Fixed test matrix (parametrized in code)
- Requires editing test file to change parameters
- Good for CI/CD validation

## Interpreting Results

### Success Metrics

**Excellent (Target):**
- Color preservation: 15-20 iterations before grayscale
- Final color score: > 30/100
- Temporal consistency: > 85/100
- Degradation rate: < 3%/iteration
- Overall score: > 70/100

**Good:**
- Color preservation: 10-15 iterations
- Final color score: > 20/100
- Temporal consistency: > 80/100
- Degradation rate: 3-5%/iteration
- Overall score: > 60/100

**Poor (Need Different Parameters):**
- Color preservation: < 10 iterations
- Final color score: < 20/100
- Temporal consistency: < 75/100
- Degradation rate: > 5%/iteration
- Overall score: < 50/100

### Results Table Columns

```
Steps | Normal Strength | KF Strength | Iterations | Final Color | Avg Temporal | Overall Score | Degradation Rate
------|-----------------|-------------|------------|-------------|--------------|---------------|------------------
20    | 0.85           | 0.15        | 18         | 35.2        | 87.5         | 75.3          | 2.8%
20    | 0.90           | 0.15        | 20         | 42.1        | 89.2         | 82.0          | 2.1%
```

**Sort by:** Overall Score (descending) to find best configuration

## Expected Timeline

**Phase 1 (16 tests):**
- ~16 tests × ~5 minutes/test = 80 minutes (1.3 hours)

**Phase 2 (4 tests):**
- ~4 tests × ~5 minutes/test = 20 minutes

**Phase 3 (5 tests):**
- ~5 tests × ~5 minutes/test = 25 minutes

**Total:** ~2-2.5 hours for comprehensive sweep

## Output Files

All results save to: `outputs/deforum-tuning/color_preservation/`

```
color_preservation/
├── test_input_rainbow.png              # Shared colorful test image
├── steps20_norm0.80_kf0.15/            # Test 1
│   ├── iteration_000_color92.png       # Iteration 0 output
│   ├── iteration_001_color87.png       # Iteration 1 output
│   ├── ...
│   ├── metrics.json                    # Full results
│   └── {timestamp}/                    # Generated frames
│       ├── 000000000.png - 000000029.png
│       └── depth-maps/
├── steps20_norm0.85_kf0.15/            # Test 2
│   └── ...
└── ...
```

## What's Different from RAFT Tests

| Aspect | RAFT Orbital Tests | I2V Chaining Tests |
|--------|-------------------|--------------------|
| **What's tested** | Depth warping stability | Img2img strength tuning |
| **Test type** | Circular camera orbit | Cascading I2V chaining |
| **Iterations** | 200-500 frames | 20 iterations (600 total frames) |
| **Metrics** | Sphere visibility | Color + temporal consistency |
| **Parameters** | rotation_factor, RAFT settings | normal/keyframe strength, steps |
| **Output** | Orbit videos | Degradation sequence |
| **Duration** | ~5 min/test | ~5 min/test |

## Next Steps After Testing

1. **Identify optimal configuration** from results table
2. **Update defaults** in `deforum/config/args.py`:
   ```python
   "strength_schedule": "0: (0.XX)",  # Best normal_strength
   "keyframe_strength_schedule": "0: (0.XX)",  # Best kf_strength
   ```
3. **Document findings** in TUNING.md or CLAUDE.md
4. **Share results** for community validation

## Troubleshooting

**Issue:** Tests run too slow
- Reduce test range (e.g., 0.85-0.95 instead of 0.80-0.95)
- Increase step size to 0.02 or 0.05
- Reduce max_iterations to 10

**Issue:** All tests degrade quickly (< 10 iterations)
- Normal strength may be too low
- Try higher range: 0.88-0.98

**Issue:** All tests maintain color (20 iterations)
- Good problem to have! Test even lower strengths to find degradation point
- Try 0.70-0.85 to explore limits

**Issue:** UI changes not appearing
- Restart WebUI completely
- Clear browser cache (Ctrl+Shift+R)

## References

- **Test implementation:** `tests/integration/test_color_preservation.py`
- **Metrics:** `tests/integration/metrics.py`
- **API backend:** `deforum/api/tuning_api.py`
- **UI:** `deforum/ui/ui_tuning.py`
- **Fractional strength:** `deforum/pipeline/fractional_strength.py`
