# Tuning Test Fix: Correct Normal Strength Testing

## Problem Summary

The color preservation tuning test was not correctly testing `normal_strength` parameter because:

1. **Original test used only 2 frames** (`max_frames=2`)
2. **With New 3D mode (REDISTRIBUTED distribution)**:
   - Frame 0: Keyframe (first frame is always a keyframe) → uses `keyframe_strength`
   - Frame 1: Keyframe (last frame is always a keyframe) → uses `keyframe_strength`
3. **Result**: `normal_strength` was NEVER tested because there were NO cadence frames!

## Understanding Strength Parameters in New 3D Mode

### Dual Strength Schedule System

New 3D mode (REDISTRIBUTED) uses TWO different strength schedules:

1. **`keyframe_strength`** (LOW = 0.15 default):
   - Applied to keyframes (frames at prompt boundaries + last frame)
   - LOW strength = MORE noise = MORE diffusion steps = DRAMATIC CHANGES
   - Example: 0.15 strength @ 20 steps = 17/20 diffusion steps executed
   - Purpose: Allow significant visual changes at keyframes

2. **`normal_strength`** (HIGH = 0.85 default):
   - Applied to cadence frames (redistributed between keyframes)
   - HIGH strength = LESS noise = FEWER diffusion steps = STABILITY
   - Example: 0.85 strength @ 20 steps = 3/20 diffusion steps executed
   - Purpose: Maintain stability and color preservation through I2I chain

### Implementation Location

`deforum/rendering/data/frame/diffusion_frame.py:294-316`:

```python
def _select_keyframe_or_cadence_strength(data: RenderData, index, is_keyframe):
    # Keyframes use keyframe_strength_schedule (LOW = change)
    # Non-keyframes use strength_schedule (HIGH = stability)

    return (keys.keyframe_strength_schedule_series[idx]
            if is_keyframe and not data.parseq_adapter.use_parseq
            else keys.strength_schedule_series[idx])
```

**The implementation was CORRECT** - the test was just not exercising it properly!

## The Fix

### Changed Test Parameters

**Before:**
```python
max_frames: 2  # Only 2 frames (both keyframes)
# Frame 0: keyframe → keyframe_strength
# Frame 1: keyframe → keyframe_strength
# normal_strength: NEVER USED ❌
```

**After:**
```python
max_frames: 30  # 30 frames with sparse keyframes
# Frame 0: keyframe → keyframe_strength
# Frames 1-28: cadence frames → normal_strength ✅
# Frame 29: keyframe → keyframe_strength
# ~27/30 frames test normal_strength!
```

### I2V Chaining Flow

Each iteration now:
1. Generates 30-frame animation
2. Most frames (1-28) use `normal_strength` for I2I diffusion
3. Takes LAST frame (frame 29) as output
4. Feeds it as input to next iteration
5. Measures color degradation through this cascading I2I chain

This correctly tests: **How many diffusion steps (via normal_strength) keep the I2I chain stable?**

### Updated Test Parameters

```python
# TEST SET 1: Normal Strength Sweep (PRIMARY GOAL)
# Fixed keyframe_strength=0.15, vary normal_strength
(20, 0.80, 0.15),  # Lower - more diffusion on cadence frames
(20, 0.85, 0.15),  # Current default
(20, 0.90, 0.15),  # Higher - fewer diffusion steps
(20, 0.95, 0.15),  # Very high - minimal diffusion

# TEST SET 2: Keyframe Strength Sweep (SECONDARY)
# Fixed normal_strength=0.85, vary keyframe_strength
(20, 0.85, 0.10),  # Lower - more change at keyframes
(20, 0.85, 0.20),  # Higher - more retention at keyframes

# TEST SET 3: Combined Optimization
(20, 0.90, 0.10),  # High cadence stability + low keyframe retention
(20, 0.95, 0.10),  # Very high cadence stability

# TEST SET 4: Schnell Viability (4 steps = coarse resolution)
(4, 0.75, 0.25),   # Similar effective steps as 0.85 @ 20
(4, 1.00, 0.25),   # No diffusion (pure I2I feed)
```

## UI Cleanup: Removed Redundant Dropdown

### The Issue

The UI had BOTH:
1. **Render Mode** radio buttons (top-level) - Correct way to select mode
2. **Keyframe Distribution** dropdown (Distribution tab) - REDUNDANT

### Why It Was Redundant

The backend correctly overrides keyframe distribution based on render mode:

`deforum/orchestration/run_deforum.py:90-103`:
```python
render_mode = RenderMode.from_string(render_mode_str)

# Override keyframe_distribution based on render_mode
distribution = render_mode.get_keyframe_distribution()
if distribution:
    args_dict['keyframe_distribution'] = distribution.value
```

**Render Mode → Keyframe Distribution mapping:**
- Classic 3D → OFF (uniform cadence)
- New 3D → REDISTRIBUTED (keyframes replace nearest cadence)
- Keyframes Only → KEYFRAMES_ONLY (only diffuse keyframes)
- Flux + Interpolation → None (separate pipeline)

### The Fix

Removed the dropdown from `deforum/ui/tabs/tab_distribution.py` and added clear documentation:

```markdown
## Distribution & Render Mode Configuration

**Keyframe distribution is automatically determined by your Render Mode selection:**
- **Classic 3D** → OFF (uniform cadence)
- **New 3D** → REDISTRIBUTED (keyframes replace nearest cadence frames)
- **Keyframes Only** → KEYFRAMES_ONLY (only diffuse at keyframes, depth-tween everything else)
- **Flux + Interpolation** → Separate pipeline (Flux keyframes + interpolation)

Change Render Mode at the top of the main UI to control distribution behavior.
```

## What the Test Actually Measures Now

### Normal Strength Test
**Question**: How many diffusion steps keep I2I chain stable?

- **Higher normal_strength** (0.90-0.95):
  - Fewer diffusion steps per cadence frame
  - Less noise added
  - Better color preservation
  - More stable I2I chain
  - But potentially less "creativity" between keyframes

- **Lower normal_strength** (0.75-0.85):
  - More diffusion steps per cadence frame
  - More noise added
  - More color drift/degradation
  - Less stable I2I chain
  - But potentially more "variation" between keyframes

### Keyframe Strength Test
**Question**: How much of previous frame is retained at keyframes?

- **Lower keyframe_strength** (0.10):
  - More diffusion steps at keyframes
  - Allows dramatic changes
  - Less retention of previous frame
  - Color can change intentionally

- **Higher keyframe_strength** (0.20):
  - Fewer diffusion steps at keyframes
  - More retention of previous frame
  - More gradual changes
  - Better continuity

Color degradation at keyframes is EXPECTED (that's where change should happen), so this test is less critical for color preservation.

## Files Modified

1. **tests/tuning/test_color_preservation.py**:
   - Changed `max_frames` from 2 to 30
   - Updated test parameters to focus on normal_strength sweep
   - Fixed return value to use LAST frame (frame 29)
   - Added comprehensive documentation

2. **deforum/api/tuning_test_helpers.py**:
   - Same changes as test file for API consistency
   - Updated `run_i2v_iteration()` function

3. **deforum/ui/tabs/tab_distribution.py**:
   - Removed redundant `keyframe_distribution` dropdown
   - Added clear documentation about render mode → distribution mapping

## How to Run the Fixed Test

```bash
# From Forge webui directory
pytest extensions/sd-forge-deforum/tests/tuning/test_color_preservation.py -v

# Or via the Tuning Lab UI (if enabled)
python webui.py --deforum-run-tuning
```

## Expected Results

With the fixed test, we should now see:
- Clear correlation between normal_strength and color preservation
- Higher normal_strength (0.90-0.95) maintains color longer
- Lower normal_strength (0.80-0.85) degrades faster
- Keyframe strength has less impact on overall color preservation

This will help us find the optimal balance between stability and quality for New 3D mode!
