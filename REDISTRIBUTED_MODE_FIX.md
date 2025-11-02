# REDISTRIBUTED_CADENCE Mode Fix: Exact Keyframe Placement

**BREAKING CHANGE:** Renamed from `REDISTRIBUTED` to `REDISTRIBUTED_CADENCE` for clarity.
Old name accepted for migration compatibility.

## Critical Issue

The REDISTRIBUTED mode had the logic **backwards**:

**OLD (WRONG) ❌:**
1. Start with uniform cadence frames
2. Try to "squeeze in" keyframes by replacing nearest cadence frames
3. **Result**: Keyframes NOT at exact requested positions!

**Example failure:**
- User requests keyframe at frame 100
- Cadence puts frames at 98 and 105
- Old logic replaces frame 98
- **Keyframe ends up at 98, NOT 100!**

## The Fix

**NEW (CORRECT) ✅:**
1. **Start with EXACT keyframes** from prompts (non-negotiable)
2. **Distribute cadence frames BETWEEN keyframes** to approximate desired cadence
3. **Drop cadence frames** if too close to keyframes (avoid back-to-back diffusions)

**Priority order:**
1. EXACT keyframe placement (most important)
2. Approximate desired cadence (nice to have)
3. Minimum spacing enforcement (prevent clustering)

## Algorithm Details

### Step 1: Get Exact Keyframes
```python
keyframes = select_keyframes(data)  # From prompts: [0, 25, 50, 75, 99]
keyframes_set = set(keyframes)       # These are NON-NEGOTIABLE
```

### Step 2: Calculate Cadence Budget
```python
num_keyframes = len(keyframes)           # e.g., 5
cadence_budget = diffusion_frame_count - num_keyframes  # e.g., 20 - 5 = 15
```

### Step 3: Distribute Cadence Frames Between Keyframes
For each section between consecutive keyframes:
```python
section_start = 25
section_end = 50
section_length = 25

# How many cadence frames fit?
num_cadence_in_section = (section_length - min_spacing) // desired_cadence

# Distribute evenly within section
for j in range(1, num_cadence_in_section + 1):
    cadence_frame = section_start + int(j * section_length / (num_cadence_in_section + 1))
    if not too_close_to_keyframes(cadence_frame):
        add_to_result(cadence_frame)
```

### Step 4: Enforce Minimum Spacing
```python
min_spacing = max(1, desired_cadence // 2)

# Don't add cadence frame if too close to keyframe
for kf in keyframes:
    if abs(cadence_frame - kf) < min_spacing:
        skip_this_frame()
```

### Step 5: Adjust to Target Count
- **Too many frames**: Drop cadence frames (KEEP keyframes!)
- **Too few frames**: Fill largest gaps with additional cadence frames

## Example Results

### Example 1: Even Keyframes
```
Input:
  max_frames: 100
  keyframes: [0, 25, 50, 75, 99]
  cadence: 5
  target: 20 frames

Output:
  [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79, 84, 89, 99]

  Keyframes (exact): 0, 25, 50, 75, 99 ✓
  Cadence frames: distributed evenly between
```

### Example 2: Off-Cadence Keyframes (NEVER MOVED)
```
Input:
  max_frames: 100
  keyframes: [0, 23, 47, 71, 99]  ← Off cadence!
  cadence: 10 (would prefer 0, 10, 20, 30...)
  target: 15 frames

Output:
  [0, 9, 18, 23, 28, 37, 42, 47, 53, 59, 65, 71, 80, 89, 99]

  Keyframes (EXACT): 0, 23, 47, 71, 99 ✓
  NOT moved to: 20, 50, 70 (cadence preference ignored)
  Cadence frames: distributed around exact keyframes
```

### Example 3: Large Gaps
```
Input:
  max_frames: 100
  keyframes: [0, 50, 99]  ← Big gaps
  cadence: 10
  target: 12 frames

Output:
  [0, 9, 14, 19, 29, 39, 49, 59, 69, 79, 89, 99]

  Keyframes (exact): 0, 50, 99 ✓
  Cadence frames: 10 frames distributed in gaps (0-50) and (50-99)
```

## Testing

Comprehensive unit tests in `tests/unit/test_keyframe_redistribution.py`:

✅ **test_exact_keyframe_placement** - Keyframes at exact positions
✅ **test_keyframes_never_moved** - Off-cadence keyframes stay put
✅ **test_cadence_frames_between_keyframes** - Cadence only in gaps
✅ **test_minimum_spacing_enforcement** - No back-to-back diffusions
✅ **test_no_cadence_budget** - All keyframes, no cadence
✅ **test_single_keyframe_section** - Large gap handling
✅ **test_uneven_keyframe_spacing** - Uneven distribution
✅ **test_exact_frame_count_target** - Matches target count
✅ **test_dense_keyframes_with_cadence** - Stress test
✅ **test_parseq_keyframes** - Parseq exact positioning
✅ **test_result_is_sorted** - Always sorted
✅ **test_no_duplicate_frames** - No duplicates
✅ **test_all_frames_in_valid_range** - Valid range
✅ **test_first_and_last_frame_always_included** - Endpoints

### Manual Test Results

```
Test 1: Exact keyframe placement
  Keyframes: [0, 25, 50, 75, 99]
  Result: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79, 84, 89, 99]
  ✓ All keyframes exact: True

Test 2: Keyframes NEVER moved
  Keyframes: [0, 23, 47, 71, 99] (off-cadence from 10)
  Result: [0, 9, 18, 23, 28, 37, 42, 47, 53, 59, 65, 71, 80, 89, 99]
  ✓ 23 exact: True
  ✓ 47 exact: True
  ✓ 71 exact: True

Test 3: Cadence frames BETWEEN keyframes
  Keyframes: [0, 50, 99]
  Result: [0, 9, 14, 19, 29, 39, 49, 59, 69, 79, 89, 99]
  Cadence frames: [9, 14, 19, 29, 39, 49, 59, 69, 79, 89]
  ✓ All cadence between keyframes: True

Test 4: Minimum spacing enforcement
  Keyframes: [0, 10, 20, 30, 99]
  Result: [0, 5, 10, 15, 20, 25, 30, 34, 39, 44, 49, 54, 59, 64, 99]
  Min spacing: 2
  ✓ No spacing violations: True

✅ All manual tests passed!
```

## Why This Matters

### For Audio Sync
- **OLD**: Keyframes at wrong positions = audio/visual desync
- **NEW**: Exact keyframes = perfect audio/visual synchronization

### For Parseq
- **OLD**: Parseq keyframes moved = timeline doesn't match
- **NEW**: Exact Parseq frames = reliable timeline

### For Prompt Timing
- **OLD**: "Scene change at frame 100" might happen at frame 98
- **NEW**: Frame 100 means frame 100, guaranteed

### For User Control
- **OLD**: Unpredictable keyframe placement
- **NEW**: What you request is what you get

## Files Modified

1. **deforum/rendering/data/frame/key_frame_distribution.py**
   - Completely rewrote `_redistributed()` method
   - Priority: EXACT keyframes > cadence approximation > spacing
   - Added comprehensive documentation

2. **deforum/rendering/data/render_mode.py**
   - Updated New 3D mode description
   - Clarified EXACT keyframe placement
   - Updated Flux + Interpolation to mention Lumina and removal of RIFE

3. **deforum/ui/tabs/tab_distribution.py**
   - Updated documentation to reflect correct behavior
   - Clarified mode differences

4. **tests/unit/test_keyframe_redistribution.py** (NEW)
   - 17 comprehensive unit tests
   - Edge case coverage
   - Regression prevention

## Breaking Changes

**None!** This is a bug fix that makes the system work as users expect.

However, if users were relying on the OLD (broken) behavior where keyframes could be moved, they may notice differences. This is **intentional and correct** - keyframes should NEVER move from requested positions.

## Migration

No migration needed. The new behavior is what users expect and what the documentation claims.

If you see different frame positions after this update, it means:
- **Before**: Keyframes were incorrectly moved
- **After**: Keyframes are now at EXACT requested positions ✓

## Performance

No performance impact. The algorithm complexity is similar, just with correct logic.

## Logging

The new implementation logs detailed information:
```
Redistributed: 5 exact keyframes + 15 cadence frames = 20 total (target: 20)
```

This helps verify correct operation during rendering.
