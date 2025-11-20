# Wormtrail Rotation Visualization Fix

## Problem

The wormtrail 3D visualization was showing only rotation without camera movement, even though the camera path schedules were correct. The camera appeared to rotate in place rather than orbit around the center point.

## Root Cause

The bug was in `deforum/utils/schedule_visualizer.py:247-273` in the dense schedule handling logic.

### How Camera Paths Work

Camera path generators output schedules in a **hybrid format**:
- **Translation schedules**: DELTAS (frame-to-frame changes)
  - Example: `0: (0.0), 1: (-0.48), 2: (-1.41), 3: (-2.27), ...`
  - Each value represents movement FROM previous frame
  - Must be ACCUMULATED to get absolute positions for visualization

- **Rotation schedules**: ABSOLUTE angles (Euler degrees)
  - Example: `0: (0.0), 1: (-5.51), 2: (-5.45), 3: (-5.33), ...`
  - Each value is the actual camera rotation at that frame
  - Must be USED DIRECTLY for visualization

### The Bug

The visualization code was treating rotation schedules the same as translation schedules:

```python
# BEFORE (INCORRECT)
cum_x, cum_y, cum_z = 0.0, 0.0, 0.0
cum_rx, cum_ry, cum_rz = 0.0, 0.0, 0.0  # ❌ Should not accumulate rotations!

for i in range(len(x_deltas)):
    cum_x += x_deltas[i]      # ✓ Correct - accumulate translation deltas
    cum_y += y_deltas[i]
    cum_z += z_deltas[i]
    cum_rx += rx_deltas[i]    # ❌ WRONG - rotations are not deltas!
    cum_ry += ry_deltas[i]    # ❌ This accumulates absolute angles
    cum_rz += rz_deltas[i]    # ❌ Causing spiral/incorrect directions

    x_coords.append(cum_x)
    y_coords.append(cum_y)
    z_coords.append(cum_z)
    rx_coords.append(cum_rx)  # ❌ Using accumulated values
    ry_coords.append(cum_ry)
    rz_coords.append(cum_rz)
```

**Example of incorrect accumulation:**
- Camera path rotation values: `0: (0), 1: (-5.51), 2: (-5.45), 3: (-5.33), ...`
- Visualization accumulated them: `0°, -5.51°, -10.96°, -16.29°, ...`
- Actual rotation angles should be: `0°, -5.51°, -5.45°, -5.33°, ...`

This caused the camera direction arrows to point completely wrong directions, making the path appear as pure rotation without movement.

## The Fix

Changed the visualization to only accumulate translation deltas, while using rotation values directly:

```python
# AFTER (CORRECT)
cum_x, cum_y, cum_z = 0.0, 0.0, 0.0

for i in range(len(x_deltas)):
    # Accumulate translation deltas ✓
    cum_x += x_deltas[i]
    cum_y += y_deltas[i]
    cum_z += z_deltas[i]

    x_coords.append(cum_x)
    y_coords.append(cum_y)
    z_coords.append(cum_z)

    # Rotations are ABSOLUTE angles from camera path generator ✓
    # Do NOT accumulate - use values directly
    rx_coords.append(rx_deltas[i])
    ry_coords.append(ry_deltas[i])
    rz_coords.append(rz_deltas[i])
```

## Verification

Created `test_wormtrail_visualization.py` which confirms:

1. **Translation deltas accumulated correctly**:
   - Frame 0: (0.00, 0.00, 0.00) dist=0.00
   - Frame 10: (-30.90, 77.54, 14.12) dist=84.66
   - Frame 333: (0.00, 0.00, 0.00) dist=0.00 (closed loop)
   - Max distance: 200.02 (full orbital movement visible)

2. **Rotation angles used directly**:
   - Rotation X range: [-5.51°, 5.52°] (11.03° variation)
   - Rotation Y range: [-1.08°, 0.00°] (1.08° variation)
   - Rotation Z range: [0.00°, 0.00°] (no roll)
   - Values stay in reasonable range (not accumulating to hundreds of degrees)

3. **Camera behavior correct**:
   - Camera orbits around center in 200-unit radius circle
   - Camera always faces center point (look-at mode)
   - Local Euler angles stay relatively constant (expected for center look-at)
   - Direction arrows now point correctly at center throughout orbit

## Why Rotation Values Stay Small

For `look_at_mode="center"`, the camera always faces the center point while orbiting. This means:

- The camera's LOCAL Euler angles stay relatively constant
- The rotation X (pitch) varies slightly (-5.51° to 5.52°) as camera height changes
- The rotation Y (yaw) stays nearly constant (-1.08°) because camera always looks inward
- The orbital rotation is achieved through POSITION changes, not LOCAL rotation changes

This is expected and correct behavior. The old visualization bug made it appear the camera was spinning wildly because it was accumulating these small local rotation adjustments into a spiral.

## Files Changed

- `deforum/utils/schedule_visualizer.py:247-273` - Fixed rotation accumulation logic
- `test_wormtrail_visualization.py` - Comprehensive test validating the fix
- `docs/WORMTRAIL_ROTATION_FIX.md` - This documentation

## Related Issues

- This was the final piece needed after fixing camera path generation and schedule conversion
- Previous fixes addressed frames_per_loop, rotation modes, and center look-at recalculation
- All three fixes together ensure proper camera path visualization with movement AND rotation
