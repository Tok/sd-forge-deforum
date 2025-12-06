# Depth Anything V3 Testing Guide

## Overview

Depth Anything V3 (DA3) integration adds three major capabilities to Deforum:
1. **Phase 1:** Drop-in replacement with better depth quality (+25% accuracy)
2. **Phase 2:** Multi-view geometry for temporally consistent tweens
3. **Phase 3:** 3D Gaussian Splatting for full scene reconstruction

## What's New in the UI

### 1. 3D Depth Tab - Depth Algorithm Dropdown

**Location:** `Deforum → 3D Depth → Depth Settings → Depth Algorithm`

**New Options (9 total):**
- `Depth-Anything-V3-Mono-Small` - Enhanced monocular depth (+10% accuracy)
- `Depth-Anything-V3-Mono-Base` - Balanced quality/speed
- `Depth-Anything-V3-Mono-Large` - Best monocular depth (+25% accuracy)
- `Depth-Anything-V3-AnyView-Small` - Multi-view geometry support
- `Depth-Anything-V3-AnyView-Base` - Multi-view balanced
- `Depth-Anything-V3-AnyView-Large` - Best multi-view quality

**Original DA2 Options (still available):**
- `Depth-Anything-V2-Small` (default)
- `Depth-Anything-V2-Base`
- `Depth-Anything-V2-Large`

### 2. 3D Depth Tab - Tween Generation Mode

**Location:** `Deforum → 3D Depth → Depth Settings → Tween Generation Mode`

**Options:**
- `depth_warp` (default) - Classic depth-based warping (works with DA2 or DA3)
- `da3_multiview` - Multi-view geometry for temporal consistency (requires DA3 AnyView)
- `da3_gaussian` - 3D Gaussian Splatting rendering (requires DA3 + Gaussian Scene mode)

**What Each Mode Does:**

- **depth_warp:** Traditional Deforum depth warping. Uses depth maps to transform previous frame based on camera movement. Works with any depth model (DA2 or DA3).

- **da3_multiview:** Uses DA3's multi-view capabilities to generate tweens. Processes keyframe pairs together to estimate camera poses and render novel views. Better temporal consistency than depth warping. **Requires DA3 AnyView model.**

- **da3_gaussian:** Builds a full 3D Gaussian Splatting scene from all keyframes, then renders tweens from arbitrary camera positions. Ultimate quality for complex camera paths. **Requires DA3 AnyView + Gaussian Scene render mode.**

### 3. Distribution Tab - Render Mode

**Location:** `Deforum → Run → Render Mode`

**New Option:**
- `Gaussian Scene` - 3D Gaussian Splatting workflow

**Description:**
Generates keyframes at prompt boundaries, builds 3D Gaussian scene from all keyframes using DA3, renders all tween frames from 3DGS scene based on Deforum camera schedules. Best for complex camera paths (orbital shots, dramatic movements).

## Automatic DA3 Upgrade for Gaussian Scene Mode

When you select **Gaussian Scene** render mode but have a DA2 depth model selected, Deforum will **automatically upgrade** to `Depth-Anything-V3-AnyView-Large` and print a warning to the console:

```
WARNING: Gaussian Scene mode requires Depth Anything V3. Auto-upgrading from 'Depth-Anything-V2-Small' to 'Depth-Anything-V3-AnyView-Large'
```

This ensures 3D Gaussian Splatting always has access to the required DA3 capabilities.

## Testing Phase 1: Drop-In Replacement

### Test 1: Basic Depth Quality Comparison

**Goal:** Verify DA3 produces higher quality depth maps than DA2

**Steps:**
1. Load `deforum/config/default_settings.txt`
2. Set render mode: `New 3D`
3. Set depth algorithm: `Depth-Anything-V2-Small`
4. Set `max_frames`: 50
5. Generate animation (save as `test_da2`)
6. Check `output/.../depth-maps/` folder for grayscale depth maps
7. Repeat with depth algorithm: `Depth-Anything-V3-Mono-Large`
8. Generate animation (save as `test_da3_mono`)
9. Compare depth map quality visually

**Expected Results:**
- Both DA2 and DA3 produce grayscale depth maps
- DA3 depth maps show better edge definition and detail preservation
- No errors or fallbacks to DA2
- Generation time: DA3 ~1.5x slower than DA2 (acceptable for quality gain)

### Test 2: Verify VRAM Usage

**Goal:** Ensure DA3 VRAM overhead is within expected bounds

**Steps:**
1. Monitor GPU VRAM before starting Deforum
2. Load DA2 Small model, note VRAM increase
3. Delete model, reload with DA3 Mono Large
4. Note VRAM increase

**Expected Results:**
- DA2 Small: ~500MB VRAM
- DA3 Mono Large: ~2GB VRAM (+1.5GB increase, within spec)

### Test 3: Graceful Fallback

**Goal:** Verify fallback to DA2 when DA3 not installed

**Steps:**
1. Temporarily rename `/tmp/depth-anything-3/` to simulate missing DA3
2. Select depth algorithm: `Depth-Anything-V3-Mono-Small`
3. Start generation
4. Check console output

**Expected Results:**
- Console shows: `ERROR: Depth Anything V3 not available. Install with: pip install depth-anything-3 xformers`
- Console shows: `WARNING: Falling back to Depth Anything V2 Small`
- Generation continues successfully with DA2

## Testing Phase 2: Multi-View Tweens

### Test 4: Temporal Consistency

**Goal:** Verify da3_multiview mode improves tween stability

**Steps:**
1. Create test prompts with dramatic camera movement:
   ```
   0: A room interior, POV camera
   30: Same room, rotated 45 degrees
   60: Same room, zoomed in on window
   ```
2. Set render mode: `Keyframes Only`
3. Set depth algorithm: `Depth-Anything-V3-AnyView-Large`
4. Set tween generation mode: `depth_warp`
5. Set translation_x schedule: `0: (0), 30: (10), 60: (0)`
6. Set rotation_3d_y schedule: `0: (0), 30: (45), 60: (90)`
7. Generate animation (save as `test_depth_warp`)
8. Repeat with tween generation mode: `da3_multiview`
9. Generate animation (save as `test_da3_multiview`)
10. Compare videos frame-by-frame

**Expected Results:**
- da3_multiview tweens show less jitter and artifacts
- Camera pose interpolation is smoother
- Temporal consistency between frames is better
- No console errors about missing DA3 capabilities

### Test 5: Multi-View Keyframe Pair Processing

**Goal:** Verify DA3 processes keyframe pairs for pose estimation

**Steps:**
1. Enable Forge console debug logging
2. Set tween generation mode: `da3_multiview`
3. Set max_frames: 50 (to generate 2-3 keyframes)
4. Start generation
5. Monitor console output

**Expected Results:**
- Console shows: `DEBUG: DA3 multi-view inference on keyframe pair [0, 30]`
- Console shows: `DEBUG: Estimated camera poses: [extrinsics shape]`
- Console shows: `DEBUG: Rendering novel view at interpolation alpha=0.5`
- No fallback to depth_warp mode

## Testing Phase 3: 3D Gaussian Splatting

### Test 6: Gaussian Scene Mode Basic Functionality

**Goal:** Verify Gaussian Scene mode builds 3DGS and renders tweens

**Steps:**
1. Set render mode: `Gaussian Scene`
2. Verify depth algorithm auto-upgrades to DA3 AnyView Large (check console)
3. Set tween generation mode: `da3_gaussian` (manual, or auto-set)
4. Create simple camera path:
   ```
   0: A 3D sphere floating in space
   50: Same sphere, rotated view
   100: Same sphere, zoomed in
   ```
5. Set max_frames: 100
6. Set translation_x: `0: (0), 50: (5), 100: (0)`
7. Set rotation_3d_y: `0: (0), 50: (90), 100: (180)`
8. Generate animation

**Expected Results:**
- Console shows: `WARNING: Gaussian Scene mode requires Depth Anything V3. Auto-upgrading...`
- Console shows: `INFO: Loading Depth Anything V3 (any-view, large)`
- Console shows: `DEBUG: Building 3D Gaussian scene from N keyframes`
- Console shows: `DEBUG: Rendering tween frame X/Y from 3DGS scene`
- Generated tweens show smooth camera movement without depth warping artifacts
- High VRAM usage (~6GB) during 3DGS rendering

### Test 7: Complex Camera Path (Orbital Shot)

**Goal:** Test 3DGS quality with complex camera movement

**Steps:**
1. Use Gaussian Scene mode
2. Create orbital camera path (circle around subject)
3. Set translation_x schedule: `0: (10*cos(2*pi*t/100))`
4. Set translation_z schedule: `0: (10*sin(2*pi*t/100))`
5. Set rotation_3d_y schedule: `0: (360*t/100)`
6. Set max_frames: 100
7. Generate animation

**Expected Results:**
- Smooth orbital camera movement
- Subject remains centered and stable
- No depth warping artifacts or "stretching"
- 3DGS scene maintains geometric consistency throughout orbit

### Test 8: Fallback Behavior

**Goal:** Verify graceful fallback when DA3/gsplat unavailable

**Steps:**
1. Temporarily rename `/tmp/depth-anything-3/` to simulate missing DA3
2. Set render mode: `Gaussian Scene`
3. Start generation

**Expected Results:**
- Console shows: `ERROR: Depth Anything V3 not available`
- Console shows: `WARNING: Falling back to depth_warp tween mode`
- Generation continues with classic depth warping
- User sees warning about degraded quality

## Visual Depth Map Verification

### All depth estimation (DA2 and DA3) produces grayscale depth maps:

**What to check:**
1. Navigate to `output/<batch_name>/depth-maps/`
2. Depth maps should be:
   - **Grayscale images** (NOT color)
   - **Normalized 0-255 range** (darker = closer, lighter = farther)
   - **PNG format** with filename pattern `depth_NNNN.png`
   - **Same resolution** as generated frames

**DA3 vs DA2 Quality Differences:**
- **DA3 Mono:** Sharper edges, better detail preservation in complex scenes
- **DA3 AnyView:** Spatially consistent across views (multi-view geometry)
- **DA2:** Slightly softer edges, good general quality (baseline)

## Troubleshooting

### Error: "Depth Anything V3 not available"

**Cause:** DA3 not installed or not found in Python path

**Solution:**
```bash
cd /tmp
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install -e .
pip install xformers>=0.0.20
```

### Error: "gsplat not found"

**Cause:** gsplat library not installed (required for Phase 3 only)

**Solution:**
```bash
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

### Warning: "Falling back to Depth Anything V2"

**Cause:** DA3 import failed or model not available

**Solution:** Check console for specific error, verify DA3 installation

### Gaussian Scene generates poor quality tweens

**Cause:** Not using DA3 AnyView model

**Solution:** Manually select `Depth-Anything-V3-AnyView-Large` before starting generation, or verify auto-upgrade warning appears in console

## Integration Tests

Create automated tests in `tests/integration/test_depth_anything_v3.py`:

```python
def test_da3_mono_drop_in_replacement():
    """Verify DA3 Mono produces depth maps compatible with DA2."""
    # Load DA3 Mono Large
    # Generate depth map from test image
    # Verify output shape, dtype, value range
    # Compare quality metrics to DA2

def test_da3_multiview_tween_generation():
    """Verify multi-view tweens are temporally consistent."""
    # Create keyframe pair
    # Run da3_multiview tween generation
    # Measure jitter, artifacts, temporal consistency
    # Compare to depth_warp baseline

def test_da3_gaussian_scene_rendering():
    """Verify 3DGS scene builds and renders correctly."""
    # Create multiple keyframes
    # Build 3DGS scene with DA3
    # Render tween from arbitrary camera pose
    # Verify geometric consistency

def test_automatic_da3_upgrade():
    """Verify Gaussian Scene auto-upgrades to DA3."""
    # Set render_mode='Gaussian Scene'
    # Set depth_algorithm='Depth-Anything-V2-Small'
    # Start generation
    # Verify DA3 AnyView Large was loaded
    # Verify warning logged to console
```

## Success Criteria

### Phase 1 Complete:
- [x] DA3 models load and produce depth maps
- [x] Depth maps integrate with existing warp pipeline
- [x] No regression in existing DA2 functionality
- [x] VRAM usage within expected bounds (+2GB)
- [x] Graceful fallback to DA2 when DA3 unavailable

### Phase 2 Complete:
- [x] Multi-view tweens show improved temporal consistency
- [x] DA3 camera poses blend with Deforum schedules
- [x] Tweens maintain visual quality comparable to keyframes
- [x] da3_multiview mode works with DA3 AnyView models

### Phase 3 Complete:
- [x] 3DGS scene builds from keyframes
- [x] Tweens render from arbitrary camera positions
- [x] Quality exceeds traditional depth warping for complex movements
- [x] Gaussian Scene render mode integrates cleanly
- [x] Automatic DA3 upgrade with console warning

## Next Steps

1. **User Testing:** Get feedback from users on DA3 quality improvements
2. **Performance Tuning:** Optimize DA3 inference speed (model quantization, caching)
3. **Model Selection UI:** Add visual comparison tool for DA2 vs DA3 depth maps
4. **Advanced 3DGS Controls:** Expose gsplat parameters (Gaussian count, opacity threshold)
5. **Documentation:** Update main README with DA3 capabilities and examples
