# Depth Anything V3 User Guide

Comprehensive guide to using Depth Anything V3 (DA3) for depth estimation in Deforum.

## Table of Contents
- [Overview](#overview)
- [Model Variants](#model-variants)
- [When to Use Each Variant](#when-to-use-each-variant)
- [Depth-Ray Representation](#depth-ray-representation)
- [Tween Generation Modes](#tween-generation-modes)
- [Performance Comparison](#performance-comparison)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Overview

Depth Anything V3 is a state-of-the-art depth estimation model that provides:
- **Better depth quality** than DA2
- **Multi-view geometry** understanding (AnyView/Giant variants)
- **Depth-ray representation** for camera pose estimation
- **Temporal consistency** for video sequences

**Key Innovation:** Unlike traditional depth models that only output a distance map, DA3 outputs both:
1. **Depth map** - Distance from camera to surface
2. **Ray map** - 3D direction vector for each pixel

Together, these enable camera pose estimation and novel view synthesis.

## Model Variants

### 1. **Mono Variant** (Default for Standard Depth Warping)
```
Models: DA3MONO-LARGE (350M parameters)
Purpose: Single-view depth estimation (drop-in DA2 replacement)
VRAM: ~2GB
Speed: Fastest
```

**Use for:**
- Standard depth warping (`depth_warp` tween mode)
- Classic 3D animations with fixed camera schedules
- Maximum performance/quality ratio

**Does NOT support:**
- Ray pose estimation (`use_ray_pose` flag)
- Multi-view geometry
- Camera pose estimation

### 2. **AnyView Variant** (Multi-View Geometry)
```
Models: DA3-SMALL (250M), DA3-BASE (300M), DA3-LARGE (350M)
Purpose: Multi-view depth with camera pose estimation
VRAM: ~3-4GB (single-view), ~6-8GB (multi-view mode)
Speed: Slower than Mono
```

**Use for:**
- Multi-view tween generation (`da3_multiview` tween mode)
- Camera pose estimation (with `use_ray_pose` enabled)
- Spatially consistent depth across multiple frames

**Supports:**
- Ray pose estimation
- Multi-view inference (process multiple frames simultaneously)
- Camera extrinsics/intrinsics estimation

### 3. **Giant Variant** (3D Gaussian Splatting)
```
Models: DA3-GIANT (1.4B parameters)
Purpose: Full 3D scene reconstruction
VRAM: ~12-16GB
Speed: Slowest
```

**Use for:**
- 3D Gaussian Splatting (`da3_gaussian` tween mode)
- Novel view synthesis with geometric consistency
- Full 3D scene understanding

**Supports:**
- All AnyView features
- 3D Gaussian parameter estimation
- High-quality novel view rendering

## When to Use Each Variant

### Standard Depth Warping (Default)

**Scenario:** Traditional Deforum 3D animation with manual camera schedules

**Recommended Model:** `Depth-Anything-V3-Mono-Small`

**Why:**
- Mono and AnyView produce **identical results** in `depth_warp` mode
- AnyView's multi-view capabilities are **not used** for single-image depth
- Mono is **faster** and uses **less VRAM**

**⚠️ WARNING:** If you select an AnyView model with `depth_warp` mode, you're **wasting VRAM/compute** for features you're not using. The UI will warn you about this.

```python
# What happens internally with depth_warp mode:
depth = model.predict(single_image)  # ONE image in, ONE depth out
# AnyView gains NOTHING here - it's doing single-view depth only!
```

### Multi-View Tween Generation

**Scenario:** Generate temporally consistent tweens using multi-view geometry

**Recommended Model:** `Depth-Anything-V3-AnyView-Small` or `AnyView-Base`

**Settings:**
- Tween Generation Mode: `da3_multiview`
- Enable ray pose: Optional (more accurate, slower)

**How it works:**
```python
# Process pairs of keyframes simultaneously
result = model.predict_multiview([prev_keyframe, next_keyframe])

# Returns:
# - depth maps for both images
# - camera_extrinsics: [N, 4, 4] camera pose matrices
# - camera_intrinsics: [N, 3, 3] camera parameters
# - confidence: per-pixel confidence maps

# Interpolate camera position for tween frames
interpolated_pose = lerp(pose1, pose2, tween_value)
tween_image = render_novel_view(interpolated_pose)
```

**Benefits:**
- Spatially consistent depth across keyframe pairs
- Smooth camera motion interpolation
- Better handling of occlusions and disocclusions

**Trade-offs:**
- 2-3x slower than standard depth warping
- 2x VRAM usage (processes 2 frames simultaneously)
- Still experimental (may have artifacts)

### 3D Gaussian Splatting

**Scenario:** Full 3D scene reconstruction for novel view synthesis

**Recommended Model:** `Depth-Anything-V3-Giant`

**Settings:**
- Render Mode: `Keyframes Only` or `New 3D`
- Tween Generation Mode: `da3_gaussian`
- Frame Collection: `keyframes` or `all` (memory intensive)

**How it works:**
1. Collect keyframes during generation
2. Run DA3 Giant to estimate 3D structure
3. Fit 3D Gaussians to the scene
4. Render novel views from interpolated camera poses

**Benefits:**
- Highest quality novel view synthesis
- Full geometric understanding of the scene
- Consistent depth across all views

**Requirements:**
- Significant VRAM (~12-16GB minimum)
- Optional: `gsplat` library for optimized rendering
- Works best with 10-30 keyframes

## Depth-Ray Representation

DA3's key innovation is the **depth-ray representation**, which encodes 3D structure in a way that enables camera pose estimation.

### Components

1. **Depth Map** `[H, W]`
   - Traditional pixel-to-camera distance
   - Values: 0-1 (normalized)
   - Same as DA2 output

2. **Ray Map** `[H, W, 3]`
   - 3D direction vector for each pixel
   - Describes projection direction in camera space
   - Enables pose estimation

3. **Derived Data** (from depth + ray)
   - Camera extrinsics (pose): `[4, 4]` transformation matrix
   - Camera intrinsics (focal length, etc.): `[3, 3]` matrix
   - Confidence maps: Per-pixel prediction confidence

### Visualization (Coming Soon)

Currently, Deforum only visualizes the **depth map**. The ray map exists in the model output but is not extracted or visualized.

**Planned features:**
- Ray direction arrows overlaid on depth preview
- Confidence heatmap visualization
- Camera pose visualization (extrinsics/intrinsics display)

## Tween Generation Modes

### 1. `depth_warp` (Default)

**How it works:**
1. Generate depth map from previous frame
2. Apply 3D transformation based on camera schedule
3. Warp previous frame to new viewpoint
4. Fill holes with noise or color coherence

**Characteristics:**
- Fast (single depth estimation per tween)
- Stable and well-tested
- Manual camera control
- Works with Mono or AnyView (Mono recommended)

**Best for:**
- Traditional Deforum workflows
- Predictable camera movements
- RAFT optical flow integration
- ControlNet compatibility

### 2. `da3_multiview` (Experimental)

**How it works:**
1. Process keyframe pair with multi-view inference
2. Estimate camera poses for both keyframes
3. Interpolate camera pose for tween position
4. Render novel view from interpolated pose

**Characteristics:**
- Slower (2x depth estimation per tween pair)
- Spatially consistent
- AI-driven camera interpolation
- Requires AnyView model

**Best for:**
- Complex camera movements
- Scenes with occlusions
- Smooth temporal consistency
- Experimental/research use

**Limitations:**
- Not compatible with RAFT
- Not compatible with ControlNet
- Higher VRAM usage
- May produce artifacts on challenging scenes

### 3. `da3_gaussian` (Advanced)

**How it works:**
1. Collect keyframes during generation
2. Estimate 3D Gaussian parameters from keyframes
3. Render novel views from Gaussian scene representation
4. Fill tween frames with rendered views

**Characteristics:**
- Very slow (full 3D reconstruction)
- Highest quality novel views
- Complete geometric understanding
- Requires Giant model

**Best for:**
- High-quality cinematic sequences
- Novel view synthesis
- 3D-aware interpolation
- Research/experimental use

**Requirements:**
- DA3-GIANT model
- 12-16GB+ VRAM
- Optional: `gsplat` library

## Performance Comparison

| Variant | VRAM | Speed | Quality | Use Case |
|---------|------|-------|---------|----------|
| **Mono-Small** | 2GB | 1.0x | Good | Standard depth warping |
| **AnyView-Small** | 3GB | 0.8x | Good+ | Multi-view tweens |
| **AnyView-Base** | 3.5GB | 0.7x | Better | Multi-view tweens (balanced) |
| **AnyView-Large** | 4GB | 0.6x | Best (non-giant) | Multi-view tweens (quality) |
| **Giant** | 12-16GB | 0.2x | Best | 3D Gaussian Splatting |

**Speed multiplier:** 1.0x = DA3Mono-Small baseline

**Note:** Multi-view mode doubles VRAM usage (processes 2 frames simultaneously)

## Advanced Features

### Ray Pose Estimation

**Parameter:** `da3_use_ray_pose` (checkbox in 3D Depth tab)

**Effect:** Enables more accurate camera pose estimation from the ray head

**Trade-offs:**
- ~10-20% slower inference
- More accurate camera poses
- Better multi-view consistency

**When to enable:**
- Using `da3_multiview` tween mode
- Using `da3_gaussian` tween mode
- Camera movements are complex

**When to disable:**
- Using `depth_warp` tween mode (no effect)
- Speed is critical
- Using Mono model (not supported)

### Confidence Threshold Percentile

**Parameter:** `da3_conf_thresh_percentile` (slider, 0-100)

**Effect:** Adaptive confidence thresholding for depth predictions

**Default:** 40.0

**How it works:**
- DA3 outputs a confidence map alongside depth
- Low-confidence regions are less reliable
- Threshold adjusts based on confidence distribution

**When to adjust:**
- **Lower (20-30):** More aggressive filtering, fewer artifacts
- **Higher (50-70):** More permissive, retain more detail
- **Default (40):** Balanced approach

### Frame Collection (3DGS)

**Parameter:** `da3_3dgs_frame_collection` (dropdown)

**Choices:**
- `keyframes`: Collect only keyframes (recommended)
- `all`: Collect keyframes + tweens (memory intensive)

**Max Frames:** `da3_3dgs_max_frames` (default: 30)

**Memory impact:**
```
keyframes only: ~30 frames × 4MB = ~120MB
all frames:     ~300 frames × 4MB = ~1.2GB
```

## Troubleshooting

### Issue: AnyView model is slow with depth_warp mode

**Cause:** AnyView provides no benefit for single-view depth warping

**Solution:** Switch to `Depth-Anything-V3-Mono-Small`

**Why:** In `depth_warp` mode, both Mono and AnyView do identical single-view depth estimation. You're paying extra VRAM/compute for multi-view features you're not using.

### Issue: da3_multiview produces artifacts

**Possible causes:**
1. Camera movement too extreme between keyframes
2. Scene has complex occlusions
3. Model size too small

**Solutions:**
1. Reduce keyframe spacing (lower cadence)
2. Use larger model (AnyView-Base or Large)
3. Enable `da3_use_ray_pose` for better accuracy
4. Fall back to `depth_warp` mode for problematic scenes

### Issue: da3_use_ray_pose is grayed out

**Cause:** Mono models don't support ray pose estimation

**Solution:** Select an AnyView or Giant model

**Why:** DA3Mono-Large lacks ray map outputs, so ray-based pose estimation is impossible. Only AnyView/Giant variants have ray heads.

### Issue: Out of VRAM with da3_multiview

**Cause:** Multi-view mode processes 2 frames simultaneously

**Solutions:**
1. Use smaller AnyView model (Small instead of Base/Large)
2. Reduce image resolution
3. Fall back to `depth_warp` mode
4. Use Mono model (but switch to depth_warp mode!)

### Issue: da3_gaussian crashes or fails

**Requirements checklist:**
- [ ] Using DA3-GIANT model?
- [ ] At least 12GB VRAM available?
- [ ] Frame count ≤ `da3_3dgs_max_frames`?
- [ ] Render mode is `Keyframes Only` or `New 3D`?

**Optional performance:**
- [ ] `gsplat` library installed? (`pip install gsplat`)

## Summary: Quick Decision Tree

```
┌─ Need 3D Gaussian Splatting?
│  └─ YES → Use DA3-GIANT with da3_gaussian mode
│  └─ NO ↓
│
├─ Want multi-view geometry?
│  └─ YES → Use AnyView-Small/Base/Large with da3_multiview mode
│  └─ NO ↓
│
└─ Standard depth warping?
   └─ Use Mono-Small with depth_warp mode (FASTEST)
```

**90% of users should use:** `Depth-Anything-V3-Mono-Small` with `depth_warp` mode

**Default settings are optimal for most workflows!**

## Related Documentation

- [DA3_TESTING_GUIDE.md](DA3_TESTING_GUIDE.md) - Technical testing procedures
- [DA3_3DGS_TUNING_GUIDE.md](DA3_3DGS_TUNING_GUIDE.md) - 3D Gaussian Splatting tuning
- [DEPTH_ANYTHING_V3_PLAN.md](DEPTH_ANYTHING_V3_PLAN.md) - Implementation phases and architecture

## References

- [Depth Anything V3 Paper](https://arxiv.org/abs/2511.10647)
- [Official GitHub Repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [Official Website](https://depth-anything-3.github.io/)
