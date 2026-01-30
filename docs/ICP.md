# Multi-View Depth Fusion & ICP Alignment

## Overview

**Problem:** Monocular depth estimation (DA3) produces independent depth maps for each frame. When camera rotates >5-10°, disocclusions (holes) appear that require diffusion-based inpainting, which can cause temporal inconsistency.

**Proposed Solution:** Fuse multiple monocular depth estimates from different camera angles using Iterative Closest Point (ICP) alignment and voxel grid averaging to create a unified 3D point cloud, then use 3D Gaussian Splatting (3DGS) for novel view rendering.

**Key Insight:** ICP doesn't need to run on every frame - apply dynamically based on:
- Movement speed (rotation/translation magnitude)
- Keyframe distance (temporal gap)
- Depth confidence (DA3 uncertainty estimates)

---

## Current Architecture

### Deforum 3D Warping Pipeline
```
Frame N → DA3 Monocular Depth → transform_image_3d_new() → Warped Frame N+1
                                 ↓
                        p3d.euler_angles_to_matrix()
                        + depth-based 3D transform
```

**Current limitations:**
- Each frame gets independent depth prediction (`deforum/animation/animation.py:123-128`)
- No cross-frame geometric consistency
- Disocclusions filled by `cv2.BORDER_WRAP` or diffusion
- Camera poses are relative deltas (schedules), not absolute

### Proposed Multi-View Fusion Pipeline
```
Keyframes 1-N → ICP Alignment → Voxel Fusion → Fused Point Cloud → 3DGS Init
                                                                        ↓
                                                              Render Novel View
                                                                        ↓
                                                           ControlNet Inpaint Holes
```

---

## Technical Components

### 1. Point Cloud Generation from Monocular Depth

**Input:**
- DA3 depth map (H, W) - metric-relative depth
- RGB image (H, W, 3)
- Camera intrinsics K (focal length, principal point)
- Camera pose (R, t) - 4x4 transformation matrix

**Implementation:**
```python
def unproject_frame(depth, image, K, pose):
    """Convert depth map + RGB to 3D point cloud in world space.

    Args:
        depth: (H, W) depth map from DA3
        image: (H, W, 3) RGB image
        K: (3, 3) camera intrinsics matrix
        pose: (4, 4) camera-to-world transformation matrix

    Returns:
        open3d.geometry.PointCloud with colors
    """
    import open3d as o3d
    import numpy as np

    # Generate pixel grid
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Unproject to camera space
    z = depth.flatten()
    x = (u.flatten() - K[0, 2]) * z / K[0, 0]
    y = (v.flatten() - K[1, 2]) * z / K[1, 1]

    # Stack into homogeneous coordinates
    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)

    # Transform to world space
    points_world = (pose @ points_cam.T).T[:, :3]

    # Filter invalid depths
    valid_mask = z > 0
    points_world = points_world[valid_mask]
    colors = image.reshape(-1, 3)[valid_mask] / 255.0

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
```

**Challenge:** Deforum schedules are **relative deltas**, not absolute poses:
```python
# Example schedule (from deforum/config/args.py)
rotation_3d_y: "0:(0), 20:(5), 40:(0)"  # Relative rotation per frame

# Need to integrate deltas → absolute poses (numerically unstable over many frames)
```

---

### 2. Colored ICP Alignment

**Purpose:** Align point clouds from consecutive frames using both geometry and color.

**Implementation:**
```python
def register_frame_pair(source_pcd, target_pcd, initial_pose):
    """Register two point clouds using Colored ICP.

    Args:
        source_pcd: Point cloud from frame N
        target_pcd: Point cloud from frame N-1
        initial_pose: Initial transformation guess from Deforum schedules

    Returns:
        Refined 4x4 transformation matrix
    """
    import open3d as o3d

    result = o3d.pipelines.registration.registration_colored_icp(
        source_pcd,
        target_pcd,
        max_correspondence_distance=0.02,  # 2cm, scene-dependent
        init=initial_pose,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=50
        )
    )

    return result.transformation
```

**For large rotations (>15°), use Trimmed ICP:**
```python
# Trimmed ICP ignores worst 30% of correspondences (disoccluded regions)
# Requires manual implementation or robust kernel in Open3D
def trimmed_icp(source, target, initial_pose, trim_fraction=0.3):
    # 1. Compute all point-to-plane correspondences
    # 2. Sort by distance
    # 3. Keep only top 70% (trim worst 30%)
    # 4. Run ICP on trimmed set
    pass  # Implementation needed
```

---

### 3. Voxel Grid Fusion

**Purpose:** Merge multiple point clouds into a single unified representation, reducing noise via averaging.

**Implementation:**
```python
def fuse_point_clouds(pcd_list, voxel_size=0.005):
    """Fuse multiple point clouds via voxel grid averaging.

    Args:
        pcd_list: List of aligned point clouds
        voxel_size: Voxel size in meters (scene-dependent)

    Returns:
        Fused point cloud with reduced noise
    """
    import numpy as np
    import open3d as o3d

    # Combine all points
    all_points = np.vstack([np.asarray(pcd.points) for pcd in pcd_list])
    all_colors = np.vstack([np.asarray(pcd.colors) for pcd in pcd_list])

    # Voxel hashing: points → voxel key → average
    voxel_dict = {}
    for pt, col in zip(all_points, all_colors):
        voxel_key = tuple(np.floor(pt / voxel_size).astype(int))
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append((pt, col))

    # Average within each voxel
    fused_points = []
    fused_colors = []
    for pts_cols in voxel_dict.values():
        pts, cols = zip(*pts_cols)
        fused_points.append(np.mean(pts, axis=0))
        fused_colors.append(np.mean(cols, axis=0))

    # Create fused point cloud
    fused = o3d.geometry.PointCloud()
    fused.points = o3d.utility.Vector3dVector(fused_points)
    fused.colors = o3d.utility.Vector3dVector(fused_colors)
    return fused
```

**Auto-tuning voxel size based on scene scale:**
```python
def estimate_scene_scale(depth_maps):
    """Estimate scene scale from depth statistics.

    Args:
        depth_maps: List of (H, W) depth arrays

    Returns:
        Recommended voxel_size in meters
    """
    all_depths = np.concatenate([d.flatten() for d in depth_maps])
    depth_p99 = np.percentile(all_depths[all_depths > 0], 99)
    voxel_size = depth_p99 / 200  # Adaptive to scene scale
    return voxel_size
```

---

### 4. 3DGS Initialization from Fused Cloud

**Current:** Deforum uses gsplat for DA3-3DGS mode (`requirements.txt:35`)

**Modification needed:**
```python
# deforum/rendering/keyframe_interp.py integration point
# Currently initializes 3DGS from single-frame DA3 depth
# Modify to accept fused multi-view point cloud

def init_3dgs_from_fused_cloud(fused_pcd, keyframe_images):
    """Initialize 3D Gaussians from fused point cloud.

    Args:
        fused_pcd: Merged point cloud from ICP fusion
        keyframe_images: List of keyframe RGB images for texture

    Returns:
        Initialized Gaussian parameters (positions, covariances, colors, opacities)
    """
    import gsplat

    positions = np.asarray(fused_pcd.points)
    colors = np.asarray(fused_pcd.colors)

    # Initialize with small covariances (tight splats)
    # Prevents "floaters" from uncertain depth
    num_points = len(positions)
    scales = np.ones((num_points, 3)) * 0.001  # 1mm initial radius
    rotations = np.zeros((num_points, 4))
    rotations[:, 0] = 1.0  # Identity quaternion
    opacities = np.ones(num_points) * 0.5

    # Disable Spherical Harmonics (SH degree=0) for single-view
    # Multi-view training would need SH for view-dependent appearance
    sh_coeffs = colors[:, :, None]  # (N, 3, 1) for degree=0

    return {
        'positions': positions,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacities,
        'sh_coeffs': sh_coeffs
    }
```

---

### 5. Scale Normalization Across Frames

**Problem:** DA3 outputs metric-relative depth per frame, but absolute scale varies.

**Solution:** Lock scale using camera translation magnitude from schedules.

```python
def normalize_depth_scales(depth_maps, poses):
    """Rescale depth maps for consistent metric scale across frames.

    Args:
        depth_maps: List of (H, W) depth arrays from DA3
        poses: List of 4x4 camera poses (integrated from Deforum deltas)

    Returns:
        List of rescaled depth maps with consistent scale
    """
    # Use first frame as reference scale
    base_translation = np.linalg.norm(poses[1][:3, 3] - poses[0][:3, 3])

    normalized_depths = [depth_maps[0]]
    for i in range(1, len(depth_maps)):
        current_trans = np.linalg.norm(poses[i][:3, 3] - poses[i-1][:3, 3])
        scale_factor = base_translation / (current_trans + 1e-8)
        normalized_depths.append(depth_maps[i] * scale_factor)

    return normalized_depths
```

**Alternative approach using depth overlap:**
```python
# Use depth consistency between frames to estimate relative scale
# Similar to ORB-SLAM's scale estimation
def estimate_scale_from_overlap(depth1, depth2, transform):
    # 1. Warp depth1 to frame2 using transform
    # 2. Find overlapping regions
    # 3. Compute ratio: depth2[overlap] / warped_depth1[overlap]
    # 4. Use median ratio as scale factor
    pass  # Implementation needed
```

---

## Dynamic ICP Application Strategy

**Key insight:** ICP is computationally expensive - apply selectively based on need.

### Triggering Conditions

```python
def should_apply_icp(frame_idx, keys, prev_depth_confidence):
    """Determine if ICP fusion is needed for this frame pair.

    Args:
        frame_idx: Current frame index
        keys: Animation keyframe schedules
        prev_depth_confidence: DA3 confidence from previous frame

    Returns:
        bool: True if ICP should be applied
    """
    # 1. Check movement magnitude
    rotation_magnitude = abs(keys.rotation_3d_y_series[frame_idx])
    translation_magnitude = abs(keys.translation_z_series[frame_idx])

    # Trigger ICP for large rotations (>10°) or fast movement
    if rotation_magnitude > 10.0 or translation_magnitude > 50.0:
        return True

    # 2. Check keyframe distance
    # Only fuse keyframes that are far apart temporally
    is_keyframe = check_is_keyframe(frame_idx)
    prev_keyframe_dist = frames_since_last_keyframe(frame_idx)
    if is_keyframe and prev_keyframe_dist > 20:
        return True

    # 3. Check depth confidence
    # Skip ICP if DA3 is already confident
    if prev_depth_confidence > 0.8:  # High confidence
        return False

    return False
```

### Batching Strategy

```python
def batch_keyframes_for_fusion(keyframe_indices, movement_schedule):
    """Group keyframes into batches for ICP fusion.

    Args:
        keyframe_indices: List of keyframe frame numbers
        movement_schedule: Camera movement data

    Returns:
        List of keyframe batches to fuse together
    """
    batches = []
    current_batch = [keyframe_indices[0]]

    for i in range(1, len(keyframe_indices)):
        prev_kf = keyframe_indices[i-1]
        curr_kf = keyframe_indices[i]

        # Calculate movement between keyframes
        rotation_change = abs(
            movement_schedule['rotation_3d_y'][curr_kf] -
            movement_schedule['rotation_3d_y'][prev_kf]
        )

        # Start new batch if rotation exceeds threshold
        if rotation_change > 30.0:  # 30° threshold
            batches.append(current_batch)
            current_batch = [curr_kf]
        else:
            current_batch.append(curr_kf)

    batches.append(current_batch)
    return batches
```

---

## Integration Points

### Files to Modify

1. **`deforum/animation/animation.py:172`** - `anim_frame_warp_3d()`
   - Add optional multi-frame fusion mode
   - Call ICP pipeline when threshold met

2. **`deforum/rendering/helpers/depth.py:32`** - `create_depth_model_and_enable_depth_map_saving_if_active()`
   - Batch process multiple frames when fusion enabled

3. **`deforum/config/args.py`**
   - Add UI controls: `enable_icp_fusion`, `icp_voxel_size`, `icp_rotation_threshold`

4. **`deforum/ui/tabs/tab_depth.py`**
   - New "Multi-View Fusion" accordion section

### New Files Needed

```
deforum/depth/point_cloud_fusion.py       # ICP + voxel fusion
deforum/depth/camera_pose_estimation.py   # Integrate Deforum deltas → poses
deforum/depth/confidence_utils.py          # DA3 confidence map handling
```

---

## Technical Assessment

### Feasibility Analysis

| Component | Complexity | Dependencies | Notes |
|-----------|-----------|--------------|-------|
| Point cloud generation | Medium | open3d | Standard CV operation |
| ICP alignment | High | open3d | Convergence issues on textureless regions |
| Voxel fusion | Medium | numpy | Hash table implementation needed |
| Pose estimation from deltas | High | None | Numerical instability over many frames |
| 3DGS initialization | Low | gsplat (exists) | Minor API change |
| Scale normalization | Medium | None | Need depth overlap method |

### Performance Impact

**Current baseline (3D mode):**
- DA3 depth estimation: ~2GB VRAM, ~3s per frame
- 3D warping: CPU-bound, <500MB, <1s
- **Total per-frame: ~2.5GB, ~3s**

**With ICP fusion (20 keyframes):**
```
Phase 1: Generate 20 keyframes          → 20 × 3s = 60s
Phase 2: ICP alignment (pairwise)       → 20 × 5s = 100s
Phase 3: Voxel fusion                   → ~10s
Phase 4: 3DGS training                  → ~120s
Total: ~290s (vs. 60s baseline) = 4.8× slower
```

**VRAM footprint:**
- 20 point clouds (921k points/frame): ~442MB
- Voxel grid (scene-dependent): ~200MB
- 3DGS Gaussians: ~1-6MB
- **Peak: ~3.1GB total** (acceptable for RTX 3060+ with 12GB)

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| ICP fails on textureless regions | High | Use Trimmed ICP + DA3 confidence weighting |
| Scale drift across frames | High | Depth overlap estimation instead of schedule integration |
| Pose integration instability | High | Use optical flow ego-motion (COLMAP-style) |
| Performance penalty too high | Medium | Apply ICP selectively (dynamic triggering) |
| Open3D dependency conflicts | Medium | Vendor specific modules or use lightweight Python ICP |
| Voxel size tuning difficulty | Low | Auto-estimate from depth percentiles |

### Dependencies Required

```python
# requirements.txt additions:
open3d>=0.18.0  # ~200MB, point cloud processing
# Note: scipy>=1.10.0 already present in requirements.txt:40
```

**Compatibility:**
- ✅ Open3D has Python 3.12 wheels (unlike xformers)
- ⚠️ Open3D may conflict with specific numpy versions pinned by diffusers
- ✅ Open3D prebuilt wheels available for CUDA 11.8/12.1

---

## Alternatives & Incremental Approaches

### Option A: Depth Temporal Smoothing (Low-Hanging Fruit)
```python
# Exponential moving average across frames
def smooth_depth_temporal(depth_current, depth_prev_warped, alpha=0.7):
    """Smooth depth estimates across frames.

    Args:
        depth_current: Current frame depth from DA3
        depth_prev_warped: Previous depth warped to current view
        alpha: Smoothing factor (0=fully previous, 1=fully current)

    Returns:
        Smoothed depth map
    """
    return alpha * depth_current + (1 - alpha) * depth_prev_warped
```

**Pros:** Trivial to implement, reduces jitter, no new dependencies
**Cons:** Doesn't solve disocclusions
**Integration:** `deforum/animation/animation.py:123` after DA3 prediction

### Option B: Confidence-Weighted Inpainting
```python
# Use DA3 confidence maps to guide inpainting priority
def confidence_weighted_inpaint(image, disocclusion_mask, confidence_map, threshold=0.5):
    """Prioritize inpainting low-confidence disoccluded regions.

    Args:
        image: Warped image with holes
        disocclusion_mask: Binary mask of disoccluded pixels
        confidence_map: DA3 confidence (0-1)
        threshold: Confidence threshold below which to inpaint

    Returns:
        Inpainted image
    """
    # Only inpaint where both disoccluded AND low confidence
    inpaint_mask = disocclusion_mask & (confidence_map < threshold)
    return cv2.inpaint(image, inpaint_mask, 3, cv2.INPAINT_TELEA)
```

**Pros:** Leverages existing DA3 confidence output, targeted fixes
**Cons:** Still requires diffusion for large holes
**Integration:** `deforum/animation/animation.py:202` after 3D transform

### Option C: RAFT Optical Flow (Already Exists!)
- Deforum already has RAFT integration for optical flow
- Flow-based warping naturally handles moderate rotations (10-20°)
- **Just needs better documentation + UI exposure**

---

## Next Steps (If Pursuing)

### Research Phase (Before Implementation)

1. **Validate pose estimation from Deforum deltas**
   - Implement delta integration → absolute poses
   - Measure drift over 100-300 frames
   - Compare with optical flow ego-motion estimation
   - **Success criteria:** <1° rotation error after 100 frames

2. **Benchmark ICP convergence on real Deforum sequences**
   - Generate test sequences with 5°, 15°, 30° rotations
   - Measure ICP convergence rates, alignment errors
   - Identify failure modes (textureless regions, occlusion)
   - **Success criteria:** >80% convergence rate on typical scenes

3. **User demand validation**
   - Poll existing users: Do they want geometric consistency over aesthetic diffusion?
   - Identify specific use cases where ICP fusion is critical
   - **Success criteria:** >20% of users express need

### Implementation Phasing (If Research Validates)

**Phase 1:** Point cloud infrastructure (no ICP yet)
- Depth → point cloud conversion
- Camera pose integration from deltas
- Scene scale auto-tuning

**Phase 2:** Simple voxel fusion (no ICP alignment)
- Multi-frame voxel averaging
- Confidence-based weighting
- Test if simple averaging helps without alignment

**Phase 3:** Full ICP pipeline
- Colored ICP integration
- Trimmed ICP for large rotations
- Dynamic triggering logic

**Phase 4:** 3DGS integration
- Use fused cloud for 3DGS initialization
- Depth supervision loss during training
- Hybrid disocclusion rendering

---

## References

- **Colored ICP:** Park et al., "Colored Point Cloud Registration Revisited" (ICCV 2017)
- **Trimmed ICP:** Chetverikov et al., "The Trimmed Iterative Closest Point Algorithm" (ICPR 2002)
- **TSDF Fusion:** Curless & Levoy, "A Volumetric Method for Building Complex Models from Range Images" (SIGGRAPH 1996)
- **Open3D:** Zhou et al., "Open3D: A Modern Library for 3D Data Processing" (arXiv 2018)
- **Deforum Depth Warping:** `deforum/animation/animation.py:172-214`
- **DA3 Integration:** `deforum/depth/depth_anything_v3.py`
- **3DGS in Deforum:** `deforum/rendering/keyframe_interp.py` (DA3-3DGS mode)

---

*Document created: 2026-01-30*
*Status: Future research - deferred pending validation studies*
