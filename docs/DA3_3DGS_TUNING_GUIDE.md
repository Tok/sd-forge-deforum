# DA3-3DGS Tuning Guide

## Current State Analysis

### Parameters That Need Tuning

#### 1. **Prompt Strategy** (CRITICAL - User's Main Concern)
**Problem:** Unknown optimal prompt diversity for 3DGS scene building
- **Similar prompts** → More consistent images → Better 3DGS reconstruction?
- **Varied prompts** → More diverse views → Better scene coverage?

**Hypothesis:**
- 3DGS needs **geometric consistency** more than **semantic diversity**
- Too much prompt variation = inconsistent scene geometry = poor splat alignment
- Too little variation = redundant keyframes = wasted computation

**Tuning Strategy:**
1. Test prompt repetition rates (1 prompt vs 5 prompts vs 10 prompts for same scene)
2. Measure: Multi-view consistency (SSIM between rendered views)
3. Measure: Geometric accuracy (depth map correlation)
4. Measure: Visual quality (perceptual hash similarity)

#### 2. **Neighbor Segments** (neighbor_segments)
**Current:** 4 segments (default)
**Range:** 0-8
**Purpose:** How many neighboring keyframes to collect for 3DGS scene building

**Trade-offs:**
- **More neighbors** → Better scene context → More VRAM → Slower
- **Fewer neighbors** → Faster → Less context → Possible gaps in scene

**Tuning Strategy:**
1. Test: 0, 2, 4, 6, 8 neighbors
2. Measure: Scene reconstruction quality (splat coverage)
3. Measure: VRAM usage
4. Measure: Rendering time

#### 3. **Densification Factor** (densification_factor)
**Current:** 3 (2.1M splats, default)
**Range:** 1-8
**Purpose:** Subdivide gaussians for finer detail

**Trade-offs:**
- **Higher** → More detail → More VRAM → Slower rendering
- **Lower** → Faster → Less detail → Possible artifacts

**Current VRAM estimates seem wrong:**
- Factor 1: 705k splats - Listed as 8GB (seems high?)
- Factor 3: 2.1M splats - Listed as 16GB
- Factor 8: 5.6M splats - Listed as 40GB+

**Tuning Strategy:**
1. Measure ACTUAL VRAM usage per factor
2. Test quality vs speed trade-offs
3. Find optimal factor for different resolutions (512, 768, 1024)

#### 4. **Near-Clip Distance** (near_clip_distance)
**Current:** 0.1 (minimal filtering, default)
**Range:** 0.0-5.0
**Purpose:** Remove "straw" artifacts close to camera

**Trade-offs:**
- **Higher** → Fewer artifacts → Risk of black frames
- **Lower** → Keep more splats → More artifacts

**Tuning Strategy:**
1. Test with different camera movements (zoom in, zoom out, orbit)
2. Measure: Artifact count (visual inspection)
3. Measure: Frame quality (perceptual metrics)

#### 5. **Deforum Motion Schedules** (EXPERIMENTAL - User's Concern)
**Current:** Disabled by default (da3_3dgs_use_deforum_motion = False)
**Issue:** "3D movement and passing deforum schedules is still way off, not sure if the rolling issue is fixed now"

**Problems to investigate:**
- Rolling/spinning artifacts when using Deforum schedules?
- Coordinate system mismatch between DA3 and Deforum?
- Cumulative drift over long sequences?

**Tuning Strategy:**
1. Test simple movements (translation_x only, rotation_y only)
2. Test complex movements (combined translation + rotation)
3. Compare DA3 auto-estimated poses vs Deforum schedules
4. Measure: Geometric stability (point cloud drift)
5. Measure: Visual artifacts (rolling, spinning, warping)

### Metrics for 3DGS Quality

#### Geometric Metrics
1. **Multi-View Consistency (SSIM)**
   - Render same scene from multiple camera angles
   - Measure SSIM between overlapping regions
   - Higher = better geometric consistency

2. **Depth Map Correlation**
   - Compare DA3 estimated depth vs rendered depth
   - Pearson correlation coefficient
   - Higher = better geometric accuracy

3. **Splat Coverage**
   - % of frame covered by gaussian splats
   - Too low = holes in scene
   - Too high = over-densification

#### Visual Metrics
1. **Perceptual Hash Similarity**
   - Compare rendered frames to original keyframes
   - Lower distance = better fidelity

2. **Color Consistency**
   - Measure color drift over interpolation sequence
   - Standard deviation of mean RGB per frame

3. **Temporal Stability**
   - Frame-to-frame SSIM for tween sequence
   - Higher = smoother interpolation

#### Performance Metrics
1. **VRAM Usage (GB)**
2. **Render Time (seconds/frame)**
3. **Scene Build Time (seconds)**

## Recommended Tuning Workflow

### Phase 1: Baseline Measurement
1. Run with default settings (neighbor=4, densification=3, near_clip=0.1)
2. Record all metrics for a standard test scene
3. Create baseline reference

### Phase 2: Individual Parameter Sweeps
1. **Prompt diversity sweep** (5 levels: 1, 3, 5, 10, 20 unique prompts)
2. **Neighbor segments sweep** (6 levels: 0, 2, 4, 6, 8)
3. **Densification sweep** (5 levels: 1, 2, 3, 4, 5)
4. **Near-clip sweep** (6 levels: 0.0, 0.1, 0.5, 1.0, 2.0, 5.0)

### Phase 3: Multi-Dimensional Optimization
1. Grid search on top 2-3 parameters
2. Find Pareto frontier (quality vs speed)
3. Recommend presets (Fast, Balanced, Quality)

### Phase 4: Motion Schedule Testing
1. Test simple translations (X, Y, Z separately)
2. Test simple rotations (X, Y, Z separately)
3. Test combined movements
4. Identify rolling/drift issues
5. Develop coordinate transformation fixes

## Test Scenes for Tuning

### Scene 1: Static Object (Geometric Accuracy)
- Single object (teapot, character model)
- Fixed lighting
- Orbit camera path
- Measure: Geometric reconstruction quality

### Scene 2: Indoor Room (Scene Complexity)
- Multiple objects, walls, furniture
- Forward/backward camera movement
- Measure: Occlusion handling, scene coverage

### Scene 3: Outdoor Landscape (Scale Variation)
- Trees, mountains, sky
- Fly-through camera path
- Measure: Depth range handling, far-field quality

### Scene 4: Character Animation (Motion Handling)
- Moving subject
- Tracking camera
- Measure: Motion vs geometry separation

## Implementation Plan for Tuning Lab

### UI Design
```
┌─────────────────────────────────────────────────────────────┐
│  🌌 DA3-3DGS Tuning Lab                                     │
├─────────────────────────────────────────────────────────────┤
│  Test Configuration                                          │
│  ├─ Test Type: [Dropdown]                                   │
│  │   - Prompt Diversity Sweep                               │
│  │   - Neighbor Segments Sweep                              │
│  │   - Densification Quality vs VRAM                        │
│  │   - Near-Clip Artifact Reduction                         │
│  │   - Motion Schedule Stability                            │
│  │                                                           │
│  ├─ Resolution: [512] [768] [1024]                         │
│  ├─ Test Scene: [Dropdown]                                  │
│  │   - Static Object (Geometric)                           │
│  │   - Indoor Room (Complexity)                            │
│  │   - Outdoor Landscape (Scale)                           │
│  │   - Character Motion (Dynamics)                         │
│  │                                                           │
│  └─ [🚀 Run Tuning Test]                                    │
│                                                              │
│  Results Visualization                                      │
│  ├─ Quality Metrics Chart (SSIM, Depth Correlation)        │
│  ├─ Performance Chart (VRAM, Time)                         │
│  ├─ Pareto Frontier (Quality vs Speed)                     │
│  └─ Best Settings Recommendation                            │
└─────────────────────────────────────────────────────────────┘
```

### Key Features
1. **Pre-defined test scenes** (no manual prompt entry needed)
2. **Automated metric calculation** (no manual inspection)
3. **Visual comparison gallery** (side-by-side renders)
4. **Recommendation engine** (suggests optimal settings)
5. **Export results to JSON** (for documentation/sharing)

## Critical Issues to Investigate

### 1. Prompt Diversity Impact
**Question:** Do similar prompts or varied prompts work better for 3DGS?

**Test Design:**
- Scene: Indoor room with furniture
- Keyframes: 10 frames
- Prompt strategies:
  1. **Identical:** "modern living room, couch, coffee table, window" (×10)
  2. **Slight variation:** Rotate through 3 descriptions
  3. **High variation:** Unique description per keyframe
- Measure: Multi-view SSIM, geometric consistency

**Expected outcome:** Similar prompts likely better (geometric consistency matters more than semantic diversity)

### 2. Rolling/Drift with Deforum Schedules
**Question:** Why does passing Deforum schedules cause rolling/spinning?

**Hypothesis:**
- Coordinate system mismatch (DA3 uses camera-to-world, Deforum uses world-to-camera?)
- Rotation quaternion vs Euler angle conversion errors
- Cumulative drift from frame-to-frame transformations

**Test Design:**
- Simple rotation_3d_y: 0:(0), 100:(360)
- Measure: Point cloud stability, rendered frame rotation
- Compare: DA3 auto-estimated poses vs Deforum-driven poses

### 3. VRAM Estimates Accuracy
**Question:** Are the current VRAM estimates (8GB for factor=1, 16GB for factor=3) accurate?

**Test Design:**
- Measure torch.cuda.mem_get_info() before/after each densification level
- Test at different resolutions (512, 768, 1024)
- Create accurate VRAM usage table

## Next Steps

1. **Implement DA3-3DGS Tuning Tab** in Tuning Lab UI
2. **Create test scene generators** (procedural scenes for consistent testing)
3. **Implement automated metrics** (SSIM, depth correlation, VRAM tracking)
4. **Run initial sweep** on all parameters
5. **Document findings** in this guide
6. **Fix rolling/drift issues** with Deforum schedules
7. **Create presets** (Fast, Balanced, Quality) based on results

## Open Questions

1. How does prompt similarity affect 3DGS reconstruction quality?
2. What's the optimal neighbor_segments for different scene types?
3. Are current VRAM estimates accurate?
4. Why does Deforum motion cause rolling artifacts?
5. What's the minimum densification_factor for acceptable quality?
6. Does near_clip_distance need to be adaptive based on scene depth?
