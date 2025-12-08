# Testing DA3-Multiview Interpolation

Quick guide to test the new DA3-Multiview interpolation method.

## Prerequisites

1. **DA3-AnyView model** (auto-downloads on first use)
   - Recommended: AnyView-Small (120MB)
   - Manual download: `./shell_scripts/download-all-models.sh` → Option 8 (Recommended set)

2. **Diffusion model** for keyframes
   - Z-Image-Turbo (recommended for speed)
   - OR Flux Dev/Schnell
   - OR Lumina 2.0

## Quick Test Settings

### 1. Basic Setup
- **Render Mode**: Keyframes + Interpolation
- **Animation Length**: 60 frames (2 seconds at 30fps)
- **Resolution**: 512x512 (fast testing)
- **Sampler**: Euler (fast)
- **Steps**: 4 (Z-Image Turbo) or 20 (others)

### 2. Prompts Tab
Create 2-3 simple keyframes with different subjects:
```
0: a red cube on a table
30: a blue sphere on a table
```

### 3. Distribution Tab (Keyframes + Interpolation Settings)
- **Interpolation Method**: DA3-Multiview
- Leave other settings default

### 4. Run Tab
- **Generate frames**: Click the button!

## What to Expect

**Phase 1: Keyframe Generation**
```
📸 Generating keyframe 1/2 (frame 0)...
✅ Keyframe 1 saved: 000000000.png
📸 Generating keyframe 2/2 (frame 30)...
✅ Keyframe 2 saved: 000000030.png
```

**Phase 2: DA3-Multiview Interpolation**
```
🔍 Initializing DA3 depth model for multi-view geometry...
✅ Depth model loaded: Depth-Anything-V3-AnyView-Small

🎞️ Interpolation Segment 1/1:
   From keyframe: 0
   To keyframe: 30
   In-between frames to generate: 29 (frames 1 to 29)

🎯 DA3-Multiview interpolation: 29 frames
   Using DA3 AnyView model for multi-view geometry
   Running DA3 multi-view inference...
   First depth: torch.Size([1, 1, 512, 512])
   Generating tween 1/29 (t=0.033)
   Generating tween 2/29 (t=0.067)
   ...
✅ DA3-Multiview generated 29 tween frames
```

**Phase 3: Video Stitching**
```
🎬 Stitching 30 total frames into video...
✅ Video stitched successfully
📁 Output: .../output/20251208123456_zit_da3-multiview.mp4
```

## Expected Results

**Good Interpolation:**
- Smooth morphing between cube → sphere
- Geometric consistency (table stays stable)
- Color transition (red → blue)
- No flickering or jitter

**Quality vs Other Methods:**
- **vs Wan FLF2V**: Faster, more geometric, less "AI creative liberty"
- **vs FILM**: More depth-aware, better 3D consistency
- **vs DA3-3DGS**: Lighter (120MB vs 4.6GB), faster, but less quality

## Troubleshooting

### "DA3-Multiview requires Depth Anything V3 (not V2)"
- DA3 package not installed
- Auto-install will trigger, then restart app

### "DA3-Multiview requires AnyView variant"
- Model auto-switches to AnyView-Small
- Check console for download progress

### Tweens look like simple blends
- This is expected for very similar keyframes
- Try more dramatic changes:
  ```
  0: a red sports car, front view
  30: a blue sports car, side view
  ```

### Tweens have artifacts
- DA3's camera pose estimation may not work well for dramatic changes
- Try:
  - Simpler movements
  - More keyframes (shorter segments)
  - Different interpolation method (Wan for semantic changes)

## Advanced Testing

### Test with Camera Movement
```
0: a house, frontal view
15: a house, 45 degree angle view
30: a house, side view
```

Expected: DA3's multi-view geometry should handle perspective changes better than simple blending.

### Test with Prompt Scheduling
Enable strength scheduling in Keyframes tab to control how much the interpolation follows the diffusion vs depth warping.

### Compare Methods
Run same keyframes with:
1. DA3-Multiview (geometric, 120MB)
2. FILM (optical flow)
3. Wan FLF2V (AI video, creative)

Compare quality, speed, and VRAM usage.

## Performance Notes

**VRAM Usage:**
- Keyframes (Z-Image): ~4-6GB
- DA3-Multiview: ~2GB additional (AnyView-Small model)
- Total: ~6-8GB for 512x512

**Speed:**
- Z-Image keyframes: ~2-3s per frame
- DA3 multi-view inference: ~1s per segment pair
- Depth warping: ~0.1s per tween
- **Total for 60 frames**: ~1-2 minutes (much faster than Wan's 5-10 min)

## Success Criteria

✅ DA3-AnyView auto-loads without errors
✅ Tweens generated between keyframes
✅ Smooth morphing (not just linear blends)
✅ Video stitches correctly
✅ Faster than Wan FLF2V
✅ Better geometry than FILM

If you see all ✅, DA3-Multiview is working correctly!
