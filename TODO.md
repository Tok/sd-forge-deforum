# TODO - Future Improvements

## Dashboard System

### Dashboard Unification (Future Consideration)
Currently, there are two separate dashboards:
- **`FixedDashboard`** (`deforum/utils/ui/dashboard.py`) - Used by 3D modes
  - Shows 5 tqdm bars (Current Tweens, Current Diffusion Steps, Total Diffusion Steps, Total Diffusion Frames, Total Frames)
  - Shows models (Diffusion, Depth) + VRAM status
  - Shows frame info with color blocks and movement indicators

- **`InterpolationDashboard`** (`deforum/utils/ui/interpolation_dashboard.py`) - Used by Flux+Interpolation modes
  - Shows Phase 1 (Diffusion Keyframes) progress
  - Shows Phase 2 progress (adaptive based on interpolation method)
  - Shows VRAM + current operation
  - Uses DA3 slopcore cyan→watermelon gradient

**Potential unification approach:**
- Create a single adaptive dashboard that shows/hides relevant progress bars based on render mode
- Mode-specific sections:
  - 3D modes → Show depth/warping/tweening tqdms
  - Interpolation modes → Show Phase 1/2 progress
  - All modes → Always show VRAM and operation status
- Benefits: Single codebase to maintain, consistent behavior across modes
- Trade-offs: More complexity in conditional rendering logic, may be harder to customize per-mode

**Status:** Deferred - Current separate dashboards work well for their specific use cases

---

## LTX-2 Improvements

### Movement Speed Enhancement
**Issue:** LTX-2 generates frames with barely any movement between them
**Potential solutions:**
- Adjust guidance scale (currently 4.0, try 3.0-3.5 for more motion)
- Increase audio influence (audio drives motion in LTX-2)
- Experiment with different prompt formulations
- Check if frame count affects motion (more frames = smaller steps = less apparent motion)

**Status:** Needs investigation

---

## Logging System

### Forge Log Color Compliance
**Issue:** Forge logs use yellow/green ANSI codes that don't match slopcore theme
**Current state:** Color replacement filter exists in `deforum/utils/system/output_filter.py`
- `replace_forge_colors()` converts Forge yellow/red/cyan to slopcore equivalents
- Filter can be installed globally with `install_global_color_filter()`

**Needs verification:**
- Is the global color filter being installed at extension init?
- Are all Forge log paths being captured (stdout, stderr, RichHandler)?
- Check if Forge's right-justified logging format (class name + log level on right) should be adopted

**Status:** Partially implemented, needs testing/verification

---

## Documentation

### Update CLAUDE.md
- Add new LTX-2 default resolution (1280x720 instead of 1024x1024)
- Document separate motion-aware prompt controls for LTX-2 and Wan
- Update dashboard documentation to reflect InterpolationDashboard method tracking
- Add notes about dashboard adaptation vs unification trade-offs

**Status:** Needs update

---

## Future Research

### Multi-View Depth Fusion & ICP Alignment
**Concept:** Use Iterative Closest Point (ICP) alignment to fuse multiple monocular depth estimates into a unified 3D point cloud, improving geometric consistency for large camera rotations.

**Key insight:** ICP doesn't need to run on all frames - could be applied dynamically based on:
- Movement speed (higher rotation/translation rates trigger fusion)
- Keyframe distance (only fuse when keyframes are far apart)
- Depth confidence (skip fusion when DA3 confidence is high)

**Status:** Deferred - See `docs/ICP.md` for detailed technical analysis

---

*Last updated: 2026-01-30*
