"""Camera Path Optimizer for Depth Warping.

Analyzes preservation/novelty metrics and suggests optimizations for better
depth warping performance. Reduces translation speed in areas with low preservation.
"""

from typing import Dict, List, Tuple
import numpy as np
from deforum.utils.frame_overlap_simulator import simulate_camera_path, FrameMetrics
from deforum.utils.parsing.schedules import parse_schedule_string, interpolate_schedule_values
from deforum.core.keyframes import FrameInterpolater


# Optimal preservation thresholds
TARGET_PRESERVATION = 0.90  # Target 90% preservation
MIN_ACCEPTABLE_PRESERVATION = 0.75  # Warn below 75%


def analyze_camera_path(
    translation_x: str,
    translation_y: str,
    translation_z: str,
    rotation_3d_x: str,
    rotation_3d_y: str,
    rotation_3d_z: str,
    max_frames: int,
    width: int,
    height: int,
) -> Tuple[List[FrameMetrics], Dict[str, float]]:
    """Analyze camera path for depth warping suitability.

    Args:
        translation_x/y/z: Translation schedule strings
        rotation_3d_x/y/z: Rotation schedule strings
        max_frames: Number of frames
        width/height: Viewport dimensions

    Returns:
        (metrics_list, analysis_summary)

    analysis_summary contains:
        - avg_preservation: Average preservation across all frames
        - min_preservation: Worst-case preservation
        - frames_below_target: % of frames below TARGET_PRESERVATION
        - frames_below_minimum: % of frames below MIN_ACCEPTABLE_PRESERVATION
        - problem_frames: List of frame indices with low preservation
    """
    # Parse schedules to get per-frame deltas
    parser = FrameInterpolater(max_frames=max_frames)

    tx_keys = parser.parse_key_frames(translation_x or "0:(0)")
    ty_keys = parser.parse_key_frames(translation_y or "0:(0)")
    ry_keys = parser.parse_key_frames(rotation_3d_y or "0:(0)")

    tx_series = parser.get_inbetweens(tx_keys, integer=False)
    ty_series = parser.get_inbetweens(ty_keys, integer=False)
    ry_series = parser.get_inbetweens(ry_keys, integer=False)

    tx_values = tx_series.tolist()
    ty_values = ty_series.tolist()
    ry_values = ry_series.tolist()

    # Calculate frame-to-frame deltas
    tx_deltas = [tx_values[i] - tx_values[i - 1] if i > 0 else 0 for i in range(max_frames)]
    ty_deltas = [ty_values[i] - ty_values[i - 1] if i > 0 else 0 for i in range(max_frames)]
    ry_deltas = [ry_values[i] - ry_values[i - 1] if i > 0 else 0 for i in range(max_frames)]
    zoom_deltas = [1.0] * max_frames

    # Run simulation
    metrics = simulate_camera_path(
        translation_x_schedule=tx_deltas,
        translation_y_schedule=ty_deltas,
        rotation_3d_y_schedule=ry_deltas,
        zoom_schedule=zoom_deltas,
        viewport_width=float(width),
        viewport_height=float(height),
    )

    # Analyze results
    preservations = [m.preservation for m in metrics]
    avg_preservation = np.mean(preservations)
    min_preservation = np.min(preservations)

    frames_below_target = sum(1 for p in preservations if p < TARGET_PRESERVATION)
    frames_below_minimum = sum(1 for p in preservations if p < MIN_ACCEPTABLE_PRESERVATION)

    problem_frames = [i for i, p in enumerate(preservations) if p < MIN_ACCEPTABLE_PRESERVATION]

    analysis = {
        "avg_preservation": float(avg_preservation),
        "min_preservation": float(min_preservation),
        "frames_below_target": frames_below_target / len(metrics),
        "frames_below_minimum": frames_below_minimum / len(metrics),
        "problem_frames": problem_frames,
        "total_frames": len(metrics),
    }

    return metrics, analysis


def generate_optimization_report(analysis: Dict[str, float]) -> str:
    """Generate human-readable optimization report.

    Args:
        analysis: Analysis summary from analyze_camera_path()

    Returns:
        Formatted markdown report with recommendations
    """
    avg = analysis["avg_preservation"]
    min_pres = analysis["min_preservation"]
    pct_below_target = analysis["frames_below_target"] * 100
    pct_below_min = analysis["frames_below_minimum"] * 100

    # Overall assessment
    if avg >= TARGET_PRESERVATION:
        status = "✅ EXCELLENT"
        color = "green"
    elif avg >= MIN_ACCEPTABLE_PRESERVATION:
        status = "⚠️ ACCEPTABLE"
        color = "yellow"
    else:
        status = "❌ NEEDS OPTIMIZATION"
        color = "red"

    report = f"""## Camera Path Analysis for Depth Warping

**Overall Status:** {status}

**Preservation Metrics:**
- Average Preservation: **{avg:.1%}** (Target: ≥{TARGET_PRESERVATION:.0%})
- Minimum Preservation: **{min_pres:.1%}** (Minimum: ≥{MIN_ACCEPTABLE_PRESERVATION:.0%})
- Frames Below Target ({TARGET_PRESERVATION:.0%}): **{pct_below_target:.1f}%**
- Frames Below Minimum ({MIN_ACCEPTABLE_PRESERVATION:.0%}): **{pct_below_min:.1f}%**

### What This Means:

**Preservation** = How much of the previous frame is still visible in the current frame.
- High preservation (>90%) = Smooth transitions, great for depth warping
- Low preservation (<75%) = Too much camera movement, depth warping will struggle

"""

    # Recommendations
    if avg >= TARGET_PRESERVATION:
        report += """### ✅ Recommendations:
Your camera path is excellent for depth warping! The movement speed maintains high frame overlap.
"""
    elif avg >= MIN_ACCEPTABLE_PRESERVATION:
        report += f"""### ⚠️ Recommendations:
Your camera path is acceptable but could be improved:
- Consider reducing translation speed by 20-30%
- Or increase cadence (render more frames) to reduce per-frame motion
- Problem frames: {len(analysis['problem_frames'])} frames need attention
"""
    else:
        report += f"""### ❌ Recommendations - Path Needs Optimization:

**Critical Issues:**
- Camera is moving too fast for effective depth warping
- {pct_below_min:.0f}% of frames have unacceptable preservation (<{MIN_ACCEPTABLE_PRESERVATION:.0%})

**Solutions:**
1. **Reduce Speed:** Cut translation values by 50% (most effective)
2. **Increase Cadence:** Render more frames to reduce per-frame delta
3. **Tighter Curves:** Use smaller radius/scale in camera path presets
4. **Lower FPS:** Slower playback = less per-frame motion

**Problem Areas:** Frames {analysis['problem_frames'][:10]}{'...' if len(analysis['problem_frames']) > 10 else ''}
"""

    return report


def auto_optimize_for_depth_warping(
    translation_x: str,
    translation_y: str,
    translation_z: str,
    rotation_3d_x: str,
    rotation_3d_y: str,
    rotation_3d_z: str,
    max_frames: int,
    width: int,
    height: int,
    target_preservation: float = TARGET_PRESERVATION,
) -> Tuple[str, str, str, str]:
    """Automatically optimize translation schedules for depth warping.

    Strategy:
    1. Analyze current path to get actual preservation metrics
    2. Calculate optimal scale factor to reach target preservation
    3. Only scale translation (preserve rotation schedules completely)

    Args:
        translation_x/y/z: Original translation schedule strings
        rotation_3d_x/y/z: Original rotation schedules (preserved, not modified)
        max_frames: Number of frames
        width/height: Viewport dimensions
        target_preservation: Target preservation (default: 0.90)

    Returns:
        (optimized_tx, optimized_ty, optimized_tz, status_message)
    """
    # First analyze current path to get baseline metrics
    try:
        _, analysis = analyze_camera_path(
            translation_x or "0:(0)",
            translation_y or "0:(0)",
            translation_z or "0:(0)",
            rotation_3d_x or "0:(0)",
            rotation_3d_y or "0:(0)",
            rotation_3d_z or "0:(0)",
            max_frames,
            width,
            height,
        )
    except Exception as e:
        return translation_x, translation_y, translation_z, f"❌ Analysis failed: {str(e)}"

    current_preservation = analysis["avg_preservation"]

    # If already at target, no optimization needed
    if current_preservation >= target_preservation:
        return (
            translation_x,
            translation_y,
            translation_z,
            f"""✅ Already Optimized!

**Current Preservation:** {current_preservation:.1%}
**Target:** {target_preservation:.0%}

No changes needed - your camera path is already suitable for depth warping!
""",
        )

    # Calculate scale factor needed to reach target
    # Preservation is roughly linear with translation magnitude
    # If we have 80% and want 90%, we need to scale translation by (80/90) = 0.89
    if current_preservation > 0.3:  # Only if we have reasonable baseline
        scale_factor = current_preservation / target_preservation
        scale_factor = max(0.3, min(0.95, scale_factor))  # Clamp to reasonable range
    else:
        # Path is too aggressive, use conservative 50% scale
        scale_factor = 0.5

    # Parse original schedules
    tx_dict = parse_schedule_string(translation_x or "0:(0)", max_frames)
    ty_dict = parse_schedule_string(translation_y or "0:(0)", max_frames)
    tz_dict = parse_schedule_string(translation_z or "0:(0)", max_frames)

    # Interpolate to get all values
    tx_interp = interpolate_schedule_values(tx_dict, max_frames)
    ty_interp = interpolate_schedule_values(ty_dict, max_frames)
    tz_interp = interpolate_schedule_values(tz_dict, max_frames)

    # Calculate magnitude of movement (for reporting)
    avg_tx = np.mean(np.abs(tx_interp))
    avg_ty = np.mean(np.abs(ty_interp))
    avg_tz = np.mean(np.abs(tz_interp))

    # Apply scaling to translation only (preserve rotation completely)
    tx_scaled = [v * scale_factor for v in tx_interp]
    ty_scaled = [v * scale_factor for v in ty_interp]
    tz_scaled = [v * scale_factor for v in tz_interp]

    # Convert back to schedule strings (sample every N frames)
    def values_to_schedule(values: List[float], max_frames: int) -> str:
        sample_interval = max(1, max_frames // 20)
        keyframes = []
        for frame in range(0, len(values), sample_interval):
            keyframes.append(f"{frame}:({values[frame]:.2f})")
        if (len(values) - 1) % sample_interval != 0:
            keyframes.append(f"{len(values)-1}:({values[-1]:.2f})")
        return ", ".join(keyframes)

    optimized_tx = values_to_schedule(tx_scaled, max_frames)
    optimized_ty = values_to_schedule(ty_scaled, max_frames)
    optimized_tz = values_to_schedule(tz_scaled, max_frames)

    # Predict new preservation (rough estimate)
    predicted_preservation = min(0.99, current_preservation / scale_factor)

    status = f"""✅ Optimized for depth warping!

**Before:**
- Preservation: {current_preservation:.1%}
- Avg translation: X={avg_tx:.1f}px, Y={avg_ty:.1f}px, Z={avg_tz:.1f}px

**After:**
- Translation scaled to {scale_factor*100:.0f}% of original
- Predicted preservation: ~{predicted_preservation:.1%}
- Avg translation: X={avg_tx*scale_factor:.1f}px, Y={avg_ty*scale_factor:.1f}px, Z={avg_tz*scale_factor:.1f}px

**Note:** Rotation schedules preserved (not modified). Click "Analyze Camera Path" to verify results.
"""

    return optimized_tx, optimized_ty, optimized_tz, status
