"""Handler for frame overlap simulator UI integration."""

from typing import Optional
import numpy as np

from deforum.utils.frame_overlap_simulator import simulate_camera_path
from deforum.utils.frame_overlap_canvas import create_canvas_html
from deforum.core.keyframes import FrameInterpolater
from deforum.utils.parsing.schedule_manipulation import get_final_schedules_with_shakify
from deforum.utils.system.logging import get_logger

logger = get_logger()


def update_frame_overlap_visualization(
    translation_x: str,
    translation_y: str,
    translation_z: str,
    rotation_3d_x: str,
    rotation_3d_y: str,
    rotation_3d_z: str,
    zoom: str = "",
    max_frames: int = 333,
    width: int = 1920,
    height: int = 1080,
    shake_name: str = "None",
    shake_intensity: float = 1.0,
    shake_speed: float = 1.0,
    target_fps: int = 60
) -> Optional[str]:
    """Update frame overlap visualization from schedule strings with shakify overlay and zoom.

    Args:
        translation_x: Translation X schedule string
        translation_y: Translation Y schedule string
        translation_z: Translation Z schedule string (unused for 2D overlap)
        rotation_3d_x: Rotation X schedule string (unused for 2D overlap)
        rotation_3d_y: Rotation Y schedule string (horizontal pan)
        rotation_3d_z: Rotation Z schedule string (unused for 2D overlap)
        zoom: Zoom schedule string (default: "" = no zoom, uses 1.0)
        max_frames: Maximum number of frames
        width: Viewport width in pixels
        height: Viewport height in pixels
        shake_name: Camera shakify pattern name (default: "None")
        shake_intensity: Shakify intensity multiplier (default: 1.0)
        shake_speed: Shakify speed multiplier (default: 1.0)
        target_fps: Target FPS for shakify interpolation (default: 60)

    Returns:
        HTML string with Canvas visualization, or None on error
    """
    try:
        # Build base schedules dict (zoom is handled separately, not processed by shakify)
        base_schedules = {
            'translation_x': translation_x or "0:(0)",
            'translation_y': translation_y or "0:(0)",
            'translation_z': translation_z or "0:(0)",
            'rotation_3d_x': rotation_3d_x or "0:(0)",
            'rotation_3d_y': rotation_3d_y or "0:(0)",
            'rotation_3d_z': rotation_3d_z or "0:(0)",
        }

        # Apply shakify overlay to get final combined schedules
        # Scale down intensity to 30% for subtle visualization
        viz_intensity = shake_intensity * 0.3 if shake_name != "None" else 0.0

        final_schedules = get_final_schedules_with_shakify(
            base_schedules=base_schedules,
            shake_name=shake_name,
            shake_intensity=viz_intensity,
            shake_speed=shake_speed,
            max_frames=max_frames,
            target_fps=target_fps
        )

        # Parse FINAL schedule strings (base + shakify) to get per-frame values
        tx_schedule = final_schedules['translation_x']
        ty_schedule = final_schedules['translation_y']
        rx_schedule = final_schedules['rotation_3d_x']
        ry_schedule = final_schedules['rotation_3d_y']

        # Create parser
        parser = FrameInterpolater(max_frames=max_frames)

        # Parse keyframes
        tx_keys = parser.parse_key_frames(tx_schedule)
        ty_keys = parser.parse_key_frames(ty_schedule)
        rx_keys = parser.parse_key_frames(rx_schedule)
        ry_keys = parser.parse_key_frames(ry_schedule)

        # Interpolate between keyframes to get per-frame values
        tx_series = parser.get_inbetweens(tx_keys, integer=False)
        ty_series = parser.get_inbetweens(ty_keys, integer=False)
        rx_series = parser.get_inbetweens(rx_keys, integer=False)
        ry_series = parser.get_inbetweens(ry_keys, integer=False)

        # Convert pandas Series to lists
        # IMPORTANT: Schedules are already deltas (from camera_path_to_schedules)
        # Do NOT calculate deltas again - just use interpolated values directly
        tx_deltas = tx_series.tolist()
        ty_deltas = ty_series.tolist()
        rx_deltas = rx_series.tolist()
        ry_deltas = ry_series.tolist()

        # IMPORTANT: Frame 0 often has a large initial rotation (e.g., -90°) that represents
        # the camera's starting orientation to face center, NOT a per-frame movement delta.
        # For frame overlap visualization, we need actual per-frame movements, so we:
        # 1. Zero out frame 0's rotation (it's just initial orientation, not movement)
        # 2. Use subsequent frames' rotations as actual movement deltas
        # This prevents the "rotating like crazy" issue in the wormtrail visualization.
        if len(rx_deltas) > 0:
            rx_deltas[0] = 0.0
        if len(ry_deltas) > 0:
            ry_deltas[0] = 0.0

        # Similarly, zero out frame 0's translation (start from rest)
        if len(tx_deltas) > 0:
            tx_deltas[0] = 0.0
        if len(ty_deltas) > 0:
            ty_deltas[0] = 0.0

        # For 2D visualization, combine 3D rotations into effective 2D rotation
        # Use pythagorean combination of rotation_x (pitch) and rotation_y (yaw)
        # This approximates the apparent rotation seen in a 2D top-down view
        combined_rotation_deltas = [
            np.sqrt(rx**2 + ry**2) * np.sign(ry) if abs(ry) > abs(rx) else np.sqrt(rx**2 + ry**2) * np.sign(rx)
            for rx, ry in zip(rx_deltas, ry_deltas)
        ]

        # Debug: Log first 20 frames of delta values
        logger.debug("Frame overlap delta schedules (first 20 frames):")
        for i in range(min(20, max_frames)):
            logger.debug(
                f"  Frame {i:3d}: tx={tx_deltas[i]:7.2f}, ty={ty_deltas[i]:7.2f}, "
                f"rx={rx_deltas[i]:7.2f}, ry={ry_deltas[i]:7.2f}, combined_rot={combined_rotation_deltas[i]:7.2f}"
            )

        # Parse zoom schedule (zoom is NOT processed by shakify, use base schedule directly)
        if zoom and zoom.strip():
            zoom_keys = parser.parse_key_frames(zoom)
            zoom_series = parser.get_inbetweens(zoom_keys, integer=False)
            zoom_deltas = zoom_series.tolist()
        else:
            # No zoom schedule provided, use 1.0 (no zoom)
            zoom_deltas = [1.0] * max_frames

        # Run frame overlap simulation
        # Use combined 3D rotation for 2D visualization
        metrics = simulate_camera_path(
            translation_x_schedule=tx_deltas,
            translation_y_schedule=ty_deltas,
            rotation_3d_y_schedule=combined_rotation_deltas,
            zoom_schedule=zoom_deltas,
            viewport_width=float(width),
            viewport_height=float(height)
        )

        # Create Canvas HTML visualization
        html = create_canvas_html(
            metrics_list=metrics,
            width=800,
            height=600,
            trail_length=30,  # Show 30 previous frames for longer worm trail
            playback_fps=10
        )

        return html

    except Exception as e:
        import traceback
        error_msg = f"Failed to update frame overlap visualization: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(error_msg)  # Also print to console for visibility
        return f'<div style="color: #FF5050; padding: 20px; background: rgba(60,60,80,0.3); border-radius: 4px;">❌ Error: {e}</div>'
