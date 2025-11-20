"""Handler for frame overlap simulator UI integration."""

from typing import Optional
import numpy as np

from deforum.utils.frame_overlap_simulator import simulate_camera_path
from deforum.utils.frame_overlap_canvas import create_canvas_html
from deforum.core.keyframes import FrameInterpolater
from deforum.utils.system.logging import get_logger

logger = get_logger()


def update_frame_overlap_visualization(
    translation_x: str,
    translation_y: str,
    translation_z: str,
    rotation_3d_x: str,
    rotation_3d_y: str,
    rotation_3d_z: str,
    max_frames: int,
    width: int,
    height: int,
    shake_name: str = "None",
    shake_intensity: float = 1.0,
    shake_speed: float = 1.0,
    target_fps: int = 60
) -> Optional[str]:
    """Update frame overlap visualization from schedule strings.

    Note: As of the dense camera path fix, shakify parameters are IGNORED.
    The shakify overlay function decimates dense schedules to ~20 keyframes,
    which breaks visualization of dense camera paths (333 frames → 22 keyframes).
    Frame Overlap is meant to show primary camera movement, not subtle shake overlay.

    Args:
        translation_x: Translation X schedule string
        translation_y: Translation Y schedule string
        translation_z: Translation Z schedule string (unused for 2D overlap)
        rotation_3d_x: Rotation X schedule string (unused for 2D overlap)
        rotation_3d_y: Rotation Y schedule string (horizontal pan)
        rotation_3d_z: Rotation Z schedule string (unused for 2D overlap)
        max_frames: Maximum number of frames
        width: Viewport width in pixels
        height: Viewport height in pixels
        shake_name: [IGNORED] Camera shakify pattern name
        shake_intensity: [IGNORED] Shakify intensity multiplier
        shake_speed: [IGNORED] Shakify speed multiplier
        target_fps: [IGNORED] Target FPS for shakify interpolation

    Returns:
        HTML string with Canvas visualization, or None on error
    """
    try:
        # For Frame Overlap visualization, use base schedules directly WITHOUT shakify.
        # The shakify overlay function decimates dense schedules to ~20 keyframes,
        # which breaks visualization of dense camera paths (333 frames → 22 keyframes).
        # Frame Overlap is meant to show primary camera movement, not subtle shake overlay.
        tx_schedule = translation_x or "0:(0)"
        ty_schedule = translation_y or "0:(0)"
        rx_schedule = rotation_3d_x or "0:(0)"
        ry_schedule = rotation_3d_y or "0:(0)"

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

        # Zoom is always 1.0 for now (no zoom schedule yet)
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
            trail_length=15,
            playback_fps=10
        )

        return html

    except Exception as e:
        logger.warning(f"Failed to update frame overlap visualization: {e}")
        return None
