"""Handler for frame overlap simulator UI integration."""

from typing import Optional

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
    max_frames: int,
    width: int,
    height: int,
    shake_name: str = "None",
    shake_intensity: float = 1.0,
    shake_speed: float = 1.0,
    target_fps: int = 60
) -> Optional[str]:
    """Update frame overlap visualization from schedule strings with shakify overlay.

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
        shake_name: Camera shakify pattern name (default: "None")
        shake_intensity: Shakify intensity multiplier (default: 1.0)
        shake_speed: Shakify speed multiplier (default: 1.0)
        target_fps: Target FPS for shakify interpolation (default: 60)

    Returns:
        HTML string with Canvas visualization, or None on error
    """
    try:
        # Build base schedules dict
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
        ry_schedule = final_schedules['rotation_3d_y']

        # Create parser
        parser = FrameInterpolater(max_frames=max_frames)

        # Parse keyframes
        tx_keys = parser.parse_key_frames(tx_schedule)
        ty_keys = parser.parse_key_frames(ty_schedule)
        ry_keys = parser.parse_key_frames(ry_schedule)

        # Interpolate between keyframes to get per-frame values
        tx_series = parser.get_inbetweens(tx_keys, integer=False)
        ty_series = parser.get_inbetweens(ty_keys, integer=False)
        ry_series = parser.get_inbetweens(ry_keys, integer=False)

        # Convert pandas Series to lists
        # IMPORTANT: Schedules are already deltas (from camera_path_to_schedules)
        # Do NOT calculate deltas again - just use interpolated values directly
        tx_deltas = tx_series.tolist()
        ty_deltas = ty_series.tolist()
        ry_deltas = ry_series.tolist()

        # Zoom is always 1.0 for now (no zoom schedule yet)
        zoom_deltas = [1.0] * max_frames

        # Run frame overlap simulation
        metrics = simulate_camera_path(
            translation_x_schedule=tx_deltas,
            translation_y_schedule=ty_deltas,
            rotation_3d_y_schedule=ry_deltas,
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
