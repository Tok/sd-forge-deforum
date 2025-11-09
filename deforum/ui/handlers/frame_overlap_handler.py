"""Handler for frame overlap simulator UI integration."""

from typing import Optional
import plotly.graph_objects as go

from deforum.utils.frame_overlap_simulator import simulate_camera_path
from deforum.utils.frame_overlap_visualizer import create_worm_trail_visualization
from deforum.core.keyframes import get_inbetweens, parse_key_frames
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
    height: int
) -> Optional[go.Figure]:
    """Update frame overlap visualization from schedule strings.

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

    Returns:
        Plotly Figure with worm trail visualization, or None on error
    """
    try:
        # Parse schedule strings to get per-frame values
        # Default to "0:(0)" if empty
        tx_schedule = translation_x or "0:(0)"
        ty_schedule = translation_y or "0:(0)"
        ry_schedule = rotation_3d_y or "0:(0)"

        # Parse keyframes
        tx_keys = parse_key_frames(tx_schedule, max_frames)
        ty_keys = parse_key_frames(ty_schedule, max_frames)
        ry_keys = parse_key_frames(ry_schedule, max_frames)

        # Interpolate between keyframes to get per-frame values
        tx_values = get_inbetweens(tx_keys, max_frames)
        ty_values = get_inbetweens(ty_keys, max_frames)
        ry_values = get_inbetweens(ry_keys, max_frames)

        # Calculate frame-to-frame deltas (what the animation engine actually uses)
        tx_deltas = [tx_values[i] - tx_values[i - 1] if i > 0 else 0 for i in range(max_frames)]
        ty_deltas = [ty_values[i] - ty_values[i - 1] if i > 0 else 0 for i in range(max_frames)]
        ry_deltas = [ry_values[i] - ry_values[i - 1] if i > 0 else 0 for i in range(max_frames)]

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

        # Create visualization
        fig = create_worm_trail_visualization(
            metrics_list=metrics,
            width=800,
            height=600,
            trail_length=15,
            playback_fps=10
        )

        return fig

    except Exception as e:
        logger.warning(f"Failed to update frame overlap visualization: {e}")
        return None
