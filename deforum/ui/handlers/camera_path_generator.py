"""Camera Path Generation Handlers

Functions to generate camera paths and populate schedules.
"""

from typing import Dict, Any, Tuple
import plotly.graph_objects as go
import numpy as np
from deforum.utils.spline_camera_path import (
    generate_rotate_around_path,
    generate_camera_path,
    generate_control_points_circle,
    generate_control_points_figure_eight,
    camera_path_to_schedules,
    SplineConfig,
    CameraPoint
)


def generate_preset_path(
    preset_type: str,
    radius: float,
    height: float,
    rotation_factor: float,
    num_frames: int,
    closed_loop: bool
) -> Tuple[str, Dict[str, str], list]:
    """Generate camera path from preset.

    Args:
        preset_type: Type of preset ("rotate-around", "circle-path", etc.)
        radius: Radius/scale of movement
        height: Vertical offset
        rotation_factor: Rotation multiplier for rotate-around
        num_frames: Total frames
        closed_loop: Whether to loop

    Returns:
        (status_message, schedules_dict, camera_path)
    """
    try:
        camera_path = []

        if preset_type == "rotate-around":
            camera_path = generate_rotate_around_path(
                num_frames=int(num_frames),
                radius=radius,
                center_x=0.0,
                center_y=0.0,
                height=height,
                rotation_factor=rotation_factor
            )
            status = f"✅ Generated rotate-around path ({len(camera_path)} frames)\n"
            status += f"Radius: {radius}, Height: {height}, Rotation Factor: {rotation_factor}"

        elif preset_type == "circle-path":
            control_points = generate_control_points_circle(
                num_points=8,
                radius=radius,
                center_x=0.0,
                center_y=height,
                center_z=0.0,
                height_variation=radius * 0.2
            )
            config = SplineConfig(
                num_frames=int(num_frames),
                num_control_points=8,
                spline_type="catmull_rom",
                closed_loop=closed_loop,
                smoothness=0.7
            )
            camera_path = generate_camera_path(config, control_points, look_at_curve=True)
            status = f"✅ Generated circle path ({len(camera_path)} frames)\n"
            status += f"Radius: {radius}, Height: {height}, Closed: {closed_loop}"

        elif preset_type == "figure-eight":
            control_points = generate_control_points_figure_eight(
                num_points=12,
                scale=radius,
                center_x=0.0,
                center_y=height,
                center_z=0.0
            )
            config = SplineConfig(
                num_frames=int(num_frames),
                num_control_points=12,
                spline_type="catmull_rom",
                closed_loop=closed_loop,
                smoothness=0.8
            )
            camera_path = generate_camera_path(config, control_points, look_at_curve=True)
            status = f"✅ Generated figure-8 path ({len(camera_path)} frames)\n"
            status += f"Scale: {radius}, Height: {height}, Closed: {closed_loop}"

        elif preset_type == "forward-zoom":
            # Simple linear zoom forward
            camera_path = [
                CameraPoint(
                    x=0.0,
                    y=height,
                    z=i * (radius / num_frames),  # Steady zoom
                    rot_x=0.0,
                    rot_y=0.0,
                    rot_z=0.0,
                    frame=i
                )
                for i in range(int(num_frames))
            ]
            status = f"✅ Generated forward zoom ({len(camera_path)} frames)\n"
            status += f"Zoom distance: {radius}, Height: {height}"

        elif preset_type == "orbit-up":
            # Circle while rising
            control_points = []
            for i in range(8):
                angle = 2 * np.pi * i / 8
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                y = height + (i / 8) * radius * 0.5  # Rise as we orbit
                control_points.append((x, y, z))

            config = SplineConfig(
                num_frames=int(num_frames),
                num_control_points=8,
                spline_type="catmull_rom",
                closed_loop=closed_loop,
                smoothness=0.7
            )
            camera_path = generate_camera_path(config, control_points, look_at_curve=True)
            status = f"✅ Generated orbit-up path ({len(camera_path)} frames)\n"
            status += f"Radius: {radius}, Rise: {radius * 0.5}, Closed: {closed_loop}"

        elif preset_type == "spiral":
            # Spiral inward/outward
            control_points = []
            for i in range(16):
                angle = 4 * np.pi * i / 16  # Two full rotations
                r = radius * (1 - i / 16)  # Shrinking radius
                x = r * np.cos(angle)
                z = r * np.sin(angle)
                y = height + (i / 16) * radius * 0.3
                control_points.append((x, y, z))

            config = SplineConfig(
                num_frames=int(num_frames),
                num_control_points=16,
                spline_type="catmull_rom",
                closed_loop=False,  # Spirals don't loop
                smoothness=0.8
            )
            camera_path = generate_camera_path(config, control_points, look_at_curve=True)
            status = f"✅ Generated spiral path ({len(camera_path)} frames)\n"
            status += f"Start radius: {radius}, End radius: 0"

        else:
            return f"❌ Unknown preset type: {preset_type}", {}, []

        # Convert to schedules
        schedules = camera_path_to_schedules(camera_path)

        return status, schedules, camera_path

    except Exception as e:
        return f"❌ Error: {str(e)}", {}, []


def generate_custom_spline_path(
    num_control_points: int,
    spline_type: str,
    smoothness: float,
    look_at_curve: bool,
    num_frames: int,
    closed_loop: bool,
    control_point_pattern: str,
    pattern_scale: float
) -> Tuple[str, Dict[str, str], list]:
    """Generate custom spline path.

    Args:
        num_control_points: Number of waypoints
        spline_type: "catmull_rom" or "linear"
        smoothness: 0-1 smoothness factor
        look_at_curve: Whether camera looks tangent to curve
        num_frames: Total frames
        closed_loop: Whether to loop
        control_point_pattern: How to distribute control points
        pattern_scale: Scale of pattern

    Returns:
        (status_message, schedules_dict, camera_path)
    """
    try:
        # Generate control points based on pattern
        if control_point_pattern == "circle":
            control_points = generate_control_points_circle(
                num_points=int(num_control_points),
                radius=pattern_scale,
                center_x=0.0,
                center_y=0.0,
                center_z=0.0,
                height_variation=pattern_scale * 0.3
            )
        elif control_point_pattern == "figure-eight":
            control_points = generate_control_points_figure_eight(
                num_points=int(num_control_points),
                scale=pattern_scale,
                center_x=0.0,
                center_y=0.0,
                center_z=0.0
            )
        elif control_point_pattern == "line":
            # Linear path
            control_points = [
                (i * (pattern_scale / num_control_points), 0.0, 0.0)
                for i in range(int(num_control_points))
            ]
        elif control_point_pattern == "random":
            # Random control points
            np.random.seed(42)  # Reproducible random
            control_points = [
                (
                    np.random.uniform(-pattern_scale, pattern_scale),
                    np.random.uniform(-pattern_scale * 0.5, pattern_scale * 0.5),
                    np.random.uniform(-pattern_scale, pattern_scale)
                )
                for _ in range(int(num_control_points))
            ]
        else:
            return f"❌ Unknown pattern: {control_point_pattern}", {}, []

        # Generate spline
        config = SplineConfig(
            num_frames=int(num_frames),
            num_control_points=int(num_control_points),
            spline_type=spline_type,
            closed_loop=closed_loop,
            smoothness=smoothness
        )

        camera_path = generate_camera_path(config, control_points, look_at_curve=look_at_curve)

        # Convert to schedules
        schedules = camera_path_to_schedules(camera_path)

        status = f"✅ Generated custom spline path ({len(camera_path)} frames)\n"
        status += f"Control points: {num_control_points}, Type: {spline_type}\n"
        status += f"Pattern: {control_point_pattern}, Scale: {pattern_scale}, Closed: {closed_loop}"

        return status, schedules, camera_path

    except Exception as e:
        return f"❌ Error: {str(e)}", {}, []


def visualize_camera_path(camera_path: list) -> Tuple[go.Figure, str]:
    """Create 3D visualization of camera path.

    Args:
        camera_path: List of CameraPoint objects

    Returns:
        (plotly_figure, stats_text)
    """
    if not camera_path:
        # Empty plot
        fig = go.Figure()
        fig.update_layout(
            title="No path generated yet",
            scene=dict(
                xaxis_title="X (Left/Right)",
                yaxis_title="Y (Up/Down)",
                zaxis_title="Z (Forward/Back)"
            )
        )
        return fig, "No path data"

    # Extract coordinates
    x_coords = [p.x for p in camera_path]
    y_coords = [p.y for p in camera_path]
    z_coords = [p.z for p in camera_path]

    # Create 3D line plot
    fig = go.Figure()

    # Path line
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines+markers',
        name='Camera Path',
        line=dict(color='purple', width=4),
        marker=dict(size=3, color='blue')
    ))

    # Start point (green)
    fig.add_trace(go.Scatter3d(
        x=[x_coords[0]],
        y=[y_coords[0]],
        z=[z_coords[0]],
        mode='markers',
        name='Start',
        marker=dict(size=10, color='green', symbol='diamond')
    ))

    # End point (red)
    fig.add_trace(go.Scatter3d(
        x=[x_coords[-1]],
        y=[y_coords[-1]],
        z=[z_coords[-1]],
        mode='markers',
        name='End',
        marker=dict(size=10, color='red', symbol='square')
    ))

    # Update layout
    fig.update_layout(
        title="3D Camera Path Visualization",
        scene=dict(
            xaxis_title="X (Left/Right)",
            yaxis_title="Y (Up/Down)",
            zaxis_title="Z (Forward/Back)",
            aspectmode='data'
        ),
        height=600
    )

    # Calculate statistics
    total_distance = 0.0
    for i in range(1, len(camera_path)):
        dx = camera_path[i].x - camera_path[i-1].x
        dy = camera_path[i].y - camera_path[i-1].y
        dz = camera_path[i].z - camera_path[i-1].z
        total_distance += np.sqrt(dx**2 + dy**2 + dz**2)

    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    z_range = max(z_coords) - min(z_coords)

    stats = f"""Path Statistics:
- Frames: {len(camera_path)}
- Total Distance: {total_distance:.2f}
- X Range: {x_range:.2f} (left/right)
- Y Range: {y_range:.2f} (up/down)
- Z Range: {z_range:.2f} (forward/back)
- Start: ({x_coords[0]:.2f}, {y_coords[0]:.2f}, {z_coords[0]:.2f})
- End: ({x_coords[-1]:.2f}, {y_coords[-1]:.2f}, {z_coords[-1]:.2f})
"""

    return fig, stats


# Global state to store current path
_current_camera_path = []


def handle_generate_preset(
    preset_type: str,
    radius: float,
    height: float,
    rotation_factor: float,
    num_frames: float,
    closed_loop: bool,
    translation_x,
    translation_y,
    translation_z,
    rotation_3d_x,
    rotation_3d_y,
    rotation_3d_z
):
    """Handle preset path generation and populate schedules."""
    global _current_camera_path

    status, schedules, camera_path = generate_preset_path(
        preset_type, radius, height, rotation_factor, num_frames, closed_loop
    )

    _current_camera_path = camera_path

    # Return updates for all components
    return [
        status,  # preset_status
        schedules.get('translation_x', ''),  # translation_x textbox
        schedules.get('translation_y', ''),  # translation_y textbox
        schedules.get('translation_z', ''),  # translation_z textbox
        schedules.get('rotation_3d_x', ''),  # rotation_3d_x textbox
        schedules.get('rotation_3d_y', ''),  # rotation_3d_y textbox
        schedules.get('rotation_3d_z', '')   # rotation_3d_z textbox
    ]


def handle_generate_custom(
    num_control_points: float,
    spline_type: str,
    smoothness: float,
    look_at_curve: bool,
    num_frames: float,
    closed_loop: bool,
    control_point_pattern: str,
    pattern_scale: float,
    translation_x,
    translation_y,
    translation_z,
    rotation_3d_x,
    rotation_3d_y,
    rotation_3d_z
):
    """Handle custom spline generation and populate schedules."""
    global _current_camera_path

    status, schedules, camera_path = generate_custom_spline_path(
        num_control_points, spline_type, smoothness, look_at_curve,
        num_frames, closed_loop, control_point_pattern, pattern_scale
    )

    _current_camera_path = camera_path

    # Return updates for all components
    return [
        status,  # custom_status
        schedules.get('translation_x', ''),
        schedules.get('translation_y', ''),
        schedules.get('translation_z', ''),
        schedules.get('rotation_3d_x', ''),
        schedules.get('rotation_3d_y', ''),
        schedules.get('rotation_3d_z', '')
    ]


def handle_visualize():
    """Handle visualization request."""
    global _current_camera_path

    fig, stats = visualize_camera_path(_current_camera_path)

    return fig, stats
