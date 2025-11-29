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
    generate_street_path,
    camera_path_to_schedules,
    SplineConfig,
    CameraPoint
)
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.system.logging import get_logger

logger = get_logger()


# ============================================================================
# PURE FUNCTIONS - Preset Type Handlers
# ============================================================================


def _generate_rotate_around(
    num_frames: int,
    radius: float,
    height: float,
    closed_loop: bool,
    rotation_mode: str,
    rotation_factor: float,
    look_at_mode: str,
    look_at_blend: float
) -> Tuple[list, str]:
    """Generate rotate-around preset path with configurable rotation."""
    camera_path = generate_rotate_around_path(
        num_frames=num_frames,
        radius=radius,
        center_x=0.0,
        center_y=0.0,
        height=height,
        use_sphere=True,
        closed_loop=closed_loop,
        rotation_mode=rotation_mode,
        rotation_factor=rotation_factor,
        look_at_mode=look_at_mode,
        look_at_blend=look_at_blend
    )

    # Build status message based on rotation mode
    if rotation_mode == "empirical":
        rotation_desc = f"Empirical (factor={rotation_factor})"
    else:
        rotation_desc = f"Quaternion ({look_at_mode}, blend={look_at_blend})"

    loop_desc = f"{num_frames} frames (1 orbit)" if closed_loop else f"{num_frames} frames (multi-orbit)"

    status = (
        f"{emoji_utils.maybe_check()} Generated rotate-around path ({loop_desc})\n"
        f"Radius: {radius}, Height: {height}\n"
        f"Rotation: {rotation_desc}"
    )
    return camera_path, status


def _generate_figure_eight(
    num_frames: int, radius: float, height: float, closed_loop: bool
) -> Tuple[list, str]:
    """Generate figure-eight preset path."""
    control_points = generate_control_points_figure_eight(
        num_points=12, scale=radius, center_x=0.0, center_y=height, center_z=0.0
    )
    config = SplineConfig(
        num_frames=num_frames,
        num_control_points=12,
        spline_type="catmull_rom",
        closed_loop=closed_loop,
        smoothness=0.8
    )
    camera_path = generate_camera_path(config, control_points, look_at_curve=True)
    status = (
        f"{emoji_utils.maybe_check()} Generated figure-8 path ({len(camera_path)} frames)\n"
        f"Scale: {radius}, Height: {height}, Closed: {closed_loop}"
    )
    return camera_path, status


def _generate_forward_zoom(num_frames: int, radius: float, height: float) -> Tuple[list, str]:
    """Generate forward zoom preset path."""
    camera_path = [
        CameraPoint(
            x=0.0,
            y=height,
            z=i * (radius / num_frames),
            rot_x=0.0,
            rot_y=0.0,
            rot_z=0.0,
            frame=i
        )
        for i in range(num_frames)
    ]
    status = (
        f"{emoji_utils.maybe_check()} Generated forward zoom ({len(camera_path)} frames)\n"
        f"Zoom distance: {radius}, Height: {height}"
    )
    return camera_path, status


def _generate_orbit_up(num_frames: int, radius: float, height: float) -> Tuple[list, str]:
    """Generate orbit-up preset path (circle while rising)."""
    num_points = 12
    control_points = []
    for i in range(num_points + 1):
        angle = 2 * np.pi * i / num_points
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = height + (i / num_points) * radius * 0.8
        control_points.append((x, y, z))

    config = SplineConfig(
        num_frames=num_frames,
        num_control_points=num_points + 1,
        spline_type="catmull_rom",
        closed_loop=False,
        smoothness=0.7
    )
    camera_path = generate_camera_path(config, control_points, look_at_curve=True)
    status = (
        f"{emoji_utils.maybe_check()} Generated orbit-up path ({len(camera_path)} frames)\n"
        f"Radius: {radius}, Rise: {radius * 0.8:.1f}\n"
        f"Note: Orbit-up cannot be closed (rising path)"
    )
    return camera_path, status


def _calculate_spiral_point(
    t: float, num_points: int, radius: float, height: float
) -> Tuple[float, float, float]:
    """Calculate single point on 3D spiral trajectory."""
    angle = 6 * np.pi * t
    r = radius * (1 - t)
    x = r * np.cos(angle)
    z = r * np.sin(angle)
    y = height + radius * 0.8 * np.sin(2 * np.pi * t)
    return x, y, z


def _calculate_look_at_rotation(
    x: float, y: float, z: float, target_height: float
) -> Tuple[float, float, float]:
    """Calculate camera rotation to look at center point."""
    dx = 0.0 - x
    dy = target_height - y
    dz = 0.0 - z

    rot_y = np.degrees(np.arctan2(dx, dz))
    horizontal_dist = np.sqrt(dx**2 + dz**2)
    rot_x = np.degrees(np.arctan2(dy, horizontal_dist + 1e-8))
    rot_z = 0.0

    return rot_x, rot_y, rot_z


def _interpolate_spiral_frame(
    frame_idx: int, num_frames: int, control_points: list, height: float
) -> CameraPoint:
    """Interpolate single frame position on spiral path."""
    t = frame_idx / (num_frames - 1) if num_frames > 1 else 0
    num_points = len(control_points)
    point_idx = int(t * (num_points - 1))

    if point_idx >= num_points - 1:
        x, y, z = control_points[-1]
    else:
        local_t = (t * (num_points - 1)) - point_idx
        p1, p2 = control_points[point_idx], control_points[point_idx + 1]
        x = p1[0] + local_t * (p2[0] - p1[0])
        y = p1[1] + local_t * (p2[1] - p1[1])
        z = p1[2] + local_t * (p2[2] - p1[2])

    rot_x, rot_y, rot_z = _calculate_look_at_rotation(x, y, z, height)
    return CameraPoint(x=x, y=y, z=z, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z, frame=frame_idx)


def _generate_spiral(num_frames: int, radius: float, height: float) -> Tuple[list, str]:
    """Generate 3D spiral preset path."""
    num_points = 24
    control_points = [
        _calculate_spiral_point(i / (num_points - 1), num_points, radius, height)
        for i in range(num_points)
    ]

    camera_path = [
        _interpolate_spiral_frame(i, num_frames, control_points, height)
        for i in range(num_frames)
    ]

    status = (
        f"{emoji_utils.maybe_check()} Generated 3D spiral path ({len(camera_path)} frames)\n"
        f"Start radius: {radius}, End radius: 0\n"
        f"Mode: 3D spiral with look-at center (fits in cube)"
    )
    return camera_path, status


def _generate_street(num_frames: int, radius: float, height: float) -> Tuple[list, str]:
    """Generate street/dashcam preset path."""
    eye_height = height if height != 0 else 10.0
    camera_path = generate_street_path(
        num_frames=num_frames,
        street_length=radius * 5,
        lane_weave=radius * 0.2,
        center_x=0.0,
        center_y=eye_height,
        center_z=0.0
    )
    status = (
        f"{emoji_utils.maybe_check()} Generated street path ({len(camera_path)} frames)\n"
        f"Distance: {radius * 5:.0f}, Eye height: {eye_height:.0f}\n"
        f"Mode: Forward-facing (dashcam/POV)"
    )
    return camera_path, status


def _generate_dashcam(num_frames: int, radius: float, height: float) -> Tuple[list, str]:
    """Generate dashcam preset path (faster with more weaving)."""
    eye_height = height if height != 0 else 8.0
    camera_path = generate_street_path(
        num_frames=num_frames,
        street_length=radius * 8,
        lane_weave=radius * 0.3,
        center_x=0.0,
        center_y=eye_height,
        center_z=0.0
    )
    status = (
        f"{emoji_utils.maybe_check()} Generated dashcam path ({len(camera_path)} frames)\n"
        f"Distance: {radius * 8:.0f}, Eye height: {eye_height:.0f}\n"
        f"Mode: Dashcam (faster, more weaving)\n"
        f"Tip: Combine with GENTLE_HANDHELD shakify pattern"
    )
    return camera_path, status


def _generate_bodycam(num_frames: int, radius: float, height: float) -> Tuple[list, str]:
    """Generate bodycam preset path (slower walking pace)."""
    eye_height = height if height != 0 else 15.0
    camera_path = generate_street_path(
        num_frames=num_frames,
        street_length=radius * 3,
        lane_weave=radius * 0.15,
        center_x=0.0,
        center_y=eye_height,
        center_z=0.0
    )
    status = (
        f"{emoji_utils.maybe_check()} Generated bodycam path ({len(camera_path)} frames)\n"
        f"Distance: {radius * 3:.0f}, Eye height: {eye_height:.0f}\n"
        f"Mode: Bodycam (walking pace)\n"
        f"Tip: Combine with INVESTIGATION or GENTLE_HANDHELD shakify pattern"
    )
    return camera_path, status


def generate_preset_path(
    preset_type: str,
    radius: float,
    height: float,
    num_frames: int,
    closed_loop: bool,
    randomize: float = 0.0,
    random_seed: int = -1,
    speed_multiplier: float = 1.0,
    speed_randomization: float = 0.0,
    rotation_mode: str = "quaternion",
    rotation_factor: float = -8.0,
    look_at_mode: str = "center",
    look_at_blend: float = 0.3
) -> Tuple[str, Dict[str, str], list]:
    """Generate camera path from preset using type-specific handlers.

    Args:
        preset_type: Type of preset ("rotate-around", "figure-eight", etc.)
        radius: Radius/scale of movement
        height: Vertical offset
        num_frames: Total frames
        closed_loop: Whether to loop (for applicable presets)
        randomize: Random offset amount (not yet implemented)
        random_seed: Seed for randomization (not yet implemented)
        speed_multiplier: Translation speed control
        speed_randomization: Speed variation amount
        rotation_mode: How to calculate rotation ("quaternion" or "empirical")
        rotation_factor: When rotation_mode="empirical", counter-rotation strength (default: -8.0)
        look_at_mode: When rotation_mode="quaternion", camera look-at behavior
                     ("center", "tangent", "inward", "blend")
        look_at_blend: When look_at_mode="blend", inward blend amount (0.0-1.0)

    Returns:
        Tuple of (status_message, schedules_dict, camera_path)
    """
    # Debug logging
    logger.debug(f"generate_preset_path: preset_type={preset_type}, radius={radius}, speed_multiplier={speed_multiplier}")

    try:
        # Type-specific handler dispatch
        num_frames_int = int(num_frames)

        if preset_type == "rotate-around":
            camera_path, status = _generate_rotate_around(
                num_frames_int, radius, height, closed_loop,
                rotation_mode, rotation_factor, look_at_mode, look_at_blend
            )
        elif preset_type == "figure-eight":
            camera_path, status = _generate_figure_eight(
                num_frames_int, radius, height, closed_loop
            )
        elif preset_type == "forward-zoom":
            camera_path, status = _generate_forward_zoom(num_frames_int, radius, height)
        elif preset_type == "orbit-up":
            camera_path, status = _generate_orbit_up(num_frames_int, radius, height)
        elif preset_type == "spiral":
            camera_path, status = _generate_spiral(num_frames_int, radius, height)
        elif preset_type == "street":
            camera_path, status = _generate_street(num_frames_int, radius, height)
        elif preset_type == "dashcam":
            camera_path, status = _generate_dashcam(num_frames_int, radius, height)
        elif preset_type == "bodycam":
            camera_path, status = _generate_bodycam(num_frames_int, radius, height)
        else:
            return f"{emoji_utils.maybe_cross()} Unknown preset type: {preset_type}", {}, []

        # Use random_seed if provided, otherwise use 0 for reproducibility
        seed = int(random_seed) if random_seed >= 0 else 0

        # Pass look_at_mode only for rotate-around with quaternion mode
        # This ensures center mode rotations are recalculated after position offset
        schedule_look_at_mode = None
        if preset_type == "rotate-around" and rotation_mode == "quaternion":
            schedule_look_at_mode = look_at_mode

        schedules = camera_path_to_schedules(
            camera_path,
            speed_multiplier=speed_multiplier,
            speed_randomization=speed_randomization,
            random_seed=seed,
            look_at_mode=schedule_look_at_mode
        )
        return status, schedules, camera_path

    except Exception as e:
        return f"{emoji_utils.maybe_cross()} Error: {str(e)}", {}, []


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
            return f"{emoji_utils.maybe_cross()} Unknown pattern: {control_point_pattern}", {}, []

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

        status = f"{emoji_utils.maybe_check()} Generated custom spline path ({len(camera_path)} frames)\n"
        status += f"Control points: {num_control_points}, Type: {spline_type}\n"
        status += f"Pattern: {control_point_pattern}, Scale: {pattern_scale}, Closed: {closed_loop}"

        return status, schedules, camera_path

    except Exception as e:
        return f"{emoji_utils.maybe_cross()} Error: {str(e)}", {}, []


def _create_empty_camera_plot() -> Tuple[go.Figure, str]:
    """Create empty 3D plot for camera path visualization."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text="No path generated yet",
            font=dict(color='#E0E7FF', size=18, family='system-ui')
        ),
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        font=dict(color='#CBD5E1', family='system-ui'),
        scene=dict(
            xaxis_title="X (Left/Right)",
            yaxis_title="Y (Up/Down)",
            zaxis_title="Z (Forward/Back)",
            xaxis=dict(
                backgroundcolor='#1E293B',
                gridcolor='#334155',
                showbackground=True,
                zerolinecolor='#475569',
                title=dict(font=dict(color='#94A3B8'))
            ),
            yaxis=dict(
                backgroundcolor='#1E293B',
                gridcolor='#334155',
                showbackground=True,
                zerolinecolor='#475569',
                title=dict(font=dict(color='#94A3B8'))
            ),
            zaxis=dict(
                backgroundcolor='#1E293B',
                gridcolor='#334155',
                showbackground=True,
                zerolinecolor='#475569',
                title=dict(font=dict(color='#94A3B8'))
            )
        )
    )
    return fig, "No path data"


def _generate_gradient_colors(num_points: int) -> list:
    """Generate gradient colors for camera path visualization."""
    return [
        f'rgb({int(102 + (118-102)*i/num_points)}, '
        f'{int(126 + (75-126)*i/num_points)}, '
        f'{int(234 + (162-234)*i/num_points)})'
        for i in range(num_points)
    ]


def _calculate_keyframe_indices(num_points: int) -> list:
    """Calculate keyframe marker indices for visualization."""
    keyframe_interval = max(20, min(40, num_points // 10))
    keyframe_indices = list(range(0, num_points, keyframe_interval))
    if keyframe_indices and keyframe_indices[-1] != num_points - 1:
        keyframe_indices.append(num_points - 1)
    return keyframe_indices


def _calculate_path_statistics(camera_path: list, x_coords: list, y_coords: list, z_coords: list) -> str:
    """Calculate and format path statistics."""
    # Calculate total distance
    total_distance = 0.0
    for i in range(1, len(camera_path)):
        dx = camera_path[i].x - camera_path[i-1].x
        dy = camera_path[i].y - camera_path[i-1].y
        dz = camera_path[i].z - camera_path[i-1].z
        total_distance += np.sqrt(dx**2 + dy**2 + dz**2)

    # Calculate ranges
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    z_range = max(z_coords) - min(z_coords)

    # Calculate keyframe info
    num_points = len(camera_path)
    keyframe_interval = max(20, min(40, num_points // 10))
    num_keyframes = (num_points + keyframe_interval - 1) // keyframe_interval

    return f"""Path Statistics:
- Frames: {len(camera_path)}
- Keyframes: ~{num_keyframes} (every {keyframe_interval} frames)
- Total Distance: {total_distance:.2f}
- X Range: {x_range:.2f} (left/right)
- Y Range: {y_range:.2f} (up/down)
- Z Range: {z_range:.2f} (forward/back)
- Start: ({x_coords[0]:.2f}, {y_coords[0]:.2f}, {z_coords[0]:.2f})
- End: ({x_coords[-1]:.2f}, {y_coords[-1]:.2f}, {z_coords[-1]:.2f})
"""


def visualize_camera_path(camera_path: list) -> Tuple[go.Figure, str]:
    """Create 3D visualization of camera path.

    Args:
        camera_path: List of CameraPoint objects

    Returns:
        (plotly_figure, stats_text)
    """
    if not camera_path:
        return _create_empty_camera_plot()

    # Extract coordinates
    x_coords = [p.x for p in camera_path]
    y_coords = [p.y for p in camera_path]
    z_coords = [p.z for p in camera_path]

    # Create 3D line plot - SLOPCORE GRIFTWAVE AESTHETIC
    fig = go.Figure()

    # Path line - PURPLE GRADIENT VIBES (simulate gradient with multiple segments)
    num_points = len(x_coords)
    colors = _generate_gradient_colors(num_points)

    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines+markers',
        name='Camera Path',
        line=dict(
            color=colors if num_points > 1 else ['#667EEA'],
            width=6,
            colorscale=[[0, '#667EEA'], [1, '#764BA2']],  # Blue to purple gradient
        ),
        marker=dict(
            size=4,
            color=colors if num_points > 1 else ['#667EEA'],
            colorscale=[[0, '#667EEA'], [1, '#764BA2']],
            line=dict(color='#1E293B', width=1),
            opacity=0.9
        ),
        hovertemplate='<b>Frame %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
        text=[str(i) for i in range(num_points)]
    ))

    # Start point - CYAN GLOW (SaaS brand color)
    fig.add_trace(go.Scatter3d(
        x=[x_coords[0]],
        y=[y_coords[0]],
        z=[z_coords[0]],
        mode='markers',
        name='Start',
        marker=dict(
            size=12,
            color='#06B6D4',  # Tailwind cyan-500
            symbol='diamond',
            line=dict(color='#0891B2', width=2),  # Tailwind cyan-600
            opacity=1.0
        ),
        hovertemplate='<b>START</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    ))

    # End point - PINK/PURPLE GLOW (startup gradient end)
    fig.add_trace(go.Scatter3d(
        x=[x_coords[-1]],
        y=[y_coords[-1]],
        z=[z_coords[-1]],
        mode='markers',
        name='End',
        marker=dict(
            size=12,
            color='#EC4899',  # Tailwind pink-500
            symbol='square',
            line=dict(color='#BE185D', width=2),  # Tailwind pink-700
            opacity=1.0
        ),
        hovertemplate='<b>END</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    ))

    # Center point - ORANGE GLOW (rotation center / look-at target)
    # Shows where rotate-around paths look at
    fig.add_trace(go.Scatter3d(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        mode='markers',
        name='Center (0,0,0)',
        marker=dict(
            size=15,
            color='#F97316',  # Tailwind orange-500
            symbol='x',
            line=dict(color='#EA580C', width=3),  # Tailwind orange-600
            opacity=0.9
        ),
        hovertemplate='<b>CENTER</b><br>Origin (0, 0, 0)<br>Rotate-around looks here<extra></extra>'
    ))

    # Keyframe markers - BRIGHT BLUE highlights at regular intervals (SLOPCORE)
    keyframe_indices = _calculate_keyframe_indices(num_points)
    keyframe_interval = max(20, min(40, num_points // 10))

    # Extract keyframe coordinates
    keyframe_x = [x_coords[i] for i in keyframe_indices]
    keyframe_y = [y_coords[i] for i in keyframe_indices]
    keyframe_z = [z_coords[i] for i in keyframe_indices]
    keyframe_labels = [str(i) for i in keyframe_indices]

    fig.add_trace(go.Scatter3d(
        x=keyframe_x,
        y=keyframe_y,
        z=keyframe_z,
        mode='markers',
        name=f'Keyframes (every {keyframe_interval})',
        marker=dict(
            size=8,
            color='#3B82F6',  # Tailwind blue-500 (bright blue, slopcore aesthetic)
            symbol='circle',
            line=dict(color='#2563EB', width=2),  # Tailwind blue-600
            opacity=0.9
        ),
        hovertemplate='<b>KEYFRAME %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
        text=keyframe_labels
    ))

    # Update layout - FULL DARKMODE SLOPCORE
    # Respect emoji toggle setting
    title_emoji = f"{emoji_utils.wan_video()} " if emoji_utils.wan_video() else ""
    fig.update_layout(
        title=dict(
            text=f'<b>{title_emoji}Camera Path Visualization</b>',
            font=dict(color='#E0E7FF', size=20, family='system-ui, -apple-system, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='#0F172A',  # Tailwind slate-900 - DARK AF
        plot_bgcolor='#1E293B',   # Tailwind slate-800
        font=dict(color='#CBD5E1', family='system-ui, -apple-system, sans-serif', size=12),
        scene=dict(
            xaxis_title="<b>X</b> (Left/Right)",
            yaxis_title="<b>Y</b> (Up/Down)",
            zaxis_title="<b>Z</b> (Forward/Back)",
            aspectmode='data',
            xaxis=dict(
                backgroundcolor='#1E293B',
                gridcolor='#334155',  # Tailwind slate-700
                showbackground=True,
                zerolinecolor='#475569',  # Tailwind slate-600
                title=dict(font=dict(color='#94A3B8', size=14)),  # Tailwind slate-400
                tickfont=dict(color='#64748B')  # Tailwind slate-500
            ),
            yaxis=dict(
                backgroundcolor='#1E293B',
                gridcolor='#334155',
                showbackground=True,
                zerolinecolor='#475569',
                title=dict(font=dict(color='#94A3B8', size=14)),
                tickfont=dict(color='#64748B')
            ),
            zaxis=dict(
                backgroundcolor='#1E293B',
                gridcolor='#334155',
                showbackground=True,
                zerolinecolor='#475569',
                title=dict(font=dict(color='#94A3B8', size=14)),
                tickfont=dict(color='#64748B')
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Better default viewing angle
            )
        ),
        height=600,
        showlegend=True,
        legend=dict(
            bgcolor='#1E293B',
            bordercolor='#475569',
            borderwidth=1,
            font=dict(color='#CBD5E1', size=11)
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        hoverlabel=dict(
            bgcolor='#1E293B',
            font=dict(color='#E0E7FF', size=12, family='system-ui'),
            bordercolor='#667EEA'
        )
    )

    # Calculate statistics
    stats = _calculate_path_statistics(camera_path, x_coords, y_coords, z_coords)

    return fig, stats


# Global state to store current path
_current_camera_path = []


def handle_generate_preset(
    preset_type: str,
    speed_multiplier: float,
    speed_randomization: float,
    radius: float,
    height: float,
    num_frames: float,
    closed_loop: bool,
    randomize: float,
    random_seed: float,
    rotation_mode: str,
    rotation_factor: float,
    look_at_mode: str,
    look_at_blend: float,
    translation_x,
    translation_y,
    translation_z,
    rotation_3d_x,
    rotation_3d_y,
    rotation_3d_z,
    animation_prompts: str = "",
):
    """Handle preset path generation and populate schedules.

    Also generates visualization immediately to avoid relying on .change() events.
    """
    global _current_camera_path
    from deforum.utils.schedule_visualizer import visualize_schedules

    # Debug logging
    logger.debug(f"handle_generate_preset: radius={radius}, speed_multiplier={speed_multiplier}, num_frames={num_frames}")

    status, schedules, camera_path = generate_preset_path(
        preset_type, radius, height, num_frames, closed_loop,
        randomize, int(random_seed), speed_multiplier, speed_randomization,
        rotation_mode, rotation_factor, look_at_mode, look_at_blend
    )

    _current_camera_path = camera_path

    # Debug: Check first few schedule values
    tx_schedule = schedules.get('translation_x', '')
    import re
    tx_matches = re.findall(r'(\d+)\s*:\s*\(\s*(-?\d+\.?\d*)\s*\)', tx_schedule)
    if len(tx_matches) >= 5:
        first_5 = [f"{frame}: ({val})" for frame, val in tx_matches[:5]]
        logger.debug(f"First 5 translation_x schedule values: {', '.join(first_5)}")

    # Generate visualization immediately
    try:
        fig, _ = visualize_schedules(
            schedules.get('translation_x', ''),
            schedules.get('translation_y', ''),
            schedules.get('translation_z', ''),
            schedules.get('rotation_3d_x', ''),
            schedules.get('rotation_3d_y', ''),
            schedules.get('rotation_3d_z', ''),
            int(num_frames),
            animation_prompts or "",
            shake_name="None",
            shake_intensity=1.0,
            shake_speed=1.0,
            target_fps=60,
            apply_shakify=False
        )
    except Exception as e:
        print(f"Failed to generate visualization: {e}")
        fig = None

    # Return schedule updates AND visualization
    return [
        status,  # preset_status
        schedules.get('translation_x', ''),  # translation_x textbox
        schedules.get('translation_y', ''),  # translation_y textbox
        schedules.get('translation_z', ''),  # translation_z textbox
        schedules.get('rotation_3d_x', ''),  # rotation_3d_x textbox
        schedules.get('rotation_3d_y', ''),  # rotation_3d_y textbox
        schedules.get('rotation_3d_z', ''),  # rotation_3d_z textbox
        fig  # camera_path_plot
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
    """Handle custom spline generation and populate schedules.

    Note: Visualization is now handled by schedule change events.
    This only updates the schedule textboxes.
    """
    global _current_camera_path

    status, schedules, camera_path = generate_custom_spline_path(
        num_control_points, spline_type, smoothness, look_at_curve,
        num_frames, closed_loop, control_point_pattern, pattern_scale
    )

    _current_camera_path = camera_path

    # Return schedule updates only - visualization will update automatically via .change() events
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
