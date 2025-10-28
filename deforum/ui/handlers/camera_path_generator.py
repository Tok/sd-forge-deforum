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


def generate_preset_path(
    preset_type: str,
    radius: float,
    height: float,
    rotation_factor: float,
    num_frames: int,
    closed_loop: bool,
    randomize: float = 0.0,
    random_seed: int = -1
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
                rotation_factor=rotation_factor,
                use_sphere=True  # Use 3D sphere rotation (not flat circle)
            )
            status = f"✅ Generated rotate-around path ({len(camera_path)} frames)\n"
            status += f"Radius: {radius}, Height: {height}, Rotation Factor: {rotation_factor}\n"
            status += "Mode: Random sphere rotation (3D)"

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
            # Circle while rising - full orbit then return to start
            control_points = []
            num_points = 12  # More control points for smoother rise
            for i in range(num_points + 1):  # +1 to complete the circle
                angle = 2 * np.pi * i / num_points
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                y = height + (i / num_points) * radius * 0.8  # Rise as we orbit
                control_points.append((x, y, z))

            config = SplineConfig(
                num_frames=int(num_frames),
                num_control_points=num_points + 1,
                spline_type="catmull_rom",
                closed_loop=False,  # Never close - rising path can't loop
                smoothness=0.7
            )
            camera_path = generate_camera_path(config, control_points, look_at_curve=True)
            status = f"✅ Generated orbit-up path ({len(camera_path)} frames)\n"
            status += f"Radius: {radius}, Rise: {radius * 0.8:.1f}\n"
            status += "Note: Orbit-up cannot be closed (rising path)"

        elif preset_type == "spiral":
            # 3D spiral inward - fits in cube, camera looks at center
            control_points = []
            num_points = 24  # More points for smoother 3D spiral
            for i in range(num_points):
                t = i / (num_points - 1)  # 0 to 1
                angle = 6 * np.pi * t  # Three full rotations

                # Shrink radius as we spiral in
                r = radius * (1 - t)

                # Circular motion in XZ plane
                x = r * np.cos(angle)
                z = r * np.sin(angle)

                # Vertical motion - rise then fall (creates 3D cube-fitting spiral)
                # Use sine wave to go up and down within the cube
                y = height + radius * 0.8 * np.sin(2 * np.pi * t)

                control_points.append((x, y, z))

            # Generate path with camera looking at center (origin)
            camera_path = []
            for frame_idx in range(int(num_frames)):
                # Interpolate position along control points
                t = frame_idx / (num_frames - 1) if num_frames > 1 else 0
                point_idx = int(t * (num_points - 1))

                if point_idx >= len(control_points) - 1:
                    x, y, z = control_points[-1]
                else:
                    # Linear interpolation between control points
                    local_t = (t * (num_points - 1)) - point_idx
                    p1 = control_points[point_idx]
                    p2 = control_points[point_idx + 1]
                    x = p1[0] + local_t * (p2[0] - p1[0])
                    y = p1[1] + local_t * (p2[1] - p1[1])
                    z = p1[2] + local_t * (p2[2] - p1[2])

                # Calculate rotation to look at center (0, height, 0)
                dx = 0.0 - x
                dy = height - y
                dz = 0.0 - z

                # Pan angle (rotation_y)
                rot_y = np.degrees(np.arctan2(dx, dz))

                # Tilt angle (rotation_x)
                horizontal_dist = np.sqrt(dx**2 + dz**2)
                rot_x = np.degrees(np.arctan2(dy, horizontal_dist + 1e-8))

                rot_z = 0.0

                camera_path.append(CameraPoint(
                    x=x, y=y, z=z,
                    rot_x=rot_x, rot_y=rot_y, rot_z=rot_z,
                    frame=frame_idx
                ))

            status = f"✅ Generated 3D spiral path ({len(camera_path)} frames)\n"
            status += f"Start radius: {radius}, End radius: 0\n"
            status += f"Mode: 3D spiral with look-at center (fits in cube)"

        elif preset_type == "street":
            # Street/dashcam forward movement with lane weaving
            camera_path = generate_street_path(
                num_frames=int(num_frames),
                street_length=radius * 5,  # Radius maps to street length
                lane_weave=radius * 0.2,  # 20% of radius for weaving
                center_x=0.0,
                center_y=height if height != 0 else 10.0,  # Default eye level at 10
                center_z=0.0
            )
            status = f"✅ Generated street path ({len(camera_path)} frames)\n"
            status += f"Distance: {radius * 5:.0f}, Eye height: {height if height != 0 else 10.0:.0f}\n"
            status += "Mode: Forward-facing (dashcam/POV)"

        elif preset_type == "dashcam":
            # Dashcam preset - faster street movement with more bumps
            camera_path = generate_street_path(
                num_frames=int(num_frames),
                street_length=radius * 8,  # Longer distance (faster movement)
                lane_weave=radius * 0.3,  # More pronounced lane changes
                center_x=0.0,
                center_y=height if height != 0 else 8.0,  # Lower eye level (car seat)
                center_z=0.0
            )
            status = f"✅ Generated dashcam path ({len(camera_path)} frames)\n"
            status += f"Distance: {radius * 8:.0f}, Eye height: {height if height != 0 else 8.0:.0f}\n"
            status += "Mode: Dashcam (faster, more weaving)\n"
            status += "Tip: Combine with GENTLE_HANDHELD shakify pattern"

        elif preset_type == "bodycam":
            # Bodycam preset - slower, at standing height, subtle wobble
            camera_path = generate_street_path(
                num_frames=int(num_frames),
                street_length=radius * 3,  # Slower walking pace
                lane_weave=radius * 0.15,  # Less weaving (walking path)
                center_x=0.0,
                center_y=height if height != 0 else 15.0,  # Standing eye level
                center_z=0.0
            )
            status = f"✅ Generated bodycam path ({len(camera_path)} frames)\n"
            status += f"Distance: {radius * 3:.0f}, Eye height: {height if height != 0 else 15.0:.0f}\n"
            status += "Mode: Bodycam (walking pace)\n"
            status += "Tip: Combine with INVESTIGATION or GENTLE_HANDHELD shakify pattern"

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
        # Empty plot - SLOPCORE DARKMODE
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="No path generated yet",
                font=dict(color='#E0E7FF', size=18, family='system-ui')
            ),
            paper_bgcolor='#0F172A',  # Tailwind slate-900
            plot_bgcolor='#1E293B',   # Tailwind slate-800
            font=dict(color='#CBD5E1', family='system-ui'),  # Tailwind slate-300
            scene=dict(
                xaxis_title="X (Left/Right)",
                yaxis_title="Y (Up/Down)",
                zaxis_title="Z (Forward/Back)",
                xaxis=dict(
                    backgroundcolor='#1E293B',
                    gridcolor='#334155',  # Tailwind slate-700
                    showbackground=True,
                    zerolinecolor='#475569',  # Tailwind slate-600
                    title=dict(font=dict(color='#94A3B8'))  # Tailwind slate-400
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

    # Extract coordinates
    x_coords = [p.x for p in camera_path]
    y_coords = [p.y for p in camera_path]
    z_coords = [p.z for p in camera_path]

    # Create 3D line plot - SLOPCORE GRIFTWAVE AESTHETIC
    fig = go.Figure()

    # Path line - PURPLE GRADIENT VIBES (simulate gradient with multiple segments)
    # Create gradient effect by varying color along path
    num_points = len(x_coords)
    colors = [f'rgb({int(102 + (118-102)*i/num_points)}, {int(126 + (75-126)*i/num_points)}, {int(234 + (162-234)*i/num_points)})'
              for i in range(num_points)]

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

    # Keyframe markers - BRIGHT BLUE highlights at regular intervals (SLOPCORE)
    # Calculate keyframe positions (every ~20-40 frames depending on path length)
    keyframe_interval = max(20, min(40, num_points // 10))
    keyframe_indices = list(range(0, num_points, keyframe_interval))
    if keyframe_indices and keyframe_indices[-1] != num_points - 1:
        keyframe_indices.append(num_points - 1)  # Ensure last frame is included

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
    total_distance = 0.0
    for i in range(1, len(camera_path)):
        dx = camera_path[i].x - camera_path[i-1].x
        dy = camera_path[i].y - camera_path[i-1].y
        dz = camera_path[i].z - camera_path[i-1].z
        total_distance += np.sqrt(dx**2 + dy**2 + dz**2)

    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    z_range = max(z_coords) - min(z_coords)

    # Calculate keyframe info for stats
    keyframe_interval = max(20, min(40, num_points // 10))
    num_keyframes = (num_points + keyframe_interval - 1) // keyframe_interval  # Ceiling division

    stats = f"""Path Statistics:
- Frames: {len(camera_path)}
- Keyframes: ~{num_keyframes} (every {keyframe_interval} frames)
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
    randomize: float,
    random_seed: float,
    translation_x,
    translation_y,
    translation_z,
    rotation_3d_x,
    rotation_3d_y,
    rotation_3d_z
):
    """Handle preset path generation and populate schedules.

    Note: Visualization is now handled by schedule change events.
    This only updates the schedule textboxes.
    """
    global _current_camera_path

    status, schedules, camera_path = generate_preset_path(
        preset_type, radius, height, rotation_factor, num_frames, closed_loop,
        randomize, int(random_seed)
    )

    _current_camera_path = camera_path

    # Return schedule updates only - visualization will update automatically via .change() events
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
