"""Schedule Visualization

Visualizes Deforum animation schedules as 3D camera paths.
Parses schedule strings from UI textboxes and creates interactive plots.
"""

from typing import Dict, List, Tuple, Optional
import re
import plotly.graph_objects as go
from deforum.utils.system.logging import emoji as emoji_utils


def parse_schedule_string(schedule_str: str) -> Dict[int, float]:
    """Parse Deforum schedule string into frame->value mapping.

    Args:
        schedule_str: Schedule string like "0: (10), 50: (20), 100: (30)"

    Returns:
        Dict mapping frame numbers to values

    Examples:
        >>> parse_schedule_string("0: (10), 50: (20)")
        {0: 10.0, 50: 20.0}
        >>> parse_schedule_string("0:(10.5),100:(20.3)")
        {0: 10.5, 100: 20.3}
    """
    if not schedule_str or not schedule_str.strip():
        return {}

    # Pattern: frame_number: (value)
    # Handles spaces and various formats: "0: (10)", "0:(10)", "0 : ( 10 )"
    pattern = r'(\d+)\s*:\s*\(\s*(-?\d+\.?\d*)\s*\)'
    matches = re.findall(pattern, schedule_str)

    schedule_dict = {}
    for frame_str, value_str in matches:
        frame = int(frame_str)
        value = float(value_str)
        schedule_dict[frame] = value

    return schedule_dict


def parse_prompt_schedule(prompt_str: str) -> set:
    """Parse animation_prompts string/JSON to extract keyframe numbers.

    Args:
        prompt_str: Either JSON dict string like '{"0": "prompt", "50": "other"}'
                    or schedule-like string

    Returns:
        Set of frame numbers that have prompts
    """
    import json

    if not prompt_str or not prompt_str.strip():
        return set()

    # Try parsing as JSON first (standard animation_prompts format)
    try:
        prompt_dict = json.loads(prompt_str)
        if isinstance(prompt_dict, dict):
            return {int(k) for k in prompt_dict.keys()}
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to regex pattern for any "frame_number:" pattern
    # This catches both "0: (text)" and "0: text" formats
    pattern = r'(\d+)\s*:'
    matches = re.findall(pattern, prompt_str)

    if matches:
        return {int(m) for m in matches}

    return set()


def interpolate_schedule(schedule_dict: Dict[int, float], max_frame: int) -> List[Tuple[int, float]]:
    """Interpolate schedule values for all frames.

    Args:
        schedule_dict: Frame->value mapping from parse_schedule_string
        max_frame: Maximum frame number to interpolate to

    Returns:
        List of (frame, value) tuples for all frames 0 to max_frame
    """
    if not schedule_dict:
        return [(i, 0.0) for i in range(max_frame + 1)]

    # Sort keyframes by frame number
    sorted_keyframes = sorted(schedule_dict.items())

    result = []
    for frame in range(max_frame + 1):
        # Find surrounding keyframes
        prev_kf = None
        next_kf = None

        for kf_frame, kf_value in sorted_keyframes:
            if kf_frame <= frame:
                prev_kf = (kf_frame, kf_value)
            if kf_frame >= frame and next_kf is None:
                next_kf = (kf_frame, kf_value)
                break

        # Interpolate value
        if prev_kf is None:
            # Before first keyframe - use first value
            value = sorted_keyframes[0][1]
        elif next_kf is None:
            # After last keyframe - use last value
            value = sorted_keyframes[-1][1]
        elif prev_kf[0] == frame:
            # Exactly on a keyframe
            value = prev_kf[1]
        else:
            # Linear interpolation between keyframes
            t = (frame - prev_kf[0]) / (next_kf[0] - prev_kf[0])
            value = prev_kf[1] + t * (next_kf[1] - prev_kf[1])

        result.append((frame, value))

    return result


def visualize_schedules(
    translation_x: str,
    translation_y: str,
    translation_z: str,
    rotation_3d_x: str,
    rotation_3d_y: str,
    rotation_3d_z: str,
    max_frames: int = 333,
    animation_prompts: str = ""
) -> Tuple[go.Figure, str]:
    """Create 3D visualization from Deforum schedule strings.

    Args:
        translation_x: Translation X schedule string
        translation_y: Translation Y schedule string
        translation_z: Translation Z schedule string
        rotation_3d_x: Rotation X schedule string
        rotation_3d_y: Rotation Y schedule string
        rotation_3d_z: Rotation Z schedule string
        max_frames: Maximum number of frames to visualize

    Returns:
        (plotly_figure, stats_text)
    """
    # Parse all schedules
    tx_dict = parse_schedule_string(translation_x)
    ty_dict = parse_schedule_string(translation_y)
    tz_dict = parse_schedule_string(translation_z)
    rx_dict = parse_schedule_string(rotation_3d_x)
    ry_dict = parse_schedule_string(rotation_3d_y)
    rz_dict = parse_schedule_string(rotation_3d_z)

    # Determine actual max frame from schedules
    all_frames = []
    for sched in [tx_dict, ty_dict, tz_dict, rx_dict, ry_dict, rz_dict]:
        if sched:
            all_frames.extend(sched.keys())

    if not all_frames:
        # No schedules defined - use defaults (single point at origin)
        actual_max_frame = 0
        # Create default single-frame path at origin
        x_coords, y_coords, z_coords = [0.0], [0.0], [0.0]
        rx_coords, ry_coords, rz_coords = [0.0], [0.0], [0.0]
        num_points = 1
        prompt_keyframes = set()
    else:
        actual_max_frame = min(max(all_frames), max_frames)

        # Interpolate all schedules
        tx_interp = interpolate_schedule(tx_dict, actual_max_frame)
        ty_interp = interpolate_schedule(ty_dict, actual_max_frame)
        tz_interp = interpolate_schedule(tz_dict, actual_max_frame)
        rx_interp = interpolate_schedule(rx_dict, actual_max_frame)
        ry_interp = interpolate_schedule(ry_dict, actual_max_frame)
        rz_interp = interpolate_schedule(rz_dict, actual_max_frame)

        # Extract coordinates
        x_coords = [val for _, val in tx_interp]
        y_coords = [val for _, val in ty_interp]
        z_coords = [val for _, val in tz_interp]
        rx_coords = [val for _, val in rx_interp]
        ry_coords = [val for _, val in ry_interp]
        rz_coords = [val for _, val in rz_interp]
        num_points = len(x_coords)

        # Parse prompt schedule to find actual keyframes (frames with prompts)
        prompt_keyframes = set()
        if animation_prompts:
            prompt_keyframes = parse_prompt_schedule(animation_prompts)

    if num_points == 0:
        # Fallback for empty data
        return _create_empty_plot(), "No schedule data"

    # Create 3D plot - SLOPCORE AESTHETIC
    fig = go.Figure()

    # Path line - PURPLE GRADIENT (thin, subtle)
    colors = [f'rgb({int(102 + (118-102)*i/num_points)}, {int(126 + (75-126)*i/num_points)}, {int(234 + (162-234)*i/num_points)})'
              for i in range(num_points)]

    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        name='Path',
        line=dict(
            color='#667EEA',
            width=3,
        ),
        hoverinfo='skip',
        showlegend=False
    ))

    # Prompt keyframes - BRIGHT BLUE (frames with prompt entries)
    if prompt_keyframes:
        keyframe_x = [x_coords[f] for f in prompt_keyframes if f < len(x_coords)]
        keyframe_y = [y_coords[f] for f in prompt_keyframes if f < len(y_coords)]
        keyframe_z = [z_coords[f] for f in prompt_keyframes if f < len(z_coords)]
        keyframe_labels = [str(f) for f in sorted(prompt_keyframes) if f < len(x_coords)]

        fig.add_trace(go.Scatter3d(
            x=keyframe_x,
            y=keyframe_y,
            z=keyframe_z,
            mode='markers',
            name='Keyframes',
            marker=dict(
                size=10,
                color='#3B82F6',  # Bright blue (slopcore)
                symbol='circle',
                line=dict(color='#2563EB', width=2),
                opacity=1.0
            ),
            hovertemplate='<b>KEYFRAME %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
            text=keyframe_labels,
            showlegend=False
        ))

    # Camera direction arrows - GREEN (classic) or PINK/PURPLE (slopcore)
    import numpy as np

    # Get theme-aware arrow color
    try:
        from deforum.rendering.options import get_log_theme
        theme = get_log_theme()
        # Slopcore: Deep purple/pink (#A353A8 = SLOPCORE_6)
        # Classic: Green (#10B981)
        arrow_color = '#A353A8' if theme == 'slopcore' else '#10B981'
    except:
        arrow_color = '#10B981'  # Fallback to green

    for idx in range(num_points):
        # Calculate forward direction from rotation angles (simplified)
        rx = np.radians(rx_coords[idx])
        ry = np.radians(ry_coords[idx])

        # Forward vector (camera -Z axis after rotation)
        # Simplified calculation - approximate direction
        arrow_length = 10  # Smaller arrows since we show all frames
        forward_x = np.sin(ry) * arrow_length
        forward_z = np.cos(ry) * arrow_length
        forward_y = -np.sin(rx) * arrow_length

        # Arrow from camera position pointing in look direction
        fig.add_trace(go.Scatter3d(
            x=[x_coords[idx], x_coords[idx] + forward_x],
            y=[y_coords[idx], y_coords[idx] + forward_y],
            z=[z_coords[idx], z_coords[idx] + forward_z],
            mode='lines',
            line=dict(color=arrow_color, width=2),  # Thinner for all frames
            hovertemplate=f'<b>Frame {idx}</b><extra></extra>',
            showlegend=False,
            hoverinfo='text'
        ))

        # Arrowhead (small marker at end)
        fig.add_trace(go.Scatter3d(
            x=[x_coords[idx] + forward_x],
            y=[y_coords[idx] + forward_y],
            z=[z_coords[idx] + forward_z],
            mode='markers',
            marker=dict(size=2, color='#10B981', symbol='diamond'),
            hoverinfo='skip',
            showlegend=False
        ))

    # Update layout - DARKMODE SLOPCORE, MAXIMUM SPACE
    fig.update_layout(
        paper_bgcolor='#0F172A',  # Tailwind slate-900
        plot_bgcolor='#1E293B',   # Tailwind slate-800
        font=dict(color='#CBD5E1', family='system-ui, -apple-system, sans-serif', size=9),
        scene=dict(
            aspectmode='data',
            xaxis=dict(
                backgroundcolor='#1E293B',
                gridcolor='#334155',
                showbackground=True,
                zerolinecolor='#475569',
                title=dict(text='', font=dict(color='#94A3B8', size=10)),  # No title to save space
                tickfont=dict(color='#64748B', size=8),
                showspikes=False
            ),
            yaxis=dict(
                backgroundcolor='#1E293B',
                gridcolor='#334155',
                showbackground=True,
                zerolinecolor='#475569',
                title=dict(text='', font=dict(color='#94A3B8', size=10)),
                tickfont=dict(color='#64748B', size=8),
                showspikes=False
            ),
            zaxis=dict(
                backgroundcolor='#1E293B',
                gridcolor='#334155',
                showbackground=True,
                zerolinecolor='#475569',
                title=dict(text='', font=dict(color='#94A3B8', size=10)),
                tickfont=dict(color='#64748B', size=8),
                showspikes=False
            )
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),  # Zero margins for maximum space
        height=650,  # Taller plot
        autosize=True,
        hovermode='closest'
    )

    # Calculate stats
    import numpy as np

    total_distance = 0.0
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        dz = z_coords[i] - z_coords[i-1]
        total_distance += np.sqrt(dx**2 + dy**2 + dz**2)

    x_range = max(x_coords) - min(x_coords) if x_coords else 0
    y_range = max(y_coords) - min(y_coords) if y_coords else 0
    z_range = max(z_coords) - min(z_coords) if z_coords else 0

    stats = f"""Path Statistics (from schedules):
- Frames: {num_points}
- Keyframes: {len(prompt_keyframes)} (with prompts)
- Total Distance: {total_distance:.2f}
- X Range: {x_range:.2f} (left/right)
- Y Range: {y_range:.2f} (up/down)
- Z Range: {z_range:.2f} (forward/back)
- Start: ({x_coords[0]:.2f}, {y_coords[0]:.2f}, {z_coords[0]:.2f})
- End: ({x_coords[-1]:.2f}, {y_coords[-1]:.2f}, {z_coords[-1]:.2f})
"""

    return fig, stats


def _create_empty_plot() -> go.Figure:
    """Create empty plot with message."""
    title_emoji = f"{emoji_utils.wan_video()} " if emoji_utils.wan_video() else ""

    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=f"<b>{title_emoji}Camera Path (Live from Schedules)</b>",
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
        ),
        annotations=[
            dict(
                text="No schedule data<br>Enter keyframes in Motion tab",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color='#94A3B8')
            )
        ]
    )
    return fig