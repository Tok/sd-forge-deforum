"""Schedule Visualization

Visualizes Deforum animation schedules as 3D camera paths.
Parses schedule strings from UI textboxes and creates interactive plots.
Uses quaternion-based rotation for accurate camera direction arrows.
"""

from typing import Dict, List, Tuple, Optional
import re
import plotly.graph_objects as go
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.math.quaternion import euler_to_forward_vector


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
    animation_prompts: str = "",
    shake_name: str = "None",
    shake_intensity: float = 1.0,
    shake_speed: float = 1.0,
    target_fps: int = 60,
    apply_shakify: bool = False
) -> Tuple[go.Figure, str]:
    """Create 3D visualization from Deforum schedule strings with optional shakify.

    Args:
        translation_x: Translation X schedule string
        translation_y: Translation Y schedule string
        translation_z: Translation Z schedule string
        rotation_3d_x: Rotation X schedule string
        rotation_3d_y: Rotation Y schedule string
        rotation_3d_z: Rotation Z schedule string
        max_frames: Maximum number of frames to visualize
        animation_prompts: Prompt schedule string for keyframe markers
        shake_name: Camera shakify pattern name
        shake_intensity: Shakify intensity multiplier
        shake_speed: Shakify speed multiplier
        target_fps: Target FPS for shakify interpolation
        apply_shakify: If True, overlay shakify (scaled down 30% for subtlety)

    Returns:
        (plotly_figure, stats_text)
    """
    # Apply shakify if requested (with 30% intensity scaling for visualization)
    if apply_shakify and shake_name and shake_name != "None":
        from deforum.utils.parsing.schedule_manipulation import get_final_schedules_with_shakify

        base_schedules = {
            'translation_x': translation_x or "0:(0)",
            'translation_y': translation_y or "0:(0)",
            'translation_z': translation_z or "0:(0)",
            'rotation_3d_x': rotation_3d_x or "0:(0)",
            'rotation_3d_y': rotation_3d_y or "0:(0)",
            'rotation_3d_z': rotation_3d_z or "0:(0)",
        }

        # Scale down intensity to 30% for subtle visualization
        viz_intensity = shake_intensity * 0.3

        final_schedules = get_final_schedules_with_shakify(
            base_schedules=base_schedules,
            shake_name=shake_name,
            shake_intensity=viz_intensity,
            shake_speed=shake_speed,
            max_frames=max_frames,
            target_fps=target_fps
        )

        translation_x = final_schedules['translation_x']
        translation_y = final_schedules['translation_y']
        translation_z = final_schedules['translation_z']
        rotation_3d_x = final_schedules['rotation_3d_x']
        rotation_3d_y = final_schedules['rotation_3d_y']
        rotation_3d_z = final_schedules['rotation_3d_z']

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
        # Use max_frames from UI - interpolate/extrapolate schedules to full animation length
        # Note: interpolate_schedule will hold last keyframe value constant after last keyframe
        actual_max_frame = max_frames

        # Interpolate all schedules
        tx_interp = interpolate_schedule(tx_dict, actual_max_frame)
        ty_interp = interpolate_schedule(ty_dict, actual_max_frame)
        tz_interp = interpolate_schedule(tz_dict, actual_max_frame)
        rx_interp = interpolate_schedule(rx_dict, actual_max_frame)
        ry_interp = interpolate_schedule(ry_dict, actual_max_frame)
        rz_interp = interpolate_schedule(rz_dict, actual_max_frame)

        # Extract delta values
        x_deltas = [val for _, val in tx_interp]
        y_deltas = [val for _, val in ty_interp]
        z_deltas = [val for _, val in tz_interp]
        rx_deltas = [val for _, val in rx_interp]
        ry_deltas = [val for _, val in ry_interp]
        rz_deltas = [val for _, val in rz_interp]

        # Detect if schedules are dense (every frame defined) = delta mode
        # Camera paths output: ALL deltas (translation + rotation)
        # Manual schedules: ALL absolutes (translation + rotation)
        is_dense_schedule = (
            len(tx_dict) > (actual_max_frame / 2) or  # More than half frames defined
            len(ty_dict) > (actual_max_frame / 2) or
            len(tz_dict) > (actual_max_frame / 2)
        )

        if is_dense_schedule:
            # Dense schedule = delta mode (camera paths, Parseq delta)
            # Accumulate ALL deltas to get absolute positions for visualization
            x_coords = []
            y_coords = []
            z_coords = []
            rx_coords = []
            ry_coords = []
            rz_coords = []

            cum_x, cum_y, cum_z = 0.0, 0.0, 0.0
            cum_rx, cum_ry, cum_rz = 0.0, 0.0, 0.0

            for i in range(len(x_deltas)):
                cum_x += x_deltas[i]
                cum_y += y_deltas[i]
                cum_z += z_deltas[i]
                cum_rx += rx_deltas[i]
                cum_ry += ry_deltas[i]
                cum_rz += rz_deltas[i]

                x_coords.append(cum_x)
                y_coords.append(cum_y)
                z_coords.append(cum_z)
                rx_coords.append(cum_rx)
                ry_coords.append(cum_ry)
                rz_coords.append(cum_rz)
        else:
            # Sparse schedule = absolute mode (manual keyframes with interpolation)
            # Use interpolated values directly for both translation and rotation
            x_coords = x_deltas
            y_coords = y_deltas
            z_coords = z_deltas
            rx_coords = rx_deltas
            ry_coords = ry_deltas
            rz_coords = rz_deltas

        num_points = len(x_coords)

        # Parse prompt schedule to find actual keyframes (frames with prompts)
        prompt_keyframes = set()
        if animation_prompts:
            prompt_keyframes = parse_prompt_schedule(animation_prompts)

    if num_points == 0:
        # Fallback for empty data
        return _create_empty_plot(), "No schedule data"

    # Skip visualization for very large animations (>5000 frames)
    # Generating visualization for 30k+ frames freezes the browser for minutes
    LARGE_ANIMATION_THRESHOLD = 5000
    if num_points > LARGE_ANIMATION_THRESHOLD:
        # Calculate basic stats for user feedback
        import numpy as np
        total_distance = sum(
            np.sqrt(
                (x_coords[i] - x_coords[i-1])**2 +
                (y_coords[i] - y_coords[i-1])**2 +
                (z_coords[i] - z_coords[i-1])**2
            )
            for i in range(1, len(x_coords))
        )

        stats = f"""⚠️ Animation Too Large for Auto-Visualization

Frames: {num_points:,} (exceeds {LARGE_ANIMATION_THRESHOLD:,} frame threshold)
Keyframes: {len(prompt_keyframes)}
Total Distance: {total_distance:.2f}

✅ Schedules have been generated and populated correctly in the textboxes above.

⚠️ Visualization skipped to prevent browser freeze (would take several minutes to load).

To view visualization:
• Reduce Max Frames to <{LARGE_ANIMATION_THRESHOLD:,} and regenerate path, OR
• Use visualization for design/preview, then increase Max Frames for final render

Note: Camera path schedules work perfectly regardless of visualization display.
"""
        return _create_empty_plot(), stats

    # Create animated 3D plot with BB0 Slopcore gradient
    # See docs/SLOPCORE.md for full palette documentation

    # BB0 Slopcore palette (solid colors, not gradient-over-time)
    BB0_VOID = '#5606FF'      # Deep purple-blue - path line, non-keyframe arrows
    BB0_DUSK = '#4C21FF'      # Purple-blue - (reserved for cadence diffusion frames)
    BB0_MIDNIGHT = '#3757FF'  # Mid blue - (reserved for tween frames)
    BB0_ZENITH = '#17A7FE'    # Cyan - keyframes, keyframe arrows
    BB0_GLITCH = '#FF1493'    # Neon pink - arrowheads

    # Frame sampling for animation to prevent browser crashes
    # Read max frames from settings (default 400 if not set)
    try:
        from modules.shared import opts
        MAX_ANIMATION_FRAMES = getattr(opts, 'deforum_max_viz_animation_frames', 400)
    except Exception:
        MAX_ANIMATION_FRAMES = 400  # Fallback if settings not available

    sample_interval = max(1, num_points // MAX_ANIMATION_FRAMES)

    # Build list of frames to animate (sampled uniformly)
    animation_frames = list(range(0, num_points, sample_interval))
    if animation_frames[-1] != num_points - 1:
        animation_frames.append(num_points - 1)  # Always include last frame

    num_displayed = len(animation_frames)

    # Build initial frame data (frame 0)
    idx = animation_frames[0]
    pitch = rx_coords[idx]
    yaw = ry_coords[idx]
    roll = rz_coords[idx]
    forward = euler_to_forward_vector(pitch, yaw, roll)
    arrow_length = 10
    forward_x = forward.x * arrow_length
    forward_y = forward.y * arrow_length
    forward_z = forward.z * arrow_length

    # Create figure with initial traces
    fig = go.Figure(
        data=[
            # Trace 0: Static path line (always visible)
            go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                name='Path',
                line=dict(color=BB0_VOID, width=4),
                hoverinfo='skip',
                showlegend=False
            ),
            # Trace 1: Static keyframe markers (always visible)
            go.Scatter3d(
                x=[x_coords[f] for f in prompt_keyframes if f < len(x_coords)],
                y=[y_coords[f] for f in prompt_keyframes if f < len(y_coords)],
                z=[z_coords[f] for f in prompt_keyframes if f < len(z_coords)],
                mode='markers',
                name='Keyframes',
                marker=dict(
                    size=12,
                    color=BB0_ZENITH,
                    symbol='circle',
                    line=dict(color=BB0_GLITCH, width=2),
                    opacity=1.0
                ),
                text=[str(f) for f in sorted(prompt_keyframes) if f < len(x_coords)],
                hovertemplate='<b>KEYFRAME %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                showlegend=False
            ) if prompt_keyframes else go.Scatter3d(x=[], y=[], z=[], mode='markers', showlegend=False),
            # Trace 2: Animated camera position marker
            go.Scatter3d(
                x=[x_coords[idx]],
                y=[y_coords[idx]],
                z=[z_coords[idx]],
                mode='markers',
                name='Camera',
                marker=dict(
                    size=15,
                    color=BB0_GLITCH,
                    symbol='diamond',
                    opacity=1.0
                ),
                hovertemplate=f'<b>Frame {idx}</b><br>X: {x_coords[idx]:.2f}<br>Y: {y_coords[idx]:.2f}<br>Z: {z_coords[idx]:.2f}<extra></extra>',
                showlegend=False
            ),
            # Trace 3: Animated forward arrow
            go.Scatter3d(
                x=[x_coords[idx], x_coords[idx] + forward_x],
                y=[y_coords[idx], y_coords[idx] + forward_y],
                z=[z_coords[idx], z_coords[idx] + forward_z],
                mode='lines',
                name='Forward',
                line=dict(color=BB0_GLITCH, width=4),
                hoverinfo='skip',
                showlegend=False
            ),
            # Trace 4: Animated arrowhead
            go.Scatter3d(
                x=[x_coords[idx] + forward_x],
                y=[y_coords[idx] + forward_y],
                z=[z_coords[idx] + forward_z],
                mode='markers',
                marker=dict(size=8, color=BB0_GLITCH, symbol='diamond'),
                hoverinfo='skip',
                showlegend=False
            )
        ]
    )

    # Build animation frames (update traces 2, 3, 4 for each frame)
    plotly_frames = []
    for idx in animation_frames:
        pitch = rx_coords[idx]
        yaw = ry_coords[idx]
        roll = rz_coords[idx]
        forward = euler_to_forward_vector(pitch, yaw, roll)
        forward_x = forward.x * arrow_length
        forward_y = forward.y * arrow_length
        forward_z = forward.z * arrow_length

        is_keyframe = idx in prompt_keyframes
        marker_color = BB0_ZENITH if is_keyframe else BB0_GLITCH

        plotly_frames.append(go.Frame(
            data=[
                {},  # Trace 0: Static path (no update)
                {},  # Trace 1: Static keyframes (no update)
                # Trace 2: Camera position
                go.Scatter3d(
                    x=[x_coords[idx]],
                    y=[y_coords[idx]],
                    z=[z_coords[idx]],
                    marker=dict(size=15, color=marker_color, symbol='diamond', opacity=1.0),
                    hovertemplate=f'<b>Frame {idx}</b><br>X: {x_coords[idx]:.2f}<br>Y: {y_coords[idx]:.2f}<br>Z: {z_coords[idx]:.2f}<extra></extra>'
                ),
                # Trace 3: Forward arrow
                go.Scatter3d(
                    x=[x_coords[idx], x_coords[idx] + forward_x],
                    y=[y_coords[idx], y_coords[idx] + forward_y],
                    z=[z_coords[idx], z_coords[idx] + forward_z],
                    line=dict(color=marker_color, width=4)
                ),
                # Trace 4: Arrowhead
                go.Scatter3d(
                    x=[x_coords[idx] + forward_x],
                    y=[y_coords[idx] + forward_y],
                    z=[z_coords[idx] + forward_z],
                    marker=dict(size=8, color=marker_color, symbol='diamond')
                )
            ],
            name=str(idx)
        ))

    fig.frames = plotly_frames

    # Add animation controls (play/pause buttons + slider)
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
        margin=dict(l=0, r=0, t=20, b=0),  # Small top margin for controls
        height=650,  # Taller plot
        autosize=True,
        hovermode='closest',
        # Animation controls
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='▶',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=50, redraw=True),  # 20 fps
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label='⏸',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ],
                x=0.5,
                xanchor='center',
                y=1.02,
                yanchor='bottom',
                bgcolor='#1E293B',
                bordercolor='#334155',
                borderwidth=1,
                font=dict(color='#CBD5E1', size=12)
            )
        ],
        sliders=[dict(
            active=0,
            yanchor='top',
            y=0,
            xanchor='left',
            x=0,
            currentvalue=dict(
                prefix='Frame: ',
                visible=True,
                xanchor='left',
                font=dict(color='#CBD5E1', size=10)
            ),
            pad=dict(b=10, t=0),
            len=1.0,
            bgcolor='#1E293B',
            bordercolor='#334155',
            borderwidth=1,
            font=dict(color='#CBD5E1', size=8),
            steps=[
                dict(
                    args=[[f.name], dict(
                        frame=dict(duration=0, redraw=True),
                        mode='immediate',
                        transition=dict(duration=0)
                    )],
                    method='animate',
                    label=str(idx)
                )
                for idx, f in enumerate(plotly_frames)
            ]
        )]
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

    # Add sampling info if frames were sampled
    sampling_info = ""
    if num_displayed < num_points:
        sampling_info = f"\n⚠️ Displaying {num_displayed} of {num_points} frames (sampled every {sample_interval} frames to prevent browser crash)"

    stats = f"""Path Statistics (from schedules):
- Frames: {num_points}
- Keyframes: {len(prompt_keyframes)} (with prompts)
- Total Distance: {total_distance:.2f}
- X Range: {x_range:.2f} (left/right)
- Y Range: {y_range:.2f} (up/down)
- Z Range: {z_range:.2f} (forward/back)
- Start: ({x_coords[0]:.2f}, {y_coords[0]:.2f}, {z_coords[0]:.2f})
- End: ({x_coords[-1]:.2f}, {y_coords[-1]:.2f}, {z_coords[-1]:.2f}){sampling_info}
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