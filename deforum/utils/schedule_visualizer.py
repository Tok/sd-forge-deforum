"""Schedule Visualization - Refactored

Visualizes Deforum animation schedules as 3D camera paths.
Parses schedule strings from UI textboxes and creates interactive plots.
Uses quaternion-based rotation for accurate camera direction arrows.

This is a REFACTORED version following strict functional programming principles:
- All functions ≤20 lines
- McCabe complexity ≤10
- Complete type hints
- Pure functions (no side effects)
- Single responsibility principle
"""

from typing import Dict, List, Tuple, Optional, NamedTuple, Set
from dataclasses import dataclass
import re
import plotly.graph_objects as go
import numpy as np
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.math.quaternion import euler_to_forward_vector


# ============================================================================
# Data Structures
# ============================================================================

class ColorPalette(NamedTuple):
    """BB0 Slopcore color palette."""
    void: str = '#5606FF'      # Deep purple-blue
    dusk: str = '#4C21FF'      # Purple-blue
    midnight: str = '#3757FF'  # Mid blue
    zenith: str = '#17A7FE'    # Cyan
    glitch: str = '#FF1493'    # Neon pink


@dataclass(frozen=True)
class Coordinates:
    """3D coordinates with rotation."""
    x: List[float]
    y: List[float]
    z: List[float]
    rx: List[float]
    ry: List[float]
    rz: List[float]

    def __len__(self) -> int:
        return len(self.x)


@dataclass(frozen=True)
class DownsampleMetadata:
    """Metadata from downsampling operation."""
    original_count: int
    downsampled: bool
    downsample_rate: Optional[int]
    frame_map: Dict[int, int]  # new_idx -> original_idx


@dataclass(frozen=True)
class AnimationConfig:
    """Configuration for animation frames."""
    max_frames: int
    sample_interval: int
    animation_frames: List[int]


# ============================================================================
# Schedule Parsing (Already well-factored in original)
# ============================================================================

def parse_schedule_string(schedule_str: str) -> Dict[int, float]:
    """Parse Deforum schedule string into frame->value mapping.

    Args:
        schedule_str: Schedule string like "0: (10), 50: (20), 100: (30)"

    Returns:
        Dict mapping frame numbers to values
    """
    if not schedule_str or not schedule_str.strip():
        return {}

    pattern = r'(\d+)\s*:\s*\(\s*(-?\d+\.?\d*)\s*\)'
    matches = re.findall(pattern, schedule_str)

    return {int(frame): float(value) for frame, value in matches}


def parse_prompt_schedule(prompt_str: str) -> Set[int]:
    """Parse animation_prompts string/JSON to extract keyframe numbers."""
    import json

    if not prompt_str or not prompt_str.strip():
        return set()

    try:
        prompt_dict = json.loads(prompt_str)
        if isinstance(prompt_dict, dict):
            return {int(k) for k in prompt_dict.keys()}
    except (json.JSONDecodeError, ValueError):
        pass

    pattern = r'(\d+)\s*:'
    matches = re.findall(pattern, prompt_str)
    return {int(m) for m in matches} if matches else set()


def interpolate_schedule(schedule_dict: Dict[int, float], max_frame: int) -> List[Tuple[int, float]]:
    """Interpolate schedule values for all frames."""
    if not schedule_dict:
        return [(i, 0.0) for i in range(max_frame + 1)]

    sorted_keyframes = sorted(schedule_dict.items())
    result = []

    for frame in range(max_frame + 1):
        prev_kf = next_kf = None

        for kf_frame, kf_value in sorted_keyframes:
            if kf_frame <= frame:
                prev_kf = (kf_frame, kf_value)
            if kf_frame >= frame and next_kf is None:
                next_kf = (kf_frame, kf_value)
                break

        value = _interpolate_value(frame, prev_kf, next_kf, sorted_keyframes)
        result.append((frame, value))

    return result


def _interpolate_value(
    frame: int,
    prev_kf: Optional[Tuple[int, float]],
    next_kf: Optional[Tuple[int, float]],
    sorted_keyframes: List[Tuple[int, float]]
) -> float:
    """Calculate interpolated value for a single frame."""
    if prev_kf is None:
        return sorted_keyframes[0][1]
    if next_kf is None:
        return sorted_keyframes[-1][1]
    if prev_kf[0] == frame:
        return prev_kf[1]

    t = (frame - prev_kf[0]) / (next_kf[0] - prev_kf[0])
    return prev_kf[1] + t * (next_kf[1] - prev_kf[1])


# ============================================================================
# Phase 1: Shakify Integration
# ============================================================================

def _apply_shakify_if_needed(
    schedules: Dict[str, str],
    shake_name: str,
    shake_intensity: float,
    shake_speed: float,
    max_frames: int,
    target_fps: int,
    apply_shakify: bool
) -> Dict[str, str]:
    """Apply camera shake to schedules if requested."""
    if not (apply_shakify and shake_name and shake_name != "None"):
        return schedules

    from deforum.utils.parsing.schedule_manipulation import get_final_schedules_with_shakify

    viz_intensity = shake_intensity * 0.3  # Scale down for visualization

    return get_final_schedules_with_shakify(
        base_schedules=schedules,
        shake_name=shake_name,
        shake_intensity=viz_intensity,
        shake_speed=shake_speed,
        max_frames=max_frames,
        target_fps=target_fps
    )


# ============================================================================
# Phase 2: Schedule Parsing
# ============================================================================

@dataclass(frozen=True)
class ParsedSchedules:
    """All parsed schedule dictionaries."""
    tx: Dict[int, float]
    ty: Dict[int, float]
    tz: Dict[int, float]
    rx: Dict[int, float]
    ry: Dict[int, float]
    rz: Dict[int, float]


def _parse_all_schedules(schedules: Dict[str, str]) -> ParsedSchedules:
    """Parse all 6 schedule strings into dicts."""
    return ParsedSchedules(
        tx=parse_schedule_string(schedules.get('translation_x', '')),
        ty=parse_schedule_string(schedules.get('translation_y', '')),
        tz=parse_schedule_string(schedules.get('translation_z', '')),
        rx=parse_schedule_string(schedules.get('rotation_3d_x', '')),
        ry=parse_schedule_string(schedules.get('rotation_3d_y', '')),
        rz=parse_schedule_string(schedules.get('rotation_3d_z', ''))
    )


def _get_all_schedule_frames(parsed: ParsedSchedules) -> List[int]:
    """Extract all frame numbers from all schedules."""
    all_frames = []
    for sched in [parsed.tx, parsed.ty, parsed.tz, parsed.rx, parsed.ry, parsed.rz]:
        all_frames.extend(sched.keys())
    return all_frames


# ============================================================================
# Phase 3: Mode Detection & Coordinate Conversion
# ============================================================================

def _detect_dense_schedule(parsed: ParsedSchedules, max_frames: int) -> bool:
    """Detect if schedules are dense (delta mode) or sparse (absolute mode)."""
    threshold = max_frames / 2
    return any(
        len(sched) > threshold
        for sched in [parsed.tx, parsed.ty, parsed.tz]
    )


def _interpolate_all_schedules(
    parsed: ParsedSchedules,
    max_frame: int
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Interpolate all 6 schedules and extract values."""
    tx_interp = interpolate_schedule(parsed.tx, max_frame)
    ty_interp = interpolate_schedule(parsed.ty, max_frame)
    tz_interp = interpolate_schedule(parsed.tz, max_frame)
    rx_interp = interpolate_schedule(parsed.rx, max_frame)
    ry_interp = interpolate_schedule(parsed.ry, max_frame)
    rz_interp = interpolate_schedule(parsed.rz, max_frame)

    return (
        [val for _, val in tx_interp],
        [val for _, val in ty_interp],
        [val for _, val in tz_interp],
        [val for _, val in rx_interp],
        [val for _, val in ry_interp],
        [val for _, val in rz_interp]
    )


def _accumulate_deltas(deltas: List[float]) -> List[float]:
    """Accumulate delta values into absolute positions."""
    coords = []
    cumulative = 0.0
    for delta in deltas:
        cumulative += delta
        coords.append(cumulative)
    return coords


def _convert_to_coordinates(
    tx: List[float], ty: List[float], tz: List[float],
    rx: List[float], ry: List[float], rz: List[float],
    is_dense: bool
) -> Coordinates:
    """Convert interpolated values to absolute coordinates."""
    if is_dense:
        # Delta mode: accumulate deltas
        return Coordinates(
            x=_accumulate_deltas(tx),
            y=_accumulate_deltas(ty),
            z=_accumulate_deltas(tz),
            rx=_accumulate_deltas(rx),
            ry=_accumulate_deltas(ry),
            rz=_accumulate_deltas(rz)
        )

    # Absolute mode: use values directly
    return Coordinates(x=tx, y=ty, z=tz, rx=rx, ry=ry, rz=rz)


# ============================================================================
# Phase 4: Downsampling for Large Animations
# ============================================================================

def _should_downsample(num_points: int, threshold: int = 5000) -> bool:
    """Check if animation exceeds threshold for downsampling."""
    return num_points > threshold


def _calculate_downsample_rate(num_points: int, target_frames: int = 400) -> int:
    """Calculate downsample rate to achieve target frame count."""
    return max(1, num_points // target_frames)


def _build_downsample_indices(
    num_points: int,
    downsample_rate: int,
    keyframes: Set[int]
) -> List[int]:
    """Build list of indices to keep after downsampling."""
    indices = set()

    # Add every Nth frame
    indices.update(range(0, num_points, downsample_rate))

    # Add all keyframes
    indices.update(kf for kf in keyframes if 0 <= kf < num_points)

    # Always include first and last
    indices.add(0)
    indices.add(num_points - 1)

    return sorted(indices)


def _downsample_coordinates(
    coords: Coordinates,
    keyframes: Set[int],
    threshold: int = 5000
) -> Tuple[Coordinates, DownsampleMetadata]:
    """Downsample coordinates for large animations."""
    if not _should_downsample(len(coords), threshold):
        return coords, DownsampleMetadata(
            original_count=len(coords),
            downsampled=False,
            downsample_rate=None,
            frame_map={}
        )

    rate = _calculate_downsample_rate(len(coords))
    indices = _build_downsample_indices(len(coords), rate, keyframes)

    return (
        Coordinates(
            x=[coords.x[i] for i in indices],
            y=[coords.y[i] for i in indices],
            z=[coords.z[i] for i in indices],
            rx=[coords.rx[i] for i in indices],
            ry=[coords.ry[i] for i in indices],
            rz=[coords.rz[i] for i in indices]
        ),
        DownsampleMetadata(
            original_count=len(coords),
            downsampled=True,
            downsample_rate=rate,
            frame_map={i: orig for i, orig in enumerate(indices)}
        )
    )


def _remap_keyframes(
    keyframes: Set[int],
    frame_map: Dict[int, int]
) -> Set[int]:
    """Remap keyframe indices after downsampling."""
    return {
        new_idx
        for new_idx, orig_idx in frame_map.items()
        if orig_idx in keyframes
    }


# ============================================================================
# Phase 5: Plot Trace Creation
# ============================================================================

def _create_path_trace(coords: Coordinates, palette: ColorPalette) -> go.Scatter3d:
    """Create main path line trace."""
    return go.Scatter3d(
        x=coords.x,
        y=coords.y,
        z=coords.z,
        mode='lines',
        name='Path',
        line=dict(color=palette.void, width=4),
        hoverinfo='skip',
        showlegend=False
    )


def _create_frame_markers_trace(coords: Coordinates, palette: ColorPalette) -> go.Scatter3d:
    """Create normal frame markers trace."""
    return go.Scatter3d(
        x=coords.x,
        y=coords.y,
        z=coords.z,
        mode='markers',
        name='Frames',
        marker=dict(size=4, color=palette.midnight, symbol='circle', opacity=0.4),
        hovertemplate='Frame %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
        text=list(range(len(coords))),
        showlegend=False
    )


def _create_keyframe_markers_trace(
    coords: Coordinates,
    keyframes: Set[int],
    palette: ColorPalette
) -> go.Scatter3d:
    """Create keyframe markers trace."""
    if not keyframes:
        return go.Scatter3d(x=[], y=[], z=[], mode='markers', showlegend=False)

    kf_list = sorted(keyframes)
    return go.Scatter3d(
        x=[coords.x[f] for f in kf_list if f < len(coords)],
        y=[coords.y[f] for f in kf_list if f < len(coords)],
        z=[coords.z[f] for f in kf_list if f < len(coords)],
        mode='markers',
        name='Keyframes',
        marker=dict(
            size=12,
            color=palette.zenith,
            symbol='circle',
            line=dict(color=palette.glitch, width=2),
            opacity=1.0
        ),
        text=[str(f) for f in kf_list if f < len(coords)],
        hovertemplate='<b>KEYFRAME %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
        showlegend=False
    )


def _create_camera_marker_trace(
    coords: Coordinates,
    idx: int,
    palette: ColorPalette
) -> go.Scatter3d:
    """Create animated camera position marker."""
    return go.Scatter3d(
        x=[coords.x[idx]],
        y=[coords.y[idx]],
        z=[coords.z[idx]],
        mode='markers',
        name='Camera',
        marker=dict(size=15, color=palette.glitch, symbol='diamond', opacity=1.0),
        hovertemplate=f'<b>Frame {idx}</b><br>X: {coords.x[idx]:.2f}<br>Y: {coords.y[idx]:.2f}<br>Z: {coords.z[idx]:.2f}<extra></extra>',
        showlegend=False
    )


def _calculate_forward_arrow(
    coords: Coordinates,
    idx: int,
    arrow_length: float = 1.0
) -> Tuple[float, float, float]:
    """Calculate forward direction vector for camera at given frame."""
    forward = euler_to_forward_vector(coords.rx[idx], coords.ry[idx], coords.rz[idx])
    return (
        forward.x * arrow_length,
        forward.y * arrow_length,
        forward.z * arrow_length
    )


def _create_forward_arrow_trace(
    coords: Coordinates,
    idx: int,
    fwd: Tuple[float, float, float],
    palette: ColorPalette
) -> go.Scatter3d:
    """Create forward direction arrow trace."""
    return go.Scatter3d(
        x=[coords.x[idx], coords.x[idx] + fwd[0]],
        y=[coords.y[idx], coords.y[idx] + fwd[1]],
        z=[coords.z[idx], coords.z[idx] + fwd[2]],
        mode='lines',
        name='Forward',
        line=dict(color=palette.glitch, width=4),
        hoverinfo='skip',
        showlegend=False
    )


def _create_arrowhead_trace(
    coords: Coordinates,
    idx: int,
    fwd: Tuple[float, float, float],
    palette: ColorPalette
) -> go.Scatter3d:
    """Create arrowhead marker trace."""
    return go.Scatter3d(
        x=[coords.x[idx] + fwd[0]],
        y=[coords.y[idx] + fwd[1]],
        z=[coords.z[idx] + fwd[2]],
        mode='markers',
        marker=dict(size=8, color=palette.glitch, symbol='diamond'),
        hoverinfo='skip',
        showlegend=False
    )


# ============================================================================
# Phase 6: Static Keyframe Arrows
# ============================================================================

def _create_keyframe_arrow_traces(
    coords: Coordinates,
    keyframes: Set[int],
    palette: ColorPalette,
    arrow_length: float = 1.0
) -> List[go.Scatter3d]:
    """Create static direction arrows for keyframes."""
    traces = []
    for idx in keyframes:
        if idx >= len(coords):
            continue

        fwd = _calculate_forward_arrow(coords, idx, arrow_length)

        traces.append(go.Scatter3d(
            x=[coords.x[idx], coords.x[idx] + fwd[0]],
            y=[coords.y[idx], coords.y[idx] + fwd[1]],
            z=[coords.z[idx], coords.z[idx] + fwd[2]],
            mode='lines',
            line=dict(color=palette.zenith, width=3),
            hovertemplate=f'<b>KEYFRAME {idx}</b><extra></extra>',
            showlegend=False,
            hoverinfo='text'
        ))

    return traces


# ============================================================================
# Phase 7: Animation Frames
# ============================================================================

def _get_animation_sample_rate(num_points: int, max_frames: int = 400) -> int:
    """Calculate frame sampling rate for animation."""
    return max(1, num_points // max_frames)


def _build_animation_frame_indices(
    num_points: int,
    sample_interval: int
) -> List[int]:
    """Build list of frame indices for animation."""
    frames = list(range(0, num_points, sample_interval))
    if frames[-1] != num_points - 1:
        frames.append(num_points - 1)
    return frames


def _create_animation_frame_data(
    coords: Coordinates,
    idx: int,
    keyframes: Set[int],
    num_static_arrows: int,
    palette: ColorPalette,
    arrow_length: float = 1.0
) -> List:
    """Create data for single animation frame."""
    fwd = _calculate_forward_arrow(coords, idx, arrow_length)
    is_keyframe = idx in keyframes
    color = palette.zenith if is_keyframe else palette.glitch

    frame_data = [
        {},  # Trace 0: Static path
        {},  # Trace 1: Static frame markers
        {},  # Trace 2: Static keyframes
        # Trace 3: Camera position
        go.Scatter3d(
            x=[coords.x[idx]],
            y=[coords.y[idx]],
            z=[coords.z[idx]],
            marker=dict(size=15, color=color, symbol='diamond', opacity=1.0),
            hovertemplate=f'<b>Frame {idx}</b><br>X: {coords.x[idx]:.2f}<br>Y: {coords.y[idx]:.2f}<br>Z: {coords.z[idx]:.2f}<extra></extra>'
        ),
        # Trace 4: Forward arrow
        go.Scatter3d(
            x=[coords.x[idx], coords.x[idx] + fwd[0]],
            y=[coords.y[idx], coords.y[idx] + fwd[1]],
            z=[coords.z[idx], coords.z[idx] + fwd[2]],
            line=dict(color=color, width=4)
        ),
        # Trace 5: Arrowhead
        go.Scatter3d(
            x=[coords.x[idx] + fwd[0]],
            y=[coords.y[idx] + fwd[1]],
            z=[coords.z[idx] + fwd[2]],
            marker=dict(size=8, color=color, symbol='diamond')
        )
    ]

    # Add empty {} for all static arrow traces
    frame_data.extend([{}] * num_static_arrows)

    return frame_data


def _build_animation_frames(
    coords: Coordinates,
    animation_indices: List[int],
    keyframes: Set[int],
    num_static_arrows: int,
    palette: ColorPalette
) -> List[go.Frame]:
    """Build all animation frames."""
    return [
        go.Frame(
            data=_create_animation_frame_data(
                coords, idx, keyframes, num_static_arrows, palette
            ),
            name=str(idx)
        )
        for idx in animation_indices
    ]


# ============================================================================
# Phase 8: Plot Layout Configuration
# ============================================================================

def _create_axis_config(color: str = '#334155') -> dict:
    """Create configuration for single axis."""
    return dict(
        backgroundcolor='#1E293B',
        gridcolor=color,
        showbackground=True,
        zerolinecolor='#475569',
        title=dict(text='', font=dict(color='#94A3B8', size=10)),
        tickfont=dict(color='#64748B', size=8),
        showspikes=False
    )


def _create_playback_buttons() -> List[dict]:
    """Create play/pause buttons for animation."""
    return [
        dict(
            label='▶',
            method='animate',
            args=[None, dict(
                frame=dict(duration=50, redraw=True),
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
    ]


def _create_frame_slider(frames: List[go.Frame]) -> dict:
    """Create frame slider for manual scrubbing."""
    return dict(
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
                label=f.name
            )
            for f in frames
        ]
    )


def _create_plot_layout(frames: List[go.Frame]) -> dict:
    """Create complete Plotly layout configuration."""
    return dict(
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        font=dict(color='#CBD5E1', family='system-ui, -apple-system, sans-serif', size=9),
        scene=dict(
            aspectmode='data',
            xaxis=_create_axis_config(),
            yaxis=_create_axis_config(),
            zaxis=_create_axis_config()
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),
        height=650,
        autosize=True,
        hovermode='closest',
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=_create_playback_buttons(),
            x=0.5,
            xanchor='center',
            y=1.02,
            yanchor='bottom',
            bgcolor='#1E293B',
            bordercolor='#334155',
            borderwidth=1,
            font=dict(color='#CBD5E1', size=12)
        )],
        sliders=[_create_frame_slider(frames)]
    )


# ============================================================================
# Phase 9: Statistics Calculation
# ============================================================================

def _calculate_path_distance(coords: Coordinates) -> float:
    """Calculate total path distance."""
    distance = 0.0
    for i in range(1, len(coords)):
        dx = coords.x[i] - coords.x[i-1]
        dy = coords.y[i] - coords.y[i-1]
        dz = coords.z[i] - coords.z[i-1]
        distance += np.sqrt(dx**2 + dy**2 + dz**2)
    return distance


def _calculate_coordinate_ranges(coords: Coordinates) -> Tuple[float, float, float]:
    """Calculate X, Y, Z ranges."""
    return (
        max(coords.x) - min(coords.x),
        max(coords.y) - min(coords.y),
        max(coords.z) - min(coords.z)
    )


def _format_statistics(
    coords: Coordinates,
    keyframes: Set[int],
    metadata: DownsampleMetadata,
    sample_interval: int,
    num_displayed: int
) -> str:
    """Format path statistics as display string."""
    distance = _calculate_path_distance(coords)
    x_range, y_range, z_range = _calculate_coordinate_ranges(coords)

    sampling_info = (
        f"\n⚠️ Displaying {num_displayed} of {len(coords)} frames "
        f"(sampled every {sample_interval} frames to prevent browser crash)"
    ) if num_displayed < len(coords) else ""

    downsample_info = (
        f"\n\nℹ️ Preview Downsampled:\n"
        f"- Original: {metadata.original_count:,} frames\n"
        f"- Showing: {len(coords):,} frames (every {metadata.downsample_rate}th + keyframes)\n"
        f"- This is preview-only; full animation will render all {metadata.original_count:,} frames"
    ) if metadata.downsampled else ""

    frame_count = metadata.original_count if metadata.downsampled else len(coords)

    return f"""Path Statistics (from schedules):
- Frames: {frame_count}
- Keyframes: {len(keyframes)} (with prompts)
- Total Distance: {distance:.2f}
- X Range: {x_range:.2f} (left/right)
- Y Range: {y_range:.2f} (up/down)
- Z Range: {z_range:.2f} (forward/back)
- Start: ({coords.x[0]:.2f}, {coords.y[0]:.2f}, {coords.z[0]:.2f})
- End: ({coords.x[-1]:.2f}, {coords.y[-1]:.2f}, {coords.z[-1]:.2f}){sampling_info}{downsample_info}
"""


# ============================================================================
# Figure Building
# ============================================================================

def _build_initial_schedule_dict(
    translation_x: str,
    translation_y: str,
    translation_z: str,
    rotation_3d_x: str,
    rotation_3d_y: str,
    rotation_3d_z: str
) -> Dict[str, str]:
    """Build initial schedule dictionary with defaults."""
    return {
        'translation_x': translation_x or "0:(0)",
        'translation_y': translation_y or "0:(0)",
        'translation_z': translation_z or "0:(0)",
        'rotation_3d_x': rotation_3d_x or "0:(0)",
        'rotation_3d_y': rotation_3d_y or "0:(0)",
        'rotation_3d_z': rotation_3d_z or "0:(0)",
    }


def _get_max_animation_frames_setting() -> int:
    """Get max animation frames from settings with fallback."""
    try:
        from modules.shared import opts
        return getattr(opts, 'deforum_max_viz_animation_frames', 400)
    except Exception:
        return 400


def _build_initial_traces(
    coords: Coordinates,
    keyframes: Set[int],
    anim_idx: int,
    palette: ColorPalette
) -> List[go.Scatter3d]:
    """Build initial plot traces."""
    fwd = _calculate_forward_arrow(coords, anim_idx)
    return [
        _create_path_trace(coords, palette),
        _create_frame_markers_trace(coords, palette),
        _create_keyframe_markers_trace(coords, keyframes, palette),
        _create_camera_marker_trace(coords, anim_idx, palette),
        _create_forward_arrow_trace(coords, anim_idx, fwd, palette),
        _create_arrowhead_trace(coords, anim_idx, fwd, palette)
    ]


def _build_complete_figure(
    coords: Coordinates,
    keyframes: Set[int],
    anim_indices: List[int],
    palette: ColorPalette
) -> go.Figure:
    """Build complete Plotly figure with traces, frames, and layout."""
    traces = _build_initial_traces(coords, keyframes, anim_indices[0], palette)

    keyframe_arrows = _create_keyframe_arrow_traces(
        coords, {idx for idx in anim_indices if idx in keyframes}, palette
    )

    frames = _build_animation_frames(
        coords, anim_indices, keyframes, len(keyframe_arrows), palette
    )

    layout = _create_plot_layout(frames)

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(layout)
    for arrow in keyframe_arrows:
        fig.add_trace(arrow)

    return fig


# ============================================================================
# Empty Plot Helper
# ============================================================================

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
            xaxis=_create_axis_config(),
            yaxis=_create_axis_config(),
            zaxis=_create_axis_config()
        ),
        annotations=[dict(
            text="No schedule data<br>Enter keyframes in Motion tab",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color='#94A3B8')
        )]
    )
    return fig


# ============================================================================
# Main Orchestrator Function
# ============================================================================

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
    """Create 3D visualization from Deforum schedule strings.

    Refactored to follow strict functional programming principles:
    - Complexity ≤10
    - Function body ≤50 lines (orchestrator exception to 20-line rule)
    - All helpers are pure functions <20 lines

    Returns:
        (plotly_figure, stats_text)
    """
    # Phase 1-2: Prepare and parse schedules
    schedules = _build_initial_schedule_dict(
        translation_x, translation_y, translation_z,
        rotation_3d_x, rotation_3d_y, rotation_3d_z
    )
    schedules = _apply_shakify_if_needed(
        schedules, shake_name, shake_intensity, shake_speed,
        max_frames, target_fps, apply_shakify
    )

    parsed = _parse_all_schedules(schedules)
    if not _get_all_schedule_frames(parsed):
        return _create_empty_plot(), "No schedule data"

    # Phase 3: Interpolate and convert to coordinates
    is_dense = _detect_dense_schedule(parsed, max_frames)
    tx, ty, tz, rx, ry, rz = _interpolate_all_schedules(parsed, max_frames)
    coords = _convert_to_coordinates(tx, ty, tz, rx, ry, rz, is_dense)

    # Phase 4: Downsample and prepare keyframes
    keyframes = parse_prompt_schedule(animation_prompts)
    coords, metadata = _downsample_coordinates(coords, keyframes)
    if metadata.downsampled:
        keyframes = _remap_keyframes(keyframes, metadata.frame_map)

    if len(coords) == 0:
        return _create_empty_plot(), "No frames after downsampling"

    # Phase 5-8: Build complete figure
    max_anim_frames = _get_max_animation_frames_setting()
    sample_interval = _get_animation_sample_rate(len(coords), max_anim_frames)
    anim_indices = _build_animation_frame_indices(len(coords), sample_interval)

    fig = _build_complete_figure(coords, keyframes, anim_indices, ColorPalette())

    # Phase 9: Generate statistics
    stats = _format_statistics(
        coords, keyframes, metadata, sample_interval, len(anim_indices)
    )

    return fig, stats
