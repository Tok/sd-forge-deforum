"""Audio Sync Timeline Visualizer - Refactored

Refactored from create_keyframe_timeline_plot (206 lines, complexity D-27)
into modular pure functions following strict functional programming principles.

Original: deforum/utils/audio/sync.py:116-321
"""

from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np


# ============================================================================
# Data Structures
# ============================================================================

class ThemeColors(NamedTuple):
    """Theme-aware color palette for timeline visualization."""
    marker: str
    line: str
    background: str
    plot_bg: str
    grid: str
    text: str


@dataclass(frozen=True)
class WaveformData:
    """Processed audio waveform data."""
    frame_x: List[int]
    normalized_amps: List[float]
    color_rgba: str


@dataclass(frozen=True)
class KeyframeMetrics:
    """Processed keyframe metrics for visualization."""
    frame_numbers: List[int]
    intensities: List[float]
    normalized_intensities: List[float]
    marker_sizes: List[float]
    timestamps: List[float]
    hover_texts: List[str]


# ============================================================================
# Theme Management
# ============================================================================

def _get_theme_name() -> str:
    """Get current log theme with fallback."""
    try:
        from deforum.rendering.options import get_log_theme
        return get_log_theme()
    except Exception:
        return "slopcore"


def _get_slopcore_colors() -> ThemeColors:
    """Get Slopcore theme colors."""
    return ThemeColors(
        marker='#667EEA',
        line='#A353A8',
        background='#0F172A',
        plot_bg='#1E293B',
        grid='#334155',
        text='#CBD5E1'
    )


def _get_classic_colors() -> ThemeColors:
    """Get Classic theme colors."""
    return ThemeColors(
        marker='#3B82F6',
        line='#10B981',
        background='#0F172A',
        plot_bg='#1E293B',
        grid='#334155',
        text='#CBD5E1'
    )


def get_theme_colors(theme: str = None) -> ThemeColors:
    """Get theme-aware colors for visualization."""
    if theme is None:
        theme = _get_theme_name()

    return _get_slopcore_colors() if theme == "slopcore" else _get_classic_colors()


# ============================================================================
# Audio Waveform Processing
# ============================================================================

def _calculate_samples_per_frame(audio_length: int, total_frames: int) -> int:
    """Calculate how many audio samples per video frame."""
    samples_per_frame = audio_length // total_frames
    return max(1, samples_per_frame)


def _calculate_frame_rms(
    audio_data: np.ndarray,
    frame_idx: int,
    samples_per_frame: int
) -> float:
    """Calculate RMS amplitude for a single frame."""
    start_idx = frame_idx * samples_per_frame
    end_idx = min(start_idx + samples_per_frame, len(audio_data))

    if start_idx >= len(audio_data):
        return 0.0

    chunk = audio_data[start_idx:end_idx]
    return float(np.sqrt(np.mean(chunk**2)))


def _extract_frame_amplitudes(
    audio_data: np.ndarray,
    total_frames: int
) -> List[float]:
    """Extract RMS amplitude for each frame."""
    samples_per_frame = _calculate_samples_per_frame(len(audio_data), total_frames)

    return [
        _calculate_frame_rms(audio_data, i, samples_per_frame)
        for i in range(total_frames)
    ]


def _normalize_amplitudes(amplitudes: List[float], scale: float = 0.4) -> List[float]:
    """Normalize amplitudes to 0-scale range."""
    if not amplitudes:
        return []

    max_amp = max(amplitudes)
    if max_amp == 0:
        return [0.0] * len(amplitudes)

    return [a / max_amp * scale for a in amplitudes]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string with alpha."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'


def process_audio_waveform(
    audio_data: np.ndarray,
    total_frames: int,
    marker_color: str
) -> WaveformData:
    """Process audio data into waveform visualization data."""
    frame_amplitudes = _extract_frame_amplitudes(audio_data, total_frames)
    normalized_amps = _normalize_amplitudes(frame_amplitudes)
    frame_x = list(range(total_frames))
    color_rgba = _hex_to_rgba(marker_color, 0.2)

    return WaveformData(
        frame_x=frame_x,
        normalized_amps=normalized_amps,
        color_rgba=color_rgba
    )


# ============================================================================
# Keyframe Processing
# ============================================================================

def _extract_keyframe_data(keyframes: List[Dict]) -> Tuple[List[int], List[float]]:
    """Extract frame numbers and intensities from keyframes."""
    frame_numbers = [kf['frame'] for kf in keyframes]
    intensities = [kf.get('intensity', 1.0) for kf in keyframes]
    return frame_numbers, intensities


def _normalize_intensities(intensities: List[float]) -> List[float]:
    """Normalize intensities to 0-1 range."""
    if not intensities:
        return []

    max_intensity = max(intensities)
    min_intensity = min(intensities)
    intensity_range = max_intensity - min_intensity

    if intensity_range == 0:
        return [1.0] * len(intensities)

    return [(i - min_intensity) / intensity_range for i in intensities]


def _calculate_marker_sizes(
    normalized_intensities: List[float],
    min_size: float = 6.0,
    size_range: float = 10.0
) -> List[float]:
    """Calculate marker sizes based on normalized intensities."""
    return [min_size + (ni * size_range) for ni in normalized_intensities]


def _calculate_timestamps(frame_numbers: List[int], fps: int) -> List[float]:
    """Convert frame numbers to timestamps in seconds."""
    return [f / fps for f in frame_numbers]


def _build_hover_text_with_prompts(
    frame_numbers: List[int],
    timestamps: List[float],
    intensities: List[float],
    prompts: List[str]
) -> List[str]:
    """Build hover text including prompt information."""
    return [
        f"<b>Frame {f}</b><br>Time: {t:.2f}s<br>Intensity: {i:.2f}<br>Prompt: {p[:50]}{'...' if len(p) > 50 else ''}"
        for f, t, i, p in zip(frame_numbers, timestamps, intensities, prompts)
    ]


def _build_hover_text_without_prompts(
    frame_numbers: List[int],
    timestamps: List[float],
    intensities: List[float]
) -> List[str]:
    """Build hover text without prompt information."""
    return [
        f"<b>Frame {f}</b><br>Time: {t:.2f}s<br>Intensity: {i:.2f}"
        for f, t, i in zip(frame_numbers, timestamps, intensities)
    ]


def _build_hover_texts(
    frame_numbers: List[int],
    timestamps: List[float],
    intensities: List[float],
    prompts: Optional[List[str]]
) -> List[str]:
    """Build hover texts based on available data."""
    if prompts and len(prompts) == len(frame_numbers):
        return _build_hover_text_with_prompts(
            frame_numbers, timestamps, intensities, prompts
        )

    return _build_hover_text_without_prompts(
        frame_numbers, timestamps, intensities
    )


def process_keyframe_metrics(
    keyframes: List[Dict],
    fps: int,
    prompts: Optional[List[str]] = None
) -> KeyframeMetrics:
    """Process keyframe data into visualization metrics."""
    frame_numbers, intensities = _extract_keyframe_data(keyframes)
    normalized_intensities = _normalize_intensities(intensities)
    marker_sizes = _calculate_marker_sizes(normalized_intensities)
    timestamps = _calculate_timestamps(frame_numbers, fps)
    hover_texts = _build_hover_texts(frame_numbers, timestamps, intensities, prompts)

    return KeyframeMetrics(
        frame_numbers=frame_numbers,
        intensities=intensities,
        normalized_intensities=normalized_intensities,
        marker_sizes=marker_sizes,
        timestamps=timestamps,
        hover_texts=hover_texts
    )


# ============================================================================
# Plotly Trace Creation
# ============================================================================

def _create_waveform_trace(waveform: WaveformData):
    """Create waveform background trace."""
    import plotly.graph_objects as go

    return go.Scatter(
        x=waveform.frame_x,
        y=waveform.normalized_amps,
        mode='lines',
        fill='tozeroy',
        line=dict(color=waveform.color_rgba, width=1),
        fillcolor=waveform.color_rgba,
        hoverinfo='skip',
        showlegend=False,
        name='Audio Waveform'
    )


def _create_keyframe_line_traces(
    metrics: KeyframeMetrics,
    marker_color: str
) -> List:
    """Create vertical line traces for each keyframe."""
    import plotly.graph_objects as go

    return [
        go.Scatter(
            x=[frame, frame],
            y=[0, 1],
            mode='lines',
            line=dict(color=marker_color, width=2.5),
            hovertemplate=f'{hover}<extra></extra>',
            showlegend=False,
            opacity=0.8
        )
        for frame, hover in zip(metrics.frame_numbers, metrics.hover_texts)
    ]


def _create_keyframe_dot_trace(metrics: KeyframeMetrics, colors: ThemeColors):
    """Create marker dots at top of keyframe lines."""
    import plotly.graph_objects as go

    return go.Scatter(
        x=metrics.frame_numbers,
        y=[0.95] * len(metrics.frame_numbers),
        mode='markers',
        marker=dict(
            size=[8 + ni * 4 for ni in metrics.normalized_intensities],
            color=colors.marker,
            symbol='circle',
            line=dict(color=colors.line, width=1.5),
            opacity=0.9
        ),
        hovertemplate='%{text}<extra></extra>',
        text=metrics.hover_texts,
        showlegend=False
    )


# ============================================================================
# Layout Configuration
# ============================================================================

def _create_xaxis_config(total_frames: int, grid_color: str) -> dict:
    """Create x-axis configuration."""
    return dict(
        title=dict(text='Frame Number', font=dict(size=11)),
        range=[0, total_frames],
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False,
        tickfont=dict(size=9),
        fixedrange=True
    )


def _create_yaxis_config() -> dict:
    """Create y-axis configuration."""
    return dict(
        title=None,
        range=[0, 1],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        fixedrange=True
    )


def create_timeline_layout(total_frames: int, colors: ThemeColors) -> dict:
    """Create complete Plotly layout configuration."""
    return dict(
        paper_bgcolor=colors.background,
        plot_bgcolor=colors.plot_bg,
        font=dict(
            color=colors.text,
            family='system-ui, -apple-system, sans-serif',
            size=10
        ),
        xaxis=_create_xaxis_config(total_frames, colors.grid),
        yaxis=_create_yaxis_config(),
        margin=dict(l=0, r=0, t=5, b=35),
        height=280,
        hovermode='closest',
        showlegend=False,
        dragmode=False
    )


# ============================================================================
# Main Orchestrator
# ============================================================================

def create_keyframe_timeline_plot(
    keyframes: List[Dict],
    total_frames: int,
    duration: float,
    fps: int,
    prompts: List[str] = None,
    audio_data=None,
    sample_rate: int = None
):
    """Create interactive Plotly timeline visualization with audio waveform.

    Refactored version with complexity ≤10 and modular structure.

    Args:
        keyframes: List of keyframe dicts with 'frame' and optional 'intensity'
        total_frames: Total number of frames in animation
        duration: Audio duration in seconds
        fps: Frames per second
        prompts: Optional list of prompts corresponding to keyframes
        audio_data: Optional numpy array of audio samples for waveform display
        sample_rate: Audio sample rate (required if audio_data provided)

    Returns:
        Plotly Figure object with theme-aware timeline visualization

    Raises:
        ImportError: If plotly is not installed
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for timeline visualization. "
            "Install with: pip install plotly"
        )

    # Get theme colors
    colors = get_theme_colors()

    # Create figure
    fig = go.Figure()

    # Add audio waveform if provided
    if audio_data is not None and sample_rate is not None:
        waveform = process_audio_waveform(audio_data, total_frames, colors.marker)
        fig.add_trace(_create_waveform_trace(waveform))

    # Process keyframe metrics
    metrics = process_keyframe_metrics(keyframes, fps, prompts)

    # Add keyframe visualizations
    for trace in _create_keyframe_line_traces(metrics, colors.marker):
        fig.add_trace(trace)

    fig.add_trace(_create_keyframe_dot_trace(metrics, colors))

    # Apply layout
    layout = create_timeline_layout(total_frames, colors)
    fig.update_layout(layout)

    return fig
