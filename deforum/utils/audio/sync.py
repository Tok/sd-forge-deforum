"""Pure business logic for audio synchronization.

This module contains pure functions extracted from ui_left.py to improve
testability and reduce complexity. All functions here are side-effect free.

Following Phase 1 of REFACTORING_STRATEGY.md:
- Pure functions only (no I/O, no state mutation)
- Full type hints
- Complexity ≤ 10
- Docstrings only where needed
"""

from typing import Tuple, List, Dict


# ============================================================================
# PURE FUNCTIONS - BPM and Keyframe Calculations
# ============================================================================


def calculate_keyframes_per_beat(bpm: float) -> float:
    """Calculate optimal keyframes per beat based on tempo.

    Slow music (60-90 BPM): 1 keyframe per beat
    Medium music (90-140 BPM): 1 keyframe per 2 beats
    Fast music (140+ BPM): 1 keyframe per 4 beats
    """
    if bpm < 90:
        return 1.0
    elif bpm < 140:
        return 0.5
    else:
        return 0.25


def calculate_bpm_based_target(duration: float, bpm: float, keyframes_per_beat: float) -> int:
    """Calculate target keyframe count from audio duration and BPM."""
    beats_per_second = bpm / 60.0
    return int(duration * beats_per_second * keyframes_per_beat)


def apply_keyframe_adjustment(base_target: int, adjustment_percent: int) -> int:
    """Apply percentage adjustment to keyframe count.

    Args:
        base_target: Original target count
        adjustment_percent: Percentage to adjust (can be negative)

    Returns:
        Adjusted target, minimum 2 keyframes

    Note:
        For small targets, percentage adjustment may round to 0 change.
        We ensure at least ±1 keyframe change when adjustment is non-zero.
    """
    if adjustment_percent == 0:
        return base_target

    # Calculate percentage-based adjustment
    adjusted = int(base_target * (1.0 + adjustment_percent / 100.0))

    # Ensure at least ±1 change when adjustment is requested
    if adjustment_percent > 0 and adjusted <= base_target:
        adjusted = base_target + 1
    elif adjustment_percent < 0 and adjusted >= base_target:
        adjusted = base_target - 1

    return max(2, adjusted)


def calculate_spacing_multiplier(adjustment_percent: int) -> float:
    """Calculate min_spacing multiplier from keyframe adjustment.

    +5% keyframes → -5% min_spacing (0.95 multiplier)
    -5% keyframes → +5% min_spacing (1.05 multiplier)
    """
    return 1.0 - (adjustment_percent / 100.0)


def calculate_adjusted_min_spacing(base_spacing: int, multiplier: float) -> int:
    """Apply spacing multiplier with minimum of 1 frame."""
    return max(1, int(base_spacing * multiplier))


def calculate_compensation_target(target: int, compensation_factor: float = 1.25) -> int:
    """Add compensation for min_spacing filtering.

    Default 1.25x accounts for ~20% of events being filtered out.
    """
    return int(target * compensation_factor)


def build_keyframe_visualization(
    keyframes: list[dict], total_frames: int, viz_width: int = 120
) -> Tuple[str, str]:
    """Build ASCII visualization of keyframe placement.

    Returns:
        Tuple of (viz_string, spacing_string) for display

    Example:
        ("[|__|_____|___|_|____|___]", "0                      331")
    """
    viz = ["_"] * viz_width
    for kf in keyframes:
        pos = int((kf["frame"] / total_frames) * (viz_width - 1))
        viz[pos] = "|"

    viz_str = "".join(viz)
    max_frame = total_frames - 1
    spacing_str = f"0{' ' * (viz_width - len(str(max_frame)) - 1)}{max_frame}"

    return viz_str, spacing_str


def create_keyframe_timeline_plot(
    keyframes: List[Dict],
    total_frames: int,
    duration: float,
    fps: int,
    prompts: List[str] = None
):
    """Create interactive Plotly timeline visualization of keyframe placement.

    Args:
        keyframes: List of keyframe dicts with 'frame' and optional 'intensity'
        total_frames: Total number of frames in animation
        duration: Audio duration in seconds
        fps: Frames per second
        prompts: Optional list of prompts corresponding to keyframes

    Returns:
        Plotly Figure object with theme-aware timeline visualization

    Raises:
        ImportError: If plotly is not installed
    """
    # Lazy import plotly (optional dependency for visualization)
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for timeline visualization. "
            "Install with: pip install plotly"
        )

    # Get theme for color selection
    try:
        from deforum.rendering.options import get_log_theme
        theme = get_log_theme()
    except:
        theme = "slopcore"  # Fallback

    # Theme-aware colors (matching schedule_visualizer.py)
    if theme == "slopcore":
        marker_color = '#667EEA'  # Purple (slopcore gradient start)
        line_color = '#A353A8'    # Deep pink/purple (SLOPCORE_6)
        bg_color = '#0F172A'      # Tailwind slate-900
        plot_bg = '#1E293B'       # Tailwind slate-800
        grid_color = '#334155'    # Tailwind slate-700
        text_color = '#CBD5E1'    # Tailwind slate-300
    else:  # classic
        marker_color = '#3B82F6'  # Bright blue
        line_color = '#10B981'    # Green
        bg_color = '#0F172A'
        plot_bg = '#1E293B'
        grid_color = '#334155'
        text_color = '#CBD5E1'

    # Create figure
    fig = go.Figure()

    # Extract frame numbers and intensities
    frame_numbers = [kf['frame'] for kf in keyframes]
    intensities = [kf.get('intensity', 1.0) for kf in keyframes]

    # Normalize intensities to 0-1 range for marker sizing
    if intensities:
        max_intensity = max(intensities)
        min_intensity = min(intensities)
        intensity_range = max_intensity - min_intensity
        if intensity_range > 0:
            normalized_intensities = [
                (i - min_intensity) / intensity_range for i in intensities
            ]
        else:
            normalized_intensities = [1.0] * len(intensities)
    else:
        normalized_intensities = [1.0] * len(frame_numbers)

    # Create marker sizes (6-16 based on intensity)
    marker_sizes = [6 + (ni * 10) for ni in normalized_intensities]

    # Timestamps for hover
    timestamps = [f / fps for f in frame_numbers]

    # Build hover text
    if prompts and len(prompts) == len(keyframes):
        hover_texts = [
            f"<b>Frame {f}</b><br>Time: {t:.2f}s<br>Intensity: {i:.2f}<br>Prompt: {p[:50]}{'...' if len(p) > 50 else ''}"
            for f, t, i, p in zip(frame_numbers, timestamps, intensities, prompts)
        ]
    else:
        hover_texts = [
            f"<b>Frame {f}</b><br>Time: {t:.2f}s<br>Intensity: {i:.2f}"
            for f, t, i in zip(frame_numbers, timestamps, intensities)
        ]

    # Add timeline markers as scatter plot (Y=0.5 for centered positioning)
    fig.add_trace(go.Scatter(
        x=frame_numbers,
        y=[0.5] * len(frame_numbers),
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=marker_color,
            symbol='line-ns-open',  # Vertical line symbol
            line=dict(color=line_color, width=2),
            opacity=0.9
        ),
        hovertemplate='%{text}<extra></extra>',
        text=hover_texts,
        showlegend=False
    ))

    # Add horizontal reference line (timeline base)
    fig.add_trace(go.Scatter(
        x=[0, total_frames],
        y=[0.5, 0.5],
        mode='lines',
        line=dict(color=grid_color, width=1, dash='dot'),
        hoverinfo='skip',
        showlegend=False
    ))

    # Update layout for compact timeline appearance
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color, family='system-ui, -apple-system, sans-serif', size=10),
        xaxis=dict(
            title=dict(text='Frame Number', font=dict(size=11)),
            range=[0, total_frames],
            showgrid=True,
            gridcolor=grid_color,
            zeroline=False,
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            title=None,
            range=[0, 1],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        margin=dict(l=40, r=20, t=10, b=40),
        height=200,  # Compact timeline height
        hovermode='closest',
        showlegend=False
    )

    return fig


def build_status_message(
    duration: float,
    fps: int,
    total_frames: int,
    bpm: float,
    events_detected: int,
    keyframes_created: int,
    prompts_used: int,
    distribution_mode: str,
    viz_str: str,
    spacing_str: str,
) -> str:
    """Build detailed status message for UI display."""
    avg_spacing = total_frames / keyframes_created if keyframes_created else 0

    return (
        f"✓ Successfully synchronized!\n"
        f"• Audio: {duration:.1f}s @ {fps} FPS ({total_frames} frames)\n"
        f"• Detected BPM: {bpm:.1f}\n"
        f"• Events detected: {events_detected}\n"
        f"• Keyframes created: {keyframes_created}\n"
        f"• Average spacing: {avg_spacing:.1f} frames (~{avg_spacing/fps:.2f}s)\n"
        f"• Prompts used: {prompts_used} (mode: {distribution_mode})\n\n"
        f"Keyframe placement:\n[{viz_str}]\n"
        f"{spacing_str}"
    )


# ============================================================================
# PURE FUNCTIONS - Keyframe Target Resolution
# ============================================================================


def resolve_keyframe_target(
    user_target: int, bpm_based_target: int, keyframe_adjustment: int
) -> Tuple[int, str]:
    """Resolve final keyframe target from user input, BPM, and adjustment.

    Returns:
        Tuple of (final_target, description)
    """
    # Convert to int if string (defensive)
    if isinstance(user_target, str):
        user_target = int(user_target) if user_target else 0
    if isinstance(bpm_based_target, str):
        bpm_based_target = int(bpm_based_target) if bpm_based_target else 0
    if isinstance(keyframe_adjustment, str):
        keyframe_adjustment = int(keyframe_adjustment) if keyframe_adjustment else 0

    if user_target and user_target > 0:
        # User specified target
        if keyframe_adjustment != 0:
            final = apply_keyframe_adjustment(user_target, keyframe_adjustment)
            desc = f"{user_target} → {final} ({keyframe_adjustment:+d}%)"
        else:
            final = user_target
            desc = f"{final} (user-specified)"
        return final, desc
    else:
        # BPM-based target
        if keyframe_adjustment != 0:
            final = apply_keyframe_adjustment(bpm_based_target, keyframe_adjustment)
            desc = f"{bpm_based_target} → {final} ({keyframe_adjustment:+d}%)"
        else:
            final = bpm_based_target
            desc = f"{final} (BPM-based)"
        return final, desc
