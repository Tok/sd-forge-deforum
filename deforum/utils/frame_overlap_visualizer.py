"""Frame Overlap Visualizer - Generate worm trail visualization for camera movements.

This module creates Plotly visualizations showing:
- Worm trail effect: Previous frame positions with fading opacity
- Preservation/novelty metrics displayed per frame
- Color-coded frames based on metrics (purple theme matching Deforum UI)
- Playable timeline with speed control
"""

from typing import List, Optional
import numpy as np
import plotly.graph_objects as go

from deforum.utils.frame_overlap_simulator import (
    FrameMetrics,
    MIN_PRESERVATION_THRESHOLD,
    MAX_NOVELTY_THRESHOLD,
    DEFAULT_TRAIL_LENGTH,
)


# Deforum purple/slopcore dark theme colors
COLOR_GOOD = 'rgb(150, 100, 255)'  # Purple - good preservation
COLOR_WARNING = 'rgb(255, 150, 100)'  # Orange - warning
COLOR_PROBLEM = 'rgb(255, 80, 80)'  # Red - problem
COLOR_VIEWPORT = 'rgb(180, 140, 255)'  # Light purple - current viewport
COLOR_BG = 'rgb(20, 20, 30)'  # Dark background
COLOR_GRID = 'rgb(60, 60, 80)'  # Dark purple grid
COLOR_TEXT = 'rgb(200, 200, 220)'  # Light text


def get_frame_color(metrics: FrameMetrics) -> str:
    """Get color for frame based on preservation/novelty metrics.

    Args:
        metrics: Frame metrics to evaluate

    Returns:
        RGB color string
    """
    if metrics.preservation < MIN_PRESERVATION_THRESHOLD:
        return COLOR_PROBLEM
    elif metrics.novelty > MAX_NOVELTY_THRESHOLD:
        return COLOR_WARNING
    else:
        return COLOR_GOOD


def calculate_opacity(age: int, max_trail_length: int) -> float:
    """Calculate opacity for a frame in the trail based on its age.

    Args:
        age: How many frames ago this frame was (0 = current, 1 = previous, etc.)
        max_trail_length: Maximum number of frames in trail

    Returns:
        Opacity value between 0.1 and 1.0
    """
    if age == 0:
        return 1.0  # Current frame fully visible

    # Exponential fade for smoother worm effect
    min_opacity = 0.1
    fade = ((max_trail_length - age) / max_trail_length) ** 2  # Quadratic fade
    return min_opacity + (1.0 - min_opacity) * fade


def create_rectangle_trace(
    rect_corners: np.ndarray,
    color: str,
    opacity: float,
    name: str,
    show_legend: bool = False
) -> go.Scatter:
    """Create a Plotly scatter trace for a rectangle.

    Args:
        rect_corners: 4x2 array of rectangle corners
        color: RGB color string
        opacity: Opacity value 0.0-1.0
        name: Trace name for legend
        show_legend: Whether to show in legend

    Returns:
        Plotly Scatter trace
    """
    # Close the rectangle by adding first point at end
    x_coords = list(rect_corners[:, 0]) + [rect_corners[0, 0]]
    y_coords = list(rect_corners[:, 1]) + [rect_corners[0, 1]]

    # Convert RGB to RGBA for fill
    rgba_fill = color.replace('rgb', 'rgba').replace(')', f', {opacity * 0.15})')

    return go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='lines',
        line=dict(color=color, width=2),
        fill='toself',
        fillcolor=rgba_fill,
        opacity=opacity,
        name=name,
        showlegend=show_legend,
        hoverinfo='skip'  # Disable hover to reduce clutter
    )


def create_worm_trail_frame(
    metrics_list: List[FrameMetrics],
    frame_index: int,
    trail_length: int = DEFAULT_TRAIL_LENGTH
) -> tuple[List[go.Scatter], tuple[float, float]]:
    """Create worm trail traces for a single frame.

    Args:
        metrics_list: Complete list of frame metrics
        frame_index: Current frame index to visualize
        trail_length: Number of previous frames to show in trail

    Returns:
        Tuple of (list of Plotly traces, (center_x, center_y) for viewport centering)
    """
    traces: List[go.Scatter] = []

    # Get current frame position for centering
    current_metrics = metrics_list[frame_index]
    center_x = current_metrics.prev_frame_rect.center_x
    center_y = current_metrics.prev_frame_rect.center_y

    # Current viewport (always shown in light purple, fixed at origin)
    viewport_corners = current_metrics.curr_viewport_rect.get_corners()
    # Translate viewport to follow current frame
    viewport_corners_translated = viewport_corners + np.array([center_x, center_y])

    traces.append(create_rectangle_trace(
        rect_corners=viewport_corners_translated,
        color=COLOR_VIEWPORT,
        opacity=0.8,
        name='Current Viewport',
        show_legend=False
    ))

    # Previous frames in trail (worm effect)
    trail_start = max(0, frame_index - trail_length + 1)
    for i in range(trail_start, frame_index + 1):
        age = frame_index - i
        metrics = metrics_list[i]

        # Get color based on metrics
        frame_color = get_frame_color(metrics)

        # Calculate opacity based on age
        opacity = calculate_opacity(age, trail_length)

        # Get previous frame rectangle
        prev_corners = metrics.prev_frame_rect.get_corners()

        # Create trace
        traces.append(create_rectangle_trace(
            rect_corners=prev_corners,
            color=frame_color,
            opacity=opacity,
            name=f'Frame {i}',
            show_legend=False
        ))

    return traces, (center_x, center_y)


def create_metrics_annotation(
    metrics: FrameMetrics,
    x_pos: float,
    y_pos: float
) -> dict:
    """Create annotation showing preservation/novelty metrics.

    Args:
        metrics: Frame metrics to display
        x_pos: X position for annotation (paper coordinates)
        y_pos: Y position for annotation (paper coordinates)

    Returns:
        Plotly annotation dict
    """
    text = (
        f"<b>Frame {metrics.frame_index}</b><br>"
        f"Preservation: {metrics.preservation * 100:.1f}%<br>"
        f"Novelty: {metrics.novelty * 100:.1f}%"
    )

    return dict(
        x=x_pos,
        y=y_pos,
        xref='paper',
        yref='paper',
        text=text,
        showarrow=False,
        align='left',
        bgcolor='rgba(40, 40, 60, 0.9)',
        bordercolor=get_frame_color(metrics),
        borderwidth=2,
        font=dict(size=11, color=COLOR_TEXT)
    )


def create_worm_trail_visualization(
    metrics_list: List[FrameMetrics],
    width: int = 800,
    height: int = 600,
    trail_length: int = DEFAULT_TRAIL_LENGTH,
    playback_fps: int = 10
) -> go.Figure:
    """Create interactive worm trail visualization with playback timeline.

    Args:
        metrics_list: List of frame metrics from simulator
        width: Figure width in pixels
        height: Figure height in pixels
        trail_length: Number of frames to show in trail
        playback_fps: Frames per second for auto-play animation

    Returns:
        Plotly Figure with animation frames
    """
    if not metrics_list:
        # Empty figure if no metrics
        fig = go.Figure()
        fig.update_layout(
            title="No metrics to display",
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_TEXT)
        )
        return fig

    # Calculate viewport bounds for consistent axis ranges
    viewport_width = metrics_list[0].curr_viewport_rect.width
    viewport_height = metrics_list[0].curr_viewport_rect.height

    # Fixed range centered on viewport (won't pan - always shows same area)
    padding_factor = 1.8
    x_range = [-viewport_width * padding_factor / 2, viewport_width * padding_factor / 2]
    y_range = [-viewport_height * padding_factor / 2, viewport_height * padding_factor / 2]

    # Create frames for animation
    frames = []
    for frame_idx in range(len(metrics_list)):
        traces, (cx, cy) = create_worm_trail_frame(metrics_list, frame_idx, trail_length)

        # Create annotation for this frame
        annotation = create_metrics_annotation(
            metrics_list[frame_idx],
            x_pos=0.02,
            y_pos=0.98
        )

        frames.append(go.Frame(
            data=traces,
            name=str(frame_idx),
            layout=dict(
                annotations=[annotation],
                # Update axis ranges to follow current frame
                xaxis=dict(range=[cx + x_range[0], cx + x_range[1]]),
                yaxis=dict(range=[cy + y_range[0], cy + y_range[1]])
            )
        ))

    # Create initial frame (frame 0)
    initial_traces, (cx0, cy0) = create_worm_trail_frame(metrics_list, 0, trail_length)
    initial_annotation = create_metrics_annotation(metrics_list[0], x_pos=0.02, y_pos=0.98)

    # Create figure with initial frame
    fig = go.Figure(data=initial_traces, frames=frames)

    # Update layout with Deforum theme
    fig.update_layout(
        title=dict(
            text=f"Frame Overlap Simulator - {len(metrics_list)} frames",
            x=0.5,
            xanchor='center',
            font=dict(size=14, color=COLOR_TEXT)
        ),
        xaxis=dict(
            title="X (pixels)",
            range=[cx0 + x_range[0], cx0 + x_range[1]],
            scaleanchor='y',
            scaleratio=1,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=COLOR_GRID,
            gridcolor=COLOR_GRID,
            showgrid=True,
            color=COLOR_TEXT
        ),
        yaxis=dict(
            title="Y (pixels)",
            range=[cy0 + y_range[0], cy0 + y_range[1]],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=COLOR_GRID,
            gridcolor=COLOR_GRID,
            showgrid=True,
            color=COLOR_TEXT
        ),
        width=width,
        height=height,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        hovermode=False,  # Disable hover
        dragmode=False,  # Disable drag/zoom
        annotations=[initial_annotation],
        # Simplified controls - just play/pause
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': int(1000 / playback_fps), 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0},
                            'mode': 'immediate'
                        }]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ],
                x=0.05,
                y=0.05,
                xanchor='left',
                yanchor='bottom',
                bgcolor='rgba(60, 60, 80, 0.8)',
                bordercolor=COLOR_VIEWPORT,
                borderwidth=1,
                font=dict(color=COLOR_TEXT, size=12)
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        args=[[f.name], dict(
                            frame=dict(duration=0, redraw=True),
                            mode='immediate',
                            transition=dict(duration=0)
                        )],
                        label=str(i),
                        method='animate'
                    )
                    for i, f in enumerate(frames)
                ],
                x=0.05,
                y=0.0,
                xanchor='left',
                yanchor='bottom',
                len=0.9,
                currentvalue=dict(
                    prefix='Frame: ',
                    visible=True,
                    xanchor='left',
                    font=dict(color=COLOR_TEXT, size=12)
                ),
                bgcolor='rgba(60, 60, 80, 0.6)',
                bordercolor=COLOR_VIEWPORT,
                borderwidth=1,
                tickcolor=COLOR_TEXT,
                font=dict(color=COLOR_TEXT)
            )
        ]
    )

    # Remove toolbar (zoom, pan, etc.)
    fig.update_layout(
        modebar=dict(
            remove=['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']
        )
    )

    return fig
