"""Frame Overlap Visualizer - Generate worm trail visualization for camera movements.

This module creates interactive Plotly visualizations showing:
- Worm trail effect: Previous frame positions with fading opacity
- Preservation/novelty metrics displayed per frame
- Color-coded frames based on metrics (green = good, yellow = warning, red = problem)
- Playable timeline to scrub through the animation
"""

from typing import List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from deforum.utils.frame_overlap_simulator import (
    FrameMetrics,
    MIN_PRESERVATION_THRESHOLD,
    MAX_NOVELTY_THRESHOLD,
    DEFAULT_TRAIL_LENGTH,
)


# Color constants
COLOR_GOOD = 'rgb(50, 205, 50)'  # Green
COLOR_WARNING = 'rgb(255, 215, 0)'  # Gold
COLOR_PROBLEM = 'rgb(255, 69, 0)'  # Red-orange
COLOR_VIEWPORT = 'rgb(100, 100, 255)'  # Blue


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

    # Linear fade from 1.0 to 0.1
    min_opacity = 0.1
    fade = (max_trail_length - age) / max_trail_length
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

    return go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='lines',
        line=dict(color=color, width=2),
        fill='toself',
        fillcolor=color.replace('rgb', 'rgba').replace(')', f', {opacity * 0.2})'),
        opacity=opacity,
        name=name,
        showlegend=show_legend,
        hoverinfo='name'
    )


def create_worm_trail_frame(
    metrics_list: List[FrameMetrics],
    frame_index: int,
    trail_length: int = DEFAULT_TRAIL_LENGTH
) -> List[go.Scatter]:
    """Create worm trail traces for a single frame.

    Args:
        metrics_list: Complete list of frame metrics
        frame_index: Current frame index to visualize
        trail_length: Number of previous frames to show in trail

    Returns:
        List of Plotly traces for this frame
    """
    traces: List[go.Scatter] = []

    # Current viewport (always shown in blue)
    current_metrics = metrics_list[frame_index]
    viewport_corners = current_metrics.curr_viewport_rect.get_corners()
    traces.append(create_rectangle_trace(
        rect_corners=viewport_corners,
        color=COLOR_VIEWPORT,
        opacity=1.0,
        name='Current Viewport',
        show_legend=True
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
        label = f'Frame {i}' if age == 0 else f'Frame {i} (-{age})'
        traces.append(create_rectangle_trace(
            rect_corners=prev_corners,
            color=frame_color,
            opacity=opacity,
            name=label,
            show_legend=(age == 0)  # Only show current frame in legend
        ))

    return traces


def create_metrics_annotation(
    metrics: FrameMetrics,
    x_pos: float,
    y_pos: float
) -> dict:
    """Create annotation showing preservation/novelty metrics.

    Args:
        metrics: Frame metrics to display
        x_pos: X position for annotation
        y_pos: Y position for annotation

    Returns:
        Plotly annotation dict
    """
    text = (
        f"Frame {metrics.frame_index}<br>"
        f"<b>Preservation:</b> {metrics.preservation * 100:.1f}%<br>"
        f"<b>Novelty:</b> {metrics.novelty * 100:.1f}%<br>"
        f"<b>Overlap:</b> {metrics.overlap_area:.0f} px²"
    )

    return dict(
        x=x_pos,
        y=y_pos,
        xref='paper',
        yref='paper',
        text=text,
        showarrow=False,
        align='left',
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor=get_frame_color(metrics),
        borderwidth=2,
        font=dict(size=12)
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
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)"
        )
        return fig

    # Calculate viewport bounds for consistent axis ranges
    viewport_width = metrics_list[0].curr_viewport_rect.width
    viewport_height = metrics_list[0].curr_viewport_rect.height

    # Add some padding
    padding_factor = 1.5
    x_range = [-viewport_width * padding_factor / 2, viewport_width * padding_factor / 2]
    y_range = [-viewport_height * padding_factor / 2, viewport_height * padding_factor / 2]

    # Create frames for animation
    frames = []
    for frame_idx in range(len(metrics_list)):
        traces = create_worm_trail_frame(metrics_list, frame_idx, trail_length)

        # Create annotation for this frame
        annotation = create_metrics_annotation(
            metrics_list[frame_idx],
            x_pos=0.02,
            y_pos=0.98
        )

        frames.append(go.Frame(
            data=traces,
            name=str(frame_idx),
            layout=dict(annotations=[annotation])
        ))

    # Create initial frame (frame 0)
    initial_traces = create_worm_trail_frame(metrics_list, 0, trail_length)
    initial_annotation = create_metrics_annotation(metrics_list[0], x_pos=0.02, y_pos=0.98)

    # Create figure with initial frame
    fig = go.Figure(data=initial_traces, frames=frames)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Frame Overlap Simulator - {len(metrics_list)} frames",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="X (pixels)",
            range=x_range,
            scaleanchor='y',
            scaleratio=1,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray'
        ),
        yaxis=dict(
            title="Y (pixels)",
            range=y_range,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray'
        ),
        width=width,
        height=height,
        hovermode='closest',
        annotations=[initial_annotation],
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=1000 / playback_fps, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ],
                x=0.1,
                y=0,
                xanchor='left',
                yanchor='top'
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
                        label=f"Frame {i}",
                        method='animate'
                    )
                    for i, f in enumerate(frames)
                ],
                x=0.1,
                y=0,
                xanchor='left',
                yanchor='top',
                len=0.8,
                currentvalue=dict(
                    prefix='Frame: ',
                    visible=True,
                    xanchor='left'
                )
            )
        ]
    )

    return fig


def create_metrics_timeline_chart(
    metrics_list: List[FrameMetrics],
    width: int = 800,
    height: int = 300
) -> go.Figure:
    """Create timeline chart showing preservation/novelty over time.

    Args:
        metrics_list: List of frame metrics from simulator
        width: Figure width in pixels
        height: Figure height in pixels

    Returns:
        Plotly Figure with timeline chart
    """
    if not metrics_list:
        fig = go.Figure()
        fig.update_layout(title="No metrics to display")
        return fig

    frame_indices = [m.frame_index for m in metrics_list]
    preservations = [m.preservation * 100 for m in metrics_list]
    novelties = [m.novelty * 100 for m in metrics_list]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Preservation %', 'Novelty %'),
        vertical_spacing=0.15
    )

    # Preservation line
    fig.add_trace(
        go.Scatter(
            x=frame_indices,
            y=preservations,
            mode='lines',
            name='Preservation',
            line=dict(color=COLOR_GOOD, width=2),
            hovertemplate='Frame %{x}<br>Preservation: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )

    # Preservation threshold line
    fig.add_hline(
        y=MIN_PRESERVATION_THRESHOLD * 100,
        line_dash='dash',
        line_color=COLOR_PROBLEM,
        annotation_text=f'Min ({MIN_PRESERVATION_THRESHOLD * 100:.0f}%)',
        row=1, col=1
    )

    # Novelty line
    fig.add_trace(
        go.Scatter(
            x=frame_indices,
            y=novelties,
            mode='lines',
            name='Novelty',
            line=dict(color=COLOR_WARNING, width=2),
            hovertemplate='Frame %{x}<br>Novelty: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )

    # Novelty threshold line
    fig.add_hline(
        y=MAX_NOVELTY_THRESHOLD * 100,
        line_dash='dash',
        line_color=COLOR_PROBLEM,
        annotation_text=f'Max ({MAX_NOVELTY_THRESHOLD * 100:.0f}%)',
        row=2, col=1
    )

    fig.update_xaxes(title_text='Frame Index', row=2, col=1)
    fig.update_yaxes(title_text='%', range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text='%', range=[0, 100], row=2, col=1)

    fig.update_layout(
        width=width,
        height=height,
        showlegend=False,
        title=dict(
            text='Preservation/Novelty Timeline',
            x=0.5,
            xanchor='center'
        )
    )

    return fig
