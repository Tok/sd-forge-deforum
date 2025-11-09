"""Canvas-based frame overlap visualizer for Gradio HTML component.

This module generates HTML5 Canvas visualizations with JavaScript animation.
Much more performant than Plotly and works reliably in Gradio.
"""

import json
from typing import List, Dict, Any
import numpy as np

from deforum.utils.frame_overlap_simulator import (
    FrameMetrics,
    MIN_PRESERVATION_THRESHOLD,
    MAX_NOVELTY_THRESHOLD,
    DEFAULT_TRAIL_LENGTH,
)


# Deforum purple/slopcore dark theme colors (hex for Canvas)
COLOR_GOOD = '#9664FF'  # Purple - good preservation
COLOR_WARNING = '#FF9664'  # Orange - warning
COLOR_PROBLEM = '#FF5050'  # Red - problem
COLOR_VIEWPORT = '#B48CFF'  # Light purple - current viewport
COLOR_BG = '#14141E'  # Dark background
COLOR_GRID = '#3C3C50'  # Dark purple grid
COLOR_TEXT = '#C8C8DC'  # Light text


def get_frame_color(metrics: FrameMetrics) -> str:
    """Get hex color for frame based on preservation/novelty metrics.

    Args:
        metrics: Frame metrics to evaluate

    Returns:
        Hex color string
    """
    if metrics.preservation < MIN_PRESERVATION_THRESHOLD:
        return COLOR_PROBLEM
    elif metrics.novelty > MAX_NOVELTY_THRESHOLD:
        return COLOR_WARNING
    else:
        return COLOR_GOOD


def serialize_frame_data(metrics_list: List[FrameMetrics], trail_length: int) -> List[Dict[str, Any]]:
    """Serialize frame metrics to JSON-compatible format.

    Args:
        metrics_list: List of frame metrics
        trail_length: Number of frames to show in trail

    Returns:
        List of frame data dicts
    """
    frames_data = []

    for frame_idx, metrics in enumerate(metrics_list):
        # Get trail frames for this frame
        trail_start = max(0, frame_idx - trail_length + 1)
        trail_frames = []

        for i in range(trail_start, frame_idx + 1):
            age = frame_idx - i
            trail_metrics = metrics_list[i]

            # Get rectangle corners
            corners = trail_metrics.prev_frame_rect.get_corners()

            trail_frames.append({
                'corners': corners.tolist(),
                'color': get_frame_color(trail_metrics),
                'age': age,
                'frameIndex': i
            })

        # Current viewport corners (translated to follow frame)
        viewport_corners = metrics.curr_viewport_rect.get_corners()
        center_x = metrics.prev_frame_rect.center_x
        center_y = metrics.prev_frame_rect.center_y
        viewport_corners_translated = viewport_corners + np.array([center_x, center_y])

        frames_data.append({
            'frameIndex': frame_idx,
            'centerX': float(center_x),
            'centerY': float(center_y),
            'preservation': float(metrics.preservation * 100),
            'novelty': float(metrics.novelty * 100),
            'viewport': viewport_corners_translated.tolist(),
            'trail': trail_frames
        })

    return frames_data


def create_canvas_html(
    metrics_list: List[FrameMetrics],
    width: int = 800,
    height: int = 600,
    trail_length: int = DEFAULT_TRAIL_LENGTH,
    playback_fps: int = 10
) -> str:
    """Create HTML with Canvas-based frame overlap visualization.

    Args:
        metrics_list: List of frame metrics from simulator
        width: Canvas width in pixels
        height: Canvas height in pixels
        trail_length: Number of frames to show in trail
        playback_fps: Frames per second for auto-play

    Returns:
        HTML string with embedded Canvas and JavaScript
    """
    if not metrics_list:
        return '<div style="color: #C8C8DC; padding: 20px;">No metrics to display</div>'

    # Serialize frame data to JSON
    frames_data = serialize_frame_data(metrics_list, trail_length)

    # Calculate viewport dimensions
    viewport_width = metrics_list[0].curr_viewport_rect.width
    viewport_height = metrics_list[0].curr_viewport_rect.height
    padding_factor = 1.8

    html = f'''
    <div style="background-color: {COLOR_BG}; padding: 20px; border-radius: 8px;">
        <div style="text-align: center; color: {COLOR_TEXT}; margin-bottom: 10px; font-size: 14px;">
            <strong>Frame Overlap Simulator - {len(metrics_list)} frames</strong>
        </div>

        <canvas id="frameCanvas" width="{width}" height="{height}"
                style="border: 1px solid {COLOR_GRID}; display: block; margin: 0 auto; background-color: {COLOR_BG};"></canvas>

        <div style="margin-top: 15px; padding: 10px; background-color: rgba(60, 60, 80, 0.6); border-radius: 5px;">
            <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px;">
                <button id="playBtn" style="background-color: {COLOR_VIEWPORT}; color: {COLOR_BG}; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: bold;">
                    ▶ Play
                </button>
                <button id="pauseBtn" style="background-color: rgba(60, 60, 80, 0.8); color: {COLOR_TEXT}; border: 1px solid {COLOR_GRID}; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px;">
                    ⏸ Pause
                </button>
                <span id="frameInfo" style="color: {COLOR_TEXT}; margin-left: 10px; font-size: 13px;">Frame: 0</span>
                <div style="flex: 1;"></div>
                <div id="metricsInfo" style="color: {COLOR_TEXT}; font-size: 12px; text-align: right;">
                    <div>Preservation: <span id="preservation">100.0</span>%</div>
                    <div>Novelty: <span id="novelty">0.0</span>%</div>
                </div>
            </div>

            <input type="range" id="frameSlider" min="0" max="{len(metrics_list) - 1}" value="0"
                   style="width: 100%; height: 6px; background: {COLOR_GRID}; outline: none; border-radius: 3px; cursor: pointer;">

            <div style="display: flex; gap: 10px; align-items: center; margin-top: 10px;">
                <label style="color: {COLOR_TEXT}; font-size: 12px;">Speed:</label>
                <input type="range" id="speedSlider" min="1" max="60" value="{playback_fps}"
                       style="width: 150px; height: 4px; background: {COLOR_GRID}; outline: none; border-radius: 2px; cursor: pointer;">
                <span id="speedInfo" style="color: {COLOR_TEXT}; font-size: 12px;">{playback_fps} fps</span>
            </div>
        </div>
    </div>

    <script>
    (function() {{
        const canvas = document.getElementById('frameCanvas');
        const ctx = canvas.getContext('2d');
        const playBtn = document.getElementById('playBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        const slider = document.getElementById('frameSlider');
        const speedSlider = document.getElementById('speedSlider');
        const frameInfo = document.getElementById('frameInfo');
        const speedInfo = document.getElementById('speedInfo');
        const preservationSpan = document.getElementById('preservation');
        const noveltySpan = document.getElementById('novelty');

        // Frame data from Python
        const frames = {json.dumps(frames_data)};
        const viewportWidth = {viewport_width};
        const viewportHeight = {viewport_height};
        const paddingFactor = {padding_factor};

        let currentFrame = 0;
        let isPlaying = false;
        let lastFrameTime = 0;
        let fps = {playback_fps};

        // Color palette
        const colors = {{
            viewport: '{COLOR_VIEWPORT}',
            text: '{COLOR_TEXT}',
            grid: '{COLOR_GRID}',
            bg: '{COLOR_BG}'
        }};

        // Calculate opacity based on age (quadratic fade)
        function calculateOpacity(age, maxAge) {{
            if (age === 0) return 1.0;
            const fade = Math.pow((maxAge - age) / maxAge, 2);
            return 0.1 + 0.9 * fade;
        }}

        // Draw a rectangle given corners
        function drawRectangle(corners, color, opacity, fill = true) {{
            ctx.globalAlpha = opacity;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;

            ctx.beginPath();
            ctx.moveTo(corners[0][0], corners[0][1]);
            for (let i = 1; i < corners.length; i++) {{
                ctx.lineTo(corners[i][0], corners[i][1]);
            }}
            ctx.closePath();

            if (fill) {{
                ctx.fillStyle = color;
                ctx.globalAlpha = opacity * 0.15;
                ctx.fill();
                ctx.globalAlpha = opacity;
            }}

            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }}

        // Transform world coordinates to canvas coordinates
        function worldToCanvas(x, y, centerX, centerY) {{
            const scale = Math.min(
                canvas.width / (viewportWidth * paddingFactor),
                canvas.height / (viewportHeight * paddingFactor)
            );

            const offsetX = canvas.width / 2;
            const offsetY = canvas.height / 2;

            return [
                (x - centerX) * scale + offsetX,
                (y - centerY) * scale + offsetY
            ];
        }}

        // Render a single frame
        function renderFrame(frameIndex) {{
            const frame = frames[frameIndex];
            if (!frame) return;

            // Clear canvas
            ctx.fillStyle = colors.bg;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw grid
            ctx.strokeStyle = colors.grid;
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.3;

            const gridSize = 50;
            const scale = Math.min(
                canvas.width / (viewportWidth * paddingFactor),
                canvas.height / (viewportHeight * paddingFactor)
            );
            const gridSpacing = gridSize * scale;

            // Vertical lines
            for (let x = 0; x < canvas.width; x += gridSpacing) {{
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }}

            // Horizontal lines
            for (let y = 0; y < canvas.height; y += gridSpacing) {{
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }}

            ctx.globalAlpha = 1.0;

            // Draw trail frames (oldest to newest)
            const maxAge = frame.trail.length - 1;
            for (const trailFrame of frame.trail) {{
                const corners = trailFrame.corners.map(
                    c => worldToCanvas(c[0], c[1], frame.centerX, frame.centerY)
                );
                const opacity = calculateOpacity(trailFrame.age, maxAge);
                drawRectangle(corners, trailFrame.color, opacity, true);
            }}

            // Draw current viewport (always centered)
            const viewportCorners = frame.viewport.map(
                c => worldToCanvas(c[0], c[1], frame.centerX, frame.centerY)
            );
            drawRectangle(viewportCorners, colors.viewport, 0.8, false);

            // Update info displays
            frameInfo.textContent = `Frame: ${{frameIndex}}`;
            preservationSpan.textContent = frame.preservation.toFixed(1);
            noveltySpan.textContent = frame.novelty.toFixed(1);

            // Color-code preservation text
            if (frame.preservation < 30) {{
                preservationSpan.style.color = '{COLOR_PROBLEM}';
            }} else if (frame.novelty > 70) {{
                preservationSpan.style.color = '{COLOR_WARNING}';
            }} else {{
                preservationSpan.style.color = '{COLOR_GOOD}';
            }}
        }}

        // Animation loop
        function animate(timestamp) {{
            if (!isPlaying) return;

            const frameInterval = 1000 / fps;
            if (timestamp - lastFrameTime >= frameInterval) {{
                currentFrame = (currentFrame + 1) % frames.length;
                slider.value = currentFrame;
                renderFrame(currentFrame);
                lastFrameTime = timestamp;
            }}

            requestAnimationFrame(animate);
        }}

        // Event listeners
        playBtn.addEventListener('click', () => {{
            isPlaying = true;
            lastFrameTime = performance.now();
            requestAnimationFrame(animate);
            playBtn.style.backgroundColor = 'rgba(60, 60, 80, 0.8)';
            pauseBtn.style.backgroundColor = '{COLOR_VIEWPORT}';
        }});

        pauseBtn.addEventListener('click', () => {{
            isPlaying = false;
            pauseBtn.style.backgroundColor = 'rgba(60, 60, 80, 0.8)';
            playBtn.style.backgroundColor = '{COLOR_VIEWPORT}';
        }});

        slider.addEventListener('input', (e) => {{
            currentFrame = parseInt(e.target.value);
            renderFrame(currentFrame);
        }});

        speedSlider.addEventListener('input', (e) => {{
            fps = parseInt(e.target.value);
            speedInfo.textContent = fps + ' fps';
        }});

        // Initial render
        renderFrame(0);
    }})();
    </script>
    '''

    return html
