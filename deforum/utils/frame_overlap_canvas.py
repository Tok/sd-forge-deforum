"""Canvas-based frame overlap visualizer using iframe data URL.

Gradio aggressively sanitizes gr.HTML() content, stripping scripts and truncating inline handlers.
Solution: Embed entire visualization as a standalone HTML page in an iframe data URL.
This completely bypasses Gradio's sanitization since iframe content is isolated.
"""

import json
from typing import List, Dict, Any
import numpy as np
import html

from deforum.utils.frame_overlap_simulator import (
    FrameMetrics,
    MIN_PRESERVATION_THRESHOLD,
    MAX_NOVELTY_THRESHOLD,
    DEFAULT_TRAIL_LENGTH,
)


# Deforum purple/slopcore dark theme colors
COLOR_GOOD = '#9664FF'
COLOR_WARNING = '#FF9664'
COLOR_PROBLEM = '#FF5050'
COLOR_VIEWPORT = '#B48CFF'
COLOR_BG = '#14141E'
COLOR_GRID = '#3C3C50'
COLOR_TEXT = '#C8C8DC'


def get_frame_color(metrics: FrameMetrics) -> str:
    """Get hex color for frame based on preservation/novelty metrics."""
    if metrics.preservation < MIN_PRESERVATION_THRESHOLD:
        return COLOR_PROBLEM
    elif metrics.novelty > MAX_NOVELTY_THRESHOLD:
        return COLOR_WARNING
    else:
        return COLOR_GOOD


def serialize_frame_data(metrics_list: List[FrameMetrics], trail_length: int) -> List[Dict[str, Any]]:
    """Serialize frame metrics to JSON-compatible format."""
    frames_data = []
    for frame_idx, metrics in enumerate(metrics_list):
        trail_start = max(0, frame_idx - trail_length + 1)
        trail_frames = []
        for i in range(trail_start, frame_idx + 1):
            age = frame_idx - i
            trail_metrics = metrics_list[i]
            corners = trail_metrics.prev_frame_rect.get_corners()
            trail_frames.append({
                'corners': corners.tolist(),
                'color': get_frame_color(trail_metrics),
                'age': age,
                'frameIndex': i
            })
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
    """Create iframe with standalone HTML visualization."""
    if not metrics_list:
        return '<div style="color: #C8C8DC; padding: 20px;">No metrics to display</div>'

    frames_data = serialize_frame_data(metrics_list, trail_length)
    viewport_width = metrics_list[0].curr_viewport_rect.width
    viewport_height = metrics_list[0].curr_viewport_rect.height
    padding_factor = 1.8

    # Create complete standalone HTML page
    standalone_html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ margin: 0; padding: 20px; background-color: {COLOR_BG}; font-family: system-ui, -apple-system, sans-serif; }}
        canvas {{ display: block; margin: 0 auto 10px auto; background-color: {COLOR_BG}; border-radius: 4px; }}
        .controls {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; max-width: {width}px; margin-left: auto; margin-right: auto; }}
        button {{ padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; border: none; }}
        .btn-play {{ background: linear-gradient(to right, #7c3aed, #a855f7); color: white; font-weight: 500; }}
        .btn-pause {{ background: transparent; color: {COLOR_TEXT}; border: 1px solid {COLOR_GRID}; }}
        input[type=range] {{ flex: 1; cursor: pointer; }}
        .info {{ color: {COLOR_TEXT}; font-size: 14px; }}
    </style>
</head>
<body>
    <canvas id="canvas" width="{width}" height="{height}"></canvas>

    <div class="controls">
        <button id="playBtn" class="btn-play" onclick="play()">▶ Play</button>
        <button id="pauseBtn" class="btn-pause" onclick="pause()">⏸ Pause</button>
        <span id="frameInfo" class="info" style="margin-left: 8px;">Frame: 0</span>
        <div style="flex: 1;"></div>
        <div class="info">
            <span>Preservation: <span id="preservation">100.0</span>%</span>
            <span style="margin-left: 12px;">Novelty: <span id="novelty">0.0</span>%</span>
        </div>
    </div>

    <input type="range" id="slider" min="0" max="{len(metrics_list) - 1}" value="0" oninput="onSliderChange(event)"
           style="width: 100%; max-width: {width}px; margin: 0 auto 8px auto; display: block;">

    <div class="controls">
        <label class="info" style="min-width: 50px;">Speed:</label>
        <input type="range" id="speedSlider" min="1" max="60" value="{playback_fps}" oninput="onSpeedChange(event)">
        <span id="speedInfo" class="info" style="min-width: 50px;">{playback_fps} fps</span>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const frames = {json.dumps(frames_data)};
        const viewportWidth = {viewport_width};
        const viewportHeight = {viewport_height};
        const paddingFactor = {padding_factor};

        let currentFrame = 0;
        let isPlaying = false;
        let lastFrameTime = 0;
        let fps = {playback_fps};

        const colors = {{
            viewport: '{COLOR_VIEWPORT}',
            grid: '{COLOR_GRID}',
            bg: '{COLOR_BG}'
        }};

        function calculateOpacity(age, maxAge) {{
            if (age === 0) return 1.0;
            const fade = Math.pow((maxAge - age) / maxAge, 2);
            return 0.1 + 0.9 * fade;
        }}

        function drawRectangle(corners, color, opacity, fill) {{
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

        function worldToCanvas(x, y, centerX, centerY) {{
            const scale = Math.min(
                canvas.width / (viewportWidth * paddingFactor),
                canvas.height / (viewportHeight * paddingFactor)
            );
            const offsetX = canvas.width / 2;
            const offsetY = canvas.height / 2;
            return [(x - centerX) * scale + offsetX, (y - centerY) * scale + offsetY];
        }}

        function renderFrame(frameIndex) {{
            const frame = frames[frameIndex];
            if (!frame) return;

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
            for (let x = 0; x < canvas.width; x += gridSpacing) {{
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }}
            for (let y = 0; y < canvas.height; y += gridSpacing) {{
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }}
            ctx.globalAlpha = 1.0;

            // Draw trail
            const maxAge = frame.trail.length - 1;
            for (const trailFrame of frame.trail) {{
                const corners = trailFrame.corners.map(c =>
                    worldToCanvas(c[0], c[1], frame.centerX, frame.centerY)
                );
                const opacity = calculateOpacity(trailFrame.age, maxAge);
                drawRectangle(corners, trailFrame.color, opacity, true);
            }}

            // Draw viewport
            const viewportCorners = frame.viewport.map(c =>
                worldToCanvas(c[0], c[1], frame.centerX, frame.centerY)
            );
            drawRectangle(viewportCorners, colors.viewport, 0.8, false);

            // Update info
            document.getElementById('frameInfo').textContent = 'Frame: ' + frameIndex;
            document.getElementById('preservation').textContent = frame.preservation.toFixed(1);
            document.getElementById('novelty').textContent = frame.novelty.toFixed(1);

            // Color-code preservation
            const preservationSpan = document.getElementById('preservation');
            if (frame.preservation < 30) {{
                preservationSpan.style.color = '{COLOR_PROBLEM}';
            }} else if (frame.novelty > 70) {{
                preservationSpan.style.color = '{COLOR_WARNING}';
            }} else {{
                preservationSpan.style.color = '{COLOR_GOOD}';
            }}
        }}

        function animate(timestamp) {{
            if (!isPlaying) return;
            const frameInterval = 1000 / fps;
            if (timestamp - lastFrameTime >= frameInterval) {{
                currentFrame = (currentFrame + 1) % frames.length;
                document.getElementById('slider').value = currentFrame;
                renderFrame(currentFrame);
                lastFrameTime = timestamp;
            }}
            requestAnimationFrame(animate);
        }}

        function play() {{
            isPlaying = true;
            lastFrameTime = performance.now();
            requestAnimationFrame(animate);
            document.getElementById('playBtn').style.background = 'transparent';
            document.getElementById('playBtn').style.border = '1px solid {COLOR_GRID}';
            document.getElementById('pauseBtn').style.background = 'linear-gradient(to right, #7c3aed, #a855f7)';
            document.getElementById('pauseBtn').style.border = 'none';
        }}

        function pause() {{
            isPlaying = false;
            document.getElementById('pauseBtn').style.background = 'transparent';
            document.getElementById('pauseBtn').style.border = '1px solid {COLOR_GRID}';
            document.getElementById('playBtn').style.background = 'linear-gradient(to right, #7c3aed, #a855f7)';
            document.getElementById('playBtn').style.border = 'none';
        }}

        function onSliderChange(e) {{
            currentFrame = parseInt(e.target.value);
            renderFrame(currentFrame);
        }}

        function onSpeedChange(e) {{
            fps = parseInt(e.target.value);
            document.getElementById('speedInfo').textContent = fps + ' fps';
        }}

        // Initial render
        renderFrame(0);
    </script>
</body>
</html>'''

    # Encode as data URL
    import base64
    html_bytes = standalone_html.encode('utf-8')
    html_b64 = base64.b64encode(html_bytes).decode('utf-8')
    data_url = f'data:text/html;base64,{html_b64}'

    # Return iframe
    return f'<iframe src="{data_url}" style="width: 100%; height: {height + 120}px; border: none; display: block;"></iframe>'
