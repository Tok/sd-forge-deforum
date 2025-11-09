"""Canvas-based frame overlap visualizer for Gradio HTML component.

Since Gradio strips <script> tags, we use inline event handlers (onclick, oninput)
and generate all JavaScript code as inline attributes.
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
    """Create HTML with Canvas-based frame overlap visualization using inline handlers."""
    if not metrics_list:
        return '<div style="color: #C8C8DC; padding: 20px;">No metrics to display</div>'

    frames_data = serialize_frame_data(metrics_list, trail_length)
    viewport_width = metrics_list[0].curr_viewport_rect.width
    viewport_height = metrics_list[0].curr_viewport_rect.height
    padding_factor = 1.8

    # Generate unique ID
    import random
    canvas_id = f"fc{random.randint(1000, 9999)}"

    # Embed all JavaScript as one big global function that gets called on load
    js_code = f'''
window.{canvas_id}_state = {{
    currentFrame: 0,
    isPlaying: false,
    fps: {playback_fps},
    lastFrameTime: 0,
    frames: {json.dumps(frames_data)},
    viewportWidth: {viewport_width},
    viewportHeight: {viewport_height},
    paddingFactor: {padding_factor},
    colors: {{
        viewport: '{COLOR_VIEWPORT}',
        text: '{COLOR_TEXT}',
        grid: '{COLOR_GRID}',
        bg: '{COLOR_BG}'
    }}
}};

window.{canvas_id}_calculateOpacity = function(age, maxAge) {{
    if (age === 0) return 1.0;
    const fade = Math.pow((maxAge - age) / maxAge, 2);
    return 0.1 + 0.9 * fade;
}};

window.{canvas_id}_drawRectangle = function(ctx, corners, color, opacity, fill) {{
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
}};

window.{canvas_id}_worldToCanvas = function(x, y, centerX, centerY, canvasW, canvasH, vpW, vpH, padFactor) {{
    const scale = Math.min(canvasW / (vpW * padFactor), canvasH / (vpH * padFactor));
    const offsetX = canvasW / 2;
    const offsetY = canvasH / 2;
    return [(x - centerX) * scale + offsetX, (y - centerY) * scale + offsetY];
}};

window.{canvas_id}_renderFrame = function(frameIndex) {{
    const state = window.{canvas_id}_state;
    const canvas = document.getElementById('{canvas_id}');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const frame = state.frames[frameIndex];
    if (!frame) return;

    // Clear canvas
    ctx.fillStyle = state.colors.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = state.colors.grid;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.3;
    const gridSize = 50;
    const scale = Math.min(
        canvas.width / (state.viewportWidth * state.paddingFactor),
        canvas.height / (state.viewportHeight * state.paddingFactor)
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

    // Draw trail frames
    const maxAge = frame.trail.length - 1;
    for (const trailFrame of frame.trail) {{
        const corners = trailFrame.corners.map(c =>
            window.{canvas_id}_worldToCanvas(c[0], c[1], frame.centerX, frame.centerY,
                canvas.width, canvas.height, state.viewportWidth, state.viewportHeight, state.paddingFactor)
        );
        const opacity = window.{canvas_id}_calculateOpacity(trailFrame.age, maxAge);
        window.{canvas_id}_drawRectangle(ctx, corners, trailFrame.color, opacity, true);
    }}

    // Draw viewport
    const viewportCorners = frame.viewport.map(c =>
        window.{canvas_id}_worldToCanvas(c[0], c[1], frame.centerX, frame.centerY,
            canvas.width, canvas.height, state.viewportWidth, state.viewportHeight, state.paddingFactor)
    );
    window.{canvas_id}_drawRectangle(ctx, viewportCorners, state.colors.viewport, 0.8, false);

    // Update displays
    document.getElementById('{canvas_id}_frameInfo').textContent = 'Frame: ' + frameIndex;
    document.getElementById('{canvas_id}_preservation').textContent = frame.preservation.toFixed(1);
    document.getElementById('{canvas_id}_novelty').textContent = frame.novelty.toFixed(1);

    // Color-code preservation
    const preservationSpan = document.getElementById('{canvas_id}_preservation');
    if (frame.preservation < 30) {{
        preservationSpan.style.color = '{COLOR_PROBLEM}';
    }} else if (frame.novelty > 70) {{
        preservationSpan.style.color = '{COLOR_WARNING}';
    }} else {{
        preservationSpan.style.color = '{COLOR_GOOD}';
    }}
}};

window.{canvas_id}_animate = function(timestamp) {{
    const state = window.{canvas_id}_state;
    if (!state.isPlaying) return;
    const frameInterval = 1000 / state.fps;
    if (timestamp - state.lastFrameTime >= frameInterval) {{
        state.currentFrame = (state.currentFrame + 1) % state.frames.length;
        document.getElementById('{canvas_id}_slider').value = state.currentFrame;
        window.{canvas_id}_renderFrame(state.currentFrame);
        state.lastFrameTime = timestamp;
    }}
    requestAnimationFrame(window.{canvas_id}_animate);
}};

window.{canvas_id}_play = function() {{
    const state = window.{canvas_id}_state;
    state.isPlaying = true;
    state.lastFrameTime = performance.now();
    requestAnimationFrame(window.{canvas_id}_animate);
    document.getElementById('{canvas_id}_playBtn').style.background = 'transparent';
    document.getElementById('{canvas_id}_playBtn').style.border = '1px solid var(--border-color-primary)';
    document.getElementById('{canvas_id}_pauseBtn').style.background = 'linear-gradient(to right, #7c3aed, #a855f7)';
    document.getElementById('{canvas_id}_pauseBtn').style.border = 'none';
}};

window.{canvas_id}_pause = function() {{
    const state = window.{canvas_id}_state;
    state.isPlaying = false;
    document.getElementById('{canvas_id}_pauseBtn').style.background = 'transparent';
    document.getElementById('{canvas_id}_pauseBtn').style.border = '1px solid var(--border-color-primary)';
    document.getElementById('{canvas_id}_playBtn').style.background = 'linear-gradient(to right, #7c3aed, #a855f7)';
    document.getElementById('{canvas_id}_playBtn').style.border = 'none';
}};

window.{canvas_id}_onSliderChange = function(e) {{
    const state = window.{canvas_id}_state;
    state.currentFrame = parseInt(e.target.value);
    window.{canvas_id}_renderFrame(state.currentFrame);
}};

window.{canvas_id}_onSpeedChange = function(e) {{
    const state = window.{canvas_id}_state;
    state.fps = parseInt(e.target.value);
    document.getElementById('{canvas_id}_speedInfo').textContent = state.fps + ' fps';
}};

// Initial render on load
setTimeout(function() {{
    window.{canvas_id}_renderFrame(0);
}}, 100);
'''

    html = f'''
<div style="width: 100%; max-width: {width}px; margin: 0 auto;">
    <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
         onload="(function(){{ {js_code} }})();"
         style="display:none;">
    <canvas id="{canvas_id}" width="{width}" height="{height}"
            style="width: 100%; height: auto; display: block; margin: 0 auto 10px auto; background-color: {COLOR_BG}; border-radius: 4px;"></canvas>

    <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 8px;">
        <button id="{canvas_id}_playBtn" onclick="window.{canvas_id}_play()"
                style="background: linear-gradient(to right, #7c3aed, #a855f7); color: white; border: none; padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500;">
            ▶ Play
        </button>
        <button id="{canvas_id}_pauseBtn" onclick="window.{canvas_id}_pause()"
                style="background: transparent; color: var(--body-text-color); border: 1px solid var(--border-color-primary); padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 14px;">
            ⏸ Pause
        </button>
        <span id="{canvas_id}_frameInfo" style="margin-left: 8px; font-size: 14px;">Frame: 0</span>
        <div style="flex: 1;"></div>
        <div style="font-size: 14px; text-align: right;">
            <span>Preservation: <span id="{canvas_id}_preservation">100.0</span>%</span>
            <span style="margin-left: 12px;">Novelty: <span id="{canvas_id}_novelty">0.0</span>%</span>
        </div>
    </div>

    <input type="range" id="{canvas_id}_slider" min="0" max="{len(metrics_list) - 1}" value="0"
           oninput="window.{canvas_id}_onSliderChange(event)"
           style="width: 100%; margin-bottom: 8px; cursor: pointer;">

    <div style="display: flex; gap: 8px; align-items: center;">
        <label style="font-size: 14px; min-width: 50px;">Speed:</label>
        <input type="range" id="{canvas_id}_speedSlider" min="1" max="60" value="{playback_fps}"
               oninput="window.{canvas_id}_onSpeedChange(event)"
               style="flex: 1; cursor: pointer;">
        <span id="{canvas_id}_speedInfo" style="font-size: 14px; min-width: 50px;">{playback_fps} fps</span>
    </div>
</div>
'''

    return html
