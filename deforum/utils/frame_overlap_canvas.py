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


def serialize_frame_data(metrics_list: List[FrameMetrics], trail_length: int, translation_amplify: float = 15.0) -> List[Dict[str, Any]]:
    """Serialize frame metrics to JSON-compatible format.

    Visualization concept (dash-cam view):
    - Current viewport is ALWAYS fixed at center (0, 0)
    - Previous frames trail AWAY showing where the camera WAS
    - Creates "worm" effect showing camera movement history

    All positions are RELATIVE to current frame position.

    Args:
        metrics_list: List of FrameMetrics from simulator
        trail_length: Number of previous frames to show
        translation_amplify: Amplification factor for translation (default: 15.0).
                            Makes small pixel-level movements more visible in preview.
                            Higher values = longer, more visible worm trail for orbital paths.
    """
    # Sample every Nth frame to reduce data size and stay under browser data URL limits
    # For 333 frames, sampling every 3rd = ~111 frames = manageable data size
    sample_interval = max(1, len(metrics_list) // 150)  # Target ~150 frames max

    frames_data = []
    for sample_idx, frame_idx in enumerate(range(0, len(metrics_list), sample_interval)):
        metrics = metrics_list[frame_idx]
        # Get current frame's position in world space
        current_center_x = metrics.prev_frame_rect.center_x
        current_center_y = metrics.prev_frame_rect.center_y
        current_rotation = metrics.prev_frame_rect.rotation

        # Build trail from sampled frames only (reduces trail density but keeps data manageable)
        # Trail should only show PREVIOUS frames (not including current frame)
        # Current frame is represented by the viewport at (0, 0)

        # Always include the immediately previous frame to avoid visual disconnect
        # Then add sampled frames going backwards
        trail_frames = []

        # First, add the frame immediately before current (if it exists)
        if frame_idx > 0:
            immediate_prev_idx = frame_idx - 1
            immediate_prev_metrics = metrics_list[immediate_prev_idx]

            # Calculate relative position for immediate previous frame
            relative_center_x = (immediate_prev_metrics.prev_frame_rect.center_x - current_center_x) * translation_amplify
            relative_center_y = (immediate_prev_metrics.prev_frame_rect.center_y - current_center_y) * translation_amplify
            relative_rotation = immediate_prev_metrics.prev_frame_rect.rotation - current_rotation

            from deforum.utils.frame_overlap_simulator import Rectangle
            relative_rect = Rectangle(
                center_x=relative_center_x,
                center_y=relative_center_y,
                width=immediate_prev_metrics.prev_frame_rect.width,
                height=immediate_prev_metrics.prev_frame_rect.height,
                rotation=relative_rotation
            )
            corners = relative_rect.get_corners()
            rounded_corners = [[round(x, 1), round(y, 1)] for x, y in corners]

            trail_frames.append({
                'corners': rounded_corners,
                'color': get_frame_color(immediate_prev_metrics),
                'age': 1,  # Most recent previous frame
                'frameIndex': immediate_prev_idx
            })

        # Then add sampled trail frames (skip if they're the same as immediate previous)
        trail_start_sample = max(0, sample_idx - trail_length)
        for trail_sample_idx in range(trail_start_sample, sample_idx):
            trail_frame_idx = trail_sample_idx * sample_interval
            if trail_frame_idx >= len(metrics_list):
                break

            # Skip if this is the same as the immediate previous frame we already added
            if frame_idx > 0 and trail_frame_idx == frame_idx - 1:
                continue

            age = sample_idx - trail_sample_idx + 1  # +1 because age=1 is now the immediate previous
            trail_metrics = metrics_list[trail_frame_idx]

            # Get trail frame's absolute world position
            trail_center_x = trail_metrics.prev_frame_rect.center_x
            trail_center_y = trail_metrics.prev_frame_rect.center_y
            trail_rotation = trail_metrics.prev_frame_rect.rotation

            # Convert to RELATIVE position (where trail frame is relative to current frame)
            # If camera moved RIGHT (+X), previous frames appear LEFT (-X)
            # Apply translation_amplify to make small pixel movements more visible
            relative_center_x = (trail_center_x - current_center_x) * translation_amplify
            relative_center_y = (trail_center_y - current_center_y) * translation_amplify
            relative_rotation = trail_rotation - current_rotation

            # Create rectangle at relative position
            from deforum.utils.frame_overlap_simulator import Rectangle
            relative_rect = Rectangle(
                center_x=relative_center_x,
                center_y=relative_center_y,
                width=trail_metrics.prev_frame_rect.width,
                height=trail_metrics.prev_frame_rect.height,
                rotation=relative_rotation
            )
            corners = relative_rect.get_corners()

            # Round coordinates to 1 decimal place to reduce JSON size
            rounded_corners = [[round(x, 1), round(y, 1)] for x, y in corners]

            trail_frames.append({
                'corners': rounded_corners,
                'color': get_frame_color(trail_metrics),
                'age': age,
                'frameIndex': trail_frame_idx
            })

        # Current viewport is ALWAYS at origin (0, 0) - never moves
        viewport_corners = metrics.curr_viewport_rect.get_corners()
        rounded_viewport = [[round(x, 1), round(y, 1)] for x, y in viewport_corners]

        frames_data.append({
            'frameIndex': frame_idx,
            'centerX': 0.0,  # Always centered in dash-cam view
            'centerY': 0.0,  # Always centered in dash-cam view
            'preservation': round(metrics.preservation * 100, 1),
            'novelty': round(metrics.novelty * 100, 1),
            'viewport': rounded_viewport,
            'trail': trail_frames
        })
    return frames_data


def create_canvas_html(
    metrics_list: List[FrameMetrics],
    width: int = 800,
    height: int = 600,
    trail_length: int = DEFAULT_TRAIL_LENGTH,
    playback_fps: int = 10,
    translation_amplify: float = 15.0
) -> str:
    """Create iframe with standalone HTML visualization.

    Args:
        metrics_list: List of FrameMetrics from simulator
        width: Canvas width in pixels
        height: Canvas height in pixels
        trail_length: Number of previous frames to show
        playback_fps: Playback speed in frames per second
        translation_amplify: Amplification factor for translation visibility (default: 15.0).
                            Higher values = more visible translation in orbital camera paths.
    """
    if not metrics_list:
        return '<div style="color: #C8C8DC; padding: 20px;">No metrics to display</div>'

    # Check if there's any actual camera movement (check consecutive frame differences)
    has_movement = any(
        abs(metrics_list[i].prev_frame_rect.center_x - metrics_list[i-1].prev_frame_rect.center_x) > 0.1 or
        abs(metrics_list[i].prev_frame_rect.center_y - metrics_list[i-1].prev_frame_rect.center_y) > 0.1 or
        abs(metrics_list[i].prev_frame_rect.rotation - metrics_list[i-1].prev_frame_rect.rotation) > 0.1
        for i in range(1, len(metrics_list))
    )

    if not has_movement:
        return '<div style="color: #FF9664; padding: 20px; background: rgba(60,60,80,0.3); border-radius: 4px;">⚠️ No camera movement detected. Use "Rotate Around" preset or enter camera schedules to see frame overlap trail.</div>'

    frames_data = serialize_frame_data(metrics_list, trail_length, translation_amplify)
    viewport_width = metrics_list[0].curr_viewport_rect.width
    viewport_height = metrics_list[0].curr_viewport_rect.height
    padding_factor = 1.8

    # Create complete standalone HTML page
    standalone_html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ margin: 0; padding: 20px; background-color: {COLOR_BG}; font-family: system-ui, -apple-system, sans-serif; overflow-x: hidden; }}
        canvas {{ display: block; margin: 0 auto 10px auto; background-color: {COLOR_BG}; border-radius: 4px; max-width: 100%; height: auto; }}
        .controls {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; max-width: 100%; margin-left: auto; margin-right: auto; flex-wrap: wrap; }}
        button {{ padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; border: none; white-space: nowrap; }}
        .btn-play {{ background: linear-gradient(to right, #7c3aed, #a855f7); color: white; font-weight: 500; }}
        .btn-pause {{ background: transparent; color: {COLOR_TEXT}; border: 1px solid {COLOR_GRID}; }}
        input[type=range] {{ flex: 1; cursor: pointer; min-width: 100px; }}
        .info {{ color: {COLOR_TEXT}; font-size: 14px; white-space: nowrap; }}
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
           style="width: 100%; max-width: 100%; margin: 0 auto 8px auto; display: block;">

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
            // Add glow effect for better edge visibility
            ctx.shadowColor = color;
            ctx.shadowBlur = 4;  // Reduced glow for thinner lines
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 0;

            // Draw thinner stroke for more frames visible
            ctx.globalAlpha = Math.min(1.0, opacity * 1.5);  // Even brighter stroke
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.0;  // Thinner for better visibility with more frames
            ctx.beginPath();
            ctx.moveTo(corners[0][0], corners[0][1]);
            for (let i = 1; i < corners.length; i++) {{
                ctx.lineTo(corners[i][0], corners[i][1]);
            }}
            ctx.closePath();

            // Very subtle fill to show overlap
            if (fill) {{
                ctx.fillStyle = color;
                ctx.shadowBlur = 0;  // No glow on fill
                ctx.globalAlpha = opacity * 0.02;  // Minimal fill (2% for thinner frames)
                ctx.fill();
                ctx.globalAlpha = Math.min(1.0, opacity * 1.5);  // Restore bright stroke
                ctx.shadowBlur = 4;  // Restore glow for stroke
            }}

            ctx.stroke();
            ctx.globalAlpha = 1.0;
            ctx.shadowBlur = 0;  // Clear glow after drawing
        }}

        function worldToCanvas(x, y) {{
            const scale = Math.min(
                canvas.width / (viewportWidth * paddingFactor),
                canvas.height / (viewportHeight * paddingFactor)
            );
            const offsetX = canvas.width / 2;
            const offsetY = canvas.height / 2;
            // Use world origin (0,0) as reference, not current frame's center
            // Trail corners already include accumulated position from simulator
            return [x * scale + offsetX, y * scale + offsetY];
        }}

        function renderFrame(frameIndex) {{
            const frame = frames[frameIndex];
            if (!frame) return;

            // Debug logging
            if (frameIndex === 0 || frameIndex === 10) {{
                console.log(`Frame ${{frameIndex}} data:`, {{
                    centerX: frame.centerX,
                    centerY: frame.centerY,
                    trailLength: frame.trail.length,
                    preservation: frame.preservation,
                    novelty: frame.novelty,
                    viewport: frame.viewport,
                    trailSample: frame.trail.map(t => ({{ frameIdx: t.frameIndex, age: t.age, center: [t.corners[0][0], t.corners[0][1]] }}))
                }});
            }}

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
            if (frameIndex === 0 || frameIndex === 10) {{
                console.log(`Drawing trail for frame ${{frameIndex}}:`, {{
                    trailLength: frame.trail.length,
                    maxAge: maxAge,
                    firstTrailCorners: frame.trail[0].corners,
                    lastTrailCorners: frame.trail[frame.trail.length - 1].corners
                }});
            }}
            for (const trailFrame of frame.trail) {{
                const corners = trailFrame.corners.map(c =>
                    worldToCanvas(c[0], c[1])
                );
                const opacity = calculateOpacity(trailFrame.age, maxAge);
                if (frameIndex === 0 || frameIndex === 10) {{
                    console.log(`  Trail frame ${{trailFrame.frameIndex}}, age=${{trailFrame.age}}, opacity=${{opacity}}, canvasCorners=${{JSON.stringify(corners)}}`);
                }}
                drawRectangle(corners, trailFrame.color, opacity, true);
            }}

            // Draw viewport
            const viewportCorners = frame.viewport.map(c =>
                worldToCanvas(c[0], c[1])
            );
            if (frameIndex === 0 || frameIndex === 10) {{
                console.log(`Viewport canvas coords:`, viewportCorners);
            }}
            drawRectangle(viewportCorners, colors.viewport, 0.8, false);

            // Update info
            document.getElementById('frameInfo').textContent = 'Frame: ' + frameIndex +
                ' | Pos: [' + frame.centerX.toFixed(1) + ', ' + frame.centerY.toFixed(1) + ']';
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

    # Return iframe scaled to fit container (max 800px width, responsive height)
    return f'<iframe src="{data_url}" style="width: 100%; max-width: 800px; height: {height + 120}px; border: none; display: block; margin: 0 auto;"></iframe>'
