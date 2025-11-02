"""Simplified terminal dashboard for Deforum rendering.

Simple console-based dashboard without Rich Live display to avoid threading issues.
Prints status updates directly to console with optional ASCII art preview.
"""

import time
import re
from collections import deque
from typing import Optional

from deforum.rendering import options as opt_utils

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
    Image = None


class MemoryStats:
    """Tracks GPU memory statistics from Forge output."""

    def __init__(self):
        self.target_model = ""
        self.free_gpu_mb = 0.0
        self.total_gpu_mb = 0.0
        self.model_require_mb = 0.0
        self.remaining_mb = 0.0

    def update_from_forge_message(self, message: str):
        """Parse memory stats from Forge console message."""
        # Example: "[Memory Management] Target: VAE, Free GPU: 8238.74 MB..."
        target_match = re.search(r'Target:\s*([^,]+)', message)
        if target_match:
            self.target_model = target_match.group(1).strip()

        free_match = re.search(r'Free GPU:\s*([\d.]+)\s*MB', message)
        if free_match:
            self.free_gpu_mb = float(free_match.group(1))

        total_match = re.search(r'Total GPU:\s*([\d.]+)\s*MB', message)
        if total_match:
            self.total_gpu_mb = float(total_match.group(1))


def image_to_ascii_art(image, width: int = 32, height: int = 18, use_color: bool = True) -> str:
    """Convert PIL Image to colored ASCII art using 2-space blocks.

    Args:
        image: PIL Image or numpy array
        width: Number of pixels wide
        height: Number of pixels tall
        use_color: Use ANSI color codes

    Returns:
        ASCII art string
    """
    if image is None or Image is None or np is None:
        return ""

    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Preserve aspect ratio
    img_width, img_height = image.size
    img_aspect = img_width / img_height
    target_aspect = width / height

    if img_aspect > target_aspect:
        final_width = width
        final_height = int(width / img_aspect)
    else:
        final_height = height
        final_width = int(height * img_aspect)

    # Resize
    image = image.resize((final_width, final_height), Image.Resampling.LANCZOS)
    pixels = np.array(image)

    lines = []
    for y in range(final_height):
        line = ""
        for x in range(final_width):
            # Get RGB
            if len(pixels.shape) == 3 and pixels.shape[2] >= 3:
                r, g, b = pixels[y, x, :3]
            else:
                gray = pixels[y, x] if pixels.ndim == 2 else pixels[y, x, 0]
                r = g = b = gray

            # 2-space pixel with background color
            if use_color:
                line += f"\033[48;2;{r};{g};{b}m  \033[0m"
            else:
                line += "  "

        lines.append(line)

    return "\n".join(lines)


class RenderDashboard:
    """Simplified dashboard - console output only (no Rich Live)."""

    def __init__(self):
        """Initialize dashboard."""
        self.memory = MemoryStats()
        self._last_print_time = 0

        # Settings
        self.theme = opt_utils.get_log_theme()
        self.use_ascii_preview = opt_utils.is_dashboard_ascii_preview_enabled()

        # State
        self.last_frame_image = None
        self.frame_info = {
            'current': 0,
            'total': 0,
            'type': 'KEYFRAME',
        }
        self.progress_data = {
            'diffusion_frames': (0, 100),
            'total_steps': (0, 100),
            'current_step': (0, 20)
        }

    def start(self):
        """Start dashboard."""
        from deforum.utils.system.logging import get_logger
        logger = get_logger()
        logger.info("")
        logger.info("=" * 80)
        logger.info("DEFORUM SIMPLIFIED DASHBOARD")
        logger.info("=" * 80)

    def stop(self):
        """Stop dashboard."""
        from deforum.utils.system.logging import get_logger
        logger = get_logger()
        logger.info("")
        logger.info("=" * 80)
        logger.info("RENDER COMPLETE")
        logger.info("=" * 80)

    def update(self):
        """Print status update (throttled to 1 per second)."""
        current_time = time.time()
        if current_time - self._last_print_time < 1.0:
            return  # Throttle

        self._last_print_time = current_time

        try:
            self._print_status()
        except Exception:
            pass  # Ignore errors

    def _print_status(self):
        """Print simple status line."""
        from deforum.utils.system.logging import get_logger
        logger = get_logger()

        # Get percentages
        df_current, df_total = self.progress_data['diffusion_frames']
        df_pct = int((df_current / df_total * 100) if df_total > 0 else 0)

        ts_current, ts_total = self.progress_data['total_steps']
        ts_pct = int((ts_current / ts_total * 100) if ts_total > 0 else 0)

        cs_current, cs_total = self.progress_data['current_step']

        # Status line
        status = f"[Frame {self.frame_info['current']}/{self.frame_info['total']}] "
        status += f"Frames:{df_pct}% Steps:{ts_pct}% Current:{cs_current}/{cs_total} "
        status += f"VRAM:{self.memory.free_gpu_mb/1024:.1f}GB"

        logger.info(status)

        # ASCII preview
        if self.use_ascii_preview and self.last_frame_image is not None:
            ascii_art = image_to_ascii_art(
                self.last_frame_image,
                width=32,
                height=18,
                use_color=(self.theme != 'simple')
            )
            if ascii_art:
                for line in ascii_art.split('\n'):
                    logger.info(line)

    def add_log(self, message: str):
        """Add log message (just pass through to logger)."""
        from deforum.utils.system.logging import get_logger
        logger = get_logger()
        logger.info(message)
