"""Fixed-position terminal dashboard using ANSI escape codes.

Similar to Claude Code CLI - logs scroll up, dashboard stays at bottom.
Uses raw ANSI cursor positioning instead of Rich Live to avoid threading issues.
"""

import sys
import time
import shutil
from typing import Optional

from deforum.rendering import options as opt_utils

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
    Image = None


def image_to_ascii_art(image, width: int = 32, height: int = 18, use_color: bool = True) -> str:
    """Convert PIL Image to colored ASCII art using 2-space blocks."""
    if image is None or Image is None or np is None:
        return ""

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

    image = image.resize((final_width, final_height), Image.Resampling.LANCZOS)
    pixels = np.array(image)

    lines = []
    for y in range(final_height):
        line = ""
        for x in range(final_width):
            if len(pixels.shape) == 3 and pixels.shape[2] >= 3:
                r, g, b = pixels[y, x, :3]
            else:
                gray = pixels[y, x] if pixels.ndim == 2 else pixels[y, x, 0]
                r = g = b = gray

            if use_color:
                line += f"\033[48;2;{r};{g};{b}m  \033[0m"
            else:
                line += "  "

        lines.append(line)

    return "\n".join(lines)


class MemoryStats:
    """Tracks GPU memory statistics."""

    def __init__(self):
        self.free_gpu_mb = 0.0


class FixedDashboard:
    """Fixed-position dashboard at bottom of terminal.

    Logs scroll up normally, dashboard stays fixed at bottom.
    Uses ANSI escape codes for cursor positioning.
    """

    def __init__(self):
        """Initialize dashboard."""
        self.memory = MemoryStats()
        self._last_update_time = 0
        self._dashboard_height = 0  # Number of lines dashboard occupies

        # Settings
        self.theme = opt_utils.get_log_theme()
        self.use_ascii_preview = opt_utils.is_dashboard_ascii_preview_enabled()

        # State
        self.last_frame_image = None
        self.frame_info = {
            'current': 0,
            'total': 0,
            'type': 'KEYFRAME',
            'seed': 0,
            'color_rgb': None,
            'movement': '',
            'prompt': ''
        }
        self.table_data = {
            'steps': '0/20',
            'cfg': '1.0',
            'dist_cfg': '3.5',
            'denoise': '0.85',
            'tr_x': '0',
            'tr_y': '0',
            'tr_z': '0',
            'ro_x': '0',
            'ro_y': '0',
            'ro_z': '0'
        }
        self.progress_data = {
            'diffusion_frames': (0, 100),
            'total_steps': (0, 100),
            'current_step': (0, 20)
        }

    def start(self):
        """Start dashboard - reserve space at bottom."""
        # Get terminal size
        self._terminal_height, self._terminal_width = self._get_terminal_size()

        # Calculate dashboard height
        self._dashboard_height = 4  # Status lines
        if self.use_ascii_preview:
            self._dashboard_height += 18  # ASCII art height

        # Print initial empty lines to reserve space
        for _ in range(self._dashboard_height):
            print()

        # Initial render
        self._render()

    def stop(self):
        """Stop dashboard - clean up."""
        # Move cursor below dashboard
        self._move_cursor_below_dashboard()
        print("\n" + "=" * 80)
        print("RENDER COMPLETE")
        print("=" * 80)

    def update(self):
        """Update dashboard (throttled to 1fps)."""
        current_time = time.time()
        if current_time - self._last_update_time < 1.0:
            return

        self._last_update_time = current_time
        self._render()

    def _render(self):
        """Render dashboard at fixed position."""
        # Save cursor position
        sys.stdout.write("\033[s")

        # Move to dashboard area (bottom of terminal)
        row = self._terminal_height - self._dashboard_height
        sys.stdout.write(f"\033[{row};1H")

        # Clear dashboard area
        for _ in range(self._dashboard_height):
            sys.stdout.write("\033[2K\n")  # Clear line and move down

        # Move back to start of dashboard area
        sys.stdout.write(f"\033[{row};1H")

        # Render content
        lines = self._build_dashboard()
        sys.stdout.write("\n".join(lines))

        # Restore cursor position
        sys.stdout.write("\033[u")
        sys.stdout.flush()

    def _build_dashboard(self) -> list:
        """Build dashboard content as list of lines."""
        lines = []

        # Separator
        lines.append("=" * min(80, self._terminal_width))

        # Status line 1: Frame info
        df_current, df_total = self.progress_data['diffusion_frames']
        df_pct = int((df_current / df_total * 100) if df_total > 0 else 0)

        line1 = f"Frame {self.frame_info['current']}/{self.frame_info['total']} [{self.frame_info['type']}]"
        line1 += f" | Progress: {df_pct}%"
        lines.append(line1)

        # Status line 2: Steps
        ts_current, ts_total = self.progress_data['total_steps']
        ts_pct = int((ts_current / ts_total * 100) if ts_total > 0 else 0)
        cs_current, cs_total = self.progress_data['current_step']

        line2 = f"Steps: {ts_pct}% | Current: {cs_current}/{cs_total}"
        line2 += f" | VRAM: {self.memory.free_gpu_mb/1024:.1f}GB"
        lines.append(line2)

        # ASCII preview
        if self.use_ascii_preview and self.last_frame_image is not None:
            ascii_art = image_to_ascii_art(
                self.last_frame_image,
                width=32,
                height=18,
                use_color=(self.theme != 'simple')
            )
            if ascii_art:
                lines.extend(ascii_art.split('\n'))

        return lines

    def _get_terminal_size(self):
        """Get terminal size (height, width)."""
        try:
            size = shutil.get_terminal_size()
            return size.lines, size.columns
        except:
            return 40, 80  # Default fallback

    def _move_cursor_below_dashboard(self):
        """Move cursor to below dashboard area."""
        row = self._terminal_height + 1
        sys.stdout.write(f"\033[{row};1H")
        sys.stdout.flush()

    def add_log(self, message: str):
        """Add log message (just pass through)."""
        # Move cursor above dashboard, print, restore position
        self._move_cursor_above_dashboard()
        print(message)
        self._render()  # Refresh dashboard after log

    def _move_cursor_above_dashboard(self):
        """Move cursor to line above dashboard."""
        row = self._terminal_height - self._dashboard_height - 1
        sys.stdout.write(f"\033[{row};1H")
        sys.stdout.flush()

    def add_ascii_art_to_log(self, image, frame_idx: int):
        """Add ASCII art to log (disabled for now)."""
        pass
