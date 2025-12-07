"""Specialized dashboard for Flux+Interpolation rendering mode.

Tracks two phases separately:
- Phase 1: Keyframe generation (diffusion)
- Phase 2: Interpolation (Wan FLF2V / FILM / DA3-3DGS)

Uses DA3 slopcore palette and respects global emoji settings.
"""

import sys
import time
import atexit
import shutil
from typing import Optional

from deforum.rendering import options as opt_utils
from deforum.utils.system.logging import emoji_if_enabled
from deforum.utils.system.logging.themes import (
    HEX_DA3_CYAN,
    HEX_DA3_RED
)


class InterpolationDashboard:
    """Simplified dashboard for Flux+Interpolation mode.

    Shows:
    - Current phase (1: Keyframes, 2: Interpolation)
    - Phase progress (X/Y)
    - VRAM usage
    - Current operation
    """

    def __init__(self):
        """Initialize interpolation dashboard."""
        self._is_active = False
        self._dashboard_height = 7  # Fixed height: header + phase1 + phase2 + vram + operation + footer
        self._last_update_time = 0
        self._update_interval = 0.1  # Update every 100ms max
        self._terminal_width = 0

        # Phase tracking
        self.phase1_current = 0
        self.phase1_total = 0
        self.phase2_current = 0
        self.phase2_total = 0
        self.current_phase = 1
        self.current_operation = ""

        # VRAM tracking
        self.vram_used_gb = 0.0
        self.vram_total_gb = 0.0
        self.vram_free_gb = 0.0

        # DA3 slopcore colors (cyan → watermelon gradient)
        self._cyan_rgb = self._hex_to_rgb(HEX_DA3_CYAN)    # #1CC4E6
        self._red_rgb = self._hex_to_rgb(HEX_DA3_RED)      # #F64A5E

        # Register cleanup
        atexit.register(self._cleanup_terminal)

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _get_terminal_width(self) -> int:
        """Get current terminal width."""
        return shutil.get_terminal_size((80, 24))[0]

    def start(self):
        """Start the dashboard."""
        self._is_active = True
        self._reserve_space()

    def stop(self):
        """Stop the dashboard and restore terminal."""
        if not self._is_active:
            return

        self._is_active = False
        self._cleanup_terminal()

    def _reserve_space(self):
        """Reserve space at bottom of terminal for dashboard."""
        # Print empty lines to push logs up
        print("\n" * (self._dashboard_height + 1), flush=True)

    def _cleanup_terminal(self):
        """Clean up terminal state on exit."""
        if not self._is_active:
            return

        # Clear dashboard area
        sys.stdout.write("\033[?25h")  # Show cursor
        sys.stdout.write(f"\033[{self._dashboard_height}A")  # Move up
        sys.stdout.write("\033[J")  # Clear from cursor to end
        sys.stdout.flush()

    def _render_dashboard(self):
        """Render the dashboard at bottom of terminal."""
        if not self._is_active:
            return

        # Throttle updates
        now = time.time()
        if now - self._last_update_time < self._update_interval:
            return
        self._last_update_time = now

        # Get terminal width for full-width display
        self._terminal_width = self._get_terminal_width()
        content_width = self._terminal_width - 4  # Account for │ borders

        # Build dashboard content
        lines = []

        # Header with DA3 cyan gradient
        phase_name = "Phase 1: Keyframes" if self.current_phase == 1 else "Phase 2: Interpolation"
        r, g, b = self._cyan_rgb
        header_color = f"\033[38;2;{r};{g};{b}m"
        reset = "\033[0m"
        header_text = f"╭─ {phase_name} "
        header_fill = "─" * (self._terminal_width - len(header_text) - 1)
        lines.append(f"{header_color}{header_text}{header_fill}╮{reset}")

        # Phase 1 progress
        p1_pct = (self.phase1_current / self.phase1_total * 100) if self.phase1_total > 0 else 0
        bar_width = max(30, self._terminal_width - 60)  # Adaptive bar width
        p1_bar = self._progress_bar(self.phase1_current, self.phase1_total, width=bar_width)
        check = emoji_if_enabled('✓') or '✓'
        dot = emoji_if_enabled('●') or '●'
        p1_status = check if self.phase1_current == self.phase1_total and self.phase1_total > 0 else dot
        p1_line = f"│ {p1_status} Phase 1: {p1_bar} {self.phase1_current:3d}/{self.phase1_total:<3d} ({p1_pct:5.1f}%)"
        p1_padding = " " * (self._terminal_width - len(p1_line.replace('\033[92m', '').replace('\033[93m', '').replace('\033[91m', '').replace('\033[0m', '')) - 1)
        lines.append(f"{p1_line}{p1_padding}│")

        # Phase 2 progress
        p2_pct = (self.phase2_current / self.phase2_total * 100) if self.phase2_total > 0 else 0
        p2_bar = self._progress_bar(self.phase2_current, self.phase2_total, width=bar_width)
        p2_status = check if self.phase2_current == self.phase2_total and self.phase2_total > 0 else dot
        p2_line = f"│ {p2_status} Phase 2: {p2_bar} {self.phase2_current:3d}/{self.phase2_total:<3d} ({p2_pct:5.1f}%)"
        p2_padding = " " * (self._terminal_width - len(p2_line.replace('\033[92m', '').replace('\033[93m', '').replace('\033[91m', '').replace('\033[0m', '')) - 1)
        lines.append(f"{p2_line}{p2_padding}│")

        # VRAM
        vram_pct = (self.vram_used_gb / self.vram_total_gb * 100) if self.vram_total_gb > 0 else 0
        vram_bar = self._vram_bar(self.vram_used_gb, self.vram_total_gb, width=bar_width)
        gpu = emoji_if_enabled('🎮') or 'GPU'
        vram_line = f"│ {gpu} VRAM:    {vram_bar} {self.vram_used_gb:5.2f}/{self.vram_total_gb:5.2f} GB ({vram_pct:5.1f}%)"
        vram_padding = " " * (self._terminal_width - len(vram_line.replace('\033[92m', '').replace('\033[93m', '').replace('\033[91m', '').replace('\033[0m', '')) - 1)
        lines.append(f"{vram_line}{vram_padding}│")

        # Current operation
        gear = emoji_if_enabled('⚙') or '>'
        op_text = self.current_operation if self.current_operation else "Idle"
        max_op_len = self._terminal_width - 6  # Account for "│ ⚙  │"
        op_truncated = op_text[:max_op_len] if len(op_text) > max_op_len else op_text
        op_padding = " " * (max_op_len - len(op_truncated))
        lines.append(f"│ {gear} {op_truncated}{op_padding} │")

        # Footer
        footer_fill = "─" * (self._terminal_width - 2)
        lines.append(f"╰{footer_fill}╯")

        # Move cursor up and render
        dashboard_text = "\n".join(lines)
        sys.stdout.write(f"\033[{self._dashboard_height}A")  # Move up
        sys.stdout.write("\033[J")  # Clear from cursor down
        sys.stdout.write(dashboard_text)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """Generate ASCII progress bar with DA3 gradient (cyan → red)."""
        if total == 0:
            return f"[{'░' * width}]"

        pct = current / total
        filled = int(width * pct)

        # Interpolate between cyan and red based on progress
        r = int(self._cyan_rgb[0] + (self._red_rgb[0] - self._cyan_rgb[0]) * pct)
        g = int(self._cyan_rgb[1] + (self._red_rgb[1] - self._cyan_rgb[1]) * pct)
        b = int(self._cyan_rgb[2] + (self._red_rgb[2] - self._cyan_rgb[2]) * pct)

        color = f"\033[38;2;{r};{g};{b}m"
        reset = "\033[0m"

        bar = color + "█" * filled + reset + "░" * (width - filled)
        return f"[{bar}]"

    def _vram_bar(self, used: float, total: float, width: int = 30) -> str:
        """Generate VRAM usage bar with color."""
        if total == 0:
            return f"[{' ' * width}]"

        pct = used / total
        filled = int(width * pct)

        # Color based on usage
        if pct > 0.9:
            color = "\033[91m"  # Red
        elif pct > 0.7:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[92m"  # Green

        bar = color + "█" * filled + "\033[0m" + "░" * (width - filled)
        return f"[{bar}]"

    def update_phase1(self, current: int, total: int):
        """Update Phase 1 progress."""
        self.phase1_current = current
        self.phase1_total = total
        self.current_phase = 1
        self._render_dashboard()

    def update_phase2(self, current: int, total: int):
        """Update Phase 2 progress."""
        self.phase2_current = current
        self.phase2_total = total
        self.current_phase = 2
        self._render_dashboard()

    def update_vram(self, used_gb: float, total_gb: float):
        """Update VRAM stats."""
        self.vram_used_gb = used_gb
        self.vram_total_gb = total_gb
        self.vram_free_gb = total_gb - used_gb
        self._render_dashboard()

    def set_operation(self, operation: str):
        """Set current operation text."""
        self.current_operation = operation
        self._render_dashboard()

    def update_vram_from_torch(self):
        """Update VRAM from PyTorch if available."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                props = torch.cuda.get_device_properties(0)
                total = props.total_memory / 1024**3

                # Use reserved (allocated + cached) as "used"
                self.update_vram(reserved, total)
        except Exception:
            pass
