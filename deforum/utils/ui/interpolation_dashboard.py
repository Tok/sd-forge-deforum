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
        self._dashboard_height = 9  # Fixed height: header + 4 progress bars + vram + operation + footer
        self._last_update_time = 0
        self._update_interval = 0.1  # Update every 100ms max
        self._terminal_width = 0
        self._terminal_height = 0

        # Phase 1 (Diffusion Keyframes) tracking
        self.phase1_current = 0
        self.phase1_total = 0

        # Phase 2 (3DGS) sub-stages tracking
        self.phase2_3dgs_build_current = 0
        self.phase2_3dgs_build_total = 0
        self.phase2_3dgs_keyframes_current = 0
        self.phase2_3dgs_keyframes_total = 0
        self.phase2_3dgs_tweens_current = 0
        self.phase2_3dgs_tweens_total = 0

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

    def _get_terminal_size(self) -> tuple:
        """Get current terminal size (height, width)."""
        size = shutil.get_terminal_size((80, 24))
        return (size[1], size[0])  # (height, width)

    def _get_terminal_width(self) -> int:
        """Get current terminal width."""
        return shutil.get_terminal_size((80, 24))[0]

    def _strip_ansi(self, text: str) -> str:
        """Strip ANSI escape codes from text for length calculation."""
        import re
        ansi_pattern = re.compile(r'\033\[[0-9;]*m')
        return ansi_pattern.sub('', text)

    def start(self):
        """Start dashboard - set up scrolling region and fixed dashboard."""
        # Get terminal size
        self._terminal_height, self._terminal_width = self._get_terminal_size()

        # Set up scrolling region (reserve bottom lines for dashboard)
        # ANSI: \033[{top};{bottom}r sets scrolling region
        scroll_bottom = self._terminal_height - self._dashboard_height
        sys.stdout.write(f"\033[1;{scroll_bottom}r")

        # Move cursor to top of scrolling region
        sys.stdout.write("\033[1;1H")
        sys.stdout.flush()

        # Mark as active
        self._is_active = True

        # Initial render
        self._render_dashboard()

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

        # Save cursor position
        sys.stdout.write("\033[s")

        # Calculate dashboard start row (bottom of terminal)
        dashboard_start = self._terminal_height - self._dashboard_height + 1

        # Move to dashboard area and clear it
        for i in range(self._dashboard_height):
            row = dashboard_start + i
            sys.stdout.write(f"\033[{row};1H")  # Move to row
            sys.stdout.write("\033[K")  # Clear line

        # Reset scrolling region to full terminal
        sys.stdout.write("\033[r")

        # Restore cursor position and show it
        sys.stdout.write("\033[u")
        sys.stdout.write("\033[?25h")
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
        phase_name = "Diffusion Keyframes" if self.current_phase == 1 else "Gaussian Splats (3DGS)"
        r, g, b = self._cyan_rgb
        header_color = f"\033[38;2;{r};{g};{b}m"
        reset = "\033[0m"
        header_text = f"╭─ {phase_name} "
        header_fill = "─" * (self._terminal_width - len(header_text) - 1)
        lines.append(f"{header_color}{header_text}{header_fill}╮{reset}")

        # Full-width progress bars (leave space for label, status, counters, percentage, units)
        # Space breakdown: "│ ● " (4) + label (21) + " " (1) + "[]" (2) + " XXX/XXX (100.0%) keyframes │" (~30)
        bar_width = max(40, self._terminal_width - 60)
        check = emoji_if_enabled('✓') or '✓'
        dot = emoji_if_enabled('●') or '●'

        # 1. Phase 1: Diffusion Keyframes (cyan start)
        p1_pct = (self.phase1_current / self.phase1_total * 100) if self.phase1_total > 0 else 0
        p1_bar = self._progress_bar(self.phase1_current, self.phase1_total, width=bar_width, phase=1)
        p1_status = check if self.phase1_current == self.phase1_total and self.phase1_total > 0 else dot
        p1_line = f"│ {p1_status} Diffusion Keyframes: {p1_bar} {self.phase1_current:3d}/{self.phase1_total:<3d} ({p1_pct:5.1f}%) keyframes"
        p1_padding = " " * (self._terminal_width - len(self._strip_ansi(p1_line)) - 1)
        lines.append(f"{p1_line}{p1_padding}│")

        # 2. Phase 2a: 3DGS Scene Build (slightly redder)
        p2a_pct = (self.phase2_3dgs_build_current / self.phase2_3dgs_build_total * 100) if self.phase2_3dgs_build_total > 0 else 0
        p2a_bar = self._progress_bar(self.phase2_3dgs_build_current, self.phase2_3dgs_build_total, width=bar_width, phase=2)
        p2a_status = check if self.phase2_3dgs_build_current == self.phase2_3dgs_build_total and self.phase2_3dgs_build_total > 0 else dot
        p2a_line = f"│ {p2a_status} 3DGS Scene Build:    {p2a_bar} {self.phase2_3dgs_build_current:3d}/{self.phase2_3dgs_build_total:<3d} ({p2a_pct:5.1f}%) keyframes"
        p2a_padding = " " * (self._terminal_width - len(self._strip_ansi(p2a_line)) - 1)
        lines.append(f"{p2a_line}{p2a_padding}│")

        # 3. Phase 2b: 3DGS Keyframes (redder)
        p2b_pct = (self.phase2_3dgs_keyframes_current / self.phase2_3dgs_keyframes_total * 100) if self.phase2_3dgs_keyframes_total > 0 else 0
        p2b_bar = self._progress_bar(self.phase2_3dgs_keyframes_current, self.phase2_3dgs_keyframes_total, width=bar_width, phase=3)
        p2b_status = check if self.phase2_3dgs_keyframes_current == self.phase2_3dgs_keyframes_total and self.phase2_3dgs_keyframes_total > 0 else dot
        p2b_line = f"│ {p2b_status} 3DGS Keyframes:      {p2b_bar} {self.phase2_3dgs_keyframes_current:3d}/{self.phase2_3dgs_keyframes_total:<3d} ({p2b_pct:5.1f}%) gs-frames"
        p2b_padding = " " * (self._terminal_width - len(self._strip_ansi(p2b_line)) - 1)
        lines.append(f"{p2b_line}{p2b_padding}│")

        # 4. Phase 2c: 3DGS Tweens (reddest/watermelon)
        p2c_pct = (self.phase2_3dgs_tweens_current / self.phase2_3dgs_tweens_total * 100) if self.phase2_3dgs_tweens_total > 0 else 0
        p2c_bar = self._progress_bar(self.phase2_3dgs_tweens_current, self.phase2_3dgs_tweens_total, width=bar_width, phase=4)
        p2c_status = check if self.phase2_3dgs_tweens_current == self.phase2_3dgs_tweens_total and self.phase2_3dgs_tweens_total > 0 else dot
        p2c_line = f"│ {p2c_status} 3DGS Tweens:         {p2c_bar} {self.phase2_3dgs_tweens_current:3d}/{self.phase2_3dgs_tweens_total:<3d} ({p2c_pct:5.1f}%) gs-frames"
        p2c_padding = " " * (self._terminal_width - len(self._strip_ansi(p2c_line)) - 1)
        lines.append(f"{p2c_line}{p2c_padding}│")

        # VRAM (right-aligned, smaller bar)
        vram_pct = (self.vram_used_gb / self.vram_total_gb * 100) if self.vram_total_gb > 0 else 0
        vram_bar_width = max(20, int(bar_width * 0.4))
        vram_bar = self._vram_bar(self.vram_used_gb, self.vram_total_gb, width=vram_bar_width)
        gpu = emoji_if_enabled('🎮') or 'GPU'
        vram_text = f"{self.vram_used_gb:5.2f}/{self.vram_total_gb:5.2f} GB ({vram_pct:5.1f}%)"
        # Right-align VRAM
        vram_content = f"{gpu} VRAM: {vram_bar} {vram_text}"
        vram_left_padding = " " * max(0, self._terminal_width - len(self._strip_ansi(vram_content)) - 4)
        lines.append(f"│{vram_left_padding} {vram_content} │")

        # Current operation
        gear = emoji_if_enabled('⚙') or '>'
        op_text = self.current_operation if self.current_operation else "Idle"
        max_op_len = self._terminal_width - 6
        op_truncated = op_text[:max_op_len] if len(op_text) > max_op_len else op_text
        op_padding = " " * (max_op_len - len(op_truncated))
        lines.append(f"│ {gear} {op_truncated}{op_padding} │")

        # Footer
        footer_fill = "─" * (self._terminal_width - 2)
        lines.append(f"╰{footer_fill}╯")

        # Save cursor position
        sys.stdout.write("\033[s")

        # Calculate dashboard start row (absolute position at bottom of terminal)
        dashboard_start = self._terminal_height - self._dashboard_height + 1

        # Render each line at its absolute position
        for idx, line in enumerate(lines):
            row = dashboard_start + idx
            sys.stdout.write(f"\033[{row};1H")  # Move to absolute row position
            sys.stdout.write("\033[K")  # Clear line
            sys.stdout.write(line)  # Write dashboard line

        # Restore cursor position
        sys.stdout.write("\033[u")
        sys.stdout.flush()

    def _progress_bar(self, current: int, total: int, width: int = 30, phase: int = 1) -> str:
        """Generate ASCII progress bar with DA3 gradient (cyan top → watermelon bottom).

        Args:
            current: Current progress value
            total: Total progress value
            width: Width of the progress bar
            phase: Which phase (1=cyan, 2=light red, 3=medium red, 4=watermelon)
        """
        if total == 0:
            return f"[{'░' * width}]"

        pct = current / total
        filled = int(width * pct)

        # Gradient from cyan (phase 1) → watermelon (phase 4)
        # phase 1: pure cyan
        # phase 2: 33% toward red
        # phase 3: 66% toward red
        # phase 4: pure red (watermelon)
        phase_pct = (phase - 1) / 3.0  # Normalize to 0.0-1.0

        r = int(self._cyan_rgb[0] + (self._red_rgb[0] - self._cyan_rgb[0]) * phase_pct)
        g = int(self._cyan_rgb[1] + (self._red_rgb[1] - self._cyan_rgb[1]) * phase_pct)
        b = int(self._cyan_rgb[2] + (self._red_rgb[2] - self._cyan_rgb[2]) * phase_pct)

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
        """Update Phase 1 (Diffusion Keyframes) progress."""
        self.phase1_current = current
        self.phase1_total = total
        self.current_phase = 1
        self._render_dashboard()

    def update_3dgs_build(self, current: int, total: int):
        """Update Phase 2a (3DGS Scene Build) progress."""
        self.phase2_3dgs_build_current = current
        self.phase2_3dgs_build_total = total
        self.current_phase = 2
        self._render_dashboard()

    def update_3dgs_keyframes(self, current: int, total: int):
        """Update Phase 2b (3DGS Keyframes) progress."""
        self.phase2_3dgs_keyframes_current = current
        self.phase2_3dgs_keyframes_total = total
        self.current_phase = 2
        self._render_dashboard()

    def update_3dgs_tweens(self, current: int, total: int):
        """Update Phase 2c (3DGS Tweens) progress."""
        self.phase2_3dgs_tweens_current = current
        self.phase2_3dgs_tweens_total = total
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
