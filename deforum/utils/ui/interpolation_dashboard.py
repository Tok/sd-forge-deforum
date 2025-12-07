"""Specialized dashboard for Flux+Interpolation rendering mode.

Tracks two phases separately:
- Phase 1: Keyframe generation (diffusion)
- Phase 2: Interpolation (Wan FLF2V / FILM / DA3-3DGS)

Simpler than the full FixedDashboard - focuses on phase progress and VRAM.
"""

import sys
import time
import atexit
from typing import Optional

from deforum.rendering import options as opt_utils


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
        self._dashboard_height = 6  # Fixed height: header + phase1 + phase2 + vram + separator
        self._last_update_time = 0
        self._update_interval = 0.1  # Update every 100ms max

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

        # Register cleanup
        atexit.register(self._cleanup_terminal)

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

        # Build dashboard content
        lines = []

        # Header
        phase_name = "Phase 1: Keyframes" if self.current_phase == 1 else "Phase 2: Interpolation"
        lines.append(f"╭─ {phase_name} {'─' * 50}╮")

        # Phase 1 progress
        p1_pct = (self.phase1_current / self.phase1_total * 100) if self.phase1_total > 0 else 0
        p1_bar = self._progress_bar(self.phase1_current, self.phase1_total, width=30)
        p1_status = "✓" if self.phase1_current == self.phase1_total and self.phase1_total > 0 else "●"
        lines.append(f"│ {p1_status} Phase 1: {p1_bar} {self.phase1_current:3d}/{self.phase1_total:<3d} ({p1_pct:5.1f}%)")

        # Phase 2 progress
        p2_pct = (self.phase2_current / self.phase2_total * 100) if self.phase2_total > 0 else 0
        p2_bar = self._progress_bar(self.phase2_current, self.phase2_total, width=30)
        p2_status = "✓" if self.phase2_current == self.phase2_total and self.phase2_total > 0 else "●"
        lines.append(f"│ {p2_status} Phase 2: {p2_bar} {self.phase2_current:3d}/{self.phase2_total:<3d} ({p2_pct:5.1f}%)")

        # VRAM
        vram_pct = (self.vram_used_gb / self.vram_total_gb * 100) if self.vram_total_gb > 0 else 0
        vram_bar = self._vram_bar(self.vram_used_gb, self.vram_total_gb, width=30)
        lines.append(f"│ 🎮 VRAM:    {vram_bar} {self.vram_used_gb:5.2f}/{self.vram_total_gb:5.2f} GB ({vram_pct:5.1f}%)")

        # Current operation
        op_text = self.current_operation[:60] if self.current_operation else "Idle"
        lines.append(f"│ ⚙ {op_text:<60} │")

        # Footer
        lines.append(f"╰{'─' * 65}╯")

        # Move cursor up and render
        dashboard_text = "\n".join(lines)
        sys.stdout.write(f"\033[{self._dashboard_height}A")  # Move up
        sys.stdout.write("\033[J")  # Clear from cursor down
        sys.stdout.write(dashboard_text)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """Generate ASCII progress bar."""
        if total == 0:
            return f"[{' ' * width}]"

        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
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
