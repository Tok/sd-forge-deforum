"""Interpolation Dashboard Renderer - Refactored

Refactored from InterpolationDashboard._render_dashboard (104 lines, complexity D-24)
into modular pure functions eliminating code duplication.

Original: deforum/utils/ui/interpolation_dashboard.py:148-251
"""

from typing import List, Tuple, NamedTuple
from dataclasses import dataclass
import sys


# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class ProgressBarData:
    """Data for a single progress bar."""
    label: str
    current: int
    total: int
    units: str  # "keyframes" or "gs-frames"
    phase: int  # For gradient color


@dataclass(frozen=True)
class DashboardMetrics:
    """All metrics for dashboard rendering."""
    phase1_current: int
    phase1_total: int
    phase2a_current: int
    phase2a_total: int
    phase2b_current: int
    phase2b_total: int
    phase2c_current: int
    phase2c_total: int
    vram_used: float
    vram_total: float
    current_operation: str


class RenderConfig(NamedTuple):
    """Rendering configuration."""
    terminal_width: int
    terminal_height: int
    dashboard_height: int
    bar_width: int


# ============================================================================
# Text Utilities
# ============================================================================

def strip_ansi_codes(text: str) -> str:
    """Strip ANSI escape codes from text."""
    import re
    return re.sub(r'\033\[[0-9;]*m', '', text)


def calculate_percentage(current: int, total: int) -> float:
    """Calculate percentage with zero-division safety."""
    return (current / total * 100) if total > 0 else 0.0


def is_complete(current: int, total: int) -> bool:
    """Check if progress is complete."""
    return current == total and total > 0


# ============================================================================
# Progress Bar Rendering
# ============================================================================

def render_progress_line(
    data: ProgressBarData,
    bar_rendered: str,
    check_icon: str,
    dot_icon: str,
    terminal_width: int,
    strip_ansi_fn
) -> str:
    """Render a single progress bar line with all components.

    Args:
        data: Progress bar data
        bar_rendered: Pre-rendered progress bar
        check_icon: Icon for completed state
        dot_icon: Icon for in-progress state
        terminal_width: Terminal width for padding
        strip_ansi_fn: Function to strip ANSI codes

    Returns:
        Complete line with borders and padding
    """
    pct = calculate_percentage(data.current, data.total)
    status_icon = check_icon if is_complete(data.current, data.total) else dot_icon

    line = (
        f"│ {status_icon} {data.label:20s} {bar_rendered} "
        f"{data.current:3d}/{data.total:<3d} ({pct:5.1f}%) {data.units}"
    )

    padding = " " * (terminal_width - len(strip_ansi_fn(line)) - 1)
    return f"{line}{padding}│"


def render_vram_line(
    vram_used: float,
    vram_total: float,
    vram_bar: str,
    gpu_icon: str,
    terminal_width: int,
    strip_ansi_fn
) -> str:
    """Render VRAM usage line (right-aligned)."""
    pct = calculate_percentage(int(vram_used * 100), int(vram_total * 100))
    vram_text = f"{vram_used:5.2f}/{vram_total:5.2f} GB ({pct:5.1f}%)"

    content = f"{gpu_icon} VRAM: {vram_bar} {vram_text}"
    left_padding = " " * max(0, terminal_width - len(strip_ansi_fn(content)) - 4)

    return f"│{left_padding} {content} │"


def render_operation_line(
    operation: str,
    gear_icon: str,
    terminal_width: int
) -> str:
    """Render current operation line."""
    op_text = operation if operation else "Idle"
    max_op_len = terminal_width - 6
    op_truncated = op_text[:max_op_len] if len(op_text) > max_op_len else op_text
    op_padding = " " * (max_op_len - len(op_truncated))

    return f"│ {gear_icon} {op_truncated}{op_padding} │"


# ============================================================================
# Border Rendering
# ============================================================================

def render_header(phase_name: str, cyan_rgb: Tuple[int, int, int], terminal_width: int) -> str:
    """Render header with gradient."""
    r, g, b = cyan_rgb
    header_color = f"\033[38;2;{r};{g};{b}m"
    reset = "\033[0m"

    header_text = f"╭─ {phase_name} "
    header_fill = "─" * (terminal_width - len(header_text) - 1)

    return f"{header_color}{header_text}{header_fill}╮{reset}"


def render_footer(terminal_width: int) -> str:
    """Render footer border."""
    footer_fill = "─" * (terminal_width - 2)
    return f"╰{footer_fill}╯"


# ============================================================================
# Dashboard Content Building
# ============================================================================

def build_progress_bars_data(metrics: DashboardMetrics) -> List[ProgressBarData]:
    """Build list of progress bar data from metrics."""
    return [
        ProgressBarData(
            label="Diffusion Keyframes:",
            current=metrics.phase1_current,
            total=metrics.phase1_total,
            units="keyframes",
            phase=1
        ),
        ProgressBarData(
            label="3DGS Scene Build:",
            current=metrics.phase2a_current,
            total=metrics.phase2a_total,
            units="keyframes",
            phase=2
        ),
        ProgressBarData(
            label="3DGS Keyframes:",
            current=metrics.phase2b_current,
            total=metrics.phase2b_total,
            units="gs-frames",
            phase=3
        ),
        ProgressBarData(
            label="3DGS Tweens:",
            current=metrics.phase2c_current,
            total=metrics.phase2c_total,
            units="gs-frames",
            phase=4
        ),
    ]


def build_dashboard_lines(
    metrics: DashboardMetrics,
    config: RenderConfig,
    phase_name: str,
    cyan_rgb: Tuple[int, int, int],
    icons: dict,
    progress_bar_fn,
    vram_bar_fn
) -> List[str]:
    """Build all dashboard content lines.

    Args:
        metrics: Dashboard metrics
        config: Render configuration
        phase_name: Current phase name
        cyan_rgb: RGB tuple for header color
        icons: Dict with 'check', 'dot', 'gpu', 'gear' icons
        progress_bar_fn: Function to render progress bar
        vram_bar_fn: Function to render VRAM bar

    Returns:
        List of rendered lines
    """
    lines = []

    # Header
    lines.append(render_header(phase_name, cyan_rgb, config.terminal_width))

    # Progress bars
    progress_data = build_progress_bars_data(metrics)

    for data in progress_data:
        bar_rendered = progress_bar_fn(data.current, data.total, config.bar_width, data.phase)
        line = render_progress_line(
            data, bar_rendered, icons['check'], icons['dot'],
            config.terminal_width, strip_ansi_codes
        )
        lines.append(line)

    # VRAM bar
    vram_bar_width = max(20, int(config.bar_width * 0.4))
    vram_bar = vram_bar_fn(metrics.vram_used, metrics.vram_total, vram_bar_width)
    lines.append(render_vram_line(
        metrics.vram_used, metrics.vram_total, vram_bar,
        icons['gpu'], config.terminal_width, strip_ansi_codes
    ))

    # Current operation
    lines.append(render_operation_line(
        metrics.current_operation, icons['gear'], config.terminal_width
    ))

    # Footer
    lines.append(render_footer(config.terminal_width))

    return lines


# ============================================================================
# Terminal Positioning
# ============================================================================

def write_dashboard_to_terminal(
    lines: List[str],
    terminal_height: int,
    dashboard_height: int
):
    """Write dashboard lines to terminal at bottom position.

    Args:
        lines: Rendered dashboard lines
        terminal_height: Total terminal height
        dashboard_height: Height of dashboard
    """
    # Save cursor position
    sys.stdout.write("\033[s")

    # Calculate dashboard start row
    dashboard_start = terminal_height - dashboard_height + 1

    # Render each line at absolute position
    for idx, line in enumerate(lines):
        row = dashboard_start + idx
        sys.stdout.write(f"\033[{row};1H")  # Move to position
        sys.stdout.write("\033[K")          # Clear line
        sys.stdout.write(line)              # Write content

    # Restore cursor position
    sys.stdout.write("\033[u")
    sys.stdout.flush()


# ============================================================================
# Main Render Function (for use in InterpolationDashboard class)
# ============================================================================

def render_dashboard(
    metrics: DashboardMetrics,
    config: RenderConfig,
    current_phase: int,
    cyan_rgb: Tuple[int, int, int],
    icon_functions: dict,
    progress_bar_fn,
    vram_bar_fn
):
    """Render interpolation dashboard.

    Refactored version with complexity ≤10.

    This is designed to be called from InterpolationDashboard._render_dashboard()
    after extracting the necessary data.

    Args:
        metrics: All dashboard metrics
        config: Rendering configuration
        current_phase: Current phase number (1 or 2)
        cyan_rgb: RGB tuple for DA3 cyan color
        icon_functions: Dict of icon getter functions
        progress_bar_fn: Function to render progress bars
        vram_bar_fn: Function to render VRAM bar
    """
    # Determine phase name
    phase_name = "Diffusion Keyframes" if current_phase == 1 else "Gaussian Splats (3DGS)"

    # Get icons
    icons = {
        'check': icon_functions['check'](),
        'dot': icon_functions['dot'](),
        'gpu': icon_functions['gpu'](),
        'gear': icon_functions['gear']()
    }

    # Build dashboard content
    lines = build_dashboard_lines(
        metrics, config, phase_name, cyan_rgb, icons,
        progress_bar_fn, vram_bar_fn
    )

    # Write to terminal
    write_dashboard_to_terminal(lines, config.terminal_height, config.dashboard_height)
