"""Startup Banner Renderer - Refactored

Refactored from print_startup_banner (336 lines, complexity E-34)
into modular pure functions following strict functional programming principles.

Original: deforum/utils/system/startup_banner.py:4-336
"""

from typing import List, Tuple, NamedTuple
from dataclasses import dataclass
import unicodedata
import shutil


# ============================================================================
# Constants
# ============================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
WHITE = "\033[97m"
TERMINAL_BLACK_BG = "\033[48;2;30;30;30m"
CORNER_COLOR = "\033[38;2;30;30;30m"

# Rounded corner characters
ROUND_TL = "◤"
ROUND_TR = "◥"
ROUND_BL = "◣"
ROUND_BR = "◢"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class BannerConfig:
    """Configuration for banner rendering."""
    fork_name: str
    github_url: str
    commit_id: str
    gradient_variant: str
    box_width: int
    use_vertical_gradient: bool


@dataclass(frozen=True)
class GradientColors:
    """Gradient color palette."""
    colors: List[str]  # Hex colors


class TerminalDimensions(NamedTuple):
    """Terminal size information."""
    width: int
    height: int


# ============================================================================
# Version & Config
# ============================================================================

def get_commit_id() -> str:
    """Get Deforum commit ID with fallback."""
    try:
        from deforum.utils.general import get_deforum_version
        return get_deforum_version()
    except Exception:
        return "Unknown"


def get_terminal_dimensions() -> TerminalDimensions:
    """Get terminal size with fallback."""
    size = shutil.get_terminal_size((120, 24))
    return TerminalDimensions(width=size.columns, height=size.rows)


def calculate_box_width(term_width: int, max_width: int = 128) -> int:
    """Calculate box width based on terminal size."""
    return min(term_width - 4, max_width)


# ============================================================================
# Text Display Width
# ============================================================================

def get_char_display_width(char: str) -> int:
    """Get display width of a single character (emojis = 2)."""
    if unicodedata.east_asian_width(char) in ('F', 'W'):
        return 2
    return 1


def calculate_display_width(text: str) -> int:
    """Calculate actual terminal display width."""
    return sum(get_char_display_width(char) for char in text)


# ============================================================================
# Color Conversion
# ============================================================================

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    )


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color."""
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_bg_ansi(hex_color: str) -> str:
    """Convert hex color to ANSI background escape code."""
    r, g, b = hex_to_rgb(hex_color)
    return f"\033[48;2;{r};{g};{b}m"


def hex_to_fg_ansi(hex_color: str) -> str:
    """Convert hex color to ANSI foreground escape code."""
    r, g, b = hex_to_rgb(hex_color)
    return f"\033[38;2;{r};{g};{b}m"


# ============================================================================
# Color Interpolation
# ============================================================================

def interpolate_color(hex1: str, hex2: str, ratio: float) -> str:
    """Interpolate between two hex colors."""
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)

    r = int(r1 + (r2 - r1) * ratio)
    g = int(g1 + (g2 - g1) * ratio)
    b = int(b1 + (b2 - b1) * ratio)

    return rgb_to_hex(r, g, b)


def generate_smooth_gradient(base_colors: List[str]) -> List[str]:
    """Generate smooth gradient by interpolating between base colors."""
    gradient = []
    for i in range(len(base_colors) - 1):
        gradient.append(base_colors[i])
        mid_color = interpolate_color(base_colors[i], base_colors[i + 1], 0.5)
        gradient.append(mid_color)
    gradient.append(base_colors[-1])
    return gradient


def get_color_by_position(gradient: List[str], position: float) -> str:
    """Get gradient color at normalized position (0.0-1.0)."""
    idx = min(int(position * len(gradient)), len(gradient) - 1)
    return gradient[idx]


# ============================================================================
# Text Formatting
# ============================================================================

def create_title_text(fork_name: str) -> str:
    """Create formatted title text with fade decorations."""
    return f"▓▓▒▒░░ ⚡ {fork_name} ⚡ ░░▒▒▓▓"


def center_text(text: str, available_width: int) -> str:
    """Center text within available width."""
    text_width = calculate_display_width(text)
    if text_width >= available_width:
        return text

    left_pad_spaces = (available_width - text_width) // 2
    return (" " * left_pad_spaces) + text


def strip_ansi_codes(text: str) -> str:
    """Strip ANSI escape codes from text."""
    import re
    return re.sub(r'\033\[[0-9;]*m', '', text)


# ============================================================================
# Content Lines Generation
# ============================================================================

def create_content_lines(fork_name: str, github_url: str, commit_id: str) -> List[str]:
    """Create banner content lines."""
    title = create_title_text(fork_name)

    return [
        title,
        "",
        "RECOMMENDED: Dedicated/Isolated Forge Neo Instance",
        "Why: Hijacks Forge output pipeline (custom dashboard, progress bars, suppressed logs)",
        "     Patches sigma timesteps for continuous 1% strength resolution (vs 5% at 20 steps)",
        "     May conflict with other extensions - optimized specifically for Deforum workflows",
        "",
        "Forge Neo: https://github.com/Haoming02/sd-webui-forge-classic/tree/neo",
        f"This Fork: {github_url} (commit: {commit_id})",
        "",
        "Primary Target: Forge Neo (fully tested and supported)",
        "Other Forge Versions: May work but remain untested"
    ]


# ============================================================================
# Gradient Position Calculation
# ============================================================================

def calculate_gradient_position(
    char_pos: int,
    row_idx: int,
    box_width: int,
    total_rows: int,
    use_vertical: bool
) -> float:
    """Calculate gradient position (0.0-1.0) based on direction."""
    if use_vertical:
        return row_idx / (total_rows - 1) if total_rows > 1 else 0.0
    return char_pos / (box_width - 1) if box_width > 1 else 0.0


# ============================================================================
# Border Rendering
# ============================================================================

def render_top_border(
    box_width: int,
    gradient: List[str],
    use_vertical: bool,
    total_rows: int
) -> str:
    """Render top border with rounded corners and gradient."""
    row_idx = 0
    line = ""

    for char_pos in range(box_width):
        gradient_pos = calculate_gradient_position(
            char_pos, row_idx, box_width, total_rows, use_vertical
        )
        bg = hex_to_bg_ansi(get_color_by_position(gradient, gradient_pos))

        if char_pos == 0:
            line += f"{bg}{CORNER_COLOR}{ROUND_TL}"
        elif char_pos == box_width - 1:
            line += f"{bg}{CORNER_COLOR}{ROUND_TR}"
        else:
            line += f"{bg} "

    return line + RESET


def render_bottom_border(
    box_width: int,
    gradient: List[str],
    use_vertical: bool,
    total_rows: int
) -> str:
    """Render bottom border with rounded corners and gradient."""
    row_idx = total_rows - 1
    line = ""

    for char_pos in range(box_width):
        gradient_pos = calculate_gradient_position(
            char_pos, row_idx, box_width, total_rows, use_vertical
        )
        bg = hex_to_bg_ansi(get_color_by_position(gradient, gradient_pos))

        if char_pos == 0:
            line += f"{bg}{CORNER_COLOR}{ROUND_BL}"
        elif char_pos == box_width - 1:
            line += f"{bg}{CORNER_COLOR}{ROUND_BR}"
        else:
            line += f"{bg} "

    return line + RESET


# ============================================================================
# Title Row Special Rendering
# ============================================================================

def _is_fade_char(char: str) -> bool:
    """Check if character is a fade shade character."""
    return char in ('▓', '▒', '░')


def _render_title_char(
    char: str,
    gradient: List[str],
    gradient_pos: float,
    inside_black: bool
) -> Tuple[str, bool, int]:
    """Render single title character with special effects.

    Returns: (rendered_char, new_inside_black, fade_increment)
    """
    if _is_fade_char(char):
        fg = hex_to_fg_ansi(get_color_by_position(gradient, gradient_pos))
        return f"{TERMINAL_BLACK_BG}{fg}{char}", inside_black, 1

    if inside_black or char == '⚡':
        if char == '⚡':
            return f"{TERMINAL_BLACK_BG}{char}", inside_black, 0
        if char == ' ':
            return f"{TERMINAL_BLACK_BG} ", inside_black, 0

        bright_color = get_color_by_position(gradient, 1.0)
        fg = hex_to_fg_ansi(bright_color)
        return f"{TERMINAL_BLACK_BG}{fg}{char}", inside_black, 0

    bg = hex_to_bg_ansi(get_color_by_position(gradient, gradient_pos))
    return f"{bg}{WHITE}{char}", inside_black, 0


def render_title_row(
    text: str,
    box_width: int,
    row_idx: int,
    total_rows: int,
    gradient: List[str],
    use_vertical: bool
) -> str:
    """Render title row with special black pill effect."""
    visible_text = strip_ansi_codes(text)
    text_width = calculate_display_width(visible_text)

    line = ""
    display_pos = 0

    # Left padding (2 spaces)
    for _ in range(2):
        grad_pos = calculate_gradient_position(
            display_pos, row_idx, box_width, total_rows, use_vertical
        )
        bg = hex_to_bg_ansi(get_color_by_position(gradient, grad_pos))
        line += f"{bg} "
        display_pos += 1

    # Render text with special effects
    inside_black = False
    fade_count = 0

    for char in visible_text:
        char_width = get_char_display_width(char)
        grad_pos = calculate_gradient_position(
            display_pos, row_idx, box_width, total_rows, use_vertical
        )

        rendered, inside_black, fade_inc = _render_title_char(
            char, gradient, grad_pos, inside_black
        )
        line += rendered

        fade_count += fade_inc
        if fade_count == 6 and not inside_black:
            inside_black = True

        display_pos += char_width

    # Right padding
    right_pad_width = box_width - 2 - text_width
    for _ in range(right_pad_width):
        grad_pos = calculate_gradient_position(
            display_pos, row_idx, box_width, total_rows, use_vertical
        )
        bg = hex_to_bg_ansi(get_color_by_position(gradient, grad_pos))
        line += f"{bg} "
        display_pos += 1

    return line + RESET


# ============================================================================
# Regular Content Row Rendering
# ============================================================================

def render_content_row(
    text: str,
    box_width: int,
    row_idx: int,
    total_rows: int,
    gradient: List[str],
    use_vertical: bool
) -> str:
    """Render regular content row with gradient background."""
    visible_text = strip_ansi_codes(text)
    text_width = calculate_display_width(visible_text)

    line = ""
    display_pos = 0

    # Left padding (2 spaces)
    for _ in range(2):
        grad_pos = calculate_gradient_position(
            display_pos, row_idx, box_width, total_rows, use_vertical
        )
        bg = hex_to_bg_ansi(get_color_by_position(gradient, grad_pos))
        line += f"{bg} "
        display_pos += 1

    # Render text
    for char in visible_text:
        char_width = get_char_display_width(char)
        grad_pos = calculate_gradient_position(
            display_pos, row_idx, box_width, total_rows, use_vertical
        )
        bg = hex_to_bg_ansi(get_color_by_position(gradient, grad_pos))
        line += f"{bg}{WHITE}{char}"
        display_pos += char_width

    # Right padding
    right_pad_width = box_width - 2 - text_width
    for _ in range(right_pad_width):
        grad_pos = calculate_gradient_position(
            display_pos, row_idx, box_width, total_rows, use_vertical
        )
        bg = hex_to_bg_ansi(get_color_by_position(gradient, grad_pos))
        line += f"{bg} "
        display_pos += 1

    return line + RESET


# ============================================================================
# Main Orchestrator
# ============================================================================

def print_startup_banner():
    """Print Deforum initialization banner with slopcore gradient.

    Refactored version with complexity ≤10 and modular structure.
    """
    from deforum.constants import FORK_NAME, GITHUB_URL
    from deforum.utils.system.logging.themes import (
        get_random_slopcore_gradient,
        set_slopcore_gradient,
        HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4,
        HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7
    )

    # Setup
    commit_id = get_commit_id()
    gradient_variant = get_random_slopcore_gradient()
    set_slopcore_gradient(gradient_variant)

    # Terminal dimensions
    term_dims = get_terminal_dimensions()
    box_width = calculate_box_width(term_dims.width)

    # Generate gradient
    base_colors = [
        HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4,
        HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7
    ]
    gradient = generate_smooth_gradient(base_colors)

    # Gradient direction
    use_vertical = (gradient_variant == 'BB0')

    # Create content
    lines = create_content_lines(FORK_NAME, GITHUB_URL, commit_id)
    total_rows = len(lines) + 2

    # Render banner
    banner_lines = []

    # Top border
    banner_lines.append(render_top_border(box_width, gradient, use_vertical, total_rows))

    # Content rows
    for content_idx, line in enumerate(lines):
        row_idx = content_idx + 1
        is_title = (content_idx == 0)

        if is_title:
            # Center title first
            centered_line = center_text(line, box_width - 4)
            rendered = render_title_row(
                centered_line, box_width, row_idx, total_rows, gradient, use_vertical
            )
        else:
            rendered = render_content_row(
                line, box_width, row_idx, total_rows, gradient, use_vertical
            )

        banner_lines.append(rendered)

    # Bottom border
    banner_lines.append(render_bottom_border(box_width, gradient, use_vertical, total_rows))

    # Print
    print("\n".join(banner_lines))
