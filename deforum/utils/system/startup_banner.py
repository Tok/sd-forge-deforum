"""Deforum startup banner with slopcore purple gradient styling."""

from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


def print_startup_banner():
    """Print Deforum initialization banner with slopcore gradient background."""
    import shutil
    import unicodedata

    # ANSI color codes for slopcore gradient
    from deforum.utils.system.logging.themes import (
        HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4,
        HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7
    )

    # Helper to get display width of text (accounts for wide chars like emojis)
    def display_width(text):
        """Calculate actual terminal display width (emojis count as 2)."""
        width = 0
        for char in text:
            if unicodedata.east_asian_width(char) in ('F', 'W'):
                width += 2  # Full-width or Wide characters (emojis, CJK, etc.)
            else:
                width += 1
        return width

    # Create extended gradient with interpolated colors for smoother effect
    def interpolate_color(hex1, hex2, ratio):
        """Interpolate between two hex colors."""
        r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
        r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        return f"#{r:02x}{g:02x}{b:02x}"

    # Generate smooth gradient (more shades for smoother background)
    gradient_colors = []
    base_colors = [HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4,
                   HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7]

    # Interpolate between each pair for smoother gradient
    for i in range(len(base_colors) - 1):
        gradient_colors.append(base_colors[i])
        mid_color = interpolate_color(base_colors[i], base_colors[i + 1], 0.5)
        gradient_colors.append(mid_color)
    gradient_colors.append(base_colors[-1])

    RESET = "\033[0m"
    BOLD = "\033[1m"
    WHITE = "\033[97m"
    # Use very dark gray for corners so they blend better with terminal background
    CORNER_COLOR = "\033[38;2;30;30;30m"

    # Helper to get terminal width
    term_width = shutil.get_terminal_size((120, 24)).columns
    box_width = min(term_width - 4, 120)  # Max 120 chars wide

    # Helper to convert hex to ANSI background RGB
    def hex_to_bg_ansi(hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"\033[48;2;{r};{g};{b}m"

    # Helper to get gradient background color by position (0.0 to 1.0)
    def get_gradient_bg_by_position(position):
        idx = min(int(position * len(gradient_colors)), len(gradient_colors) - 1)
        return hex_to_bg_ansi(gradient_colors[idx])

    # Slopcore rounded button characters (anti-tailwind aesthetic)
    ROUND_TL = "◤"  # Top-left rounded
    ROUND_TR = "◥"  # Top-right rounded
    ROUND_BL = "◣"  # Bottom-left rounded
    ROUND_BR = "◢"  # Bottom-right rounded

    # Title with text gradient and bolt emojis
    title_text = "⚡ Zirteq's Fluxabled Fork of the Deforum Extension for Forge Neo Fork of Forge WebUI Fork of Automatic1111 ⚡"

    # Center the title based on its display width
    title_width = display_width(title_text)
    # Use box_width - 4 as available width (2 padding on each side)
    available_width = box_width - 4
    if title_width < available_width:
        left_pad_spaces = (available_width - title_width) // 2
        title_text = (" " * left_pad_spaces) + title_text

    # Prepare content lines (plain text, background will have gradient)
    lines = [
        title_text,
        "",  # Separator
        "RECOMMENDED: Dedicated/Isolated Forge Neo Instance",
        "Why: Hijacks Forge output pipeline (custom dashboard, progress bars, suppressed logs)",
        "     May conflict with other extensions - optimized specifically for Deforum workflows",
        "",
        "Forge Neo: https://github.com/Haoming02/sd-webui-forge-classic/tree/neo",
        "This Fork: https://github.com/Tok/sd-forge-deforum",
        "",
        "Primary Target: Forge Neo (fully tested and supported)",
        "Other Forge Versions: May work but remain untested"
    ]

    # Draw box with diagonal slopcore gradient background
    banner_lines = []
    total_rows = len(lines) + 2  # +2 for top and bottom borders

    # Diagonal shift amount (shift gradient start position for each row)
    # Smaller shift = more vertical, larger = more diagonal
    diagonal_shift = 0.4  # 40% shift for stronger diagonal, no wrap-around

    # Top border with slopcore rounded corners and diagonal gradient
    import re

    # Top line: ◤ with gradient bg, then spaces with gradient, ending with ◥
    top_line = ""
    for char_pos in range(box_width):
        gradient_pos = min(1.0, max(0.0, 0 + char_pos * diagonal_shift / box_width))
        bg = get_gradient_bg_by_position(gradient_pos)

        if char_pos == 0:
            # Left corner with gradient bg
            top_line += f"{bg}{CORNER_COLOR}{ROUND_TL}"
        elif char_pos == box_width - 1:
            # Right corner with gradient bg
            top_line += f"{bg}{CORNER_COLOR}{ROUND_TR}"
        else:
            # Middle space with gradient bg
            top_line += f"{bg} "
    top_line += RESET  # Single reset at end of line
    banner_lines.append(top_line)

    # Content lines with diagonal gradient background
    for row_idx, line in enumerate(lines):
        # Strip ANSI codes to get visible text
        visible_text = re.sub(r'\033\[[0-9;]*m', '', line)
        text_display_width = display_width(visible_text)  # Actual terminal width

        # Build line with diagonal gradient
        row_base = (row_idx + 1) / total_rows  # Vertical position

        # Build content: 2 spaces + text + padding to fill box_width
        left_padding = "  "
        right_padding_width = box_width - 2 - text_display_width
        right_padding = " " * right_padding_width

        # Now apply gradient to each character position
        content_line = ""
        char_idx = 0
        display_pos = 0

        # Left padding (2 spaces)
        for i in range(2):
            gradient_pos = min(1.0, max(0.0, row_base + display_pos * diagonal_shift / box_width))
            bg = get_gradient_bg_by_position(gradient_pos)
            content_line += f"{bg} "
            display_pos += 1

        # Text content (accounting for wide chars)
        text_idx = 0
        while text_idx < len(visible_text):
            char = visible_text[text_idx]
            char_width = 2 if unicodedata.east_asian_width(char) in ('F', 'W') else 1

            gradient_pos = min(1.0, max(0.0, row_base + display_pos * diagonal_shift / box_width))
            bg = get_gradient_bg_by_position(gradient_pos)
            content_line += f"{bg}{WHITE}{char}"

            display_pos += char_width
            text_idx += 1

        # Right padding
        for i in range(right_padding_width):
            gradient_pos = min(1.0, max(0.0, row_base + display_pos * diagonal_shift / box_width))
            bg = get_gradient_bg_by_position(gradient_pos)
            content_line += f"{bg} "
            display_pos += 1

        content_line += RESET  # Single reset at end of line
        banner_lines.append(content_line)

    # Bottom border with slopcore rounded corners and diagonal gradient
    bottom_line = ""
    for char_pos in range(box_width):
        gradient_pos = min(1.0, max(0.0, (total_rows - 1) / total_rows + char_pos * diagonal_shift / box_width))
        bg = get_gradient_bg_by_position(gradient_pos)

        if char_pos == 0:
            # Left corner with gradient bg
            bottom_line += f"{bg}{CORNER_COLOR}{ROUND_BL}"
        elif char_pos == box_width - 1:
            # Right corner with gradient bg
            bottom_line += f"{bg}{CORNER_COLOR}{ROUND_BR}"
        else:
            # Middle space with gradient bg
            bottom_line += f"{bg} "
    bottom_line += RESET  # Single reset at end of line
    banner_lines.append(bottom_line)

    # Print the banner (no extra empty lines)
    print("\n".join(banner_lines))
