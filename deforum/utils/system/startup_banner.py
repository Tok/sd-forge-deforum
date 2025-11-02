"""Deforum startup banner with slopcore purple gradient styling."""

from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


def print_startup_banner():
    """Print Deforum initialization banner with slopcore gradient background."""
    import shutil

    # ANSI color codes for slopcore gradient
    from deforum.utils.system.logging.themes import (
        HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4,
        HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7
    )

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

    # Rounded border characters (slopcore anti-tailwind aesthetic)
    ROUND_TL = "╭"  # Top-left rounded
    ROUND_TR = "╮"  # Top-right rounded
    ROUND_BL = "╰"  # Bottom-left rounded
    ROUND_BR = "╯"  # Bottom-right rounded
    HORIZONTAL = "─"
    VERTICAL = "│"

    # Title with text gradient and bolt emojis
    title_text = "⚡ Zirteq's Fluxabled Fork of the Deforum Extension for Forge Neo Fork of Forge WebUI Fork of Automatic1111 ⚡"

    # Prepare content lines (plain text, background will have gradient)
    lines = [
        title_text,
        "",  # Separator
        "Primary Target: Forge Neo (fully tested and supported)",
        "Other Forge Versions: May work but remain untested",
        "More Info: https://github.com/Tok/sd-forge-deforum"
    ]

    # Draw box with diagonal slopcore gradient background
    banner_lines = []
    total_rows = len(lines) + 2  # +2 for top and bottom borders

    # Diagonal shift amount (shift gradient start position for each row)
    # Smaller shift = more vertical, larger = more diagonal
    diagonal_shift = 0.4  # 40% shift for stronger diagonal, no wrap-around

    # Top border with rounded corners and diagonal gradient
    import re
    row_gradient = []
    for char_pos in range(box_width):
        # Diagonal: row 0, but gradient position shifts with horizontal position
        # Clamp to 0-1 range instead of modulo to prevent wrap-around
        gradient_pos = min(1.0, max(0.0, 0 + char_pos * diagonal_shift / box_width))
        row_gradient.append(get_gradient_bg_by_position(gradient_pos))

    top_line = f"{row_gradient[0]}{WHITE}{ROUND_TL}"
    for char_pos in range(1, box_width - 1):
        top_line += f"{row_gradient[char_pos]}{HORIZONTAL}"
    top_line += f"{row_gradient[-1]}{ROUND_TR}{RESET}"
    banner_lines.append(top_line)

    # Content lines with diagonal gradient background
    for row_idx, line in enumerate(lines):
        # Strip ANSI codes to get visible length
        visible_text = re.sub(r'\033\[[0-9;]*m', '', line)
        text_len = len(visible_text)

        # Build line with diagonal gradient
        row_base = (row_idx + 1) / total_rows  # Vertical position

        content_line = ""

        # Calculate gradient for each character position in this row
        for char_pos in range(box_width):
            # Diagonal gradient: combine row and column position
            # Clamp instead of modulo to prevent wrap-around from purple back to blue
            gradient_pos = min(1.0, max(0.0, row_base + char_pos * diagonal_shift / box_width))
            bg = get_gradient_bg_by_position(gradient_pos)

            if char_pos == 0:
                # Left border
                content_line += f"{bg}{WHITE}{VERTICAL}"
            elif char_pos == 1:
                # Space after left border
                content_line += f"{bg} "
            elif char_pos < text_len + 2:
                # Text content
                content_line += f"{bg}{visible_text[char_pos - 2]}"
            elif char_pos < box_width - 1:
                # Padding
                content_line += f"{bg} "
            else:
                # Right border (last char)
                content_line += f"{bg}{VERTICAL}{RESET}"

        banner_lines.append(content_line)

    # Bottom border with rounded corners and diagonal gradient
    row_gradient = []
    for char_pos in range(box_width):
        # Clamp to prevent wrap-around
        gradient_pos = min(1.0, max(0.0, (total_rows - 1) / total_rows + char_pos * diagonal_shift / box_width))
        row_gradient.append(get_gradient_bg_by_position(gradient_pos))

    bottom_line = f"{row_gradient[0]}{WHITE}{ROUND_BL}"
    for char_pos in range(1, box_width - 1):
        bottom_line += f"{row_gradient[char_pos]}{HORIZONTAL}"
    bottom_line += f"{row_gradient[-1]}{ROUND_BR}{RESET}"
    banner_lines.append(bottom_line)

    # Print the banner
    print("")
    print("\n".join(banner_lines))
    print("")
