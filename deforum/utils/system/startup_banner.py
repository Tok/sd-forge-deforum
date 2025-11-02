"""Deforum startup banner with slopcore purple gradient styling."""

from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


def print_startup_banner():
    """Print Deforum initialization banner with slopcore 2D diagonal gradient table."""
    import os
    import shutil
    from pathlib import Path

    # ANSI color codes for extended slopcore gradient (more shades for 2D effect)
    from deforum.utils.system.logging.themes import (
        HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4,
        HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7
    )

    # Create extended gradient with interpolated colors for smoother diagonal effect
    def interpolate_color(hex1, hex2, ratio):
        """Interpolate between two hex colors."""
        r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
        r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        return f"#{r:02x}{g:02x}{b:02x}"

    # Generate smooth gradient with more shades (14 shades for smoother diagonal)
    gradient_colors = []
    base_colors = [HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4,
                   HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7]

    # Interpolate between each pair
    for i in range(len(base_colors) - 1):
        gradient_colors.append(base_colors[i])
        # Add one interpolated color between each pair
        mid_color = interpolate_color(base_colors[i], base_colors[i + 1], 0.5)
        gradient_colors.append(mid_color)
    gradient_colors.append(base_colors[-1])

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Helper to get terminal width
    term_width = shutil.get_terminal_size((120, 24)).columns
    box_width = min(term_width - 4, 120)  # Max 120 chars wide

    # Helper to convert hex to ANSI RGB
    def hex_to_rgb_ansi(hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"\033[38;2;{r};{g};{b}m"

    # Helper to get gradient color by position (0.0 to 1.0)
    def get_gradient_color(position):
        idx = min(int(position * len(gradient_colors)), len(gradient_colors) - 1)
        return hex_to_rgb_ansi(gradient_colors[idx])

    # Draw box with diagonal gradient borders
    def draw_gradient_border(text_lines, width):
        """Draw a box with diagonal gradient borders around text lines."""
        # Top border with gradient
        top_border = ""
        for i in range(width):
            color = get_gradient_color(i / width)
            if i == 0:
                top_border += f"{color}╔"
            elif i == width - 1:
                top_border += f"╗{RESET}"
            else:
                top_border += "═"

        # Middle rows
        middle_rows = []
        for row_idx, line in enumerate(text_lines):
            # Calculate diagonal position for left and right borders
            left_pos = row_idx / (len(text_lines) + 1)
            right_pos = (row_idx + width) / (len(text_lines) + width)

            left_color = get_gradient_color(left_pos)
            right_color = get_gradient_color(right_pos)

            # Pad text to exact width
            visible_len = len(line) - line.count('\033[') * 10  # Rough ANSI length estimate
            # More accurate: strip ANSI codes
            import re
            visible_text = re.sub(r'\033\[[0-9;]*m', '', line)
            padding_needed = width - 2 - len(visible_text)

            middle_rows.append(f"{left_color}║{RESET} {line}{' ' * padding_needed}{right_color}║{RESET}")

        # Bottom border with gradient
        bottom_border = ""
        for i in range(width):
            color = get_gradient_color((i + len(text_lines)) / (width + len(text_lines)))
            if i == 0:
                bottom_border += f"{color}╚"
            elif i == width - 1:
                bottom_border += f"╝{RESET}"
            else:
                bottom_border += "═"

        return "\n".join([top_border] + middle_rows + [bottom_border])

    # Title with diagonal gradient effect and bolt emojis
    title_text = "⚡ Zirteq's Fluxabled Fork of the Deforum Extension for Forge Neo Fork of Forge WebUI Fork of Automatic1111 ⚡"

    # Create diagonal gradient for title
    gradient_title = ""
    for i, char in enumerate(title_text):
        color = get_gradient_color(i / len(title_text))
        gradient_title += f"{BOLD}{color}{char}"
    gradient_title += RESET

    # Prepare content lines
    lines = [
        gradient_title,
        "",  # Separator
        f"{BOLD}Primary Target:{RESET} Forge Neo (fully tested and supported)",
        f"{BOLD}Other Forge Versions:{RESET} May work but remain untested",
        f"{BOLD}More Info:{RESET} https://github.com/Tok/sd-forge-deforum"
    ]

    # Draw the banner
    print("")
    print(draw_gradient_border(lines, box_width))
    print("")
