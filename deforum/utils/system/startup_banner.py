"""Deforum startup banner with slopcore gradient styling."""


def print_startup_banner():
    """Print Deforum initialization banner with slopcore gradient background."""
    import shutil
    import unicodedata
    from deforum.constants import FORK_NAME, GITHUB_URL

    # Get commit ID (may fail during early preload if modules.extensions not ready)
    try:
        from deforum.utils.general import get_deforum_version
        commit_id = get_deforum_version()
    except:
        commit_id = "Unknown"

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

    # Generate smooth slopcore gradient (more shades for smoother background)
    # Slopcore gradient order: darkest purple → bright blue (7→6→5→4→3→2→1)
    slopcore_gradient = []
    slopcore_base_colors = [HEX_SLOPCORE_7, HEX_SLOPCORE_6, HEX_SLOPCORE_5, HEX_SLOPCORE_4,
                            HEX_SLOPCORE_3, HEX_SLOPCORE_2, HEX_SLOPCORE_1]

    # Interpolate between each pair for smoother slopcore gradient
    for i in range(len(slopcore_base_colors) - 1):
        slopcore_gradient.append(slopcore_base_colors[i])
        mid_color = interpolate_color(slopcore_base_colors[i], slopcore_base_colors[i + 1], 0.5)
        slopcore_gradient.append(mid_color)
    slopcore_gradient.append(slopcore_base_colors[-1])

    RESET = "\033[0m"
    BOLD = "\033[1m"
    WHITE = "\033[97m"
    # Terminal black for fade background (same as corners/edges)
    TERMINAL_BLACK = "\033[48;2;0;0;0m"
    # Dark grays for fade foreground (progressively darker toward black)
    DARK_GRAY_1 = "\033[38;2;60;60;60m"  # ▓ (dark shade) - very dark gray
    DARK_GRAY_2 = "\033[38;2;40;40;40m"  # ▒ (medium shade) - darker gray
    # Use very dark gray for corners so they blend better with terminal background
    CORNER_COLOR = "\033[38;2;30;30;30m"

    # Helper to get terminal width
    term_width = shutil.get_terminal_size((120, 24)).columns
    box_width = min(term_width - 4, 124)  # Max 124 chars wide (increased for longer title)

    # Helper to convert hex to ANSI background RGB
    def hex_to_bg_ansi(hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"\033[48;2;{r};{g};{b}m"

    # Helper to get slopcore gradient background color by position (0.0 to 1.0)
    def get_slopcore_bg_by_position(position):
        idx = min(int(position * len(slopcore_gradient)), len(slopcore_gradient) - 1)
        return hex_to_bg_ansi(slopcore_gradient[idx])

    # Slopcore rounded button characters (tailwind-hegemony punk, bootstrap default-css-wave)
    ROUND_TL = "◤"  # Top-left rounded
    ROUND_TR = "◥"  # Top-right rounded
    ROUND_BL = "◣"  # Bottom-left rounded
    ROUND_BR = "◢"  # Bottom-right rounded

    # Brightest slopcore blue for title text foreground (end of spectrum)
    SLOPCORE_BRIGHT_BLUE_FG = f"\033[38;2;{int(HEX_SLOPCORE_1[1:3], 16)};{int(HEX_SLOPCORE_1[3:5], 16)};{int(HEX_SLOPCORE_1[5:7], 16)}m"

    # Title with smooth fade-to-black using block shades
    # Format: ▓▓▒▒ ⚡ FORK NAME ⚡ ▒▒▓▓
    # Fade pattern: gradient → dark shade → medium shade → terminal black → medium shade → dark shade → gradient
    # Whole banner is slopcore button (rounded corners), title has fade effect (not pill)
    title_text = f"▓▓▒▒ ⚡ {FORK_NAME} ⚡ ▒▒▓▓"

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
        f"This Fork: {GITHUB_URL} (commit: {commit_id})",
        "",
        "Primary Target: Forge Neo (fully tested and supported)",
        "Other Forge Versions: May work but remain untested"
    ]

    # Draw box with diagonal slopcore gradient background
    banner_lines = []
    total_rows = len(lines) + 2  # +2 for top and bottom borders

    # Double Fibonacci spacing: Fibonacci sequence * 2 for varied diagonal increments
    # Generate enough entries to cover all rows (13 total) without cycling
    # Fibonacci: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233...
    # Doubled: 2, 2, 4, 6, 10, 16, 26, 42, 68, 110, 178, 288, 466...
    # We'll use first 15 entries scaled down to fit banner width
    double_fib_raw = [2, 2, 4, 6, 10, 16, 26, 42, 68, 110, 178, 288, 466, 754, 1220]

    # Normalize to small increments (divide by 100) for smooth slopcore gradient
    double_fib_pattern = [x / 100.0 for x in double_fib_raw[:15]]

    # Cumulative shifts for smooth diagonal slopcore gradient
    cumulative_shifts = [0]  # Start at 0
    for increment in double_fib_pattern:
        cumulative_shifts.append(cumulative_shifts[-1] + increment)

    def get_row_shift(row_idx):
        """Get cumulative horizontal shift for this row using double Fibonacci spacing."""
        if row_idx >= len(cumulative_shifts):
            return cumulative_shifts[-1]  # Max shift for overflow
        return cumulative_shifts[row_idx]

    # Top border with slopcore rounded corners and diagonal gradient
    import re

    # Maximum shift value for normalization (last cumulative shift)
    max_shift = cumulative_shifts[-1]

    # Top line: ◤ with slopcore gradient bg, then spaces with slopcore gradient, ending with ◥
    top_line = ""
    top_shift = get_row_shift(0)
    for char_pos in range(box_width):
        # Slopcore gradient: emphasize row shift more for fuller range
        # Use row position as primary driver, char position as secondary
        gradient_pos = min(1.0, max(0.0, (top_shift * 3 + char_pos * 0.5) / (max_shift * 3 + box_width * 0.5)))
        bg = get_slopcore_bg_by_position(gradient_pos)

        if char_pos == 0:
            # Left corner with slopcore gradient bg
            top_line += f"{bg}{CORNER_COLOR}{ROUND_TL}"
        elif char_pos == box_width - 1:
            # Right corner with slopcore gradient bg
            top_line += f"{bg}{CORNER_COLOR}{ROUND_TR}"
        else:
            # Middle space with slopcore gradient bg
            top_line += f"{bg} "
    top_line += RESET  # Single reset at end of line
    banner_lines.append(top_line)

    # Content lines with diagonal slopcore gradient background
    for row_idx, line in enumerate(lines):
        # Strip ANSI codes to get visible text
        visible_text = re.sub(r'\033\[[0-9;]*m', '', line)
        text_display_width = display_width(visible_text)  # Actual terminal width

        # Get row shift using double Fibonacci pattern
        row_shift = get_row_shift(row_idx + 1)  # +1 because row 0 is top border

        # Build content: 2 spaces + text + padding to fill box_width
        left_padding = "  "
        right_padding_width = box_width - 2 - text_display_width
        right_padding = " " * right_padding_width

        # Now apply gradient to each character position
        content_line = ""
        display_pos = 0

        # Special handling for title row (row_idx == 0): black pill button effect
        is_title_row = (row_idx == 0)

        # Helper to get gradient foreground color (for brackets)
        def hex_to_fg_ansi(hex_color):
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            return f"\033[38;2;{r};{g};{b}m"

        def get_slopcore_fg_by_position(position):
            idx = min(int(position * len(slopcore_gradient)), len(slopcore_gradient) - 1)
            return hex_to_fg_ansi(slopcore_gradient[idx])

        # Track if we're inside the black section (after ▓▓▒▒ fade-in, before ▒▒▓▓ fade-out)
        # Pattern: ▓▓▒▒ ⚡ FORK ⚡ ▒▒▓▓
        inside_pill = False  # Variable name kept for simplicity
        fade_chars_seen = 0  # Count ▓ and ▒ characters
        exiting_pill = False  # Flag when we start seeing ▒ on the right side

        # Left padding (2 spaces)
        for i in range(2):
            gradient_pos = min(1.0, max(0.0, (row_shift * 3 + display_pos * 0.5) / (max_shift * 3 + box_width * 0.5)))
            bg = get_slopcore_bg_by_position(gradient_pos)
            content_line += f"{bg} "
            display_pos += 1

        # Text content (accounting for wide chars)
        text_idx = 0
        while text_idx < len(visible_text):
            char = visible_text[text_idx]
            char_width = 2 if unicodedata.east_asian_width(char) in ('F', 'W') else 1

            if is_title_row:
                # Calculate gradient position for this character
                gradient_pos = min(1.0, max(0.0, (row_shift * 3 + display_pos * 0.5) / (max_shift * 3 + box_width * 0.5)))

                if char in ('▓', '▒'):
                    # Shade characters: gradient background, dark gray foreground (fade to black)
                    bg = get_slopcore_bg_by_position(gradient_pos)
                    # Use progressively darker grays: ▓ is darker than ▒
                    fg = DARK_GRAY_1 if char == '▓' else DARK_GRAY_2
                    content_line += f"{bg}{fg}{char}"

                    if not inside_pill and not exiting_pill:
                        fade_chars_seen += 1
                        if fade_chars_seen == 4:  # After ▓▓▒▒, next char starts black section
                            inside_pill = True
                    elif inside_pill:
                        exiting_pill = True  # First ▒ on right side
                        inside_pill = False
                elif inside_pill or (fade_chars_seen == 4 and not exiting_pill):
                    # Inside the black section (space, emoji, text)
                    if char == '⚡':
                        # Bolt emoji: natural yellow color, terminal-black background
                        content_line += f"{TERMINAL_BLACK}{char}"
                    elif char == ' ':
                        # Space inside black section: terminal-black background
                        content_line += f"{TERMINAL_BLACK} "
                    else:
                        # Fork name text: blue foreground, terminal-black background
                        content_line += f"{TERMINAL_BLACK}{SLOPCORE_BRIGHT_BLUE_FG}{char}"
                else:
                    # Outside black section: normal gradient background, white text
                    bg = get_slopcore_bg_by_position(gradient_pos)
                    content_line += f"{bg}{WHITE}{char}"
            else:
                # Regular content: slopcore gradient background, white text
                gradient_pos = min(1.0, max(0.0, (row_shift * 3 + display_pos * 0.5) / (max_shift * 3 + box_width * 0.5)))
                bg = get_slopcore_bg_by_position(gradient_pos)
                content_line += f"{bg}{WHITE}{char}"

            display_pos += char_width
            text_idx += 1

        # Right padding
        for i in range(right_padding_width):
            gradient_pos = min(1.0, max(0.0, (row_shift * 3 + display_pos * 0.5) / (max_shift * 3 + box_width * 0.5)))
            bg = get_slopcore_bg_by_position(gradient_pos)
            content_line += f"{bg} "
            display_pos += 1

        content_line += RESET  # Single reset at end of line
        banner_lines.append(content_line)

    # Bottom border with slopcore rounded corners and diagonal slopcore gradient
    bottom_line = ""
    bottom_shift = get_row_shift(len(lines) + 1)  # Last row
    for char_pos in range(box_width):
        gradient_pos = min(1.0, max(0.0, (bottom_shift * 3 + char_pos * 0.5) / (max_shift * 3 + box_width * 0.5)))
        bg = get_slopcore_bg_by_position(gradient_pos)

        if char_pos == 0:
            # Left corner with slopcore gradient bg
            bottom_line += f"{bg}{CORNER_COLOR}{ROUND_BL}"
        elif char_pos == box_width - 1:
            # Right corner with slopcore gradient bg
            bottom_line += f"{bg}{CORNER_COLOR}{ROUND_BR}"
        else:
            # Middle space with slopcore gradient bg
            bottom_line += f"{bg} "
    bottom_line += RESET  # Single reset at end of line
    banner_lines.append(bottom_line)

    # Print the banner (no extra empty lines)
    print("\n".join(banner_lines))
