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

    # Randomly select slopcore gradient variant (BB0 or DA3)
    from deforum.utils.system.logging.themes import (
        get_random_slopcore_gradient, set_slopcore_gradient, get_active_gradient
    )

    gradient_variant = get_random_slopcore_gradient()
    set_slopcore_gradient(gradient_variant)

    # ANSI color codes for active slopcore gradient (now selected)
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
    # BB0 original order: Purple (#5606FF) → Cyan (#17A7FE)
    # DA3 original order: Cyan (#1CC4E6) → Red/Pink (#F64A5E)
    slopcore_base_colors = [HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4,
                            HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7]

    # No reversal needed - original orders work for vertical BB0 and horizontal DA3
    # BB0 vertical: Purple(top) → Cyan(bottom)
    # DA3 horizontal: Cyan(left) → Red(right)

    slopcore_gradient = []
    # Interpolate between each pair for smoother slopcore gradient
    for i in range(len(slopcore_base_colors) - 1):
        slopcore_gradient.append(slopcore_base_colors[i])
        mid_color = interpolate_color(slopcore_base_colors[i], slopcore_base_colors[i + 1], 0.5)
        slopcore_gradient.append(mid_color)
    slopcore_gradient.append(slopcore_base_colors[-1])

    RESET = "\033[0m"
    BOLD = "\033[1m"
    WHITE = "\033[97m"
    # Very dark gray (30,30,30) for terminal black and corners (same color)
    TERMINAL_BLACK_BG = "\033[48;2;30;30;30m"  # Background for fade and black section
    CORNER_COLOR = "\033[38;2;30;30;30m"        # Foreground for corner triangles

    # Helper to get terminal width
    term_width = shutil.get_terminal_size((120, 24)).columns
    box_width = min(term_width - 4, 128)  # Max 128 chars wide (3-step fade is 4 chars wider)

    # Helper to convert hex to ANSI background RGB
    def hex_to_bg_ansi(hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"\033[48;2;{r};{g};{b}m"

    # Helper to get slopcore gradient background color by position (0.0 to 1.0)
    # DA3: Horizontal (left→right), BB0: Vertical (top→bottom)
    def get_slopcore_bg_by_position(position):
        idx = min(int(position * len(slopcore_gradient)), len(slopcore_gradient) - 1)
        return hex_to_bg_ansi(slopcore_gradient[idx])

    # Determine gradient direction based on variant
    # DA3: left-to-right (horizontal), BB0: top-to-bottom (vertical)
    use_vertical_gradient = (gradient_variant == 'BB0')

    # Slopcore rounded button characters (tailwind-hegemony punk, bootstrap default-css-wave)
    ROUND_TL = "◤"  # Top-left rounded
    ROUND_TR = "◥"  # Top-right rounded
    ROUND_BL = "◣"  # Bottom-left rounded
    ROUND_BR = "◢"  # Bottom-right rounded

    # Brightest slopcore cyan for title text foreground (end of spectrum)
    SLOPCORE_BRIGHT_BLUE_FG = f"\033[38;2;{int(HEX_SLOPCORE_7[1:3], 16)};{int(HEX_SLOPCORE_7[3:5], 16)};{int(HEX_SLOPCORE_7[5:7], 16)}m"

    # Title with smooth 3-step fade-to-black using block shades
    # Format: ▓▓▒▒░░ ⚡ FORK NAME ⚡ ░░▒▒▓▓
    # Fade effect: gradient foreground on terminal-black background
    # Pattern: ▓ (high density) → ▒ (medium) → ░ (low) → pure black → ░ → ▒ → ▓
    # Whole banner is slopcore button (rounded corners), title has smooth colored fade
    title_text = f"▓▓▒▒░░ ⚡ {FORK_NAME} ⚡ ░░▒▒▓▓"

    # Center the title based on its display width
    title_width = display_width(title_text)
    # Use box_width - 4 as available width (2 padding on each side)
    available_width = box_width - 4
    if title_width < available_width:
        left_pad_spaces = (available_width - title_width) // 2
        title_text = (" " * left_pad_spaces) + title_text

    # Get gradient metadata for footer
    from deforum.utils.system.logging.themes import SLOPCORE_GRADIENTS
    gradient_meta = SLOPCORE_GRADIENTS[gradient_variant]

    # Prepare content lines (plain text, background will have gradient)
    lines = [
        title_text,
        "",  # Separator
        "RECOMMENDED: Dedicated/Isolated Forge Neo Instance",
        "Why: Hijacks Forge output pipeline (custom dashboard, progress bars, suppressed logs)",
        "     Patches sigma timesteps for continuous 1% strength resolution (vs 5% at 20 steps)",
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
    # DA3: horizontal (left→right), BB0: vertical (top→bottom, so top row = start of gradient)
    top_line = ""
    row_idx = 0  # Top border row
    for char_pos in range(box_width):
        if use_vertical_gradient:
            # BB0: Vertical gradient - use row position (top row = 0.0)
            gradient_pos = row_idx / (total_rows - 1) if total_rows > 1 else 0.0
        else:
            # DA3: Horizontal gradient - use char position (left = 0.0)
            gradient_pos = char_pos / (box_width - 1) if box_width > 1 else 0.0
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

    # Content lines with gradient background
    # DA3: horizontal (left to right), BB0: vertical (top to bottom)
    for content_row_idx, line in enumerate(lines):
        # Content rows start after top border (row 1 in total_rows)
        actual_row_idx = content_row_idx + 1
        # Strip ANSI codes to get visible text
        visible_text = re.sub(r'\033\[[0-9;]*m', '', line)
        text_display_width = display_width(visible_text)  # Actual terminal width

        # Build content: 2 spaces + text + padding to fill box_width
        left_padding = "  "
        right_padding_width = box_width - 2 - text_display_width
        right_padding = " " * right_padding_width

        # Now apply gradient to each character position (simple horizontal)
        content_line = ""
        display_pos = 0

        # Helper to calculate gradient position based on gradient direction
        def calc_gradient_pos(char_pos):
            if use_vertical_gradient:
                # BB0: Vertical - use row position
                return actual_row_idx / (total_rows - 1) if total_rows > 1 else 0.0
            else:
                # DA3: Horizontal - use char position
                return char_pos / (box_width - 1) if box_width > 1 else 0.0

        # Special handling for title row (content_row_idx == 0): black pill button effect
        is_title_row = (content_row_idx == 0)

        # Helper to get gradient foreground color (for brackets)
        def hex_to_fg_ansi(hex_color):
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            return f"\033[38;2;{r};{g};{b}m"

        def get_slopcore_fg_by_position(position):
            idx = min(int(position * len(slopcore_gradient)), len(slopcore_gradient) - 1)
            return hex_to_fg_ansi(slopcore_gradient[idx])

        # Track if we're inside the black section (after ▓▓▒▒░░ fade-in, before ░░▒▒▓▓ fade-out)
        # Pattern: ▓▓▒▒░░ ⚡ FORK ⚡ ░░▒▒▓▓
        inside_black_section = False
        fade_chars_seen = 0  # Count ▓, ▒, and ░ characters
        exiting_black_section = False  # Flag when we start seeing ░ on the right side

        # Left padding (2 spaces)
        for i in range(2):
            gradient_pos = calc_gradient_pos(display_pos)
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
                gradient_pos = calc_gradient_pos(display_pos)

                if char in ('▓', '▒', '░'):
                    # Shade characters: gradient foreground on terminal-black background
                    # Creates colored fade effect - denser characters show more color
                    fg = get_slopcore_fg_by_position(gradient_pos)
                    content_line += f"{TERMINAL_BLACK_BG}{fg}{char}"

                    if not inside_black_section and not exiting_black_section:
                        fade_chars_seen += 1
                        if fade_chars_seen == 6:  # After ▓▓▒▒░░, next char starts black section
                            inside_black_section = True
                    elif inside_black_section:
                        exiting_black_section = True  # First ░ on right side
                        inside_black_section = False
                elif inside_black_section or (fade_chars_seen == 6 and not exiting_black_section):
                    # Inside the black section (space, emoji, text)
                    if char == '⚡':
                        # Bolt emoji: natural yellow color, terminal-black background
                        content_line += f"{TERMINAL_BLACK_BG}{char}"
                    elif char == ' ':
                        # Space inside black section: terminal-black background
                        content_line += f"{TERMINAL_BLACK_BG} "
                    else:
                        # Fork name text: blue foreground, terminal-black background
                        content_line += f"{TERMINAL_BLACK_BG}{SLOPCORE_BRIGHT_BLUE_FG}{char}"
                else:
                    # Outside black section: normal gradient background, white text
                    bg = get_slopcore_bg_by_position(gradient_pos)
                    content_line += f"{bg}{WHITE}{char}"
            else:
                # Regular content: slopcore gradient background, white text
                gradient_pos = calc_gradient_pos(display_pos)
                bg = get_slopcore_bg_by_position(gradient_pos)
                content_line += f"{bg}{WHITE}{char}"

            display_pos += char_width
            text_idx += 1

        # Right padding
        for i in range(right_padding_width):
            gradient_pos = calc_gradient_pos(display_pos)
            bg = get_slopcore_bg_by_position(gradient_pos)
            content_line += f"{bg} "
            display_pos += 1

        content_line += RESET  # Single reset at end of line
        banner_lines.append(content_line)

    # Bottom border with slopcore rounded corners and gradient
    # DA3: horizontal (left→right), BB0: vertical (bottom row = end of gradient)
    bottom_line = ""
    bottom_row_idx = total_rows - 1  # Bottom border row
    for char_pos in range(box_width):
        if use_vertical_gradient:
            # BB0: Vertical gradient - use row position (bottom row = 1.0)
            gradient_pos = bottom_row_idx / (total_rows - 1) if total_rows > 1 else 0.0
        else:
            # DA3: Horizontal gradient - use char position
            gradient_pos = char_pos / (box_width - 1) if box_width > 1 else 0.0
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
