"""Console output theme definitions for Deforum.

Defines color palettes and styling for three themes:
- Slopcore: Modern blue→purple gradient (new default)
- Classic: Vibrant multi-color palette (legacy)
- Simple: Plain text, no colors
"""

from deforum.utils.image.color import hex_to_ansi_foreground as from_hex_color

# ============================================================================
# SLOPCORE THEME - Authentic BLANK BANSHEE 0 gradient
# ============================================================================
# Exact colors pipetted from BB0 album cover
# Original album gradient: #5606ff (top) → #17a7fe (bottom), straight vertical

# 7-shade gradient (for CLI banner, charts, general use)
HEX_SLOPCORE_1 = '#5606FF'  # Deep purple-blue (album top)
HEX_SLOPCORE_2 = '#4C21FF'  # Purple-blue
HEX_SLOPCORE_3 = '#413CFF'  # Blue-purple (banner start)
HEX_SLOPCORE_4 = '#3757FF'  # Mid blue
HEX_SLOPCORE_5 = '#2C71FE'  # Blue
HEX_SLOPCORE_6 = '#228CFE'  # Bright blue
HEX_SLOPCORE_7 = '#17A7FE'  # Cyan (album bottom, banner end)

# 5-shade gradient (specifically for tqdm progress bars)
# Maps to the 5 parallel dashboard bars in display order
HEX_SLOPCORE_TQDM_1 = '#5606FF'  # Deep purple-blue [Current Tweens - FASTEST]
HEX_SLOPCORE_TQDM_2 = '#462EFF'  # Purple-blue [Current Steps - FAST]
HEX_SLOPCORE_TQDM_3 = '#3757FF'  # Mid blue [Total Steps - MEDIUM]
HEX_SLOPCORE_TQDM_4 = '#277FFE'  # Bright blue [Total Diffusion Frames - SLOW]
HEX_SLOPCORE_TQDM_5 = '#17A7FE'  # Cyan [Total Frames - SLOWEST]

# Functional colors (borrowed from classic theme for slopcore mode)
# These are NOT slopcore colors, but used for practical UX purposes
HEX_FUNCTIONAL_YELLOW = '#FFEA56'  # Yellow for warnings (classic yellow, matches Precalculations tqdm)
HEX_FUNCTIONAL_RED = '#FE797B'     # Red for errors (classic red)

SLOPCORE_1 = from_hex_color(HEX_SLOPCORE_1)  # Bright blue
SLOPCORE_2 = from_hex_color(HEX_SLOPCORE_2)  # Blue-purple
SLOPCORE_3 = from_hex_color(HEX_SLOPCORE_3)  # Light purple
SLOPCORE_4 = from_hex_color(HEX_SLOPCORE_4)  # Mid purple
SLOPCORE_5 = from_hex_color(HEX_SLOPCORE_5)  # Purple
SLOPCORE_6 = from_hex_color(HEX_SLOPCORE_6)  # Deep purple
SLOPCORE_7 = from_hex_color(HEX_SLOPCORE_7)  # Darkest purple

# Functional color conversions (not part of slopcore palette)
FUNCTIONAL_YELLOW = from_hex_color(HEX_FUNCTIONAL_YELLOW)  # For warnings
FUNCTIONAL_RED = from_hex_color(HEX_FUNCTIONAL_RED)  # For errors

# ============================================================================
# CLASSIC THEME - Original vibrant colors
# ============================================================================

HEX_CLASSIC_RED = '#FE797B'
HEX_CLASSIC_ORANGE = '#FFB750'
HEX_CLASSIC_YELLOW = '#FFEA56'
HEX_CLASSIC_GREEN = '#8FE968'
HEX_CLASSIC_BLUE = '#36CEDC'
HEX_CLASSIC_PURPLE = '#A587CA'

CLASSIC_RED = from_hex_color(HEX_CLASSIC_RED)
CLASSIC_ORANGE = from_hex_color(HEX_CLASSIC_ORANGE)
CLASSIC_YELLOW = from_hex_color(HEX_CLASSIC_YELLOW)
CLASSIC_GREEN = from_hex_color(HEX_CLASSIC_GREEN)
CLASSIC_BLUE = from_hex_color(HEX_CLASSIC_BLUE)
CLASSIC_PURPLE = from_hex_color(HEX_CLASSIC_PURPLE)

# ============================================================================
# ANSI Control Codes
# ============================================================================

ESC = "\033["
TERM = "m"
RESET_COLOR = f"{ESC}0{TERM}"
BOLD = f"{ESC}1{TERM}"
ITALIC = f"{ESC}3{TERM}"
UNDERLINE = f"{ESC}4{TERM}"


# ============================================================================
# Theme Color Maps
# ============================================================================

def get_tqdm_color_for_theme(classic_color_hex: str, theme: str) -> str:
    """Map a classic tqdm color to the appropriate color for the given theme.

    This function translates the classic vibrant tqdm colors (used in 5 parallel bars)
    to colors appropriate for the selected theme.

    Args:
        classic_color_hex: Original color hex (e.g., '#FE797B' for red)
        theme: Theme name ('slopcore', 'classic', 'simple')

    Returns:
        Color hex appropriate for the theme, or None for no color
    """
    if theme == 'slopcore':
        # Map classic rainbow colors to 5-shade slopcore tqdm gradient
        # Evenly spaced interpolation from dark purple-blue → bright cyan
        color_map = {
            HEX_CLASSIC_PURPLE: HEX_SLOPCORE_TQDM_1,  # Purple → Deep purple-blue [Current Tweens - FASTEST]
            HEX_CLASSIC_BLUE: HEX_SLOPCORE_TQDM_2,    # Blue → Purple-blue [Current Steps - FAST]
            HEX_CLASSIC_GREEN: HEX_SLOPCORE_TQDM_3,   # Green → Mid blue [Total Steps - MEDIUM]
            HEX_CLASSIC_ORANGE: HEX_SLOPCORE_TQDM_4,  # Orange → Bright blue [Total Diffusion Frames - SLOW]
            HEX_CLASSIC_RED: HEX_SLOPCORE_TQDM_5,     # Red → Cyan [Total Frames - SLOWEST]
        }
        return color_map.get(classic_color_hex, HEX_SLOPCORE_TQDM_3)  # Default to mid blue
    elif theme == 'simple':
        # Simple theme: no color
        return None
    else:  # classic
        # Classic theme: keep original vibrant colors
        return classic_color_hex


def get_theme_colors(theme: str) -> dict:
    """Get color palette for specified theme.

    Args:
        theme: One of 'slopcore', 'classic', 'simple'

    Returns:
        Dictionary mapping semantic names to ANSI color codes
    """
    if theme == 'slopcore':
        return {
            'trace': SLOPCORE_1,         # Brightest purple-blue (ultra-verbose internals)
            'debug': SLOPCORE_2,         # Blue-purple (debugging info)
            'info': SLOPCORE_7,          # Bright cyan (normal operation, stands out)
            'warning': FUNCTIONAL_YELLOW, # Yellow (NOT slopcore - functional color)
            'error': FUNCTIONAL_RED,      # Red (NOT slopcore - functional color)
            'critical': SLOPCORE_5,      # Purple (critical failures)
            'header': SLOPCORE_3,        # Light purple
            'emphasis': SLOPCORE_4,      # Mid purple
            'reset': RESET_COLOR,
            'bold': BOLD,
            # All 7 shades available for gradients
            'shade_1': SLOPCORE_1,
            'shade_2': SLOPCORE_2,
            'shade_3': SLOPCORE_3,
            'shade_4': SLOPCORE_4,
            'shade_5': SLOPCORE_5,
            'shade_6': SLOPCORE_6,
            'shade_7': SLOPCORE_7,
        }
    elif theme == 'classic':
        return {
            'trace': CLASSIC_PURPLE,     # Purple (ultra-verbose)
            'debug': CLASSIC_YELLOW,     # Yellow (debugging)
            'info': CLASSIC_BLUE,        # Blue (normal operation)
            'warning': CLASSIC_ORANGE,   # Orange (warnings)
            'error': CLASSIC_RED,        # Red (errors)
            'critical': CLASSIC_RED,     # Red (critical)
            'header': CLASSIC_BLUE,
            'emphasis': CLASSIC_PURPLE,
            'reset': RESET_COLOR,
            'bold': BOLD,
        }
    else:  # simple
        return {
            'trace': '',
            'debug': '',
            'info': '',
            'warning': '',
            'error': '',
            'critical': '',
            'header': '',
            'emphasis': '',
            'reset': '',
            'bold': '',
        }
