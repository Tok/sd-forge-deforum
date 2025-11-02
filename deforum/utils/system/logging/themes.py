"""Console output theme definitions for Deforum.

Defines color palettes and styling for three themes:
- Slopcore: Modern blue→purple gradient (new default)
- Classic: Vibrant multi-color palette (legacy)
- Simple: Plain text, no colors
"""

from deforum.utils.image.color import hex_to_ansi_foreground as from_hex_color

# ============================================================================
# SLOPCORE THEME - Blue to Purple gradient (7 shades)
# ============================================================================
# Core slopcore identity: 7-shade gradient from bright blue through purple

HEX_SLOPCORE_1 = '#4A90E2'  # Bright blue
HEX_SLOPCORE_2 = '#5883D8'  # Blue-purple
HEX_SLOPCORE_3 = '#667EEA'  # Light purple (banner start)
HEX_SLOPCORE_4 = '#7B6DB8'  # Mid purple
HEX_SLOPCORE_5 = '#8F5CA0'  # Purple
HEX_SLOPCORE_6 = '#A353A8'  # Deep purple
HEX_SLOPCORE_7 = '#764BA2'  # Darkest purple (banner end)

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
        # Map classic rainbow colors to slopcore blue→purple gradient
        # Fast-moving "Current" bars (purple, blue) → bright blue side
        # Slow-moving "Total" bars (green, orange, red) → purple side
        color_map = {
            HEX_CLASSIC_PURPLE: HEX_SLOPCORE_1,    # Purple (#A587CA) → Bright blue (#4A90E2) [Current Tweens - FASTEST]
            HEX_CLASSIC_BLUE: HEX_SLOPCORE_2,      # Blue (#36CEDC) → Light blue (#5B9FD8) [Current Steps - FAST]
            HEX_CLASSIC_GREEN: HEX_SLOPCORE_5,     # Green (#8FE968) → Mid-deep purple (#8F5DA8) [Total Steps - MEDIUM]
            HEX_CLASSIC_ORANGE: HEX_SLOPCORE_6,    # Orange (#FFB750) → Deep purple (#A353A8) [Total Diffusion Frames - SLOW]
            HEX_CLASSIC_RED: HEX_SLOPCORE_7,       # Red (#FE797B) → Darkest purple (#764BA2) [Total Frames - SLOWEST]
        }
        return color_map.get(classic_color_hex, HEX_SLOPCORE_4)  # Default to mid purple
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
            'debug': SLOPCORE_1,        # Bright blue (lightest)
            'info': SLOPCORE_3,          # Light purple
            'warning': FUNCTIONAL_YELLOW, # Yellow (NOT slopcore - functional color)
            'error': FUNCTIONAL_RED,      # Red (NOT slopcore - functional color)
            'critical': SLOPCORE_7,      # Darkest purple
            'header': SLOPCORE_2,       # Blue-purple
            'emphasis': SLOPCORE_4,     # Mid purple
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
            'debug': CLASSIC_YELLOW,
            'info': CLASSIC_BLUE,
            'warning': CLASSIC_ORANGE,
            'error': CLASSIC_RED,
            'critical': CLASSIC_RED,
            'header': CLASSIC_BLUE,
            'emphasis': CLASSIC_PURPLE,
            'reset': RESET_COLOR,
            'bold': BOLD,
        }
    else:  # simple
        return {
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
