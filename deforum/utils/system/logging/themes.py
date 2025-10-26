"""Console output theme definitions for Deforum.

Defines color palettes and styling for three themes:
- Slopcore: Modern blue→purple gradient (new default)
- Classic: Vibrant multi-color palette (legacy)
- Simple: Plain text, no colors
"""

from deforum.utils.image.color import hex_to_ansi_foreground as from_hex_color

# ============================================================================
# SLOPCORE THEME - Blue to Purple gradient
# ============================================================================
# 7-shade gradient from bright blue through purple (inspired by banner gradient)

HEX_SLOPCORE_1 = '#4A90E2'  # Bright blue
HEX_SLOPCORE_2 = '#5883D8'  # Blue-purple
HEX_SLOPCORE_3 = '#667EEA'  # Light purple (banner start)
HEX_SLOPCORE_4 = '#7B6DB8'  # Mid purple
HEX_SLOPCORE_5 = '#8F5CA0'  # Purple
HEX_SLOPCORE_6 = '#A353A8'  # Deep purple
HEX_SLOPCORE_7 = '#764BA2'  # Darkest purple (banner end)

SLOPCORE_1 = from_hex_color(HEX_SLOPCORE_1)  # Bright blue
SLOPCORE_2 = from_hex_color(HEX_SLOPCORE_2)  # Blue-purple
SLOPCORE_3 = from_hex_color(HEX_SLOPCORE_3)  # Light purple
SLOPCORE_4 = from_hex_color(HEX_SLOPCORE_4)  # Mid purple
SLOPCORE_5 = from_hex_color(HEX_SLOPCORE_5)  # Purple
SLOPCORE_6 = from_hex_color(HEX_SLOPCORE_6)  # Deep purple
SLOPCORE_7 = from_hex_color(HEX_SLOPCORE_7)  # Darkest purple

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

def get_theme_colors(theme: str) -> dict:
    """Get color palette for specified theme.

    Args:
        theme: One of 'slopcore', 'classic', 'simple'

    Returns:
        Dictionary mapping semantic names to ANSI color codes
    """
    if theme == 'slopcore':
        return {
            'debug': SLOPCORE_1,      # Bright blue (lightest)
            'info': SLOPCORE_3,        # Light purple
            'warning': SLOPCORE_5,     # Mid purple
            'error': SLOPCORE_6,       # Deep purple
            'critical': SLOPCORE_7,    # Darkest purple
            'header': SLOPCORE_2,      # Blue-purple
            'emphasis': SLOPCORE_4,    # Mid purple
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
