"""Console output theme definitions for Deforum.

Defines color palettes and styling for three themes:
- Slopcore: Modern blue→purple gradient (new default)
- Classic: Vibrant multi-color palette (legacy)
- Simple: Plain text, no colors
"""

from deforum.utils.image.color import hex_to_ansi_foreground as from_hex_color

# ============================================================================
# SLOPCORE THEME - Authentic BLANK BANSHEE 0 gradient (Default)
# ============================================================================
# Exact colors pipetted from BB0 album cover
# Original album gradient: #5606ff (electric purple, top) → #17a7fe (azure cyan, bottom), straight vertical

# 7-shade gradient (for CLI banner, charts, general use)
HEX_SLOPCORE_BB0_1 = '#5606FF'  # Electric purple (album top)
HEX_SLOPCORE_BB0_2 = '#4C21FF'  # Purple-blue
HEX_SLOPCORE_BB0_3 = '#413CFF'  # Blue-purple (banner start)
HEX_SLOPCORE_BB0_4 = '#3757FF'  # Mid blue
HEX_SLOPCORE_BB0_5 = '#2C71FE'  # Blue
HEX_SLOPCORE_BB0_6 = '#228CFE'  # Bright blue
HEX_SLOPCORE_BB0_7 = '#17A7FE'  # Azure cyan (album bottom, banner end)

# 5-shade gradient (specifically for tqdm progress bars)
# Maps to the 5 parallel dashboard bars in display order
HEX_SLOPCORE_BB0_TQDM_1 = '#5606FF'  # Electric purple [Current Tweens - FASTEST]
HEX_SLOPCORE_BB0_TQDM_2 = '#462EFF'  # Purple-blue [Current Steps - FAST]
HEX_SLOPCORE_BB0_TQDM_3 = '#3757FF'  # Mid blue [Total Steps - MEDIUM]
HEX_SLOPCORE_BB0_TQDM_4 = '#277FFE'  # Bright blue [Total Diffusion Frames - SLOW]
HEX_SLOPCORE_BB0_TQDM_5 = '#17A7FE'  # Azure cyan [Total Frames - SLOWEST]

# ============================================================================
# SLOPCORE THEME - Depth Anything V3 gradient (DA3 mode)
# ============================================================================
# Exact colors pipetted from Depth Anything V3 website header
# Original gradient: #1cc4e6 (electric cyan) → #f64a5e (coral red), left to right
# Reference: https://depth-anything-3.github.io/

# 7-shade gradient (for CLI banner, charts, general use)
HEX_SLOPCORE_DA3_1 = '#1CC4E6'  # Electric cyan (gradient start)
HEX_SLOPCORE_DA3_2 = '#40AFCF'  # Cyan-blue blend
HEX_SLOPCORE_DA3_3 = '#649BB8'  # Blue-teal
HEX_SLOPCORE_DA3_4 = '#8987A2'  # Mid purple-grey
HEX_SLOPCORE_DA3_5 = '#AD728B'  # Purple-pink
HEX_SLOPCORE_DA3_6 = '#D15E74'  # Rose pink
HEX_SLOPCORE_DA3_7 = '#F64A5E'  # Coral red (gradient end)

# 5-shade gradient (specifically for tqdm progress bars)
# Maps to the 5 parallel dashboard bars in display order
HEX_SLOPCORE_DA3_TQDM_1 = '#1CC4E6'  # Electric cyan [Current Tweens - FASTEST]
HEX_SLOPCORE_DA3_TQDM_2 = '#52A5C4'  # Blue-teal [Current Steps - FAST]
HEX_SLOPCORE_DA3_TQDM_3 = '#8987A2'  # Mid purple-grey [Total Steps - MEDIUM]
HEX_SLOPCORE_DA3_TQDM_4 = '#BF6880'  # Pink-purple [Total Diffusion Frames - SLOW]
HEX_SLOPCORE_DA3_TQDM_5 = '#F64A5E'  # Coral red [Total Frames - SLOWEST]

# Global DA3 colors for UI buttons and DA3/3DGS-related features
# Using key gradient points for visual consistency
HEX_DA3_CYAN = HEX_SLOPCORE_DA3_1      # '#1CC4E6' - Electric cyan (DA3 start)
HEX_DA3_BLUE = HEX_SLOPCORE_DA3_2      # '#40AFCF' - Cyan-blue
HEX_DA3_PURPLE = HEX_SLOPCORE_DA3_4    # '#8987A2' - Mid purple-grey
HEX_DA3_PINK = HEX_SLOPCORE_DA3_6      # '#D15E74' - Rose pink
HEX_DA3_RED = HEX_SLOPCORE_DA3_7       # '#F64A5E' - Coral red (DA3 end)

# ============================================================================
# Active Slopcore Gradient Selection
# ============================================================================
# Default to BB0 gradient (backward compatibility)
# Can be dynamically switched to DA3 gradient when DA3-3DGS is active

HEX_SLOPCORE_1 = HEX_SLOPCORE_BB0_1
HEX_SLOPCORE_2 = HEX_SLOPCORE_BB0_2
HEX_SLOPCORE_3 = HEX_SLOPCORE_BB0_3
HEX_SLOPCORE_4 = HEX_SLOPCORE_BB0_4
HEX_SLOPCORE_5 = HEX_SLOPCORE_BB0_5
HEX_SLOPCORE_6 = HEX_SLOPCORE_BB0_6
HEX_SLOPCORE_7 = HEX_SLOPCORE_BB0_7

HEX_SLOPCORE_TQDM_1 = HEX_SLOPCORE_BB0_TQDM_1
HEX_SLOPCORE_TQDM_2 = HEX_SLOPCORE_BB0_TQDM_2
HEX_SLOPCORE_TQDM_3 = HEX_SLOPCORE_BB0_TQDM_3
HEX_SLOPCORE_TQDM_4 = HEX_SLOPCORE_BB0_TQDM_4
HEX_SLOPCORE_TQDM_5 = HEX_SLOPCORE_BB0_TQDM_5

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
        # TRACE, DEBUG, INFO, WARNING, ERROR always use BB0 colors (fixed, not gradient-dependent)
        # This ensures consistent log colors regardless of active gradient (BB0/DA3)
        # Gradient progression: Purple → Blue → Cyan (matching BB0 album cover)
        trace_color = from_hex_color(HEX_SLOPCORE_BB0_3)   # Always BB0 blue-purple (#413CFF)
        debug_color = from_hex_color(HEX_SLOPCORE_BB0_2)   # Always BB0 purple-blue (#4C21FF)
        info_color = from_hex_color(HEX_SLOPCORE_BB0_7)    # Always BB0 cyan (#17A7FE)

        return {
            'trace': trace_color,        # BB0 bright blue (ultra-verbose internals)
            'debug': debug_color,        # BB0 blue (debugging info)
            'info': info_color,          # BB0 cyan (normal operation)
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


# ============================================================================
# Slopcore Gradient Switching
# ============================================================================

# Available slopcore gradient variants
SLOPCORE_GRADIENTS = {
    'BB0': {
        '7_shade': [HEX_SLOPCORE_BB0_1, HEX_SLOPCORE_BB0_2, HEX_SLOPCORE_BB0_3,
                    HEX_SLOPCORE_BB0_4, HEX_SLOPCORE_BB0_5, HEX_SLOPCORE_BB0_6, HEX_SLOPCORE_BB0_7],
        '5_tqdm': [HEX_SLOPCORE_BB0_TQDM_1, HEX_SLOPCORE_BB0_TQDM_2, HEX_SLOPCORE_BB0_TQDM_3,
                   HEX_SLOPCORE_BB0_TQDM_4, HEX_SLOPCORE_BB0_TQDM_5],
        'description': 'BLANK BANSHEE 0 - Electric purple → Azure cyan',
        'reference': 'BB0 album cover gradient (#5606ff → #17a7fe)'
    },
    'DA3': {
        '7_shade': [HEX_SLOPCORE_DA3_1, HEX_SLOPCORE_DA3_2, HEX_SLOPCORE_DA3_3,
                    HEX_SLOPCORE_DA3_4, HEX_SLOPCORE_DA3_5, HEX_SLOPCORE_DA3_6, HEX_SLOPCORE_DA3_7],
        '5_tqdm': [HEX_SLOPCORE_DA3_TQDM_1, HEX_SLOPCORE_DA3_TQDM_2, HEX_SLOPCORE_DA3_TQDM_3,
                   HEX_SLOPCORE_DA3_TQDM_4, HEX_SLOPCORE_DA3_TQDM_5],
        'description': 'Depth Anything V3 - Electric cyan → Coral red',
        'reference': 'https://depth-anything-3.github.io/ (#1cc4e6 → #f64a5e)'
    }
}

_active_gradient = 'BB0'  # Default gradient


def set_slopcore_gradient(gradient_name: str) -> None:
    """Dynamically switch the active slopcore gradient.

    Args:
        gradient_name: Name of gradient variant ('BB0', 'DA3', etc.)

    Raises:
        ValueError: If gradient_name is not recognized
    """
    global _active_gradient
    global HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4
    global HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7
    global HEX_SLOPCORE_TQDM_1, HEX_SLOPCORE_TQDM_2, HEX_SLOPCORE_TQDM_3
    global HEX_SLOPCORE_TQDM_4, HEX_SLOPCORE_TQDM_5
    global SLOPCORE_1, SLOPCORE_2, SLOPCORE_3, SLOPCORE_4
    global SLOPCORE_5, SLOPCORE_6, SLOPCORE_7

    if gradient_name not in SLOPCORE_GRADIENTS:
        raise ValueError(f"Unknown gradient: {gradient_name}. Available: {list(SLOPCORE_GRADIENTS.keys())}")

    _active_gradient = gradient_name
    gradient = SLOPCORE_GRADIENTS[gradient_name]

    # Update 7-shade gradient
    shades = gradient['7_shade']
    HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4, \
    HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7 = shades

    # Update 5-shade tqdm gradient
    tqdm_shades = gradient['5_tqdm']
    HEX_SLOPCORE_TQDM_1, HEX_SLOPCORE_TQDM_2, HEX_SLOPCORE_TQDM_3, \
    HEX_SLOPCORE_TQDM_4, HEX_SLOPCORE_TQDM_5 = tqdm_shades

    # Re-convert hex to ANSI
    SLOPCORE_1 = from_hex_color(HEX_SLOPCORE_1)
    SLOPCORE_2 = from_hex_color(HEX_SLOPCORE_2)
    SLOPCORE_3 = from_hex_color(HEX_SLOPCORE_3)
    SLOPCORE_4 = from_hex_color(HEX_SLOPCORE_4)
    SLOPCORE_5 = from_hex_color(HEX_SLOPCORE_5)
    SLOPCORE_6 = from_hex_color(HEX_SLOPCORE_6)
    SLOPCORE_7 = from_hex_color(HEX_SLOPCORE_7)


def get_active_gradient() -> str:
    """Get the name of the currently active slopcore gradient.

    Returns:
        Gradient name ('BB0', 'DA3', etc.)
    """
    return _active_gradient


def get_random_slopcore_gradient() -> str:
    """Randomly select a slopcore gradient variant.

    Returns:
        Random gradient name from available gradients
    """
    import random
    return random.choice(list(SLOPCORE_GRADIENTS.keys()))


def list_slopcore_gradients() -> dict:
    """Get all available slopcore gradient variants with metadata.

    Returns:
        Dictionary of gradient names to metadata (description, reference)
    """
    return {
        name: {
            'description': data['description'],
            'reference': data['reference']
        }
        for name, data in SLOPCORE_GRADIENTS.items()
    }
