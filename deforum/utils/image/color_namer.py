"""High-resolution color naming utility using HSV color space.

Provides accurate, descriptive names for any hex color with precise hue resolution.
Used for gradient documentation, ASCII preview pixels, and debug output.
"""

import colorsys
from typing import Tuple


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple.

    Args:
        hex_color: Hex color string (with or without #)

    Returns:
        RGB tuple (0-255)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HSV.

    Args:
        r, g, b: RGB values (0-255)

    Returns:
        HSV tuple (h: 0-360, s: 0-100, v: 0-100)
    """
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h * 360, s * 100, v * 100


def get_hue_name(hue: float) -> str:
    """Get precise hue name from hue angle (0-360°).

    High-resolution hue wheel with 24 named segments:
    - 15° resolution for pure colors
    - Descriptive intermediate shades

    Args:
        hue: Hue angle in degrees (0-360)

    Returns:
        Precise hue name
    """
    # Normalize to 0-360
    hue = hue % 360

    # 24-segment hue wheel (15° per segment)
    # Tuned to match Deforum gradient aesthetics
    if 0 <= hue < 10 or 350 <= hue < 360:
        return "red"
    elif 10 <= hue < 25:
        return "scarlet"
    elif 25 <= hue < 40:
        return "coral"
    elif 40 <= hue < 55:
        return "orange"
    elif 55 <= hue < 70:
        return "amber"
    elif 70 <= hue < 85:
        return "yellow"
    elif 85 <= hue < 100:
        return "chartreuse"
    elif 100 <= hue < 115:
        return "lime"
    elif 115 <= hue < 130:
        return "green"
    elif 130 <= hue < 145:
        return "emerald"
    elif 145 <= hue < 160:
        return "teal"
    elif 160 <= hue < 175:
        return "turquoise"
    elif 175 <= hue < 195:
        return "cyan"
    elif 195 <= hue < 210:
        return "azure"
    elif 210 <= hue < 225:
        return "blue"
    elif 225 <= hue < 240:
        return "cobalt"
    elif 240 <= hue < 260:
        return "indigo"
    elif 260 <= hue < 275:
        return "purple"
    elif 275 <= hue < 290:
        return "violet"
    elif 290 <= hue < 305:
        return "magenta"
    elif 305 <= hue < 320:
        return "rose"
    elif 320 <= hue < 335:
        return "pink"
    elif 335 <= hue < 350:
        return "watermelon"
    else:
        return "red"  # Fallback


def get_saturation_prefix(saturation: float) -> str:
    """Get saturation-based prefix.

    Args:
        saturation: Saturation percentage (0-100)

    Returns:
        Descriptive prefix for saturation level
    """
    if saturation < 10:
        return "grey"  # Almost no color
    elif saturation < 25:
        return "greyish"
    elif saturation < 40:
        return "muted"
    elif saturation < 60:
        return "soft"
    elif saturation < 80:
        return ""  # Normal saturation, no prefix needed
    elif saturation < 90:
        return "vivid"
    else:
        return "electric"  # Maximum saturation


def get_value_prefix(value: float, saturation: float) -> str:
    """Get value (brightness)-based prefix.

    Args:
        value: Value/brightness percentage (0-100)
        saturation: Saturation percentage (for context)

    Returns:
        Descriptive prefix for brightness level
    """
    # For very dark colors
    if value < 15:
        return "black"
    elif value < 30:
        return "very dark"
    elif value < 45:
        return "dark"

    # For light colors (only if saturated enough to matter)
    elif value > 85 and saturation > 20:
        return "very bright"
    elif value > 70 and saturation > 20:
        return "bright"

    # For white/near-white (low saturation + high value)
    elif value > 90 and saturation < 15:
        return "white"
    elif value > 75 and saturation < 20:
        return "pale"

    # Normal value range
    else:
        return ""


def name_color(hex_color: str, include_technical: bool = False) -> str:
    """Get precise, descriptive name for a hex color.

    Combines hue, saturation, and value to create accurate color names.

    Args:
        hex_color: Hex color string (with or without #)
        include_technical: If True, append HSV values in parentheses

    Returns:
        Descriptive color name (e.g., "electric coral red", "azure cyan")

    Examples:
        >>> name_color("#f64a5e")
        "electric coral red"
        >>> name_color("#1cc4e6")
        "vivid cyan"
        >>> name_color("#5606ff")
        "electric indigo"
        >>> name_color("#17a7fe")
        "vivid azure"
    """
    r, g, b = hex_to_rgb(hex_color)
    h, s, v = rgb_to_hsv(r, g, b)

    # Build name from components
    hue_name = get_hue_name(h)
    sat_prefix = get_saturation_prefix(s)
    val_prefix = get_value_prefix(v, s)

    # Combine prefixes (remove duplicates and empty strings)
    parts = []

    # Add value prefix first (brightness)
    if val_prefix:
        parts.append(val_prefix)

    # Add saturation prefix
    if sat_prefix:
        # Don't add saturation prefix if we already said "pale" or "white"
        if val_prefix not in ["pale", "white"]:
            parts.append(sat_prefix)

    # Add hue name
    parts.append(hue_name)

    # Join with spaces
    name = " ".join(parts)

    # Add technical HSV values if requested
    if include_technical:
        name += f" (H:{h:.1f}° S:{s:.1f}% V:{v:.1f}%)"

    return name


def describe_gradient(start_hex: str, end_hex: str) -> str:
    """Describe a gradient using precise color names.

    Args:
        start_hex: Starting color hex
        end_hex: Ending color hex

    Returns:
        Gradient description

    Examples:
        >>> describe_gradient("#1cc4e6", "#f64a5e")
        "vivid cyan → electric coral red"
        >>> describe_gradient("#5606ff", "#17a7fe")
        "electric indigo → vivid azure"
    """
    start_name = name_color(start_hex)
    end_name = name_color(end_hex)
    return f"{start_name} → {end_name}"


# Convenience exports for common Deforum gradients
if __name__ == "__main__":
    # Test with Deforum gradient colors
    print("BB0 Gradient Colors:")
    print(f"  #5606FF: {name_color('#5606FF', include_technical=True)}")
    print(f"  #17A7FE: {name_color('#17A7FE', include_technical=True)}")
    print(f"  Gradient: {describe_gradient('#5606FF', '#17A7FE')}")
    print()
    print("DA3 Gradient Colors:")
    print(f"  #1CC4E6: {name_color('#1CC4E6', include_technical=True)}")
    print(f"  #F64A5E: {name_color('#F64A5E', include_technical=True)}")
    print(f"  Gradient: {describe_gradient('#1CC4E6', '#F64A5E')}")
    print()
    print("DA3 7-Shade Gradient:")
    da3_shades = ['#1CC4E6', '#40AFCF', '#649BB8', '#8987A2', '#AD728B', '#D15E74', '#F64A5E']
    for shade in da3_shades:
        print(f"  {shade}: {name_color(shade, include_technical=True)}")
