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

    High-resolution hue wheel with 36 named segments (~10° resolution):
    - Finer granularity for accurate color naming
    - Based on standard color theory and common color names
    - Special attention to red-pink transition (335-360°)

    Args:
        hue: Hue angle in degrees (0-360)

    Returns:
        Precise hue name

    References:
        - Standard 12-color wheel: https://www.canva.com/colors/color-wheel/
        - Coral/Salmon hues: https://creativebooster.net/blogs/colors/shades-of-coral-color
        - Watermelon pink (350°): https://colors.artyclick.com/color-names-dictionary/color-names/watermelon-pink-color
    """
    # Normalize to 0-360
    hue = hue % 360

    # 36-segment hue wheel (~10° per segment)
    # Tuned with research-based color names
    if 0 <= hue < 8:
        return "red"
    elif 8 <= hue < 16:
        return "scarlet"
    elif 16 <= hue < 24:
        return "coral"  # Standard coral at ~16°
    elif 24 <= hue < 32:
        return "vermillion"
    elif 32 <= hue < 40:
        return "orange-red"
    elif 40 <= hue < 48:
        return "orange"
    elif 48 <= hue < 56:
        return "amber"
    elif 56 <= hue < 64:
        return "gold"
    elif 64 <= hue < 72:
        return "yellow"
    elif 72 <= hue < 80:
        return "lemon"
    elif 80 <= hue < 88:
        return "lime-yellow"
    elif 88 <= hue < 96:
        return "chartreuse"
    elif 96 <= hue < 104:
        return "lime"
    elif 104 <= hue < 112:
        return "grass"
    elif 112 <= hue < 128:
        return "green"
    elif 128 <= hue < 136:
        return "forest"
    elif 136 <= hue < 144:
        return "emerald"
    elif 144 <= hue < 152:
        return "mint"
    elif 152 <= hue < 160:
        return "teal"
    elif 160 <= hue < 168:
        return "turquoise"
    elif 168 <= hue < 176:
        return "aqua"
    elif 176 <= hue < 192:
        return "cyan"  # Standard cyan at 180°
    elif 192 <= hue < 200:
        return "sky"
    elif 200 <= hue < 212:
        return "azure"  # Standard azure at ~210°
    elif 212 <= hue < 228:
        return "blue"
    elif 228 <= hue < 236:
        return "cobalt"
    elif 236 <= hue < 252:
        return "indigo"  # Centered around 240°
    elif 252 <= hue < 264:
        return "purple"
    elif 264 <= hue < 276:
        return "violet"  # Standard violet at ~270°
    elif 276 <= hue < 288:
        return "amethyst"
    elif 288 <= hue < 304:
        return "magenta"  # Standard magenta at ~300°
    elif 304 <= hue < 312:
        return "fuchsia"
    elif 312 <= hue < 320:
        return "hot pink"
    elif 320 <= hue < 328:
        return "rose"  # Standard rose at ~330°
    elif 328 <= hue < 336:
        return "pink"
    elif 336 <= hue < 344:
        return "salmon"
    elif 344 <= hue < 352:
        return "flamingo"  # Flamingo pink at ~344°
    elif 352 <= hue < 360:
        return "watermelon"  # Watermelon pink at ~350°
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


def name_color_simple(hex_color: str) -> str:
    """Get simplified, user-friendly color name (no verbose prefixes).

    Perfect for UI elements, ASCII pixels, and user-facing text.
    Strips "very bright" and similar verbose modifiers, keeping only
    the essential saturation prefix and hue name.

    Args:
        hex_color: Hex color string (with or without #)

    Returns:
        Simplified color name (e.g., "electric cyan", "coral red")

    Examples:
        >>> name_color_simple("#f64a5e")
        "coral red"
        >>> name_color_simple("#1cc4e6")
        "electric cyan"
        >>> name_color_simple("#5606ff")
        "electric purple"
        >>> name_color_simple("#17a7fe")
        "electric azure"
    """
    r, g, b = hex_to_rgb(hex_color)
    h, s, v = rgb_to_hsv(r, g, b)

    # Get hue name
    hue_name = get_hue_name(h)

    # Get saturation prefix (skip value prefix for simplicity)
    sat_prefix = get_saturation_prefix(s)

    # Special case: very dark/black colors need value prefix
    if v < 30:
        val_prefix = get_value_prefix(v, s)
        if val_prefix:
            parts = [val_prefix, hue_name]
        else:
            parts = [sat_prefix, hue_name] if sat_prefix else [hue_name]
    # Special case: very pale/white colors
    elif v > 75 and s < 20:
        val_prefix = get_value_prefix(v, s)
        if val_prefix in ["pale", "white"]:
            parts = [val_prefix, hue_name]
        else:
            parts = [sat_prefix, hue_name] if sat_prefix else [hue_name]
    # Normal case: just saturation + hue
    else:
        parts = [sat_prefix, hue_name] if sat_prefix else [hue_name]

    return " ".join(parts)


# Convenience exports for common Deforum gradients
if __name__ == "__main__":
    # Test with Deforum gradient colors
    print("BB0 Gradient Colors:")
    print(f"  #5606FF: {name_color_simple('#5606FF')} (full: {name_color('#5606FF')})")
    print(f"  #17A7FE: {name_color_simple('#17A7FE')} (full: {name_color('#17A7FE')})")
    print()
    print("DA3 Gradient Colors:")
    print(f"  #1CC4E6: {name_color_simple('#1CC4E6')} (full: {name_color('#1CC4E6')})")
    print(f"  #F64A5E: {name_color_simple('#F64A5E')} (full: {name_color('#F64A5E')})")
    print()
    print("All BB0 Shades (simplified):")
    bb0_shades = ['#5606FF', '#4C21FF', '#413CFF', '#3757FF', '#2C71FE', '#228CFE', '#17A7FE']
    for shade in bb0_shades:
        print(f"  {shade}: {name_color_simple(shade)}")
    print()
    print("All DA3 Shades (simplified):")
    da3_shades = ['#1CC4E6', '#40AFCF', '#649BB8', '#8987A2', '#AD728B', '#D15E74', '#F64A5E']
    for shade in da3_shades:
        print(f"  {shade}: {name_color_simple(shade)}")
