"""Pure functions for type conversion and random name generation.

This module contains utility functions extracted from
scripts/deforum_helpers/wan/utils/core_utils.py, following functional
programming principles with no side effects.
"""

import binascii
import os


def generate_random_name(length: int = 8, suffix: str = "") -> str:
    """Generate random filename using secure random bytes.

    Creates a random hexadecimal name of specified length using os.urandom()
    for cryptographically secure random generation. Optionally adds a suffix
    with automatic dot prepending.

    Args:
        length: Number of random bytes to generate (hex string will be 2x this)
        suffix: Optional file extension/suffix (dot auto-prepended if missing)

    Returns:
        Random hexadecimal string with optional suffix

    Examples:
        >>> name = generate_random_name()
        >>> len(name)
        16
        >>> name = generate_random_name(4, '.mp4')
        >>> name.endswith('.mp4')
        True
        >>> name = generate_random_name(4, 'mp4')  # Auto-prepends dot
        >>> name.endswith('.mp4')
        True
        >>> # Actual random output varies:
        >>> # generate_random_name(4, '.txt')
        >>> # 'a3f5b2c1.txt'
    """
    hex_name = binascii.b2a_hex(os.urandom(length)).decode("utf-8")

    if suffix:
        if not suffix.startswith("."):
            suffix = "." + suffix
        return hex_name + suffix

    return hex_name


def string_to_boolean(value: str | bool) -> bool:
    """Convert string to boolean value.

    Converts common string representations to boolean values.
    If input is already boolean, returns it unchanged.
    Case-insensitive matching for string values.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        value: String or boolean to convert

    Returns:
        Boolean value

    Raises:
        ValueError: If string cannot be converted to boolean

    Examples:
        >>> string_to_boolean('yes')
        True
        >>> string_to_boolean('NO')
        False
        >>> string_to_boolean('1')
        True
        >>> string_to_boolean('0')
        False
        >>> string_to_boolean(True)
        True
        >>> string_to_boolean('maybe')
        Traceback (most recent call last):
        ...
        ValueError: Boolean value expected (True/False), got: maybe
    """
    if isinstance(value, bool):
        return value

    value_lower = value.lower()

    if value_lower in ("yes", "true", "t", "y", "1"):
        return True
    elif value_lower in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"Boolean value expected (True/False), got: {value}")
