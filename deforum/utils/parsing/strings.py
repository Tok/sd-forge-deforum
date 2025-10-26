"""Pure functions for string manipulation and formatting.

This module contains string-related pure functions extracted from various
deforum_helpers modules, following functional programming principles with
no side effects.
"""

import platform
from typing import Any


def get_os() -> str:
    """Get operating system name.

    Returns:
        Operating system name: "Windows", "Linux", "Mac", or "Unknown"

    Examples:
        >>> get_os() in ["Windows", "Linux", "Mac", "Unknown"]
        True
    """
    return {"Windows": "Windows", "Linux": "Linux", "Darwin": "Mac"}.get(
        platform.system(), "Unknown"
    )


def custom_placeholder_format(value_dict: dict, placeholder_match) -> str:
    """Format placeholder value for string substitution.

    Extracts value from dictionary based on regex match and formats it:
    - Converts key to lowercase for lookup
    - Returns "_" if value is None
    - For dict values, uses first key's value
    - Limits output to 50 characters

    Args:
        value_dict: Dictionary of values keyed by placeholder names
        placeholder_match: Regex match object with group(1) = placeholder key

    Returns:
        Formatted string value (max 50 chars)

    Examples:
        >>> import re
        >>> match = re.match(r'{(\\w+)}', '{name}')
        >>> custom_placeholder_format({'name': 'test'}, match)
        'test'
        >>> custom_placeholder_format({'name': None}, match)
        '_'
        >>> custom_placeholder_format({'name': 'a' * 60}, match)[:50]
        'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    """
    key = placeholder_match.group(1).lower()
    value = value_dict.get(key, key) or "_"

    if isinstance(value, dict) and value:
        first_key = list(value.keys())[0]
        value = (
            str(value[first_key][0])
            if isinstance(value[first_key], list) and value[first_key]
            else str(value[first_key])
        )

    return str(value)[:50]


def clean_gradio_path_strings(input_str: str | Any) -> str | Any:
    """Remove surrounding quotes from Gradio path strings.

    Gradio may wrap paths in double quotes. This function removes them.

    Args:
        input_str: Input string (or any other type)

    Returns:
        String without surrounding quotes, or original value if not a quoted string

    Examples:
        >>> clean_gradio_path_strings('"/path/to/file"')
        '/path/to/file'
        >>> clean_gradio_path_strings('regular string')
        'regular string'
        >>> clean_gradio_path_strings(42)
        42
    """
    if isinstance(input_str, str) and input_str.startswith('"') and input_str.endswith('"'):
        return input_str[1:-1]
    else:
        return input_str


def tick_or_cross(value: bool, use_simple_symbols: bool = True) -> str:
    """Convert boolean to tick/cross symbol.

    Args:
        value: Boolean value to convert
        use_simple_symbols: If True, use simple ✔/✖. If False, use emoji ✅/❌

    Returns:
        Tick symbol if True, cross symbol if False

    Examples:
        >>> tick_or_cross(True)
        '✔'
        >>> tick_or_cross(False)
        '✖'
        >>> tick_or_cross(True, use_simple_symbols=False)
        '\u00002705'
        >>> tick_or_cross(False, use_simple_symbols=False)
        '\u0000274C'
    """
    tick = "✔" if use_simple_symbols else "\U00002705"  # Check mark ✅
    cross = "✖" if use_simple_symbols else "\U0000274c"  # Cross mark ❌
    return tick if value else cross


def sanitize_keyframe_value(value: str) -> str:
    """Remove quotes and parentheses from keyframe value string.

    Used for cleaning animation keyframe values before parsing.

    Args:
        value: Raw keyframe value string

    Returns:
        Sanitized string with quotes and parens removed

    Examples:
        >>> sanitize_keyframe_value("'(3.14)'")
        '3.14'
        >>> sanitize_keyframe_value('"test"')
        'test'
        >>> sanitize_keyframe_value('(1+2)')
        '1+2'
    """
    return value.replace("'", "").replace('"', "").replace("(", "").replace(")", "")
