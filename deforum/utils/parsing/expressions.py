"""Pure functions for expression parsing and evaluation.

This module contains functions for parsing and evaluating embedded expressions
in strings, following functional programming principles with no side effects.
"""

import re
from typing import Dict, Any

try:
    import numexpr

    NUMEXPR_AVAILABLE = True
except ImportError:
    NUMEXPR_AVAILABLE = False
    numexpr = None


def parse_embedded_expressions(value: str, variables: Dict[str, Any], delimiter: str = "`") -> str:
    """Parse and evaluate embedded expressions in a string.

    Finds expressions delimited by backticks and evaluates them using numexpr,
    substituting provided variables into the expressions.

    Args:
        value: String containing embedded expressions
        variables: Dictionary of variable names to values for substitution
        delimiter: Character used to delimit expressions (default: backtick)

    Returns:
        String with expressions evaluated and substituted

    Raises:
        RuntimeError: If numexpr is not available
        ValueError: If expression evaluation fails

    Examples:
        >>> parse_embedded_expressions("frame `t*2`", {"t": 5})
        'frame 10'
        >>> parse_embedded_expressions("total `max_f/2`", {"max_f": 100})
        'total 50'
        >>> parse_embedded_expressions("value `t+10` end", {"t": 5})
        'value 15 end'
        >>> parse_embedded_expressions("no expression", {"t": 5})
        'no expression'
    """
    if not NUMEXPR_AVAILABLE:
        raise RuntimeError("numexpr module is required for expression parsing but is not installed")

    # Create regex pattern to match delimited expressions
    pattern = re.escape(delimiter) + r".*?" + re.escape(delimiter)
    regex = re.compile(pattern)

    parsed_value = value
    for match in regex.finditer(value):
        matched_string = match.group(0)
        # Remove delimiters to get expression
        expression = matched_string.strip(delimiter)

        # Substitute variables in expression
        for var_name, var_value in variables.items():
            expression = expression.replace(var_name, str(var_value))

        # Strip whitespace before evaluation
        expression = expression.strip()

        # Evaluate expression
        try:
            result = numexpr.evaluate(expression)
            # Convert result to appropriate string representation
            if hasattr(result, "item"):  # numpy scalar
                result_str = str(result.item())
            else:
                result_str = str(result)

            parsed_value = parsed_value.replace(matched_string, result_str, 1)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {str(e)}") from e

    return parsed_value


def parse_frame_expression(value: str, frame_idx: int, max_frames: int) -> str:
    """Parse frame-specific expressions in a string.

    Convenience wrapper around parse_embedded_expressions specifically for
    frame-based expressions using 't' for current frame and 'max_f' for
    maximum frame count.

    Args:
        value: String containing frame expressions
        frame_idx: Current frame index
        max_frames: Maximum number of frames

    Returns:
        String with frame expressions evaluated

    Examples:
        >>> parse_frame_expression("frame `t`", 10, 100)
        'frame 10'
        >>> parse_frame_expression("progress `t/max_f`", 25, 100)
        'progress 0.25'
        >>> parse_frame_expression("calc `t*2 + max_f/10`", 5, 100)
        'calc 20.0'
        >>> parse_frame_expression("static text", 10, 100)
        'static text'
    """
    variables = {
        "t": frame_idx,
        "max_f": max_frames,
    }
    return parse_embedded_expressions(value, variables)


def has_embedded_expressions(value: str, delimiter: str = "`") -> bool:
    """Check if string contains embedded expressions.

    Args:
        value: String to check
        delimiter: Character used to delimit expressions

    Returns:
        True if string contains delimited expressions

    Examples:
        >>> has_embedded_expressions("frame `t`")
        True
        >>> has_embedded_expressions("static text")
        False
        >>> has_embedded_expressions("`expr1` and `expr2`")
        True
        >>> has_embedded_expressions("one ` but not two")
        False
    """
    pattern = re.escape(delimiter) + r".*?" + re.escape(delimiter)
    return bool(re.search(pattern, value))


def extract_expressions(value: str, delimiter: str = "`") -> list[str]:
    """Extract all embedded expressions from a string.

    Args:
        value: String containing expressions
        delimiter: Character used to delimit expressions

    Returns:
        List of expression strings (without delimiters)

    Examples:
        >>> extract_expressions("frame `t` at `t/max_f`")
        ['t', 't/max_f']
        >>> extract_expressions("no expressions here")
        []
        >>> extract_expressions("`a` `b` `c`")
        ['a', 'b', 'c']
        >>> extract_expressions("partial ` expression")
        []
    """
    pattern = re.escape(delimiter) + r"(.*?)" + re.escape(delimiter)
    matches = re.findall(pattern, value)
    return matches


def validate_expression(expression: str, variables: Dict[str, Any] = None) -> bool:
    """Validate if an expression can be evaluated.

    Args:
        expression: Expression string to validate
        variables: Optional dictionary of variables for substitution

    Returns:
        True if expression is valid and can be evaluated

    Examples:
        >>> validate_expression("1 + 1")
        True
        >>> validate_expression("t * 2", {"t": 5})
        True
        >>> validate_expression("invalid syntax !")
        False
        >>> validate_expression("")
        True
    """
    if not NUMEXPR_AVAILABLE:
        return False

    if not expression.strip():
        return True  # Empty expression is technically valid

    # Substitute variables if provided
    test_expr = expression
    if variables:
        for var_name, var_value in variables.items():
            test_expr = test_expr.replace(var_name, str(var_value))

    try:
        numexpr.evaluate(test_expr)
        return True
    except Exception:
        return False
