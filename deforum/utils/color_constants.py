"""
Color constants for terminal output.
Extracted from log_utils to avoid circular dependencies.
"""

ESC = "\033["  # ANSI escape character with bracket. Same as "\x1b[".
TERM = "m"  # ANSI terminator

TEXT = "38;2;"
RESET_COLOR = f"{ESC}0{TERM}"

HEX_RED = '#FE797B'
HEX_ORANGE = '#FFB750'
HEX_YELLOW = '#FFEA56'
HEX_GREEN = '#8FE968'
HEX_BLUE = '#36CEDC'
HEX_CYAN = '#00FFFF'
HEX_PURPLE = '#A587CA'


def from_hex_color(hex_color):
    def _hex_to_rgb(color):
        color = color.lstrip('#')  # Remove '#' if present
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    r, g, b = _hex_to_rgb(hex_color)
    return f"{ESC}{TEXT}{r};{g};{b}{TERM}"


RED = from_hex_color(HEX_RED)
ORANGE = from_hex_color(HEX_ORANGE)
YELLOW = from_hex_color(HEX_YELLOW)
GREEN = from_hex_color(HEX_GREEN)
BLUE = from_hex_color(HEX_BLUE)
CYAN = from_hex_color(HEX_CYAN)
PURPLE = from_hex_color(HEX_PURPLE)

BOLD = f"{ESC}1{TERM}"
UNDERLINE = f"{ESC}4{TERM}" 