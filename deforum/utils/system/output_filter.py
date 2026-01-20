"""Output filtering utilities for suppressing unwanted Forge messages.

Provides context managers to filter stdout/stderr during Deforum operations,
allowing us to suppress redundant or unnecessary output from Forge while
keeping Deforum's own output clean and informative.

Also provides message interception for dashboard integration and color
replacement for theme consistency.
"""

import sys
import io
import re
from contextlib import contextmanager
from typing import List, Optional, Callable


# Forge Neo's hardcoded ANSI color codes (from modules/launch_utils.py)
FORGE_RESET = "\033[0m"      # Reset
FORGE_RED = "\033[0;31m"     # Red (errors)
FORGE_YELLOW = "\033[0;33m"  # Yellow (warnings/alerts)
FORGE_CYAN = "\033[0;36m"    # Cyan (info)
FORGE_GREY = "\033[0;90m"    # Grey (dim text)

# RichHandler also uses these ANSI codes
RICH_YELLOW = "\033[33m"     # RichHandler warning color
RICH_RED = "\033[31m"        # RichHandler error color


def replace_forge_colors(text: str) -> str:
    """Replace Forge's ANSI colors with theme-appropriate colors.

    When slopcore theme is active, replaces Forge's yellow/red/cyan with
    slopcore gradient colors to maintain visual consistency.

    Args:
        text: Text potentially containing Forge ANSI color codes

    Returns:
        Text with colors replaced according to active theme
    """
    try:
        from deforum.rendering.options import get_cli_theme
        theme = get_cli_theme()
    except:
        # During early initialization, default to not replacing
        return text

    if theme == 'slopcore':
        # Get slopcore theme colors
        try:
            from deforum.utils.system.logging.themes import get_theme_colors
            colors = get_theme_colors('slopcore')

            # Replace Forge's colors with slopcore equivalents
            # Yellow (warnings/alerts) → slopcore info color (cyan)
            text = text.replace(FORGE_YELLOW, colors['info'])
            text = text.replace(RICH_YELLOW, colors['info'])

            # Red (errors) → slopcore error color (functional red)
            text = text.replace(FORGE_RED, colors['error'])
            text = text.replace(RICH_RED, colors['error'])

            # Cyan (info) → slopcore info color (cyan)
            text = text.replace(FORGE_CYAN, colors['info'])

            # Grey (dim) → slopcore debug color
            text = text.replace(FORGE_GREY, colors['debug'])

        except Exception:
            # If theme colors unavailable, strip colors entirely
            text = re.sub(r'\033\[[0-9;]+m', '', text)

    elif theme == 'simple':
        # Simple theme: strip all ANSI color codes
        text = re.sub(r'\033\[[0-9;]+m', '', text)

    # Classic theme: pass through unchanged (keep Forge's colors)
    return text


class FilteredOutput(io.StringIO):
    """StringIO wrapper that filters out lines matching specific patterns and replaces colors."""

    def __init__(self, original_stream, filter_patterns, important_patterns=None, callback=None, replace_colors=True):
        """Initialize filtered output stream.

        Args:
            original_stream: Original sys.stdout or sys.stderr
            filter_patterns: List of strings to suppress completely
            important_patterns: List of patterns to route to callback instead of stdout
            callback: Optional function to call with intercepted messages
            replace_colors: Whether to replace Forge's colors with theme-appropriate colors
        """
        super().__init__()
        self.original_stream = original_stream
        self.filter_patterns = filter_patterns
        self.important_patterns = important_patterns or []
        self.callback = callback
        self.replace_colors = replace_colors
        self.last_suppressed = False  # Track if last write was suppressed

    def write(self, text):
        """Write text to original stream unless it matches filter patterns."""
        # Suppress standalone newlines that follow suppressed content
        if self.last_suppressed and text == '\n':
            self.last_suppressed = False
            return len(text)

        # Check if should be completely suppressed
        should_suppress = any(pattern in text for pattern in self.filter_patterns)

        if should_suppress:
            self.last_suppressed = True
            return len(text)  # Suppress completely

        self.last_suppressed = False

        # Check if should be intercepted for dashboard
        is_important = any(pattern in text for pattern in self.important_patterns)

        if is_important and self.callback:
            # Send to callback instead of stdout
            self.callback(text)
            return len(text)

        # Replace Forge's colors with theme-appropriate colors
        if self.replace_colors:
            text = replace_forge_colors(text)

        # Pass through to original stream
        self.original_stream.write(text)
        self.original_stream.flush()

        return len(text)

    def flush(self):
        """Flush the original stream."""
        self.original_stream.flush()


@contextmanager
def suppress_forge_output(patterns=None, important_patterns=None, callback=None):
    """Context manager to suppress specific Forge output patterns.

    Args:
        patterns: List of string patterns to filter out. Defaults to common
                 redundant Forge messages that Deforum already displays.
        important_patterns: Patterns to intercept and route to callback
        callback: Function to call with intercepted important messages

    Example:
        >>> with suppress_forge_output():
        ...     processing.process_images(p)  # Won't print "Distilled CFG Scale: 3.5"

        >>> def handle_memory(msg):
        ...     dashboard.memory.update_from_forge_message(msg)
        >>> with suppress_forge_output(important_patterns=["[Memory Management]"], callback=handle_memory):
        ...     processing.process_images(p)
    """
    if patterns is None:
        # Default patterns to suppress (info already shown in Deforum's table or too verbose)
        patterns = [
            "Distilled CFG Scale:",
            "Distilled CFG Scale will be ignored for Schnell",
            "[Unload]",  # VRAM unload messages (too verbose during generation)
            "Skipping unconditional conditioning",  # CFG=1 message (user knows this from table)
            "Done.\n",  # Forge backend memory management completion messages
            "Done.",  # Also catch without trailing newline
            "Unload model",  # Model unload messages
            "Memory cleanup has taken",  # Memory cleanup timing
            "Moving model(s) has taken",  # Model moving timing
        ]

    # Save original stdout
    original_stdout = sys.stdout

    try:
        # Replace stdout with filtered version
        sys.stdout = FilteredOutput(original_stdout, patterns, important_patterns, callback)
        yield
    finally:
        # Restore original stdout
        sys.stdout = original_stdout


def install_global_color_filter():
    """Install global color filter to replace Forge's colors with theme-appropriate colors.

    This replaces sys.stdout and sys.stderr with filtered versions that automatically
    convert Forge Neo's yellow/red/cyan ANSI codes to slopcore theme colors when active.

    Should be called once during Deforum initialization.
    """
    import sys

    # Only install if not already installed
    if hasattr(sys.stdout, '_deforum_color_filter'):
        return

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class ColorReplacementFilter:
        """Transparent wrapper that replaces Forge colors with theme colors."""

        def __init__(self, original_stream):
            self.original_stream = original_stream
            self._deforum_color_filter = True  # Marker to prevent double-installation

        def write(self, text):
            """Replace colors and forward to original stream."""
            text = replace_forge_colors(text)
            return self.original_stream.write(text)

        def flush(self):
            """Forward flush to original stream."""
            return self.original_stream.flush()

        def __getattr__(self, name):
            """Forward all other attributes to original stream."""
            return getattr(self.original_stream, name)

    sys.stdout = ColorReplacementFilter(original_stdout)
    sys.stderr = ColorReplacementFilter(original_stderr)
