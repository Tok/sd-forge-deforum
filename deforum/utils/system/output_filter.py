"""Output filtering utilities for suppressing unwanted Forge messages.

Provides context managers to filter stdout/stderr during Deforum operations,
allowing us to suppress redundant or unnecessary output from Forge while
keeping Deforum's own output clean and informative.

Also provides message interception for dashboard integration.
"""

import sys
import io
from contextlib import contextmanager
from typing import List, Optional, Callable


class FilteredOutput(io.StringIO):
    """StringIO wrapper that filters out lines matching specific patterns."""

    def __init__(self, original_stream, filter_patterns, important_patterns=None, callback=None):
        """Initialize filtered output stream.

        Args:
            original_stream: Original sys.stdout or sys.stderr
            filter_patterns: List of strings to suppress completely
            important_patterns: List of patterns to route to callback instead of stdout
            callback: Optional function to call with intercepted messages
        """
        super().__init__()
        self.original_stream = original_stream
        self.filter_patterns = filter_patterns
        self.important_patterns = important_patterns or []
        self.callback = callback

    def write(self, text):
        """Write text to original stream unless it matches filter patterns."""
        # Check if should be completely suppressed
        should_suppress = any(pattern in text for pattern in self.filter_patterns)

        if should_suppress:
            return len(text)  # Suppress completely

        # Check if should be intercepted for dashboard
        is_important = any(pattern in text for pattern in self.important_patterns)

        if is_important and self.callback:
            # Send to callback instead of stdout
            self.callback(text)
            return len(text)

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
        # Default patterns to suppress (info already shown in Deforum's table)
        patterns = [
            "Distilled CFG Scale:",
            "Distilled CFG Scale will be ignored for Schnell",
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
