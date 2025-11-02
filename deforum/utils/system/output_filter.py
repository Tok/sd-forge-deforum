"""Output filtering utilities for suppressing unwanted Forge messages.

Provides context managers to filter stdout/stderr during Deforum operations,
allowing us to suppress redundant or unnecessary output from Forge while
keeping Deforum's own output clean and informative.
"""

import sys
import io
from contextlib import contextmanager


class FilteredOutput(io.StringIO):
    """StringIO wrapper that filters out lines matching specific patterns."""

    def __init__(self, original_stream, filter_patterns):
        """Initialize filtered output stream.

        Args:
            original_stream: Original sys.stdout or sys.stderr
            filter_patterns: List of strings to filter out (exact match or substring)
        """
        super().__init__()
        self.original_stream = original_stream
        self.filter_patterns = filter_patterns

    def write(self, text):
        """Write text to original stream unless it matches filter patterns."""
        # Check if any filter pattern matches this text
        should_filter = any(pattern in text for pattern in self.filter_patterns)

        if not should_filter:
            # Pass through to original stream
            self.original_stream.write(text)
            self.original_stream.flush()

        return len(text)

    def flush(self):
        """Flush the original stream."""
        self.original_stream.flush()


@contextmanager
def suppress_forge_output(patterns=None):
    """Context manager to suppress specific Forge output patterns.

    Args:
        patterns: List of string patterns to filter out. Defaults to common
                 redundant Forge messages that Deforum already displays.

    Example:
        >>> with suppress_forge_output():
        ...     processing.process_images(p)  # Won't print "Distilled CFG Scale: 3.5"
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
        sys.stdout = FilteredOutput(original_stdout, patterns)
        yield
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
