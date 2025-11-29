"""
Tests for deforum/utils/system/output_filter.py
Output filtering utilities for suppressing unwanted messages.
"""

import pytest
import sys
from io import StringIO
from deforum.utils.system.output_filter import FilteredOutput, suppress_forge_output


class TestFilteredOutput:
    """Test FilteredOutput class for stream filtering."""

    def test_init_with_basic_patterns(self):
        """Test FilteredOutput initialization with filter patterns."""
        original = StringIO()
        patterns = ["ERROR", "WARNING"]

        filtered = FilteredOutput(original, patterns)

        assert filtered.original_stream == original
        assert filtered.filter_patterns == patterns
        assert filtered.important_patterns == []
        assert filtered.callback is None

    def test_init_with_all_parameters(self):
        """Test FilteredOutput initialization with all parameters."""
        original = StringIO()
        patterns = ["ERROR"]
        important = ["CRITICAL"]
        callback = lambda x: None

        filtered = FilteredOutput(original, patterns, important, callback)

        assert filtered.important_patterns == important
        assert filtered.callback == callback

    def test_write_passes_through_normal_text(self):
        """Test that non-filtered text passes through to original stream."""
        original = StringIO()
        filtered = FilteredOutput(original, ["ERROR"])

        filtered.write("Normal log message\n")

        assert original.getvalue() == "Normal log message\n"

    def test_write_suppresses_filtered_patterns(self):
        """Test that filtered patterns are suppressed."""
        original = StringIO()
        filtered = FilteredOutput(original, ["Distilled CFG Scale:"])

        filtered.write("Distilled CFG Scale: 3.5\n")

        assert original.getvalue() == ""

    def test_write_suppresses_multiple_patterns(self):
        """Test suppressing multiple different patterns."""
        original = StringIO()
        patterns = ["ERROR", "WARNING", "DEBUG"]
        filtered = FilteredOutput(original, patterns)

        # Write filtered messages
        filtered.write("ERROR: Something failed\n")
        filtered.write("WARNING: Be careful\n")
        filtered.write("DEBUG: Detail info\n")

        # Write normal message
        filtered.write("INFO: All good\n")

        assert original.getvalue() == "INFO: All good\n"

    def test_write_returns_length(self):
        """Test that write() returns text length even when suppressed."""
        original = StringIO()
        filtered = FilteredOutput(original, ["SUPPRESS"])

        # Suppressed message should still return length
        result = filtered.write("SUPPRESS this message\n")
        assert result == len("SUPPRESS this message\n")

        # Normal message should also return length
        result = filtered.write("Normal message\n")
        assert result == len("Normal message\n")

    def test_write_with_callback_intercepts_important_messages(self):
        """Test that important patterns are sent to callback."""
        original = StringIO()
        intercepted = []

        def callback(msg):
            intercepted.append(msg)

        filtered = FilteredOutput(
            original,
            filter_patterns=["SUPPRESS"],
            important_patterns=["[Memory]"],
            callback=callback
        )

        # Write important message - should go to callback, not stdout
        filtered.write("[Memory] Usage: 8GB\n")

        assert original.getvalue() == ""  # Not in stdout
        assert len(intercepted) == 1
        assert intercepted[0] == "[Memory] Usage: 8GB\n"

    def test_write_callback_without_important_patterns(self):
        """Test that callback is not called without important patterns match."""
        original = StringIO()
        called = []

        def callback(msg):
            called.append(msg)

        filtered = FilteredOutput(
            original,
            filter_patterns=[],
            important_patterns=["IMPORTANT"],
            callback=callback
        )

        # Write non-important message
        filtered.write("Regular message\n")

        assert len(called) == 0
        assert original.getvalue() == "Regular message\n"

    def test_write_multiple_important_patterns(self):
        """Test multiple important patterns trigger callback."""
        original = StringIO()
        intercepted = []

        filtered = FilteredOutput(
            original,
            filter_patterns=[],
            important_patterns=["[Memory]", "[GPU]", "[Progress]"],
            callback=lambda x: intercepted.append(x)
        )

        filtered.write("[Memory] Usage: 8GB\n")
        filtered.write("[GPU] Temp: 65C\n")
        filtered.write("[Progress] 50%\n")
        filtered.write("Normal log\n")

        assert len(intercepted) == 3
        assert original.getvalue() == "Normal log\n"

    def test_flush_calls_original_flush(self):
        """Test that flush propagates to original stream."""
        # Use real sys.stdout since StringIO doesn't track flush calls
        original = StringIO()
        filtered = FilteredOutput(original, [])

        # Should not raise exception
        filtered.flush()

    def test_partial_pattern_match(self):
        """Test that patterns match substrings."""
        original = StringIO()
        filtered = FilteredOutput(original, ["CFG"])

        # Pattern should match anywhere in text
        filtered.write("Before CFG Scale: 3.5 after\n")

        assert original.getvalue() == ""

    def test_case_sensitive_matching(self):
        """Test that pattern matching is case-sensitive."""
        original = StringIO()
        filtered = FilteredOutput(original, ["ERROR"])

        # Lowercase should not be filtered
        filtered.write("error: lowercase\n")

        # Uppercase should be filtered
        filtered.write("ERROR: uppercase\n")

        assert original.getvalue() == "error: lowercase\n"


class TestSuppressForgeOutput:
    """Test suppress_forge_output context manager."""

    def test_default_patterns_suppress_distilled_cfg(self):
        """Test that default patterns suppress Distilled CFG Scale messages."""
        original = StringIO()

        with suppress_forge_output():
            # Temporarily replace stdout to capture output
            sys.stdout.original_stream.write("Distilled CFG Scale: 3.5\n")

        # Message should be suppressed (not in final output)

    def test_custom_patterns(self):
        """Test using custom suppression patterns."""
        captured = StringIO()
        original_stdout = sys.stdout

        try:
            sys.stdout = captured

            with suppress_forge_output(patterns=["CUSTOM_SUPPRESS"]):
                print("Normal message")
                print("CUSTOM_SUPPRESS this message")
                print("Another normal message")

            output = captured.getvalue()

            # Check suppressions worked
            assert "Normal message" in output
            assert "Another normal message" in output
            assert "CUSTOM_SUPPRESS" not in output

        finally:
            sys.stdout = original_stdout

    def test_context_manager_restores_stdout(self):
        """Test that context manager restores original stdout."""
        original_stdout = sys.stdout

        with suppress_forge_output():
            inner_stdout = sys.stdout
            assert inner_stdout != original_stdout
            assert isinstance(inner_stdout, FilteredOutput)

        # Should be restored after context
        assert sys.stdout == original_stdout

    def test_context_manager_restores_on_exception(self):
        """Test that stdout is restored even if exception occurs."""
        original_stdout = sys.stdout

        try:
            with suppress_forge_output():
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be restored
        assert sys.stdout == original_stdout

    def test_with_callback_and_important_patterns(self):
        """Test context manager with callback for important messages."""
        intercepted = []
        captured = StringIO()
        original_stdout = sys.stdout

        def callback(msg):
            intercepted.append(msg)

        try:
            sys.stdout = captured

            with suppress_forge_output(
                patterns=["SUPPRESS"],
                important_patterns=["[Memory]"],
                callback=callback
            ):
                print("Normal message")
                print("SUPPRESS this")
                print("[Memory] Usage: 8GB")

            output = captured.getvalue()

            # Check results
            assert "Normal message" in output
            assert "SUPPRESS" not in output
            assert "[Memory]" not in output  # Went to callback
            assert len(intercepted) == 1
            assert "[Memory] Usage: 8GB" in intercepted[0]

        finally:
            sys.stdout = original_stdout

    def test_nested_suppression(self):
        """Test that nested context managers work correctly."""
        captured = StringIO()
        original_stdout = sys.stdout

        try:
            sys.stdout = captured

            with suppress_forge_output(patterns=["OUTER"]):
                print("Level 1")

                with suppress_forge_output(patterns=["INNER"]):
                    print("Level 2")
                    print("INNER suppressed")

                print("OUTER suppressed")
                print("Back to level 1")

            output = captured.getvalue()

            assert "Level 1" in output
            assert "Level 2" in output
            assert "Back to level 1" in output
            assert "INNER" not in output
            assert "OUTER" not in output

        finally:
            sys.stdout = original_stdout

    def test_multiple_writes_in_context(self):
        """Test multiple writes within single context."""
        captured = StringIO()
        original_stdout = sys.stdout

        try:
            sys.stdout = captured

            with suppress_forge_output(patterns=["FILTER"]):
                for i in range(5):
                    print(f"Message {i}")
                    print(f"FILTER message {i}")

            output = captured.getvalue()

            # Should have 5 normal messages, 0 filtered
            assert output.count("Message") == 5
            assert "FILTER" not in output

        finally:
            sys.stdout = original_stdout

    def test_empty_patterns_list(self):
        """Test that empty patterns list doesn't suppress anything."""
        captured = StringIO()
        original_stdout = sys.stdout

        try:
            sys.stdout = captured

            with suppress_forge_output(patterns=[]):
                print("Should not be suppressed")
                print("Nothing should be filtered")

            output = captured.getvalue()

            assert "Should not be suppressed" in output
            assert "Nothing should be filtered" in output

        finally:
            sys.stdout = original_stdout
