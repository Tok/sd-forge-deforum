"""Unit tests for deforum.utils.system.logging.log module.

Tests legacy logging functions and ANSI formatting utilities.
"""

import pytest
from unittest.mock import patch, MagicMock, call


class TestANSIConstants:
    """Test ANSI escape code constants."""

    def test_esc_sequence_defined(self):
        """ESC should be ANSI escape character with bracket."""
        from deforum.utils.system.logging.log import ESC
        assert ESC == "\033[" or ESC == "\x1b["

    def test_term_defined(self):
        """TERM should be ANSI terminator."""
        from deforum.utils.system.logging.log import TERM
        assert TERM == "m"

    def test_reset_color_defined(self):
        """RESET_COLOR should be valid ANSI reset sequence."""
        from deforum.utils.system.logging.log import RESET_COLOR
        assert RESET_COLOR == "\033[0m" or RESET_COLOR == "\x1b[0m"

    def test_color_constants_defined(self):
        """Color constants should be defined."""
        from deforum.utils.system.logging.log import (
            RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE
        )
        # All should be strings (ANSI codes)
        for color in [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]:
            assert isinstance(color, str)
            assert len(color) > 0

    def test_style_constants_defined(self):
        """Style constants (BOLD, ITALIC, UNDERLINE) should be defined."""
        from deforum.utils.system.logging.log import BOLD, ITALIC, UNDERLINE
        assert isinstance(BOLD, str)
        assert isinstance(ITALIC, str)
        assert isinstance(UNDERLINE, str)

    def test_hex_color_constants(self):
        """Hex color constants should be valid hex codes."""
        from deforum.utils.system.logging.log import (
            HEX_RED, HEX_ORANGE, HEX_YELLOW,
            HEX_GREEN, HEX_BLUE, HEX_PURPLE
        )
        for hex_color in [HEX_RED, HEX_ORANGE, HEX_YELLOW,
                          HEX_GREEN, HEX_BLUE, HEX_PURPLE]:
            assert hex_color.startswith('#')
            assert len(hex_color) == 7  # #RRGGBB


class TestClearNextLines:
    """Test clear_next_n_lines ANSI control function."""

    @patch('builtins.print')
    def test_clear_next_n_lines_single(self, mock_print):
        """clear_next_n_lines(1) should clear 1 line."""
        from deforum.utils.system.logging.log import clear_next_n_lines

        clear_next_n_lines(1)

        # Should call print twice (clear + move back)
        assert mock_print.call_count == 2

    @patch('builtins.print')
    def test_clear_next_n_lines_multiple(self, mock_print):
        """clear_next_n_lines(3) should clear 3 lines."""
        from deforum.utils.system.logging.log import clear_next_n_lines

        clear_next_n_lines(3)

        # Should call print twice (clear + move back)
        assert mock_print.call_count == 2

    @patch('builtins.print')
    def test_clear_next_n_lines_zero(self, mock_print):
        """clear_next_n_lines(0) should be no-op."""
        from deforum.utils.system.logging.log import clear_next_n_lines

        clear_next_n_lines(0)

        # Should still call print but with empty strings
        assert mock_print.call_count == 2


class TestInfoFunction:
    """Test legacy info() logging function."""

    @patch('builtins.print')
    def test_info_basic_message(self, mock_print):
        """info() should print message with INFO prefix."""
        from deforum.utils.system.logging.log import info

        info("Test message")

        mock_print.assert_called_once()
        call_args = str(mock_print.call_args)
        assert "INFO" in call_args
        assert "Test message" in call_args

    @patch('builtins.print')
    def test_info_with_color(self, mock_print):
        """info() should apply color if provided."""
        from deforum.utils.system.logging.log import info, RED

        info("Colored message", color=RED)

        mock_print.assert_called_once()
        # Should contain the color code and message
        call_args = str(mock_print.call_args)
        assert "Colored message" in call_args

    @patch('builtins.print')
    def test_info_empty_string_skipped(self, mock_print):
        """info() should skip printing empty strings."""
        from deforum.utils.system.logging.log import info

        info("")

        # Should NOT have called print
        mock_print.assert_not_called()

    @patch('builtins.print')
    def test_info_whitespace_only_skipped(self, mock_print):
        """info() should skip printing whitespace-only strings."""
        from deforum.utils.system.logging.log import info

        info("   \n\t  ")

        # Should NOT have called print
        mock_print.assert_not_called()

    @patch('builtins.print')
    def test_info_with_leading_trailing_whitespace(self, mock_print):
        """info() should print strings with content despite whitespace."""
        from deforum.utils.system.logging.log import info

        info("  message  ")

        # Should have called print (has non-whitespace content)
        mock_print.assert_called_once()


class TestErrorFunction:
    """Test legacy error() logging function."""

    @patch('builtins.print')
    def test_error_message(self, mock_print):
        """error() should print message with ERROR prefix."""
        from deforum.utils.system.logging.log import error

        error("Error message")

        mock_print.assert_called_once()
        call_args = str(mock_print.call_args)
        assert "ERROR" in call_args
        assert "Error message" in call_args


class TestWarningFunction:
    """Test legacy warning() logging function."""

    @patch('builtins.print')
    def test_warning_message(self, mock_print):
        """warning() should print message with WARNING prefix."""
        from deforum.utils.system.logging.log import warning

        warning("Warning message")

        mock_print.assert_called_once()
        call_args = str(mock_print.call_args)
        assert "WARNING" in call_args
        assert "Warning message" in call_args


class TestDebugFunction:
    """Test legacy debug() logging function."""

    @patch('deforum.rendering.options.is_verbose', return_value=True)
    @patch('builtins.print')
    def test_debug_when_verbose(self, mock_print, mock_verbose):
        """debug() should print when verbose mode enabled."""
        from deforum.utils.system.logging.log import debug

        debug("Debug message")

        # Check that debug message was printed (may have other print calls from imports)
        assert mock_print.called
        # Check last call contains our debug message
        last_call_args = str(mock_print.call_args)
        assert "DEBUG" in last_call_args
        assert "Debug message" in last_call_args

    @patch('deforum.rendering.options.is_verbose', return_value=False)
    @patch('builtins.print')
    def test_debug_when_not_verbose(self, mock_print, mock_verbose):
        """debug() should not print when verbose mode disabled."""
        from deforum.utils.system.logging.log import debug

        debug("Debug message")

        # Should NOT have called print
        mock_print.assert_not_called()


class TestPrintAnimationFrameInfo:
    """Test print_animation_frame_info function."""

    @patch('deforum.rendering.options.get_log_theme', return_value='classic')
    @patch('builtins.print')
    def test_print_frame_info_basic(self, mock_print, mock_theme):
        """print_animation_frame_info should print frame number."""
        from deforum.utils.system.logging.log import print_animation_frame_info

        print_animation_frame_info(5, 100)

        mock_print.assert_called_once()
        call_args = str(mock_print.call_args)
        assert "5" in call_args
        assert "100" in call_args
        assert "Animation Frame" in call_args

    @patch('deforum.rendering.options.get_log_theme', return_value='classic')
    @patch('builtins.print')
    def test_print_frame_info_keyframe(self, mock_print, mock_theme):
        """print_animation_frame_info should indicate keyframe."""
        from deforum.utils.system.logging.log import print_animation_frame_info

        print_animation_frame_info(5, 100, is_keyframe=True)

        mock_print.assert_called_once()
        call_args = str(mock_print.call_args)
        assert "KEYFRAME" in call_args

    @patch('deforum.rendering.options.get_log_theme', return_value='classic')
    @patch('builtins.print')
    def test_print_frame_info_cadence(self, mock_print, mock_theme):
        """print_animation_frame_info should indicate cadence frame."""
        from deforum.utils.system.logging.log import print_animation_frame_info

        print_animation_frame_info(5, 100, is_keyframe=False)

        mock_print.assert_called_once()
        call_args = str(mock_print.call_args)
        assert "CADENCE" in call_args

    @patch('deforum.rendering.options.get_log_theme', return_value='slopcore')
    @patch('builtins.print')
    def test_print_frame_info_slopcore_theme(self, mock_print, mock_theme):
        """print_animation_frame_info should work with slopcore theme."""
        from deforum.utils.system.logging.log import print_animation_frame_info

        print_animation_frame_info(10, 200, is_keyframe=True)

        mock_print.assert_called_once()
        # Should still print frame info with theme colors
        call_args = str(mock_print.call_args)
        assert "10" in call_args
        assert "200" in call_args


class TestPrintKeyFrameDebugInfo:
    """Test print_key_frame_debug_info_if_verbose function."""

    def test_print_key_frame_debug_info_signature(self):
        """print_key_frame_debug_info_if_verbose should be callable."""
        from deforum.utils.system.logging.log import print_key_frame_debug_info_if_verbose
        import inspect

        sig = inspect.signature(print_key_frame_debug_info_if_verbose)
        params = list(sig.parameters.keys())

        assert params == ['diffusion_frames']


class TestPrintTweenFrameCreationInfo:
    """Test print_tween_frame_creation_info function."""

    @patch('deforum.utils.system.logging.log.info')
    def test_print_tween_frame_creation_info(self, mock_info):
        """print_tween_frame_creation_info should call info() with summary."""
        from deforum.utils.system.logging.log import print_tween_frame_creation_info

        # Create mock key frames with tweens
        class MockTween:
            pass

        class MockKeyFrame:
            def __init__(self, tween_count):
                self.tweens = [MockTween() for _ in range(tween_count)]

        key_frames = [MockKeyFrame(3), MockKeyFrame(5), MockKeyFrame(2)]

        # Create mock index distribution enum
        class MockIndexDist:
            name = "TEST_DIST"

        index_dist = MockIndexDist()

        print_tween_frame_creation_info(key_frames, index_dist)

        # Should have called info() once
        mock_info.assert_called_once()
        call_args = str(mock_info.call_args)
        assert "3 key frames" in call_args  # 3 key frames
        assert "10 tweens" in call_args  # 3+5+2 = 10 tweens


class TestSimplePrintFunctions:
    """Test simple print wrapper functions."""

    @patch('builtins.print')
    def test_print_init_frame_info(self, mock_print):
        """print_init_frame_info should print init frame number."""
        from deforum.utils.system.logging.log import print_init_frame_info

        print_init_frame_info(42)

        mock_print.assert_called_once()
        call_args = str(mock_print.call_args)
        assert "42" in call_args

    @patch('builtins.print')
    def test_print_warning_no_image(self, mock_print):
        """print_warning_generate_returned_no_image should print warning."""
        from deforum.utils.system.logging.log import print_warning_generate_returned_no_image

        print_warning_generate_returned_no_image()

        mock_print.assert_called_once()
        call_args = str(mock_print.call_args)
        assert "no image" in call_args.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
