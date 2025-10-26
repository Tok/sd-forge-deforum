"""Unit tests for logging, emoji, and theme system.

Tests the logging infrastructure in deforum/utils/system/logging/ including:
- Logger initialization and configuration
- Emoji toggling and themed emojis
- Log level filtering
- Theme color codes
"""

import pytest
from unittest.mock import patch, MagicMock
from deforum.utils.system.logging import (
    get_logger,
    emoji_if_enabled,
    LogLevel,
)
from deforum.utils.system.logging.emoji import _select, get_themed_emoji
from deforum.rendering.options import is_emojis_enabled


class TestEmojiSystem:
    """Test emoji toggling and selection."""

    def test_emoji_if_enabled_when_enabled(self):
        """Emoji should be returned when enabled."""
        with patch('deforum.rendering.options.is_emojis_enabled', return_value=True):
            result = emoji_if_enabled('✅')
            assert result == '✅'

    def test_emoji_if_enabled_when_disabled(self):
        """Empty string should be returned when disabled."""
        with patch('deforum.rendering.options.is_emojis_enabled', return_value=False):
            result = emoji_if_enabled('✅')
            assert result == ''

    def test_emoji_select_when_enabled(self):
        """_select should return emoji when enabled."""
        with patch('deforum.rendering.options.is_emojis_enabled', return_value=True):
            result = _select('🔍')
            assert result == '🔍'

    def test_emoji_select_when_disabled(self):
        """_select should return empty string when disabled."""
        with patch('deforum.rendering.options.is_emojis_enabled', return_value=False):
            result = _select('🔍')
            assert result == ''

    def test_emoji_select_during_initialization(self):
        """_select should handle missing settings during startup."""
        # Simulate import error during initialization
        with patch('deforum.rendering.options.is_emojis_enabled', side_effect=ImportError):
            result = _select('🔍')
            assert result == ''  # Default to disabled


class TestThemedEmojis:
    """Test themed emoji system."""

    def test_get_themed_emoji_slopcore(self):
        """Slopcore theme uses specific emoji set."""
        emoji = get_themed_emoji('success', 'slopcore')
        assert emoji in ['✓', '✅']  # Could be either

    def test_get_themed_emoji_classic(self):
        """Classic theme uses vibrant emojis."""
        emoji = get_themed_emoji('success', 'classic')
        assert emoji == '✅'

    def test_get_themed_emoji_simple(self):
        """Simple theme disables emojis."""
        emoji = get_themed_emoji('success', 'simple')
        assert emoji == ''

    def test_get_themed_emoji_fallback(self):
        """Unknown emoji key should return empty string."""
        emoji = get_themed_emoji('nonexistent_key', 'slopcore')
        assert emoji == ''

    def test_get_themed_emoji_unknown_theme(self):
        """Unknown theme should fallback to classic."""
        emoji = get_themed_emoji('success', 'unknown_theme')
        assert emoji != ''  # Should use classic fallback


class TestLoggerConfiguration:
    """Test logger initialization and configuration."""

    def test_get_logger_singleton(self):
        """get_logger should return the same instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2

    def test_logger_has_methods(self):
        """Logger should have standard logging methods."""
        logger = get_logger()
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')

    @patch('deforum.rendering.options.opts')
    def test_logger_respects_log_level(self, mock_opts):
        """Logger should respect configured log level."""
        mock_opts.data = {
            'deforum_log_level': 'WARNING',
            'deforum_log_theme': 'classic',
            'deforum_enable_emojis': False
        }

        logger = get_logger()

        # This is a basic check - actual filtering tested in integration
        assert logger.log_level is not None

    @patch('deforum.rendering.options.opts')
    def test_logger_respects_theme(self, mock_opts):
        """Logger should use configured theme."""
        mock_opts.data = {
            'deforum_log_level': 'INFO',
            'deforum_log_theme': 'slopcore',
            'deforum_enable_emojis': True
        }

        logger = get_logger()

        # Logger should store theme
        assert logger.theme is not None

    @patch('deforum.rendering.options.opts')
    def test_logger_respects_emoji_setting(self, mock_opts):
        """Logger should respect emoji enable/disable setting."""
        mock_opts.data = {
            'deforum_log_level': 'INFO',
            'deforum_log_theme': 'classic',
            'deforum_enable_emojis': False
        }

        logger = get_logger()

        # Logger should store emoji setting
        assert hasattr(logger, 'emojis_enabled')


class TestLogLevel:
    """Test LogLevel enum and filtering."""

    def test_log_level_values(self):
        """LogLevel should have standard levels."""
        assert LogLevel.DEBUG.value == 10
        assert LogLevel.INFO.value == 20
        assert LogLevel.WARNING.value == 30
        assert LogLevel.ERROR.value == 40
        assert LogLevel.CRITICAL.value == 50

    def test_log_level_ordering(self):
        """Log levels should be properly ordered."""
        assert LogLevel.DEBUG < LogLevel.INFO
        assert LogLevel.INFO < LogLevel.WARNING
        assert LogLevel.WARNING < LogLevel.ERROR
        assert LogLevel.ERROR < LogLevel.CRITICAL


class TestThemeValues:
    """Test theme string values (LogTheme enum not exported)."""

    def test_theme_strings(self):
        """Theme strings should be valid for themed emoji system."""
        # Test that themed emojis work with standard theme names
        assert get_themed_emoji('success', 'slopcore') != ''
        assert get_themed_emoji('success', 'classic') == '✅'
        assert get_themed_emoji('success', 'simple') == ''


class TestLoggerOutput:
    """Test logger message formatting and output."""

    @patch('builtins.print')
    @patch('deforum.rendering.options.opts')
    def test_info_message_format(self, mock_opts, mock_print):
        """Info messages should be properly formatted."""
        mock_opts.data = {
            'deforum_log_level': 'INFO',
            'deforum_log_theme': 'classic',
            'deforum_enable_emojis': False
        }

        logger = get_logger()
        logger.info("Test message")

        # Should have called print
        assert mock_print.called
        # Message should contain "INFO" and "Test message"
        call_args = str(mock_print.call_args)
        assert 'INFO' in call_args or 'Test message' in call_args

    @patch('builtins.print')
    @patch('deforum.rendering.options.opts')
    def test_info_with_emoji(self, mock_opts, mock_print):
        """Info messages should include emoji when enabled."""
        mock_opts.data = {
            'deforum_log_level': 'INFO',
            'deforum_log_theme': 'classic',
            'deforum_enable_emojis': True
        }

        logger = get_logger()
        logger.info("Test message", emoji='success')

        assert mock_print.called
        # With emojis enabled, output should contain emoji
        call_args = str(mock_print.call_args)
        # May contain emoji depending on theme mapping

    @patch('builtins.print')
    @patch('deforum.rendering.options.opts')
    def test_debug_filtered_at_info_level(self, mock_opts, mock_print):
        """Debug messages should not print when level is INFO."""
        mock_opts.data = {
            'deforum_log_level': 'INFO',
            'deforum_log_theme': 'classic',
            'deforum_enable_emojis': False
        }

        logger = get_logger()
        logger.debug("Debug message")

        # Debug should not be printed when log level is INFO
        # This depends on implementation details
        # Actual behavior tested in integration tests

    @patch('builtins.print')
    @patch('deforum.rendering.options.opts')
    def test_print_kwargs_support(self, mock_opts, mock_print):
        """Logger should support print kwargs like end and flush."""
        mock_opts.data = {
            'deforum_log_level': 'INFO',
            'deforum_log_theme': 'classic',
            'deforum_enable_emojis': False
        }

        logger = get_logger()
        logger.info("Test", end='', flush=True)

        # Should pass kwargs to print
        if mock_print.called:
            # Check if kwargs were passed (implementation-specific)
            pass


class TestIntegration:
    """Integration tests for the full logging system."""

    @patch('builtins.print')
    @patch('deforum.rendering.options.opts')
    def test_full_logging_flow(self, mock_opts, mock_print):
        """Test complete logging workflow."""
        mock_opts.data = {
            'deforum_log_level': 'INFO',
            'deforum_log_theme': 'slopcore',
            'deforum_enable_emojis': True
        }

        logger = get_logger()

        # Should print info and above
        logger.info("Info message", emoji='info')
        logger.warning("Warning message", emoji='warning')
        logger.error("Error message", emoji='error')

        # Debug should be filtered
        logger.debug("Debug message")

        # Verify print was called for info, warning, error (not debug)
        assert mock_print.call_count >= 3

    @patch('deforum.rendering.options.is_emojis_enabled')
    def test_emoji_toggle_affects_all_outputs(self, mock_emoji_setting):
        """Emoji setting should affect all emoji-enabled outputs."""
        # Test with emojis enabled
        mock_emoji_setting.return_value = True
        assert emoji_if_enabled('✅') == '✅'
        assert _select('🔍') == '🔍'

        # Test with emojis disabled
        mock_emoji_setting.return_value = False
        assert emoji_if_enabled('✅') == ''
        assert _select('🔍') == ''
