"""Unit tests for _LazyLogger proxy pattern.

Tests that the lazy logger proxy correctly defers initialization
until first method call, avoiding module-level opts access issues.
"""
import pytest
from unittest.mock import Mock, patch
from deforum.utils.system.logging.logger import _LazyLogger, get_logger, reset_logger


class TestLazyLogger:
    """Test lazy initialization behavior of logger proxy."""

    def setup_method(self):
        """Reset logger singleton before each test."""
        reset_logger()

    def test_lazy_logger_defers_initialization(self):
        """Logger proxy should not initialize DeforumLogger until first method call."""
        logger = _LazyLogger()
        assert logger._real_logger is None, "Logger should not be initialized on creation"

    def test_lazy_logger_initializes_on_first_use(self):
        """Logger should initialize on first method call."""
        with patch('deforum.utils.system.logging.logger.DeforumLogger') as MockLogger:
            logger = _LazyLogger()
            logger.info("test message")
            MockLogger.assert_called_once()

    def test_lazy_logger_only_initializes_once(self):
        """Logger should only initialize once, not on every method call."""
        with patch('deforum.utils.system.logging.logger.DeforumLogger') as MockLogger:
            logger = _LazyLogger()
            logger.info("message 1")
            logger.info("message 2")
            logger.info("message 3")
            # Should only be called once despite 3 method calls
            assert MockLogger.call_count == 1

    def test_get_logger_returns_singleton(self):
        """get_logger() should return same instance every time."""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2

    def test_lazy_logger_forwards_debug(self):
        """debug() method should forward to real logger."""
        with patch('deforum.utils.system.logging.logger.DeforumLogger') as MockLogger:
            mock_instance = Mock()
            MockLogger.return_value = mock_instance

            logger = _LazyLogger()
            logger.debug("test debug", emoji="bug")

            mock_instance.debug.assert_called_once_with("test debug", "bug")

    def test_lazy_logger_forwards_info(self):
        """info() method should forward to real logger."""
        with patch('deforum.utils.system.logging.logger.DeforumLogger') as MockLogger:
            mock_instance = Mock()
            MockLogger.return_value = mock_instance

            logger = _LazyLogger()
            logger.info("test info", emoji="check")

            mock_instance.info.assert_called_once_with("test info", "check")

    def test_lazy_logger_forwards_warning(self):
        """warning() method should forward to real logger."""
        with patch('deforum.utils.system.logging.logger.DeforumLogger') as MockLogger:
            mock_instance = Mock()
            MockLogger.return_value = mock_instance

            logger = _LazyLogger()
            logger.warning("test warning")

            mock_instance.warning.assert_called_once_with("test warning", None)

    def test_lazy_logger_forwards_error(self):
        """error() method should forward to real logger."""
        with patch('deforum.utils.system.logging.logger.DeforumLogger') as MockLogger:
            mock_instance = Mock()
            MockLogger.return_value = mock_instance

            logger = _LazyLogger()
            logger.error("test error")

            mock_instance.error.assert_called_once_with("test error", None)

    def test_lazy_logger_forwards_critical(self):
        """critical() method should forward to real logger."""
        with patch('deforum.utils.system.logging.logger.DeforumLogger') as MockLogger:
            mock_instance = Mock()
            MockLogger.return_value = mock_instance

            logger = _LazyLogger()
            logger.critical("test critical")

            mock_instance.critical.assert_called_once_with("test critical", None)

    def test_lazy_logger_forwards_header(self):
        """header() method should forward to real logger."""
        with patch('deforum.utils.system.logging.logger.DeforumLogger') as MockLogger:
            mock_instance = Mock()
            MockLogger.return_value = mock_instance

            logger = _LazyLogger()
            logger.header("Test Header", width=80)

            mock_instance.header.assert_called_once_with("Test Header", 80)

    def test_lazy_logger_forwards_separator(self):
        """separator() method should forward to real logger."""
        with patch('deforum.utils.system.logging.logger.DeforumLogger') as MockLogger:
            mock_instance = Mock()
            MockLogger.return_value = mock_instance

            logger = _LazyLogger()
            logger.separator(char='-', width=60)

            mock_instance.separator.assert_called_once_with('-', 60)

    def test_lazy_logger_handles_opts_unavailable(self):
        """Logger should fall back to defaults when opts is not available."""
        # Patch the import inside _ensure_initialized to raise AttributeError
        with patch('deforum.rendering.options.get_log_theme', side_effect=AttributeError):
            logger = _LazyLogger()
            # Should not raise exception, should use defaults
            logger.info("test with unavailable opts")
            assert logger._real_logger is not None

    def test_lazy_logger_kwargs_passthrough(self):
        """Logger should pass through **kwargs like flush=True."""
        with patch('deforum.utils.system.logging.logger.DeforumLogger') as MockLogger:
            mock_instance = Mock()
            MockLogger.return_value = mock_instance

            logger = _LazyLogger()
            logger.info("test", emoji="check", flush=True, end="")

            mock_instance.info.assert_called_once_with("test", "check", flush=True, end="")
