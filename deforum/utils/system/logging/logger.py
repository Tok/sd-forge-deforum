"""Centralized logging system for Deforum with theme support.

Provides unified console output with three themes:
- Slopcore: Modern blue→purple gradient (default)
- Classic: Vibrant multi-color palette (legacy)
- Simple: Plain text, no colors

Features:
- Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Theme-aware emoji support
- Styled tqdm progress bars
- Clean, consistent output formatting
"""

from enum import Enum
from typing import Iterator, Optional

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from deforum.utils.system.logging.themes import get_theme_colors, RESET_COLOR, BOLD
from deforum.utils.system.logging.emoji import get_themed_emoji


class LogLevel(Enum):
    """Log verbosity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class DeforumLogger:
    """Centralized logger with theme support and log levels."""

    def __init__(
        self,
        theme: str = 'slopcore',
        log_level: str = 'INFO',
        emojis_enabled: bool = True
    ):
        """Initialize logger.

        Args:
            theme: Output theme ('slopcore', 'classic', 'simple')
            log_level: Minimum log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            emojis_enabled: Whether to show emojis
        """
        self.theme = theme
        self.log_level = LogLevel[log_level]
        self.emojis_enabled = emojis_enabled
        self.colors = get_theme_colors(theme)

    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on level."""
        return level.value >= self.log_level.value

    def _format_message(self, level: str, msg: str, emoji: Optional[str] = None) -> str:
        """Format log message with colors and optional emoji.

        Args:
            level: Log level name ('debug', 'info', 'warning', 'error', 'critical')
            msg: Message text
            emoji: Optional emoji name (e.g., 'run', 'key')

        Returns:
            Formatted message string
        """
        color = self.colors.get(level, '')
        reset = self.colors['reset']
        bold = self.colors['bold']

        # Get themed emoji if provided
        emoji_str = ''
        if emoji and self.emojis_enabled:
            emoji_str = get_themed_emoji(emoji, self.theme)
            if emoji_str:
                emoji_str = f"{emoji_str} "

        # Format: [LEVEL] emoji message
        level_label = level.upper()
        return f"{color}{bold}{level_label}:{reset} {emoji_str}{msg}"

    def debug(self, msg: str, emoji: Optional[str] = None):
        """Log debug message (verbose internal details).

        Args:
            msg: Message text
            emoji: Optional emoji name
        """
        if self._should_log(LogLevel.DEBUG):
            print(self._format_message('debug', msg, emoji))

    def info(self, msg: str, emoji: Optional[str] = None):
        """Log info message (normal operation).

        Args:
            msg: Message text
            emoji: Optional emoji name
        """
        if self._should_log(LogLevel.INFO):
            print(self._format_message('info', msg, emoji))

    def warning(self, msg: str, emoji: Optional[str] = None):
        """Log warning message.

        Args:
            msg: Message text
            emoji: Optional emoji name
        """
        if self._should_log(LogLevel.WARNING):
            print(self._format_message('warning', msg, emoji))

    def error(self, msg: str, emoji: Optional[str] = None):
        """Log error message.

        Args:
            msg: Message text
            emoji: Optional emoji name
        """
        if self._should_log(LogLevel.ERROR):
            print(self._format_message('error', msg, emoji))

    def critical(self, msg: str, emoji: Optional[str] = None):
        """Log critical error message.

        Args:
            msg: Message text
            emoji: Optional emoji name
        """
        if self._should_log(LogLevel.CRITICAL):
            print(self._format_message('critical', msg, emoji))

    def header(self, msg: str, width: int = 80):
        """Print styled header/section divider.

        Args:
            msg: Header text
            width: Total width of header (including borders)
        """
        if not self._should_log(LogLevel.INFO):
            return

        color = self.colors['header']
        reset = self.colors['reset']
        bold = self.colors['bold']

        # Create border based on theme
        if self.theme == 'slopcore':
            # Gradient border using different shades
            border_chars = ['='] * width
            border = ''.join(border_chars)
            print(f"{color}{border}{reset}")
            print(f"{color}{bold}{msg.center(width)}{reset}")
            print(f"{color}{border}{reset}")
        elif self.theme == 'classic':
            border = '=' * width
            print(f"{color}{border}{reset}")
            print(f"{bold}{msg.center(width)}{reset}")
            print(f"{color}{border}{reset}")
        else:  # simple
            border = '=' * width
            print(border)
            print(msg.center(width))
            print(border)

    def progress(
        self,
        iterable: Iterator,
        desc: str = '',
        total: Optional[int] = None,
        **tqdm_kwargs
    ) -> Iterator:
        """Create themed tqdm progress bar.

        Args:
            iterable: Iterable to wrap
            desc: Description text
            total: Total iterations (optional)
            **tqdm_kwargs: Additional tqdm arguments

        Returns:
            tqdm iterator (or plain iterator if tqdm unavailable)
        """
        if not TQDM_AVAILABLE:
            # Fallback to plain iterator if tqdm not available
            return iterable

        # Get theme-specific tqdm styling
        style = self._get_tqdm_style()
        style.update(tqdm_kwargs)  # Allow overrides

        return tqdm(iterable, desc=desc, total=total, **style)

    def _get_tqdm_style(self) -> dict:
        """Get tqdm style configuration for current theme.

        Returns:
            Dictionary of tqdm parameters
        """
        if self.theme == 'slopcore':
            # Use mid-range purple from slopcore gradient
            return {
                'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                'colour': '#7B6DB8',  # Mid purple from slopcore gradient (shade 4)
                'ncols': 100,
                'ascii': False,
            }
        elif self.theme == 'classic':
            # IMPORTANT: Keep classic colorful style unchanged!
            # Classic uses default tqdm colors which are vibrant
            return {
                'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}',
                # Don't override colour - let classic tqdm use default vibrant colors
            }
        else:  # simple
            return {
                'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}',
                'ascii': True,
                'colour': None,
            }

    def separator(self, char: str = '-', width: int = 80):
        """Print a separator line.

        Args:
            char: Character to use for separator
            width: Width of separator
        """
        if not self._should_log(LogLevel.INFO):
            return

        color = self.colors.get('emphasis', '')
        reset = self.colors['reset']

        if self.theme == 'simple':
            print(char * width)
        else:
            print(f"{color}{char * width}{reset}")


# ============================================================================
# Global Logger Instance
# ============================================================================

_logger_instance: Optional[DeforumLogger] = None


def emoji_if_enabled(emoji_str: str) -> str:
    """Return emoji string only if emojis are enabled in settings, otherwise empty string.

    This allows inline emojis in logger messages to respect the emoji toggle:
    logger.info(f"{emoji_if_enabled('✅')} Task complete")

    Args:
        emoji_str: The emoji character(s) to conditionally show

    Returns:
        The emoji string if emojis enabled, empty string otherwise
    """
    global _logger_instance
    if _logger_instance and hasattr(_logger_instance, 'emojis_enabled'):
        return emoji_str if _logger_instance.emojis_enabled else ''
    # If no logger yet, check settings directly
    try:
        from deforum.rendering.options import is_emojis_disabled
        return '' if is_emojis_disabled() else emoji_str
    except:
        return emoji_str  # Fallback: show emoji


def get_logger() -> DeforumLogger:
    """Get global logger instance (singleton).

    Returns:
        DeforumLogger instance
    """
    global _logger_instance
    if _logger_instance is None:
        # Try to get settings, fall back to defaults
        try:
            from deforum.rendering.options import (
                get_log_theme,
                get_log_level,
                is_emojis_disabled
            )
            theme = get_log_theme()
            log_level = get_log_level()
            emojis_enabled = not is_emojis_disabled()
        except ImportError:
            # Fallback to defaults if settings not available
            theme = 'slopcore'
            log_level = 'INFO'
            emojis_enabled = True

        _logger_instance = DeforumLogger(
            theme=theme,
            log_level=log_level,
            emojis_enabled=emojis_enabled
        )

    return _logger_instance


def set_logger_config(
    theme: Optional[str] = None,
    log_level: Optional[str] = None,
    emojis_enabled: Optional[bool] = None
):
    """Update global logger configuration.

    Args:
        theme: New theme ('slopcore', 'classic', 'simple')
        log_level: New log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        emojis_enabled: Whether to enable emojis
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = get_logger()

    if theme is not None:
        _logger_instance.theme = theme
        _logger_instance.colors = get_theme_colors(theme)

    if log_level is not None:
        _logger_instance.log_level = LogLevel[log_level]

    if emojis_enabled is not None:
        _logger_instance.emojis_enabled = emojis_enabled


def reset_logger():
    """Reset logger instance (useful for testing)."""
    global _logger_instance
    _logger_instance = None
