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
    TRACE = -1
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
            log_level: Minimum log level ('TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
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
            level: Log level name ('trace', 'debug', 'info', 'warning', 'error', 'critical')
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

    def trace(self, msg: str, emoji: Optional[str] = None, **kwargs):
        """Log trace message (ultra-verbose internal details, algorithm steps).

        Args:
            msg: Message text
            emoji: Optional emoji name
            **kwargs: Additional arguments passed to print() (e.g., end='', flush=True)
        """
        if self._should_log(LogLevel.TRACE):
            print(self._format_message('trace', msg, emoji), **kwargs)

    def debug(self, msg: str, emoji: Optional[str] = None, **kwargs):
        """Log debug message (debugging information, function entry/exit).

        Args:
            msg: Message text
            emoji: Optional emoji name
            **kwargs: Additional arguments passed to print() (e.g., end='', flush=True)
        """
        if self._should_log(LogLevel.DEBUG):
            print(self._format_message('debug', msg, emoji), **kwargs)

    def info(self, msg: str, emoji: Optional[str] = None, **kwargs):
        """Log info message (normal operation).

        Args:
            msg: Message text
            emoji: Optional emoji name
            **kwargs: Additional arguments passed to print() (e.g., end='', flush=True)
        """
        if self._should_log(LogLevel.INFO):
            print(self._format_message('info', msg, emoji), **kwargs)

    def warning(self, msg: str, emoji: Optional[str] = None, **kwargs):
        """Log warning message.

        Args:
            msg: Message text
            emoji: Optional emoji name
            **kwargs: Additional arguments passed to print() (e.g., end='', flush=True)
        """
        if self._should_log(LogLevel.WARNING):
            print(self._format_message('warning', msg, emoji), **kwargs)

    def error(self, msg: str, emoji: Optional[str] = None, **kwargs):
        """Log error message.

        Args:
            msg: Message text
            emoji: Optional emoji name
            **kwargs: Additional arguments passed to print() (e.g., end='', flush=True)
                     exc_info=True will print traceback
        """
        if self._should_log(LogLevel.ERROR):
            # Extract exc_info if present (print() doesn't support it)
            exc_info = kwargs.pop('exc_info', False)
            print(self._format_message('error', msg, emoji), **kwargs)
            if exc_info:
                import traceback
                traceback.print_exc()

    def critical(self, msg: str, emoji: Optional[str] = None, **kwargs):
        """Log critical error message.

        Args:
            msg: Message text
            emoji: Optional emoji name
            **kwargs: Additional arguments passed to print() (e.g., end='', flush=True)
        """
        if self._should_log(LogLevel.CRITICAL):
            print(self._format_message('critical', msg, emoji), **kwargs)

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
    """Return emoji string only if emojis are enabled, with theme-aware substitutions.

    This allows inline emojis in logger messages to respect the emoji toggle:
    logger.info(f"{emoji_if_enabled('✅')} Task complete")

    In slopcore mode, certain emojis are substituted for aesthetic consistency:
    - ✅ (green check) → ✓ (white/black check mark)

    Args:
        emoji_str: The emoji character(s) to conditionally show

    Returns:
        The themed emoji string if emojis enabled, empty string otherwise
    """
    global _logger_instance

    # Check if emojis are enabled
    emojis_enabled = False
    theme = 'classic'

    if _logger_instance and hasattr(_logger_instance, 'emojis_enabled'):
        emojis_enabled = _logger_instance.emojis_enabled
        theme = getattr(_logger_instance, 'theme', 'classic')
    else:
        # If no logger yet, check settings directly
        try:
            from deforum.rendering.options import is_emojis_enabled, get_log_theme
            emojis_enabled = is_emojis_enabled()
            theme = get_log_theme()
        except:
            emojis_enabled = False  # Fallback: no emoji (match UI default)

    if not emojis_enabled:
        return ''

    # Apply theme-specific emoji substitutions
    if theme == 'slopcore':
        # Map colored emojis to monochrome equivalents for aesthetic consistency
        SLOPCORE_EMOJI_MAP = {
            '✅': '✓',  # Green check → White/black check mark
            '❌': '✗',  # Red X → White/black X
            '⚠️': '⚠',  # Warning (remove variation selector for cleaner look)
            '🚨': '⚠',  # Alert → Warning (simpler)
        }
        return SLOPCORE_EMOJI_MAP.get(emoji_str, emoji_str)

    return emoji_str


class _LazyLogger:
    """Lazy-loading logger proxy that defers initialization until first method call.

    This allows 'logger = get_logger()' at module level without opts being ready.
    The actual DeforumLogger is only created when you call logger.info(), etc.
    """
    def __init__(self) -> None:
        self._real_logger: Optional[DeforumLogger] = None

    def _ensure_initialized(self) -> None:
        """Initialize real logger on first use."""
        if self._real_logger is None:
            # Try to get settings, fall back to defaults
            try:
                from deforum.rendering.options import (
                    get_log_theme,
                    get_log_level,
                    is_emojis_enabled
                )
                theme = get_log_theme()
                log_level = get_log_level()
                emojis_enabled = is_emojis_enabled()
            except (ImportError, AttributeError):
                # Fallback to defaults if settings not available or opts not initialized yet
                theme = 'slopcore'
                log_level = 'INFO'
                emojis_enabled = False  # Match UI default (unchecked = disabled)

            self._real_logger = DeforumLogger(
                theme=theme,
                log_level=log_level,
                emojis_enabled=emojis_enabled
            )

    def trace(self, msg: str, emoji: Optional[str] = None, **kwargs) -> None:
        self._ensure_initialized()
        return self._real_logger.trace(msg, emoji, **kwargs)

    def debug(self, msg: str, emoji: Optional[str] = None, **kwargs) -> None:
        self._ensure_initialized()
        return self._real_logger.debug(msg, emoji, **kwargs)

    def info(self, msg: str, emoji: Optional[str] = None, **kwargs) -> None:
        self._ensure_initialized()
        return self._real_logger.info(msg, emoji, **kwargs)

    def warning(self, msg: str, emoji: Optional[str] = None, **kwargs) -> None:
        self._ensure_initialized()
        return self._real_logger.warning(msg, emoji, **kwargs)

    def error(self, msg: str, emoji: Optional[str] = None, **kwargs) -> None:
        self._ensure_initialized()
        return self._real_logger.error(msg, emoji, **kwargs)

    def critical(self, msg: str, emoji: Optional[str] = None, **kwargs) -> None:
        self._ensure_initialized()
        return self._real_logger.critical(msg, emoji, **kwargs)

    def header(self, msg: str, width: int = 80) -> None:
        self._ensure_initialized()
        return self._real_logger.header(msg, width)

    def progress(self, iterable: Iterator, desc: str = '', total: Optional[int] = None, **tqdm_kwargs) -> Iterator:
        self._ensure_initialized()
        return self._real_logger.progress(iterable, desc, total, **tqdm_kwargs)

    def separator(self, char: str = '-', width: int = 80) -> None:
        self._ensure_initialized()
        return self._real_logger.separator(char, width)

    # Property accessors to expose underlying logger attributes
    @property
    def log_level(self) -> 'LogLevel':
        """Access log_level from real logger."""
        self._ensure_initialized()
        return self._real_logger.log_level

    @property
    def theme(self) -> str:
        """Access theme from real logger."""
        self._ensure_initialized()
        return self._real_logger.theme

    @property
    def emojis_enabled(self) -> bool:
        """Access emojis_enabled from real logger."""
        self._ensure_initialized()
        return self._real_logger.emojis_enabled

    @property
    def colors(self) -> dict:
        """Access colors from real logger."""
        self._ensure_initialized()
        return self._real_logger.colors


def get_logger() -> _LazyLogger:
    """Get global lazy logger instance (singleton).

    Returns a lazy proxy that defers DeforumLogger initialization until first use.
    Safe to call at module level even if opts is not ready.

    Returns:
        _LazyLogger proxy instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = _LazyLogger()
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
