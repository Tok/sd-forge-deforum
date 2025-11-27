"""Context manager for temporarily overriding WebUI options.

Provides a clean way to temporarily modify Forge/A1111 opts and restore them after execution.
"""

from typing import Any, Optional
from types import TracebackType

from modules.shared import opts
from deforum.utils.system.logging import get_logger

logger = get_logger()


class A1111OptionsOverrider:
    """Context manager that temporarily overrides WebUI options.

    Usage:
        with A1111OptionsOverrider({'option_name': new_value}):
            # Code runs with overridden options
            pass
        # Options automatically restored here
    """

    def __init__(self, opts_overrides: Optional[dict[str, Any]] = None) -> None:
        """Initialize the options overrider.

        Args:
            opts_overrides: Dictionary mapping option names to new values
        """
        self.opts_overrides = opts_overrides or {}
        self.original_opts: Optional[dict[str, Any]] = None

    def __enter__(self) -> "A1111OptionsOverrider":
        """Enter context: save original options and apply overrides.

        Returns:
            Self for context manager protocol
        """
        if not self.opts_overrides:
            return self

        # Save original values only for options that exist
        self.original_opts = {
            key: opts.data[key]
            for key in self.opts_overrides
            if key in opts.data
        }

        if self.original_opts:
            logger.debug(f"Captured options to override: {self.original_opts}")

        # Try to set each option, gracefully skipping any that fail
        successfully_set = {}
        failed_to_set = {}

        for key, value in self.opts_overrides.items():
            try:
                # Try to set the option via setattr (validates option exists in Forge)
                setattr(opts, key, value)
                opts.data[key] = value
                successfully_set[key] = value
            except (AttributeError, KeyError) as e:
                # Option doesn't exist or isn't registered properly in Forge
                failed_to_set[key] = str(e)
                logger.debug(f"Failed to set option '{key}': {e}")

        if failed_to_set:
            logger.warning(f"Skipping non-existent options: {list(failed_to_set.keys())}")

        if successfully_set:
            logger.info(f"Successfully set options: {list(successfully_set.keys())}")

        return self

    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Exit context: restore original options.

        Args:
            exception_type: Type of exception if one occurred
            exception_value: Exception instance if one occurred
            traceback: Traceback if exception occurred
        """
        if exception_type is not None:
            logger.warning(
                f"Error during execution with overridden opts: {exception_type.__name__} - "
                f"{exception_value}"
            )
            logger.debug(f"Traceback: {traceback}")

        if self.original_opts:
            logger.info(f"Restoring options: {list(self.original_opts.keys())}")
            for key, value in self.original_opts.items():
                try:
                    setattr(opts, key, value)
                    opts.data[key] = value
                except (AttributeError, KeyError) as e:
                    logger.warning(f"Failed to restore option '{key}': {e}")
