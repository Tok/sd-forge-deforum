"""Forge Neo detection utility.

Forge Neo is a maintained fork with Wan 2.2 and Flux built-in, so we need to:
1. Skip all compatibility patches (diffusers, huggingface-hub, peft, accelerate)
2. Skip Flux blocker (Neo has Flux out-of-the-box)
"""

import os
import sys
from typing import Optional

from deforum.utils.system.logging import get_logger

logger = get_logger()

# Marker files that indicate Forge Neo installation
NEO_MARKER_FILES = ('neo.txt', 'FORGE_NEO.txt', '.forge_neo')

# Module names specific to Forge Neo
NEO_SPECIFIC_MODULES = ('modules_neo', 'backend_neo')


def _get_webui_root() -> Optional[str]:
    """Get webui root directory from paths_internal.

    Returns:
        Webui root path or None if unavailable
    """
    try:
        from modules import paths_internal
        return paths_internal.script_path
    except ImportError:
        logger.debug("modules.paths_internal not available")
        return None


def _check_marker_files(webui_root: str) -> bool:
    """Check if Neo marker files exist in webui root.

    Args:
        webui_root: Path to webui root directory

    Returns:
        True if any marker file exists, False otherwise
    """
    return any(
        os.path.exists(os.path.join(webui_root, marker))
        for marker in NEO_MARKER_FILES
    )


def _check_directory_name(webui_root: str) -> bool:
    """Check if directory name contains 'neo'.

    Args:
        webui_root: Path to webui root directory

    Returns:
        True if directory name contains 'neo', False otherwise
    """
    return 'neo' in os.path.basename(webui_root).lower()


def _check_sys_path() -> bool:
    """Check if sys.path contains neo-related directories.

    Returns:
        True if neo directory found in sys.path, False otherwise
    """
    return any(
        'forge-neo' in path.lower() or 'forge_neo' in path.lower()
        for path in sys.path
    )


def _check_neo_modules() -> bool:
    """Check if Neo-specific modules are importable.

    Returns:
        True if any Neo-specific module found, False otherwise
    """
    try:
        import importlib.util

        return any(
            importlib.util.find_spec(module_name) is not None
            for module_name in NEO_SPECIFIC_MODULES
        )
    except (ImportError, ValueError):
        return False


def is_forge_neo() -> bool:
    """Detect if we're running on Forge Neo vs original Forge.

    Detection methods (in order):
    1. Check for Neo-specific marker files in webui root
    2. Check if webui root directory name contains 'neo'
    3. Check sys.path for neo-related directories
    4. Check for Neo-specific importable modules

    Returns:
        True if running on Forge Neo, False otherwise
    """
    try:
        # Method 1 & 2: Check webui root for markers and name
        webui_root = _get_webui_root()
        if webui_root:
            if _check_marker_files(webui_root):
                logger.debug("Detected Neo via marker file")
                return True

            if _check_directory_name(webui_root):
                logger.debug("Detected Neo via directory name")
                return True

        # Method 3: Check sys.path
        if _check_sys_path():
            logger.debug("Detected Neo via sys.path")
            return True

        # Method 4: Check for Neo-specific modules
        if _check_neo_modules():
            logger.debug("Detected Neo via importable modules")
            return True

        return False

    except Exception as e:
        # If detection fails, assume NOT Neo (safe default for compatibility patches)
        logger.warning(f"Neo detection failed: {e}")
        return False


def get_forge_variant() -> str:
    """Get human-readable Forge variant name.

    Returns:
        "Forge Neo" or "Forge" depending on detected variant
    """
    return "Forge Neo" if is_forge_neo() else "Forge"


# Cached result for performance (detection is expensive)
_neo_detection_cache: Optional[bool] = None


def is_forge_neo_cached() -> bool:
    """Cached version of is_forge_neo() for repeated calls.

    Returns:
        True if running on Forge Neo, False otherwise
    """
    global _neo_detection_cache

    if _neo_detection_cache is None:
        _neo_detection_cache = is_forge_neo()

    return _neo_detection_cache


def clear_neo_cache() -> None:
    """Clear the Neo detection cache.

    Useful for testing or when environment changes during runtime.
    """
    global _neo_detection_cache
    _neo_detection_cache = None
