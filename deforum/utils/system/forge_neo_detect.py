"""
Forge Neo detection utility.

Forge Neo is a maintained fork with Wan 2.2 and Flux built-in, so we need to:
1. Skip all compatibility patches (diffusers, huggingface-hub, peft, accelerate)
2. Skip Flux blocker (Neo has Flux out-of-the-box)
"""

import os
import sys


def is_forge_neo() -> bool:
    """
    Detect if we're running on Forge Neo vs original Forge.

    Detection methods:
    1. Check for Neo-specific marker files
    2. Check webui root directory name contains 'neo'
    3. Check for Neo-specific modules/packages

    Returns:
        True if running on Forge Neo, False otherwise
    """
    try:
        # Method 1: Check for Neo marker file in webui root
        try:
            from modules import paths_internal
            webui_root = paths_internal.script_path

            # Check for neo-specific files
            neo_markers = [
                'neo.txt',
                'FORGE_NEO.txt',
                '.forge_neo',
            ]

            for marker in neo_markers:
                if os.path.exists(os.path.join(webui_root, marker)):
                    return True

            # Check if directory name contains 'neo'
            if 'neo' in os.path.basename(webui_root).lower():
                return True

        except ImportError:
            pass

        # Method 2: Check sys.path for neo directory
        for path in sys.path:
            if 'forge-neo' in path.lower() or 'forge_neo' in path.lower():
                return True

        # Method 3: Check if Neo-specific modules are available
        try:
            import importlib.util
            # Neo might have specific modules not in original Forge
            # This is a placeholder - adjust based on actual Neo differences
            neo_modules = ['modules_neo', 'backend_neo']
            for module_name in neo_modules:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    return True
        except:
            pass

        return False

    except Exception as e:
        # If detection fails, assume NOT Neo (safe default for compatibility patches)
        print(f"[Deforum] Warning: Neo detection failed: {e}")
        return False


def get_forge_variant() -> str:
    """
    Get human-readable Forge variant name.

    Returns:
        "Forge Neo" or "Forge" depending on detected variant
    """
    return "Forge Neo" if is_forge_neo() else "Forge"


# Cache the result to avoid repeated detection
_IS_NEO_CACHE = None


def is_forge_neo_cached() -> bool:
    """
    Cached version of is_forge_neo() for repeated calls.

    Returns:
        True if running on Forge Neo, False otherwise
    """
    global _IS_NEO_CACHE
    if _IS_NEO_CACHE is None:
        _IS_NEO_CACHE = is_forge_neo()
    return _IS_NEO_CACHE
