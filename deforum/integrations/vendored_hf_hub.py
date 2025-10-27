"""Vendored huggingface-hub wrapper for compatibility.

Forge uses huggingface-hub 0.26.2, but diffusers (needed for Wan) requires >=0.34.0.
This module provides a compatibility layer by installing a newer version in a separate
location and managing imports to avoid conflicts.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Optional

VENDORED_HF_HUB_VERSION = "0.36.0"  # Last version before 1.0 breaking changes
_vendored_path: Optional[Path] = None
_import_patched = False


def get_vendored_path() -> Path:
    """Get path to vendored huggingface-hub installation."""
    global _vendored_path

    if _vendored_path is None:
        # Install in extension's local directory
        ext_dir = Path(__file__).parent.parent.parent
        _vendored_path = ext_dir / ".vendored" / "huggingface_hub"

    return _vendored_path


def ensure_vendored_hf_hub() -> bool:
    """Ensure vendored huggingface-hub is installed.

    Returns:
        True if available, False if installation failed
    """
    vendored_path = get_vendored_path()

    # Check if already installed
    marker_file = vendored_path / f"._installed_{VENDORED_HF_HUB_VERSION}"
    if marker_file.exists():
        return True

    try:
        print(f"[Deforum] Installing vendored huggingface-hub {VENDORED_HF_HUB_VERSION}...")

        # Create vendored directory
        vendored_path.parent.mkdir(parents=True, exist_ok=True)

        # Install to vendored location using pip with --target
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            f"huggingface-hub=={VENDORED_HF_HUB_VERSION}",
            "--target", str(vendored_path.parent),
            "--no-deps",  # Don't install dependencies (they're already in Forge)
            "--upgrade",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Create marker file
        marker_file.touch()

        print(f"[Deforum] ✓ Vendored huggingface-hub {VENDORED_HF_HUB_VERSION} installed")
        return True

    except Exception as e:
        print(f"[Deforum] ⚠️ Failed to install vendored huggingface-hub: {e}")
        print("[Deforum]    Wan video features may not work correctly")
        return False


def use_vendored_hf_hub() -> bool:
    """Switch to using vendored huggingface-hub for imports.

    This must be called BEFORE importing diffusers or any code that uses huggingface-hub.

    Returns:
        True if successfully switched, False otherwise
    """
    global _import_patched

    if _import_patched:
        return True

    if not ensure_vendored_hf_hub():
        return False

    vendored_path = get_vendored_path().parent

    # Insert vendored path at the FRONT of sys.path so it takes precedence
    vendored_str = str(vendored_path)
    if vendored_str not in sys.path:
        sys.path.insert(0, vendored_str)
        print(f"[Deforum] Using vendored huggingface-hub from {vendored_path}")

    _import_patched = True
    return True


def restore_system_hf_hub():
    """Restore system huggingface-hub by removing vendored path from sys.path.

    Use this after diffusers import is complete to avoid affecting other extensions.
    """
    global _import_patched

    if not _import_patched:
        return

    vendored_path = get_vendored_path().parent
    vendored_str = str(vendored_path)

    if vendored_str in sys.path:
        sys.path.remove(vendored_str)
        print(f"[Deforum] Restored system huggingface-hub")

    _import_patched = False
