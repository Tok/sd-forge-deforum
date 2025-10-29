"""Lumina Integration Package

Compatibility patches and utilities for Lumina 2.0 model support.
"""

from .compat_patch import apply_lumina_patch_if_needed, ensure_num_tokens_for_lumina

__all__ = [
    'apply_lumina_patch_if_needed',
    'ensure_num_tokens_for_lumina',
]
