"""Flux 2 Compatibility Integration

Provides automatic compatibility patches for Flux 2 models:
- Adds vec_in_dim parameter fallback (768) for GGUF models
- Enables Flux 2 support without modifying Forge Neo
"""

from .compat_patch import ensure_flux2_compatibility

__all__ = [
    'ensure_flux2_compatibility',
]
