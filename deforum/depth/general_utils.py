"""
General utilities for depth processing.
Re-exports common utility functions from core_utilities for backward compatibility.
"""

from ..utils.core_utilities import (
    download_file_with_checksum
)

__all__ = [
    'download_file_with_checksum'
] 