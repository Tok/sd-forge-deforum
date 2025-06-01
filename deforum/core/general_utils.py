"""
General utilities for core functionality.
Re-exports common utility functions from core_utilities for backward compatibility.
"""

from ..utils.core_utilities import (
    get_deforum_version,
    get_commit_date,
    debug_print
)

__all__ = [
    'get_deforum_version',
    'get_commit_date',
    'debug_print'
] 