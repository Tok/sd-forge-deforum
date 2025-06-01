"""
General utilities for configuration.
Re-exports common utility functions from core_utilities for backward compatibility.
"""

from ..utils.core_utilities import (
    get_os,
    substitute_placeholders,
    get_deforum_version,
    clean_gradio_path_strings
)

__all__ = [
    'get_os',
    'substitute_placeholders', 
    'get_deforum_version',
    'clean_gradio_path_strings'
] 