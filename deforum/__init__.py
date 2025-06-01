"""
Deforum: AI-Powered Animation Generation

A modern, functional programming-based system for creating AI-generated animations
with support for multiple animation modes, prompt enhancement, and advanced integrations.

This package follows contemporary Python standards with:
- Immutable data structures
- Pure functions
- Side effect isolation
- Modular architecture
- Comprehensive type hints
"""

# Import only modules that don't depend on WebUI
from .models import data_models, schedule_models

# Lazy imports for WebUI-dependent modules
def _lazy_import_config():
    """Lazy import of config modules that depend on WebUI modules"""
    try:
        from .config import arguments, settings, defaults
        return arguments, settings, defaults
    except ImportError as e:
        # WebUI modules not available yet, return None
        return None, None, None

def _lazy_import_ui():
    """Lazy import of UI modules"""
    try:
        from . import ui
        return ui
    except ImportError:
        return None

__version__ = "2.0.0"
__author__ = "Deforum LLC"
__license__ = "AGPL-3.0"

__all__ = [
    # Core functionality
    'render_animation',
    'generate_frame',
    'process_animation_sequence',
    
    # Data models
    'data_models',
    'schedule_models',
    
    # Lazy loading functions
    '_lazy_import_config',
    '_lazy_import_ui',
] 