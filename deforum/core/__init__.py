"""
Core generation and rendering functionality.
"""

# Import only modules that don't depend on WebUI for basic functionality
from .keyframe_animation import DeformAnimKeys, ControlNetKeys, LooperAnimKeys, FrameInterpolater

# Lazy imports for WebUI-dependent modules
def _lazy_import_generation():
    """Lazy import of generation modules that depend on WebUI"""
    try:
        from . import main_generation_pipeline
        return main_generation_pipeline
    except ImportError:
        return None

def _lazy_import_rendering():
    """Lazy import of rendering modules that depend on WebUI"""
    try:
        from . import rendering_engine, rendering_modes
        return rendering_engine, rendering_modes
    except ImportError:
        return None, None

def _lazy_import_all():
    """Lazy import of all core modules"""
    try:
        from . import animation_controller, run_deforum
        return animation_controller, run_deforum
    except ImportError:
        return None, None

__all__ = [
    # Always available (no WebUI dependency)
    'DeformAnimKeys',
    'ControlNetKeys', 
    'LooperAnimKeys',
    'FrameInterpolater',
    
    # Lazy loading functions
    '_lazy_import_generation',
    '_lazy_import_rendering',
    '_lazy_import_all',
] 