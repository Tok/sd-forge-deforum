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

from .models import data_models, schedule_models
from .config import arguments, settings, defaults
from .core import *
from .animation import *
from .depth import *
from .media import *
from .ui import *
from .prompt import *
from .utils import *

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
    
    # Configuration
    'arguments',
    'settings',
    'defaults',
    
    # Animation system
    'animation_controller',
    'keyframe_animation',
    'movement_analysis',
    
    # Depth processing
    'depth_estimation',
    'depth_warping',
    
    # Media processing
    'video_processing',
    'audio_processing',
    'frame_interpolation',
    
    # UI components
    'interface_panels',
    'gradio_components',
    
    # Prompt system
    'prompt_enhancement',
    'prompt_processing',
] 