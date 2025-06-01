"""
Animation system for keyframes, movement analysis, and scheduling.
"""

from .movement_analysis import *
from .schedule_system import *

__all__ = [
    # Movement analysis
    'analyze_movement_patterns',
    'detect_movement_changes',
    'calculate_movement_vectors',
    
    # Schedule system
    'parse_animation_schedule',
    'interpolate_schedule_values',
    'validate_schedule_syntax',
    'build_frame_schedule',
] 