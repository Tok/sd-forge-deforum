#!/usr/bin/env python3
"""
Enhanced Movement Analyzer for Deforum (Backward Compatibility Layer)
Main entry point that delegates to the new modular movement analysis system
"""

from typing import Dict, List, Tuple, Optional

# Import new modular components
from .movement_detection import (
    parse_schedule_string, interpolate_schedule, detect_movement_segments,
    detect_circular_motion, detect_pattern_type, calculate_movement_intensity,
    group_similar_segments
)

from .movement_analysis import (
    MovementAnalyzer, analyze_deforum_movement
)

from .movement_utils import (
    create_shakify_data, apply_shakify_to_schedule, 
    generate_wan_motion_intensity_schedule,
    validate_shakify_pattern, get_available_shake_patterns,
    analyze_shake_pattern_properties, create_custom_shake_pattern,
    smooth_schedule_values, normalize_movement_values,
    detect_movement_peaks, calculate_movement_velocity,
    calculate_movement_acceleration
)

# ============================================================================
# Backward Compatibility Exports
# ============================================================================

# Re-export all functions for backward compatibility
__all__ = [
    # Detection functions
    'parse_schedule_string', 'interpolate_schedule', 'detect_movement_segments',
    'detect_circular_motion', 'detect_pattern_type', 'calculate_movement_intensity',
    'group_similar_segments',
    
    # Analysis classes and functions
    'MovementAnalyzer', 'analyze_deforum_movement',
    
    # Utility functions
    'create_shakify_data', 'apply_shakify_to_schedule',
    'generate_wan_motion_intensity_schedule',
    'validate_shakify_pattern', 'get_available_shake_patterns',
    'analyze_shake_pattern_properties', 'create_custom_shake_pattern',
    'smooth_schedule_values', 'normalize_movement_values',
    'detect_movement_peaks', 'calculate_movement_velocity',
    'calculate_movement_acceleration'
]

# Maintain compatibility with existing imports
if __name__ == "__main__":
    # Test the modular system
    print("🎬 Movement Analyzer modular system loaded successfully!")
    print("✅ Available modules:")
    print("  - movement_detection: Core detection algorithms")
    print("  - movement_analysis: High-level analysis and description generation")
    print("  - movement_utils: Camera Shakify integration and utilities")
    
    # Test basic functionality
    try:
        # Test schedule parsing
        test_schedule = "0:(0), 30:(10), 60:(5)"
        keyframes = parse_schedule_string(test_schedule)
        values = interpolate_schedule(keyframes, 100)
        
        # Test movement detection
        segments = detect_movement_segments(values, 'translation_x', 1.0)
        
        print(f"✅ Basic functionality test passed: {len(segments)} segments detected")
        
        # Test analyzer
        from types import SimpleNamespace
        test_args = SimpleNamespace(
            translation_x="0:(0), 30:(10), 60:(5)",
            translation_y="0:(0)",
            translation_z="0:(0)",
            rotation_3d_x="0:(0)",
            rotation_3d_y="0:(0)", 
            rotation_3d_z="0:(0)",
            zoom="0:(1.0)"
        )
        
        description, intensity = analyze_deforum_movement(test_args, max_frames=100)
        print(f"✅ Analysis test passed: '{description}' (intensity: {intensity:.2f})")
        
    except Exception as e:
        print(f"⚠️ Test failed: {e}")
    
    print("🎯 Movement Analyzer modular system ready for use!") 