#!/usr/bin/env python3
"""
Movement Utilities and Camera Shakify Integration
Utility functions for movement processing and Camera Shakify integration
"""

import re
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from types import SimpleNamespace

# Try to import Camera Shakify components for integration
try:
    # Direct import of shake data to avoid dependency issues
    from ....rendering.data.shakify.shake_data import SHAKE_LIST
    from ....defaults import get_camera_shake_list
    import pandas as pd
    from scipy.interpolate import CubicSpline
    SHAKIFY_AVAILABLE = True
    print("🎬 Camera Shakify integration available")
except ImportError:
    SHAKIFY_AVAILABLE = False
    print("⚠️ Camera Shakify integration not available")

# Try to import DeformAnimKeys, but fall back to standalone parsing if not available
try:
    from ....core.keyframe_animation import DeformAnimKeys
    DEFORUM_AVAILABLE = True
except ImportError:
    DEFORUM_AVAILABLE = False


def create_shakify_data(shake_name: str, shake_intensity: float, shake_speed: float, target_fps: int = 30, max_frames: int = 120, frame_start: int = 0) -> Optional[Dict]:
    """
    Create Camera Shakify data using the actual SHAKE_LIST from shakify integration
    
    Args:
        shake_name: Name of the shake pattern
        shake_intensity: Intensity multiplier
        shake_speed: Speed multiplier  
        target_fps: Target frames per second
        max_frames: Number of frames to generate
        frame_start: Starting frame offset for frame-specific analysis
    """
    if not shake_name or shake_name.lower() in ['none', 'off', ''] or shake_intensity <= 0:
        return None
    
    print(f"🎬 Creating Camera Shakify data: {shake_name} (intensity: {shake_intensity}, speed: {shake_speed}, frame_start: {frame_start})")
    
    # Check if shakify is available and shake pattern exists
    if not SHAKIFY_AVAILABLE or not SHAKE_LIST:
        print(f"⚠️ Camera Shakify not available, skipping shake generation")
        return None
    
    # Look for the shake pattern in SHAKE_LIST
    shake_pattern = None
    shake_key = shake_name.upper()
    
    if shake_key in SHAKE_LIST:
        shake_pattern = SHAKE_LIST[shake_key]
        print(f"✅ Found shake pattern: {shake_pattern[0]} at {shake_pattern[1]} fps")
    else:
        print(f"⚠️ Shake pattern '{shake_name}' not found in available patterns: {list(SHAKE_LIST.keys())}")
        return None
    
    # Extract shake data from the pattern
    pattern_name, pattern_fps, pattern_data = shake_pattern
    
    # Create result structure
    result = {
        'translation': {'x': [], 'y': [], 'z': []},
        'rotation_3d': {'x': [], 'y': [], 'z': []}
    }
    
    # Map shakify data structure to our format
    for frame in range(max_frames):
        for transform_type in ['translation', 'rotation_3d']:
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                # Map to shakify keys
                if transform_type == 'translation':
                    shakify_key = ('location', axis_idx)
                else:  # rotation_3d
                    shakify_key = ('rotation_euler', axis_idx)
                
                if shakify_key in pattern_data:
                    # Get the pattern data
                    keyframe_data = pattern_data[shakify_key]
                    
                    # Calculate source frame with speed scaling and frame offset
                    source_frame = int(((frame + frame_start) * shake_speed * pattern_fps / target_fps) % len(keyframe_data))
                    
                    # Get base value from pattern
                    if source_frame < len(keyframe_data):
                        base_value = keyframe_data[source_frame][1]  # (frame, value) tuple
                    else:
                        base_value = 0.0
                    
                    # Apply intensity scaling
                    scaled_value = base_value * shake_intensity
                    result[transform_type][axis].append(scaled_value)
                else:
                    # No data for this axis, use zero
                    result[transform_type][axis].append(0.0)
    
    print(f"✅ Generated frame-specific Camera Shakify data for frames {frame_start}-{frame_start + max_frames - 1} using pattern '{pattern_name}'")
    return result


def apply_shakify_to_schedule(base_schedule: str, shake_values: List[float], max_frames: int) -> str:
    """
    Apply Camera Shakify values to a base movement schedule to create combined schedule
    This mimics the _maybe_shake function from the experimental render core
    """
    if not shake_values or len(shake_values) == 0:
        return base_schedule
    
    # Parse base schedule
    from .movement_detection import parse_schedule_string, interpolate_schedule
    base_keyframes = parse_schedule_string(base_schedule, max_frames)
    base_values = interpolate_schedule(base_keyframes, max_frames)
    
    # Apply shake to base values (additive)
    combined_values = []
    for frame in range(min(len(base_values), len(shake_values))):
        combined_value = base_values[frame] + shake_values[frame]
        combined_values.append(combined_value)
    
    # Convert back to schedule string format
    # Sample every few frames to avoid overly long schedule strings
    sample_interval = max(1, max_frames // 20)  # Sample ~20 points
    schedule_parts = []
    
    for i in range(0, len(combined_values), sample_interval):
        frame_num = i
        value = combined_values[i]
        schedule_parts.append(f"{frame_num}:({value:.6f})")
    
    # Always include the last frame if not already sampled
    if len(combined_values) > 0 and (len(combined_values) - 1) % sample_interval != 0:
        schedule_parts.append(f"{len(combined_values) - 1}:({combined_values[-1]:.6f})")
    
    return ", ".join(schedule_parts)


def generate_wan_motion_intensity_schedule(anim_args, max_frames: int = 100, sensitivity: float = 1.0) -> str:
    """
    Generate a motion intensity schedule for Wan based on Deforum movement analysis
    
    Args:
        anim_args: Deforum animation arguments
        max_frames: Number of frames to analyze
        sensitivity: Movement detection sensitivity
        
    Returns:
        Motion intensity schedule string suitable for Wan
    """
    try:
        from .movement_detection import parse_schedule_string, interpolate_schedule, calculate_movement_intensity
        
        # Extract movement schedules
        translation_x = getattr(anim_args, 'translation_x', "0:(0)")
        translation_y = getattr(anim_args, 'translation_y', "0:(0)")
        translation_z = getattr(anim_args, 'translation_z', "0:(0)")
        
        rotation_3d_x = getattr(anim_args, 'rotation_3d_x', "0:(0)")
        rotation_3d_y = getattr(anim_args, 'rotation_3d_y', "0:(0)")
        rotation_3d_z = getattr(anim_args, 'rotation_3d_z', "0:(0)")
        
        zoom = getattr(anim_args, 'zoom', "0:(1.0)")
        
        # Interpolate all schedules
        x_values = interpolate_schedule(parse_schedule_string(translation_x), max_frames)
        y_values = interpolate_schedule(parse_schedule_string(translation_y), max_frames)
        z_values = interpolate_schedule(parse_schedule_string(translation_z), max_frames)
        
        rx_values = interpolate_schedule(parse_schedule_string(rotation_3d_x), max_frames)
        ry_values = interpolate_schedule(parse_schedule_string(rotation_3d_y), max_frames)
        rz_values = interpolate_schedule(parse_schedule_string(rotation_3d_z), max_frames)
        
        zoom_values = interpolate_schedule(parse_schedule_string(zoom), max_frames)
        
        # Calculate frame-by-frame motion intensity
        motion_intensities = []
        
        for frame in range(max_frames):
            # Calculate movement intensity for this frame
            frame_intensity = 0.0
            
            # Translation intensity (change from previous frame)
            if frame > 0:
                tx_change = abs(x_values[frame] - x_values[frame-1])
                ty_change = abs(y_values[frame] - y_values[frame-1])
                tz_change = abs(z_values[frame] - z_values[frame-1])
                
                translation_intensity = (tx_change + ty_change + tz_change) / 3.0
                frame_intensity += translation_intensity * 0.4  # Weight translation
            
            # Rotation intensity
            if frame > 0:
                rx_change = abs(rx_values[frame] - rx_values[frame-1])
                ry_change = abs(ry_values[frame] - ry_values[frame-1])
                rz_change = abs(rz_values[frame] - rz_values[frame-1])
                
                rotation_intensity = (rx_change + ry_change + rz_change) / 3.0
                frame_intensity += rotation_intensity * 0.4  # Weight rotation
            
            # Zoom intensity
            if frame > 0:
                zoom_change = abs(zoom_values[frame] - zoom_values[frame-1])
                frame_intensity += zoom_change * 0.2  # Weight zoom
            
            # Apply sensitivity scaling
            frame_intensity *= sensitivity
            
            # Normalize to 0-1 range
            frame_intensity = min(frame_intensity, 1.0)
            
            motion_intensities.append(frame_intensity)
        
        # Convert to schedule string format
        # Sample key frames to create a reasonable schedule
        schedule_parts = []
        sample_interval = max(1, max_frames // 15)  # ~15 keyframes
        
        for i in range(0, max_frames, sample_interval):
            intensity = motion_intensities[i]
            schedule_parts.append(f"{i}:({intensity:.3f})")
        
        # Always include the last frame
        if max_frames > 0 and (max_frames - 1) % sample_interval != 0:
            schedule_parts.append(f"{max_frames - 1}:({motion_intensities[-1]:.3f})")
        
        schedule_string = ", ".join(schedule_parts)
        print(f"🎯 Generated motion intensity schedule: {schedule_string}")
        
        return schedule_string
        
    except Exception as e:
        print(f"⚠️ Error generating motion intensity schedule: {e}")
        # Fallback to constant low intensity
        return f"0:(0.2), {max_frames-1}:(0.2)"


def validate_shakify_pattern(shake_name: str) -> bool:
    """
    Validate that a Camera Shakify pattern exists
    
    Args:
        shake_name: Name of the shake pattern to validate
        
    Returns:
        True if pattern exists, False otherwise
    """
    if not shake_name or not SHAKIFY_AVAILABLE:
        return False
    
    shake_key = shake_name.upper()
    return shake_key in SHAKE_LIST


def get_available_shake_patterns() -> List[str]:
    """
    Get list of available Camera Shakify patterns
    
    Returns:
        List of available shake pattern names
    """
    if not SHAKIFY_AVAILABLE or not SHAKE_LIST:
        return []
    
    return list(SHAKE_LIST.keys())


def analyze_shake_pattern_properties(shake_name: str) -> Dict:
    """
    Analyze properties of a Camera Shakify pattern
    
    Args:
        shake_name: Name of the shake pattern
        
    Returns:
        Dictionary with pattern properties
    """
    if not validate_shakify_pattern(shake_name):
        return {}
    
    try:
        shake_key = shake_name.upper()
        pattern_name, pattern_fps, pattern_data = SHAKE_LIST[shake_key]
        
        # Analyze pattern data
        properties = {
            'name': pattern_name,
            'fps': pattern_fps,
            'has_translation': any(key[0] == 'location' for key in pattern_data.keys()),
            'has_rotation': any(key[0] == 'rotation_euler' for key in pattern_data.keys()),
            'duration_frames': 0,
            'axes': []
        }
        
        # Find duration and axes
        for key, data in pattern_data.items():
            if data:
                properties['duration_frames'] = max(properties['duration_frames'], len(data))
                
                transform_type, axis_idx = key
                axis_name = ['x', 'y', 'z'][axis_idx]
                properties['axes'].append(f"{transform_type}_{axis_name}")
        
        # Calculate intensity statistics
        all_values = []
        for data in pattern_data.values():
            all_values.extend([frame_data[1] for frame_data in data])
        
        if all_values:
            properties['min_value'] = min(all_values)
            properties['max_value'] = max(all_values)
            properties['avg_magnitude'] = sum(abs(v) for v in all_values) / len(all_values)
        
        return properties
        
    except Exception as e:
        print(f"⚠️ Error analyzing shake pattern {shake_name}: {e}")
        return {}


def create_custom_shake_pattern(name: str, translation_data: Dict, rotation_data: Dict, fps: int = 30) -> bool:
    """
    Create a custom shake pattern for testing purposes
    
    Args:
        name: Name for the custom pattern
        translation_data: Translation data {x: [...], y: [...], z: [...]}
        rotation_data: Rotation data {x: [...], y: [...], z: [...]}
        fps: Pattern frame rate
        
    Returns:
        True if pattern created successfully, False otherwise
    """
    try:
        if not SHAKIFY_AVAILABLE:
            print("⚠️ Camera Shakify not available for custom patterns")
            return False
        
        # Create pattern data structure
        pattern_data = {}
        
        # Add translation data
        for axis_idx, axis in enumerate(['x', 'y', 'z']):
            if axis in translation_data and translation_data[axis]:
                key = ('location', axis_idx)
                # Convert to (frame, value) tuples
                pattern_data[key] = [(i, value) for i, value in enumerate(translation_data[axis])]
        
        # Add rotation data
        for axis_idx, axis in enumerate(['x', 'y', 'z']):
            if axis in rotation_data and rotation_data[axis]:
                key = ('rotation_euler', axis_idx)
                # Convert to (frame, value) tuples
                pattern_data[key] = [(i, value) for i, value in enumerate(rotation_data[axis])]
        
        # Add to SHAKE_LIST (for this session only)
        custom_key = name.upper()
        SHAKE_LIST[custom_key] = (name, fps, pattern_data)
        
        print(f"✅ Created custom shake pattern: {name}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating custom shake pattern: {e}")
        return False


def smooth_schedule_values(values: List[float], window_size: int = 3) -> List[float]:
    """
    Apply smoothing to schedule values to reduce jitter
    
    Args:
        values: List of values to smooth
        window_size: Size of smoothing window
        
    Returns:
        Smoothed values
    """
    if len(values) < window_size:
        return values.copy()
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(values)):
        # Calculate window bounds
        start_idx = max(0, i - half_window)
        end_idx = min(len(values), i + half_window + 1)
        
        # Calculate average over window
        window_values = values[start_idx:end_idx]
        avg_value = sum(window_values) / len(window_values)
        smoothed.append(avg_value)
    
    return smoothed


def normalize_movement_values(values: List[float], target_range: Tuple[float, float] = (-1.0, 1.0)) -> List[float]:
    """
    Normalize movement values to a target range
    
    Args:
        values: Values to normalize
        target_range: Target (min, max) range
        
    Returns:
        Normalized values
    """
    if not values:
        return values
    
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        # All values are the same
        mid_point = (target_range[0] + target_range[1]) / 2
        return [mid_point] * len(values)
    
    # Scale to target range
    target_min, target_max = target_range
    scale = (target_max - target_min) / (max_val - min_val)
    
    normalized = []
    for value in values:
        normalized_value = target_min + (value - min_val) * scale
        normalized.append(normalized_value)
    
    return normalized


def detect_movement_peaks(values: List[float], threshold: float = 0.1) -> List[int]:
    """
    Detect peaks in movement values
    
    Args:
        values: Movement values to analyze
        threshold: Minimum peak height
        
    Returns:
        List of frame indices where peaks occur
    """
    if len(values) < 3:
        return []
    
    peaks = []
    
    for i in range(1, len(values) - 1):
        # Check if this is a local maximum
        if (values[i] > values[i-1] and values[i] > values[i+1] and 
            values[i] > threshold):
            peaks.append(i)
    
    return peaks


def calculate_movement_velocity(values: List[float]) -> List[float]:
    """
    Calculate velocity (first derivative) of movement values
    
    Args:
        values: Position values
        
    Returns:
        Velocity values (one less element than input)
    """
    if len(values) < 2:
        return []
    
    velocities = []
    for i in range(1, len(values)):
        velocity = values[i] - values[i-1]
        velocities.append(velocity)
    
    return velocities


def calculate_movement_acceleration(values: List[float]) -> List[float]:
    """
    Calculate acceleration (second derivative) of movement values
    
    Args:
        values: Position values
        
    Returns:
        Acceleration values (two less elements than input)
    """
    velocities = calculate_movement_velocity(values)
    return calculate_movement_velocity(velocities) 