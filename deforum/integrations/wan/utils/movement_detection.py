#!/usr/bin/env python3
"""
Movement Detection Algorithms
Core algorithms for detecting movement patterns and analyzing camera motion
"""

import re
import math
from typing import Dict, List, Tuple, Optional
import numpy as np


def parse_schedule_string(schedule_str: str, max_frames: int = 100) -> List[Tuple[int, float]]:
    """
    Parse Deforum schedule string into frame-value pairs
    
    Format: "0:(1.0), 30:(2.0), 60:(1.5)"
    Returns: [(0, 1.0), (30, 2.0), (60, 1.5)]
    """
    if not schedule_str or schedule_str.strip() == "":
        return [(0, 0.0)]
        
    # Clean the string
    schedule_str = schedule_str.strip()
    
    # Pattern to match frame:(value) pairs
    pattern = r'(\d+):\s*\(([^)]+)\)'
    matches = re.findall(pattern, schedule_str)
    
    if not matches:
        # Try to parse as single value
        try:
            value = float(schedule_str)
            return [(0, value)]
        except ValueError:
            return [(0, 0.0)]
    
    # Convert matches to frame-value pairs
    keyframes = []
    for frame_str, value_str in matches:
        try:
            frame = int(frame_str)
            value = float(value_str)
            keyframes.append((frame, value))
        except ValueError:
            continue
    
    # Sort by frame number
    keyframes.sort(key=lambda x: x[0])
    
    return keyframes if keyframes else [(0, 0.0)]


def interpolate_schedule(keyframes: List[Tuple[int, float]], max_frames: int) -> List[float]:
    """
    Interpolate schedule values across all frames
    """
    if not keyframes:
        return [0.0] * max_frames
    
    values = []
    
    for frame in range(max_frames):
        # Find surrounding keyframes
        prev_kf = None
        next_kf = None
        
        for kf in keyframes:
            if kf[0] <= frame:
                prev_kf = kf
            if kf[0] >= frame and next_kf is None:
                next_kf = kf
        
        if prev_kf is None:
            values.append(keyframes[0][1])
        elif next_kf is None:
            values.append(prev_kf[1])
        elif prev_kf[0] == next_kf[0]:
            values.append(prev_kf[1])
        else:
            # Linear interpolation
            t = (frame - prev_kf[0]) / (next_kf[0] - prev_kf[0])
            value = prev_kf[1] + t * (next_kf[1] - prev_kf[1])
            values.append(value)
    
    return values


def detect_movement_segments(values: List[float], movement_type: str, sensitivity: float = 1.0) -> List[Dict]:
    """
    Detect movement segments in a value sequence
    
    Args:
        values: List of movement values per frame
        movement_type: Type of movement (translation_x, rotation_y, zoom, etc.)
        sensitivity: Detection sensitivity threshold
        
    Returns:
        List of movement segments with start/end frames and properties
    """
    if not values:
        return []
    
    # Detect significant changes based on movement type
    if movement_type in ['translation_x', 'translation_y']:
        # For translation, use absolute change detection
        threshold = 5.0 * sensitivity  # Minimum significant movement
        derivative_threshold = 0.1 * sensitivity  # Minimum speed
    elif movement_type == 'translation_z':
        # Z movement (dolly) is more sensitive
        threshold = 2.0 * sensitivity
        derivative_threshold = 0.05 * sensitivity
    elif movement_type in ['rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z']:
        # Rotation in degrees
        threshold = 1.0 * sensitivity
        derivative_threshold = 0.02 * sensitivity
    elif movement_type == 'zoom':
        # Zoom is relative change
        threshold = 0.02 * sensitivity  # 2% change
        derivative_threshold = 0.001 * sensitivity
    else:
        # Default thresholds
        threshold = 1.0 * sensitivity
        derivative_threshold = 0.01 * sensitivity
    
    segments = []
    current_segment = None
    
    for i in range(len(values)):
        # Calculate movement properties at this frame
        current_value = values[i]
        
        # Calculate derivative (speed of change)
        if i > 0:
            derivative = abs(values[i] - values[i-1])
        else:
            derivative = 0.0
        
        # Determine if this frame has significant movement
        has_movement = derivative > derivative_threshold or abs(current_value) > threshold
        
        if has_movement:
            if current_segment is None:
                # Start new segment
                current_segment = {
                    'start_frame': i,
                    'end_frame': i,
                    'movement_type': movement_type,
                    'start_value': current_value,
                    'end_value': current_value,
                    'max_derivative': derivative,
                    'total_change': 0.0
                }
            else:
                # Extend current segment
                current_segment['end_frame'] = i
                current_segment['end_value'] = current_value
                current_segment['max_derivative'] = max(current_segment['max_derivative'], derivative)
                current_segment['total_change'] = abs(current_value - current_segment['start_value'])
        else:
            # No significant movement - end current segment if it exists
            if current_segment is not None:
                # Only add segment if it's long enough or has significant change
                if (current_segment['end_frame'] - current_segment['start_frame'] >= 2 or 
                    current_segment['total_change'] > threshold):
                    segments.append(current_segment)
                current_segment = None
    
    # Don't forget the last segment
    if current_segment is not None:
        if (current_segment['end_frame'] - current_segment['start_frame'] >= 2 or 
            current_segment['total_change'] > threshold):
            segments.append(current_segment)
    
    return segments


def detect_circular_motion(x_values: List[float], y_values: List[float], min_frames: int = 10) -> bool:
    """
    Detect if X and Y movement values form a circular pattern
    
    Args:
        x_values: X movement values per frame
        y_values: Y movement values per frame  
        min_frames: Minimum frames required for circular motion
        
    Returns:
        True if circular motion detected, False otherwise
    """
    if len(x_values) != len(y_values) or len(x_values) < min_frames:
        return False
    
    try:
        # Convert to numpy arrays for easier computation
        x_arr = np.array(x_values)
        y_arr = np.array(y_values)
        
        # Remove DC component (center the motion)
        x_centered = x_arr - np.mean(x_arr)
        y_centered = y_arr - np.mean(y_arr)
        
        # Calculate distances from center
        distances = np.sqrt(x_centered**2 + y_centered**2)
        
        # For circular motion, distances should be relatively constant
        if len(distances) < 3:
            return False
            
        distance_std = np.std(distances)
        distance_mean = np.mean(distances)
        
        # Check if standard deviation is small relative to mean (consistent radius)
        if distance_mean == 0:
            return False
            
        radius_consistency = distance_std / distance_mean
        
        # Also check if there's enough variation in angles
        angles = np.arctan2(y_centered, x_centered)
        angle_range = np.max(angles) - np.min(angles)
        
        # Circular motion should have:
        # 1. Consistent radius (low radius_consistency)
        # 2. Significant angular variation
        # 3. Sufficient magnitude
        
        is_circular = (
            radius_consistency < 0.3 and  # Radius varies by less than 30%
            angle_range > math.pi/2 and   # At least 90 degrees of rotation
            distance_mean > 1.0           # Sufficient movement magnitude
        )
        
        return is_circular
        
    except Exception:
        return False


def detect_pattern_type(x_values: List[float], y_values: List[float], z_values: List[float]) -> str:
    """
    Detect the type of movement pattern from 3D values
    
    Args:
        x_values: X movement values
        y_values: Y movement values  
        z_values: Z movement values
        
    Returns:
        Pattern type string ('linear', 'circular', 'complex', 'oscillating', 'static')
    """
    try:
        # Check for static (no movement)
        x_range = max(x_values) - min(x_values) if x_values else 0
        y_range = max(y_values) - min(y_values) if y_values else 0
        z_range = max(z_values) - min(z_values) if z_values else 0
        
        total_range = x_range + y_range + z_range
        if total_range < 0.1:
            return 'static'
        
        # Check for circular motion in XY plane
        if detect_circular_motion(x_values, y_values):
            return 'circular'
        
        # Check for oscillating motion (back and forth)
        def is_oscillating(values):
            if len(values) < 6:
                return False
            
            # Count direction changes
            direction_changes = 0
            for i in range(2, len(values)):
                prev_diff = values[i-1] - values[i-2]
                curr_diff = values[i] - values[i-1]
                if (prev_diff > 0 and curr_diff < 0) or (prev_diff < 0 and curr_diff > 0):
                    direction_changes += 1
            
            # Oscillating if there are multiple direction changes
            oscillation_ratio = direction_changes / (len(values) - 2)
            return oscillation_ratio > 0.3
        
        x_oscillating = is_oscillating(x_values)
        y_oscillating = is_oscillating(y_values)
        z_oscillating = is_oscillating(z_values)
        
        if x_oscillating or y_oscillating or z_oscillating:
            return 'oscillating'
        
        # Check for linear motion (consistent direction)
        def is_linear(values):
            if len(values) < 3:
                return True
            
            # Count direction changes
            direction_changes = 0
            for i in range(2, len(values)):
                prev_diff = values[i-1] - values[i-2]
                curr_diff = values[i] - values[i-1]
                if abs(prev_diff) > 0.01 and abs(curr_diff) > 0.01:
                    if (prev_diff > 0 and curr_diff < 0) or (prev_diff < 0 and curr_diff > 0):
                        direction_changes += 1
            
            # Linear if few direction changes
            if len(values) <= 2:
                return True
            change_ratio = direction_changes / (len(values) - 2)
            return change_ratio < 0.2
        
        x_linear = is_linear(x_values)
        y_linear = is_linear(y_values)
        z_linear = is_linear(z_values)
        
        # Count axes with significant linear movement
        linear_axes = sum([x_linear and x_range > 1.0, 
                          y_linear and y_range > 1.0, 
                          z_linear and z_range > 1.0])
        
        if linear_axes >= 1:
            return 'linear'
        
        # Default to complex if no clear pattern
        return 'complex'
        
    except Exception:
        return 'complex'


def calculate_movement_intensity(values: List[float], movement_type: str) -> float:
    """
    Calculate the intensity/magnitude of movement for a value sequence
    
    Args:
        values: Movement values per frame
        movement_type: Type of movement for scaling
        
    Returns:
        Normalized intensity value (0.0 to 1.0+)
    """
    if not values:
        return 0.0
    
    try:
        # Calculate total variation
        total_variation = sum(abs(values[i] - values[i-1]) for i in range(1, len(values)))
        
        # Calculate range
        value_range = max(values) - min(values)
        
        # Movement-specific scaling
        if movement_type in ['translation_x', 'translation_y', 'translation_z']:
            # Translation intensity based on total distance
            intensity = total_variation / len(values) / 10.0  # Scale for typical camera movement
        elif movement_type in ['rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z']:
            # Rotation intensity based on total rotation
            intensity = total_variation / len(values) / 5.0  # Scale for typical rotation
        elif movement_type == 'zoom':
            # Zoom intensity based on relative change
            intensity = value_range * 2.0  # Zoom is typically small values
        else:
            # Generic intensity calculation
            intensity = total_variation / len(values)
        
        # Normalize to 0-1 range with some allowance for high intensity
        return min(intensity, 2.0)
        
    except Exception:
        return 0.0


def group_similar_segments(segments: List[Dict], max_frames: int, similarity_threshold: float = 0.3) -> List[List[Dict]]:
    """
    Group similar movement segments together for better analysis
    
    Args:
        segments: List of movement segments to group
        max_frames: Total number of frames
        similarity_threshold: Threshold for grouping similar segments
        
    Returns:
        List of segment groups
    """
    if not segments:
        return []
    
    groups = []
    remaining_segments = segments.copy()
    
    while remaining_segments:
        # Start new group with first remaining segment
        current_group = [remaining_segments.pop(0)]
        reference_segment = current_group[0]
        
        # Find similar segments to add to this group
        i = 0
        while i < len(remaining_segments):
            segment = remaining_segments[i]
            
            # Calculate similarity based on:
            # 1. Duration similarity
            # 2. Movement magnitude similarity
            # 3. Direction similarity (if applicable)
            
            ref_duration = reference_segment['end_frame'] - reference_segment['start_frame']
            seg_duration = segment['end_frame'] - segment['start_frame']
            
            duration_similarity = 1.0 - abs(ref_duration - seg_duration) / max(ref_duration, seg_duration, 1)
            
            ref_magnitude = abs(reference_segment['total_change'])
            seg_magnitude = abs(segment['total_change'])
            
            magnitude_similarity = 1.0 - abs(ref_magnitude - seg_magnitude) / max(ref_magnitude, seg_magnitude, 1)
            
            # Overall similarity
            overall_similarity = (duration_similarity + magnitude_similarity) / 2.0
            
            if overall_similarity > similarity_threshold:
                current_group.append(remaining_segments.pop(i))
            else:
                i += 1
        
        groups.append(current_group)
    
    return groups 