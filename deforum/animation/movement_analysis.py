"""
Functional movement analysis system for Deforum.

This module provides pure functions for analyzing camera movement patterns from
animation schedules. It replaces the existing MovementAnalyzer class with a
functional approach that emphasizes immutability and composability.

Key principles:
- Pure functions with no side effects
- Immutable data structures
- Functional composition using map, filter, reduce
- Small, well-named functions
- Isolated side effects (logging, external dependencies)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum

from .schedule_system import (
    parse_and_interpolate_schedule, InterpolatedSchedule
)
from ..models.data_models import AnimationArgs


class MovementType(Enum):
    """Movement type enumeration"""
    TRANSLATION_X = "translation_x"
    TRANSLATION_Y = "translation_y"
    TRANSLATION_Z = "translation_z"
    ROTATION_X = "rotation_x"
    ROTATION_Y = "rotation_y"
    ROTATION_Z = "rotation_z"
    ZOOM = "zoom"
    ANGLE = "angle"


class MovementDirection(Enum):
    """Movement direction enumeration"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STATIC = "static"


class MovementIntensity(Enum):
    """Movement intensity enumeration"""
    SUBTLE = "subtle"
    GENTLE = "gentle"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


@dataclass(frozen=True)
class MovementSegment:
    """Immutable movement segment representation"""
    start_frame: int
    end_frame: int
    movement_type: MovementType
    direction: MovementDirection
    max_change: float
    total_range: float
    intensity: MovementIntensity


@dataclass(frozen=True)
class MovementData:
    """Immutable movement data extracted from animation args"""
    translation_x_values: Tuple[float, ...]
    translation_y_values: Tuple[float, ...]
    translation_z_values: Tuple[float, ...]
    rotation_x_values: Tuple[float, ...]
    rotation_y_values: Tuple[float, ...]
    rotation_z_values: Tuple[float, ...]
    zoom_values: Tuple[float, ...]
    angle_values: Tuple[float, ...]
    max_frames: int


@dataclass(frozen=True)
class MovementAnalysisResult:
    """Immutable result of movement analysis"""
    description: str
    strength: float
    segments: Tuple[MovementSegment, ...]
    movement_data: MovementData


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable analysis configuration"""
    sensitivity: float = 1.0
    min_translation_threshold: float = 0.001
    min_rotation_threshold: float = 0.001
    min_zoom_threshold: float = 0.0001
    segment_grouping_window: float = 0.1  # 10% of total frames


# Pure utility functions for movement calculation
def calculate_frame_changes(values: Tuple[float, ...]) -> Tuple[float, ...]:
    """Pure function: values -> frame-to-frame changes"""
    if len(values) < 2:
        return tuple()
    
    return tuple(
        values[i] - values[i-1]
        for i in range(1, len(values))
    )


def determine_movement_intensity(total_range: float, movement_type: MovementType) -> MovementIntensity:
    """Pure function: range + type -> intensity"""
    if movement_type == MovementType.ZOOM:
        thresholds = [0.01, 0.05, 0.2, 1.0]
    elif movement_type.value.startswith('rotation'):
        thresholds = [1.0, 5.0, 20.0, 90.0]
    else:  # translation
        thresholds = [1.0, 5.0, 20.0, 100.0]
    
    if total_range < thresholds[0]:
        return MovementIntensity.SUBTLE
    elif total_range < thresholds[1]:
        return MovementIntensity.GENTLE
    elif total_range < thresholds[2]:
        return MovementIntensity.MODERATE
    elif total_range < thresholds[3]:
        return MovementIntensity.STRONG
    else:
        return MovementIntensity.EXTREME


def get_movement_threshold(movement_type: MovementType, config: AnalysisConfig) -> float:
    """Pure function: movement type + config -> threshold"""
    base_thresholds = {
        MovementType.ZOOM: config.min_zoom_threshold,
        MovementType.ROTATION_X: config.min_rotation_threshold,
        MovementType.ROTATION_Y: config.min_rotation_threshold,
        MovementType.ROTATION_Z: config.min_rotation_threshold,
    }
    
    return base_thresholds.get(movement_type, config.min_translation_threshold) * config.sensitivity


def extract_movement_data(animation_args: AnimationArgs) -> MovementData:
    """Pure function: animation args -> movement data"""
    max_frames = animation_args.max_frames
    
    # Parse all schedules using our functional schedule system
    tx_schedule = parse_and_interpolate_schedule(animation_args.translation_x, max_frames)
    ty_schedule = parse_and_interpolate_schedule(animation_args.translation_y, max_frames)
    tz_schedule = parse_and_interpolate_schedule(animation_args.translation_z, max_frames)
    rx_schedule = parse_and_interpolate_schedule(animation_args.rotation_3d_x, max_frames)
    ry_schedule = parse_and_interpolate_schedule(animation_args.rotation_3d_y, max_frames)
    rz_schedule = parse_and_interpolate_schedule(animation_args.rotation_3d_z, max_frames)
    zoom_schedule = parse_and_interpolate_schedule(animation_args.zoom, max_frames)
    angle_schedule = parse_and_interpolate_schedule(animation_args.angle, max_frames)
    
    return MovementData(
        translation_x_values=tuple(float(v) for v in tx_schedule.values),
        translation_y_values=tuple(float(v) for v in ty_schedule.values),
        translation_z_values=tuple(float(v) for v in tz_schedule.values),
        rotation_x_values=tuple(float(v) for v in rx_schedule.values),
        rotation_y_values=tuple(float(v) for v in ry_schedule.values),
        rotation_z_values=tuple(float(v) for v in rz_schedule.values),
        zoom_values=tuple(float(v) for v in zoom_schedule.values),
        angle_values=tuple(float(v) for v in angle_schedule.values),
        max_frames=max_frames
    )


# Core movement analysis functions (functional style)
def detect_movement_segments_for_axis(values: Tuple[float, ...], 
                                     movement_type: MovementType,
                                     config: AnalysisConfig) -> Tuple[MovementSegment, ...]:
    """Pure function: values + type + config -> movement segments"""
    if len(values) < 2:
        return tuple()
    
    changes = calculate_frame_changes(values)
    threshold = get_movement_threshold(movement_type, config)
    
    segments = []
    current_segment = None
    
    for frame_idx, change in enumerate(changes):
        abs_change = abs(change)
        
        if abs_change > threshold:
            direction = MovementDirection.INCREASING if change > 0 else MovementDirection.DECREASING
            
            # Start new segment or continue existing one
            if current_segment is None or current_segment['direction'] != direction:
                # End previous segment
                if current_segment is not None:
                    current_segment['end_frame'] = frame_idx
                    segments.append(_create_movement_segment(current_segment, values, movement_type))
                
                # Start new segment
                current_segment = {
                    'start_frame': frame_idx,
                    'end_frame': frame_idx,
                    'direction': direction,
                    'max_change': abs_change,
                    'start_value': values[frame_idx],
                }
            else:
                # Continue current segment
                current_segment['end_frame'] = frame_idx
                current_segment['max_change'] = max(current_segment['max_change'], abs_change)
        
        elif current_segment is not None:
            # End current segment if movement stopped
            current_segment['end_frame'] = frame_idx
            segments.append(_create_movement_segment(current_segment, values, movement_type))
            current_segment = None
    
    # Close final segment if needed
    if current_segment is not None:
        current_segment['end_frame'] = len(changes)
        segments.append(_create_movement_segment(current_segment, values, movement_type))
    
    return tuple(segments)


def _create_movement_segment(segment_data: Dict, values: Tuple[float, ...], 
                           movement_type: MovementType) -> MovementSegment:
    """Pure function: segment data -> MovementSegment"""
    start_frame = segment_data['start_frame']
    end_frame = segment_data['end_frame']
    direction = segment_data['direction']
    max_change = segment_data['max_change']
    
    # Calculate total range
    start_value = values[start_frame] if start_frame < len(values) else 0.0
    end_value = values[min(end_frame, len(values) - 1)]
    total_range = abs(end_value - start_value)
    
    intensity = determine_movement_intensity(total_range, movement_type)
    
    return MovementSegment(
        start_frame=start_frame,
        end_frame=end_frame,
        movement_type=movement_type,
        direction=direction,
        max_change=max_change,
        total_range=total_range,
        intensity=intensity
    )


def detect_all_movement_segments(movement_data: MovementData, 
                               config: AnalysisConfig) -> Tuple[MovementSegment, ...]:
    """Pure function: movement data + config -> all movement segments"""
    all_segments = []
    
    # Define axis mappings using functional approach
    axis_mappings = [
        (movement_data.translation_x_values, MovementType.TRANSLATION_X),
        (movement_data.translation_y_values, MovementType.TRANSLATION_Y),
        (movement_data.translation_z_values, MovementType.TRANSLATION_Z),
        (movement_data.rotation_x_values, MovementType.ROTATION_X),
        (movement_data.rotation_y_values, MovementType.ROTATION_Y),
        (movement_data.rotation_z_values, MovementType.ROTATION_Z),
        (movement_data.zoom_values, MovementType.ZOOM),
        (movement_data.angle_values, MovementType.ANGLE),
    ]
    
    # Use functional composition to detect segments for all axes
    for values, movement_type in axis_mappings:
        segments = detect_movement_segments_for_axis(values, movement_type, config)
        all_segments.extend(segments)
    
    # Sort by start frame for chronological order
    return tuple(sorted(all_segments, key=lambda s: s.start_frame))


# Segment grouping functions (functional style)
def should_group_segments(seg1: MovementSegment, seg2: MovementSegment, 
                         max_frames: int, grouping_window: float) -> bool:
    """Pure function: two segments + config -> should group boolean"""
    # Check if segments should be grouped together
    same_movement_type = seg1.movement_type == seg2.movement_type
    same_direction = seg1.direction == seg2.direction
    frame_gap = seg2.start_frame - seg1.end_frame
    max_gap = max_frames * grouping_window
    
    return same_movement_type and same_direction and frame_gap <= max_gap


def group_similar_segments(segments: Tuple[MovementSegment, ...],
                         max_frames: int,
                         grouping_window: float = 0.1) -> Tuple[Tuple[MovementSegment, ...], ...]:
    """Pure function: segments -> grouped segments using functional approach"""
    if not segments:
        return tuple()
    
    groups = []
    current_group = [segments[0]]
    
    for i in range(1, len(segments)):
        current_seg = segments[i]
        prev_seg = segments[i-1]
        
        if should_group_segments(prev_seg, current_seg, max_frames, grouping_window):
            current_group.append(current_seg)
        else:
            groups.append(tuple(current_group))
            current_group = [current_seg]
    
    # Add the last group
    groups.append(tuple(current_group))
    
    return tuple(groups)


# Description generation functions (functional style)
def generate_movement_description(movement_type: MovementType, 
                                direction: MovementDirection,
                                intensity: MovementIntensity) -> str:
    """Pure function: movement characteristics -> description"""
    # Movement direction descriptions
    direction_map = {
        (MovementType.TRANSLATION_X, MovementDirection.INCREASING): "panning right",
        (MovementType.TRANSLATION_X, MovementDirection.DECREASING): "panning left",
        (MovementType.TRANSLATION_Y, MovementDirection.INCREASING): "moving up",
        (MovementType.TRANSLATION_Y, MovementDirection.DECREASING): "moving down",
        (MovementType.TRANSLATION_Z, MovementDirection.INCREASING): "dolly forward",
        (MovementType.TRANSLATION_Z, MovementDirection.DECREASING): "dolly backward",
        (MovementType.ROTATION_X, MovementDirection.INCREASING): "tilting up",
        (MovementType.ROTATION_X, MovementDirection.DECREASING): "tilting down",
        (MovementType.ROTATION_Y, MovementDirection.INCREASING): "rotating right",
        (MovementType.ROTATION_Y, MovementDirection.DECREASING): "rotating left",
        (MovementType.ROTATION_Z, MovementDirection.INCREASING): "rolling clockwise",
        (MovementType.ROTATION_Z, MovementDirection.DECREASING): "rolling counter-clockwise",
        (MovementType.ZOOM, MovementDirection.INCREASING): "zooming in",
        (MovementType.ZOOM, MovementDirection.DECREASING): "zooming out",
        (MovementType.ANGLE, MovementDirection.INCREASING): "rotating clockwise",
        (MovementType.ANGLE, MovementDirection.DECREASING): "rotating counter-clockwise",
    }
    
    motion = direction_map.get((movement_type, direction), f"{movement_type.value} {direction.value}")
    
    return f"{intensity.value} {motion}"


def describe_segment_group(segment_group: Tuple[MovementSegment, ...],
                         max_frames: int) -> str:
    """Pure function: segment group -> description"""
    if not segment_group:
        return ""
    
    # Use the first segment to determine movement characteristics
    main_segment = segment_group[0]
    
    # Calculate group duration
    start_frame = min(seg.start_frame for seg in segment_group)
    end_frame = max(seg.end_frame for seg in segment_group)
    duration = end_frame - start_frame + 1
    
    # Calculate combined intensity
    total_range = sum(seg.total_range for seg in segment_group)
    combined_intensity = determine_movement_intensity(total_range, main_segment.movement_type)
    
    # Generate base description
    base_desc = generate_movement_description(
        main_segment.movement_type,
        main_segment.direction,
        combined_intensity
    )
    
    # Add duration qualifier
    if duration < max_frames * 0.2:
        duration_desc = "brief"
    elif duration < max_frames * 0.5:
        duration_desc = "extended"
    else:
        duration_desc = "sustained"
    
    return f"{base_desc} ({duration_desc})"


def generate_overall_description(segment_groups: Tuple[Tuple[MovementSegment, ...], ...],
                               max_frames: int) -> str:
    """Pure function: segment groups -> overall description"""
    if not segment_groups:
        return "static camera position"
    
    # Generate descriptions for each group
    group_descriptions = tuple(filter(
        lambda desc: desc,  # Filter out empty descriptions
        map(lambda group: describe_segment_group(group, max_frames), segment_groups)
    ))
    
    if not group_descriptions:
        return "static camera position"
    
    # Combine descriptions using functional approach
    if len(group_descriptions) == 1:
        return f"camera movement with {group_descriptions[0]}"
    elif len(group_descriptions) <= 4:
        return f"camera movement with {', '.join(group_descriptions[:-1])} and {group_descriptions[-1]}"
    else:
        return f"complex camera movement with {', '.join(group_descriptions[:3])} and {len(group_descriptions)-3} additional motion phases"


def calculate_movement_strength(segment_groups: Tuple[Tuple[MovementSegment, ...], ...],
                              max_frames: int) -> float:
    """Pure function: segment groups -> movement strength"""
    if not segment_groups:
        return 0.0
    
    total_strength = 0.0
    
    for group in segment_groups:
        for segment in group:
            # Calculate strength based on range and duration
            duration = segment.end_frame - segment.start_frame + 1
            strength = (segment.total_range * duration) / (max_frames * 10.0)
            total_strength += strength
    
    # Cap at 2.0 and ensure positive
    return max(0.0, min(2.0, total_strength))


# High-level analysis functions
def analyze_movement(animation_args: AnimationArgs, 
                   config: Optional[AnalysisConfig] = None) -> MovementAnalysisResult:
    """
    Pure function: animation args + config -> movement analysis result
    Main entry point for movement analysis using functional composition
    """
    if config is None:
        config = AnalysisConfig()
    
    # Functional composition: extract -> detect -> group -> describe
    movement_data = extract_movement_data(animation_args)
    all_segments = detect_all_movement_segments(movement_data, config)
    segment_groups = group_similar_segments(all_segments, movement_data.max_frames, config.segment_grouping_window)
    
    description = generate_overall_description(segment_groups, movement_data.max_frames)
    strength = calculate_movement_strength(segment_groups, movement_data.max_frames)
    
    return MovementAnalysisResult(
        description=description,
        strength=strength,
        segments=all_segments,
        movement_data=movement_data
    )


def analyze_movement_with_sensitivity(animation_args: AnimationArgs, 
                                    sensitivity: float = 1.0) -> MovementAnalysisResult:
    """Pure function: animation args + sensitivity -> analysis result"""
    config = AnalysisConfig(
        sensitivity=sensitivity,
        min_translation_threshold=0.001 * sensitivity,
        min_rotation_threshold=0.001 * sensitivity,
        min_zoom_threshold=0.0001 * sensitivity
    )
    
    return analyze_movement(animation_args, config)


# Statistics and analysis functions
def get_movement_statistics(analysis_result: MovementAnalysisResult) -> Dict[str, Any]:
    """Pure function: analysis result -> statistics"""
    segments = analysis_result.segments
    movement_data = analysis_result.movement_data
    
    if not segments:
        return {
            "total_segments": 0,
            "movement_types": [],
            "total_duration": 0,
            "average_intensity": 0.0,
            "coverage_percentage": 0.0
        }
    
    # Calculate statistics using functional approach
    movement_types = list(set(seg.movement_type.value for seg in segments))
    total_duration = sum(seg.end_frame - seg.start_frame + 1 for seg in segments)
    intensities = [seg.intensity.value for seg in segments]
    
    # Calculate coverage percentage
    active_frames = set()
    for seg in segments:
        active_frames.update(range(seg.start_frame, seg.end_frame + 1))
    
    coverage_percentage = len(active_frames) / movement_data.max_frames * 100 if movement_data.max_frames > 0 else 0
    
    return {
        "total_segments": len(segments),
        "movement_types": sorted(movement_types),
        "total_duration": total_duration,
        "average_intensity": len(intensities) and intensities.count(max(set(intensities), key=intensities.count)) / len(intensities) or 0.0,
        "coverage_percentage": coverage_percentage,
        "max_frames": movement_data.max_frames,
        "strength": analysis_result.strength
    }


# Functional helpers for specific movement types
def detect_circular_motion(movement_data: MovementData) -> Tuple[bool, float]:
    """Pure function: movement data -> (is_circular, strength)"""
    x_values = movement_data.rotation_x_values
    y_values = movement_data.rotation_y_values
    
    if len(x_values) < 4 or len(y_values) < 4:
        return False, 0.0
    
    # Calculate ranges
    x_range = max(x_values) - min(x_values)
    y_range = max(y_values) - min(y_values)
    
    # Check if both axes have significant motion
    if x_range < 2.0 or y_range < 2.0:
        return False, 0.0
    
    # Simple correlation check for circular motion
    # In a true circular motion, X and Y should have a phase relationship
    try:
        import numpy as np
        x_array = np.array(x_values)
        y_array = np.array(y_values)
        
        # Normalize to remove DC offset
        x_norm = x_array - np.mean(x_array)
        y_norm = y_array - np.mean(y_array)
        
        # Calculate correlation with 90-degree phase shift
        quarter_period = len(x_values) // 4
        if quarter_period > 0:
            x_shifted = np.roll(x_norm, quarter_period)
            correlation = np.corrcoef(y_norm, x_shifted)[0, 1]
            
            # High correlation indicates circular motion
            is_circular = abs(correlation) > 0.7
            strength = (x_range + y_range) / 2 if is_circular else 0.0
            
            return is_circular, strength
    except ImportError:
        pass
    
    return False, 0.0


def detect_camera_shake_pattern(movement_data: MovementData) -> Tuple[bool, str, float]:
    """Pure function: movement data -> (has_shake, description, strength)"""
    # Look for high-frequency, low-amplitude movements across multiple axes
    all_values = [
        movement_data.translation_x_values,
        movement_data.translation_y_values,
        movement_data.translation_z_values,
        movement_data.rotation_x_values,
        movement_data.rotation_y_values,
        movement_data.rotation_z_values,
    ]
    
    shake_indicators = 0
    total_intensity = 0.0
    
    for values in all_values:
        if len(values) < 2:
            continue
            
        changes = calculate_frame_changes(values)
        
        # Count direction changes (indicates shake)
        direction_changes = 0
        prev_direction = None
        
        for change in changes:
            if abs(change) > 0.001:  # Threshold for meaningful change
                current_direction = "pos" if change > 0 else "neg"
                if prev_direction and current_direction != prev_direction:
                    direction_changes += 1
                prev_direction = current_direction
        
        # High direction change rate indicates shake
        change_rate = direction_changes / len(changes) if changes else 0
        if change_rate > 0.3:  # More than 30% direction changes
            shake_indicators += 1
            total_intensity += change_rate
    
    # Require shake in at least 2 axes
    has_shake = shake_indicators >= 2
    strength = total_intensity / len(all_values) if has_shake else 0.0
    
    if has_shake:
        if strength > 0.8:
            description = "intense camera shake"
        elif strength > 0.5:
            description = "moderate camera shake"
        else:
            description = "subtle camera shake"
    else:
        description = ""
    
    return has_shake, description, strength


# Legacy compatibility functions
def create_movement_schedule_series(animation_args: AnimationArgs) -> Dict[str, Tuple[float, ...]]:
    """Pure function: animation args -> schedule series (legacy compatibility)"""
    movement_data = extract_movement_data(animation_args)
    
    return {
        "translation_x": movement_data.translation_x_values,
        "translation_y": movement_data.translation_y_values,
        "translation_z": movement_data.translation_z_values,
        "rotation_x": movement_data.rotation_x_values,
        "rotation_y": movement_data.rotation_y_values,
        "rotation_z": movement_data.rotation_z_values,
        "zoom": movement_data.zoom_values,
        "angle": movement_data.angle_values,
    } 