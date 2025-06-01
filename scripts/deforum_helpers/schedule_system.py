"""
Functional schedule parsing and interpolation system for Deforum.

This module provides pure functions for parsing schedule strings and interpolating
values across animation frames. It replaces the existing FrameInterpolater with
a functional approach that emphasizes immutability and composability.

Key principles:
- Pure functions with no side effects
- Immutable data structures
- Functional composition using map, filter, reduce
- Small, well-named functions
- Isolated side effects (numexpr evaluation)
"""

import re
import numexpr
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from enum import Enum


class InterpolationMethod(Enum):
    """Interpolation method enumeration"""
    LINEAR = "linear"
    CUBIC = "cubic"
    QUADRATIC = "quadratic"


@dataclass(frozen=True)
class ScheduleKeyframe:
    """Immutable keyframe representation"""
    frame: int
    value: Union[float, str]
    is_numeric: bool


@dataclass(frozen=True)
class ParsedSchedule:
    """Immutable result of schedule parsing"""
    keyframes: Tuple[ScheduleKeyframe, ...]
    raw_schedule: str
    max_frames: int
    is_single_string: bool = False


@dataclass(frozen=True)
class InterpolatedSchedule:
    """Immutable result of schedule interpolation"""
    values: Tuple[Union[float, str], ...]
    method: InterpolationMethod
    original_schedule: ParsedSchedule


# Pure utility functions
def sanitize_value(value: str) -> str:
    """Pure function: string -> sanitized string"""
    return value.replace("'", "").replace('"', "").replace('(', "").replace(')', "")


def check_is_number(value: str) -> bool:
    """Pure function: string -> boolean (is numeric)"""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def safe_numexpr_evaluate(expression: str, frame: int = 0, max_frames: int = 100, seed: int = -1) -> Union[float, str]:
    """
    Pure function: expression string -> evaluated value
    Isolates the side effect of numexpr evaluation
    """
    try:
        # Set up variables for numexpr
        t = frame
        max_f = max_frames - 1
        s = seed
        
        result = numexpr.evaluate(expression)
        
        # Convert numpy arrays to scalar values
        if hasattr(result, 'item'):
            return float(result.item())
        elif isinstance(result, (np.ndarray, np.generic)):
            return float(result)
        else:
            return float(result)
    except Exception:
        # Return the original expression if evaluation fails
        return expression


# Core parsing functions (functional style)
def tokenize_schedule(schedule_str: str) -> Tuple[str, ...]:
    """Pure function: schedule string -> tokens"""
    if not schedule_str or not schedule_str.strip():
        return tuple()
    
    return tuple(
        token.strip() 
        for token in schedule_str.split(",")
        if token.strip()
    )


def parse_keyframe_token(token: str) -> Optional[Tuple[int, str]]:
    """Pure function: token -> (frame, value) or None"""
    if ":" not in token:
        return None
    
    frame_part, value_part = token.split(":", 1)
    frame_part = frame_part.strip()
    value_part = value_part.strip()
    
    # Handle quoted frame numbers (like "max_f-1")
    if frame_part.startswith('"') and frame_part.endswith('"'):
        frame_part = frame_part[1:-1]
    if frame_part.startswith("'") and frame_part.endswith("'"):
        frame_part = frame_part[1:-1]
    
    try:
        # Try to parse frame as number first
        if check_is_number(sanitize_value(frame_part)):
            frame = int(sanitize_value(frame_part))
        else:
            # Evaluate mathematical expressions for frame numbers
            frame = int(safe_numexpr_evaluate(frame_part))
            
        return (frame, value_part)
    except (ValueError, TypeError):
        return None


def parse_schedule_tokens(tokens: Tuple[str, ...]) -> Tuple[ScheduleKeyframe, ...]:
    """Pure function: tokens -> keyframes using functional composition"""
    def token_to_keyframe(token: str) -> Optional[ScheduleKeyframe]:
        parsed = parse_keyframe_token(token)
        if parsed is None:
            return None
        
        frame, value = parsed
        
        # Remove outer parentheses from value if present
        if value.startswith('(') and value.endswith(')'):
            value = value[1:-1]
            
        sanitized_value = sanitize_value(value)
        is_numeric = check_is_number(sanitized_value)
        
        return ScheduleKeyframe(
            frame=frame,
            value=float(sanitized_value) if is_numeric else value,
            is_numeric=is_numeric
        )
    
    # Use functional composition: map + filter
    keyframes = tuple(filter(
        lambda kf: kf is not None,
        map(token_to_keyframe, tokens)
    ))
    
    # Sort by frame number
    return tuple(sorted(keyframes, key=lambda kf: kf.frame))


def parse_schedule_string(schedule_str: str, max_frames: int = 100, is_single_string: bool = False) -> ParsedSchedule:
    """
    Pure function: schedule string -> parsed schedule
    Main entry point for schedule parsing using functional composition
    """
    if not schedule_str or not schedule_str.strip():
        # Default keyframe at frame 0
        default_keyframe = ScheduleKeyframe(frame=0, value=0.0, is_numeric=True)
        return ParsedSchedule(
            keyframes=(default_keyframe,),
            raw_schedule=schedule_str,
            max_frames=max_frames,
            is_single_string=is_single_string
        )
    
    # Functional composition: tokenize -> parse -> create result
    tokens = tokenize_schedule(schedule_str)
    keyframes = parse_schedule_tokens(tokens)
    
    # Handle empty keyframes case
    if not keyframes:
        try:
            # Try to parse as single value
            value = float(schedule_str.strip())
            default_keyframe = ScheduleKeyframe(frame=0, value=value, is_numeric=True)
            keyframes = (default_keyframe,)
        except ValueError:
            # Use default
            default_keyframe = ScheduleKeyframe(frame=0, value=0.0, is_numeric=True)
            keyframes = (default_keyframe,)
    
    return ParsedSchedule(
        keyframes=keyframes,
        raw_schedule=schedule_str,
        max_frames=max_frames,
        is_single_string=is_single_string
    )


# Interpolation functions (functional style)
def interpolate_linear(kf1: ScheduleKeyframe, kf2: ScheduleKeyframe, frame: int) -> Union[float, str]:
    """Pure function: two keyframes + frame -> interpolated value"""
    if not (kf1.is_numeric and kf2.is_numeric):
        # For non-numeric values, use the nearest keyframe
        distance_to_kf1 = abs(frame - kf1.frame)
        distance_to_kf2 = abs(frame - kf2.frame)
        return kf1.value if distance_to_kf1 <= distance_to_kf2 else kf2.value
    
    if kf1.frame == kf2.frame:
        return kf1.value
    
    # Linear interpolation
    t = (frame - kf1.frame) / (kf2.frame - kf1.frame)
    return kf1.value + t * (kf2.value - kf1.value)


def find_surrounding_keyframes(keyframes: Tuple[ScheduleKeyframe, ...], frame: int) -> Tuple[Optional[ScheduleKeyframe], Optional[ScheduleKeyframe]]:
    """Pure function: keyframes + frame -> (prev_keyframe, next_keyframe)"""
    prev_kf = None
    next_kf = None
    
    for kf in keyframes:
        if kf.frame <= frame:
            prev_kf = kf
        if kf.frame >= frame and next_kf is None:
            next_kf = kf
            
    return prev_kf, next_kf


def interpolate_for_frame(keyframes: Tuple[ScheduleKeyframe, ...], frame: int, max_frames: int, seed: int = -1) -> Union[float, str]:
    """Pure function: keyframes + frame -> interpolated value for that frame"""
    if not keyframes:
        return 0.0
    
    prev_kf, next_kf = find_surrounding_keyframes(keyframes, frame)
    
    # Handle edge cases
    if prev_kf is None:
        # Before first keyframe
        return evaluate_keyframe_value(keyframes[0], frame, max_frames, seed)
    elif next_kf is None:
        # After last keyframe
        return evaluate_keyframe_value(prev_kf, frame, max_frames, seed)
    elif prev_kf.frame == next_kf.frame:
        # Exact match
        return evaluate_keyframe_value(prev_kf, frame, max_frames, seed)
    else:
        # Interpolate between keyframes
        prev_value = evaluate_keyframe_value(prev_kf, frame, max_frames, seed)
        next_value = evaluate_keyframe_value(next_kf, frame, max_frames, seed)
        
        if isinstance(prev_value, (int, float)) and isinstance(next_value, (int, float)):
            # Linear interpolation for numeric values
            t = (frame - prev_kf.frame) / (next_kf.frame - prev_kf.frame)
            return prev_value + t * (next_value - prev_value)
        else:
            # For non-numeric, use nearest
            distance_to_prev = abs(frame - prev_kf.frame)
            distance_to_next = abs(frame - next_kf.frame)
            return prev_value if distance_to_prev <= distance_to_next else next_value


def evaluate_keyframe_value(keyframe: ScheduleKeyframe, frame: int, max_frames: int, seed: int = -1) -> Union[float, str]:
    """Pure function: keyframe + context -> evaluated value"""
    if keyframe.is_numeric:
        return keyframe.value
    else:
        # Evaluate mathematical expression
        return safe_numexpr_evaluate(str(keyframe.value), frame, max_frames, seed)


def interpolate_schedule(parsed_schedule: ParsedSchedule, 
                        interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR,
                        seed: int = -1) -> InterpolatedSchedule:
    """
    Pure function: parsed schedule -> interpolated values using functional composition
    Main entry point for schedule interpolation
    """
    keyframes = parsed_schedule.keyframes
    max_frames = parsed_schedule.max_frames
    is_single_string = parsed_schedule.is_single_string
    
    # Use functional approach: map over all frame indices
    values = tuple(
        interpolate_for_frame(keyframes, frame, max_frames, seed)
        for frame in range(max_frames)
    )
    
    # Handle single string mode (replicate last valid value)
    if is_single_string:
        values = fill_string_values(values)
    
    return InterpolatedSchedule(
        values=values,
        method=interpolation_method,
        original_schedule=parsed_schedule
    )


def fill_string_values(values: Tuple[Union[float, str], ...]) -> Tuple[Union[float, str], ...]:
    """Pure function: values with gaps -> filled values"""
    filled_values = list(values)
    last_valid_value = None
    
    for i, value in enumerate(filled_values):
        if value is not None and value != "" and not (isinstance(value, float) and np.isnan(value)):
            last_valid_value = value
        elif last_valid_value is not None:
            filled_values[i] = last_valid_value
    
    return tuple(filled_values)


# High-level convenience functions
def parse_and_interpolate_schedule(schedule_str: str, 
                                  max_frames: int = 100, 
                                  is_single_string: bool = False,
                                  interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR,
                                  seed: int = -1) -> InterpolatedSchedule:
    """
    Pure function: schedule string -> interpolated schedule
    Convenience function that combines parsing and interpolation
    """
    parsed = parse_schedule_string(schedule_str, max_frames, is_single_string)
    return interpolate_schedule(parsed, interpolation_method, seed)


def create_schedule_series(schedule_str: str, 
                          max_frames: int = 100, 
                          is_single_string: bool = False,
                          seed: int = -1) -> Tuple[Union[float, str], ...]:
    """
    Pure function: schedule string -> value series
    Legacy compatibility function that returns just the values
    """
    interpolated = parse_and_interpolate_schedule(schedule_str, max_frames, is_single_string, InterpolationMethod.LINEAR, seed)
    return interpolated.values


# Functional helpers for working with multiple schedules
def parse_multiple_schedules(schedule_dict: Dict[str, str], 
                           max_frames: int = 100) -> Dict[str, ParsedSchedule]:
    """Pure function: dict of schedules -> dict of parsed schedules"""
    return {
        key: parse_schedule_string(schedule_str, max_frames)
        for key, schedule_str in schedule_dict.items()
    }


def interpolate_multiple_schedules(parsed_schedules: Dict[str, ParsedSchedule],
                                 interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR,
                                 seed: int = -1) -> Dict[str, InterpolatedSchedule]:
    """Pure function: dict of parsed schedules -> dict of interpolated schedules"""
    return {
        key: interpolate_schedule(parsed, interpolation_method, seed)
        for key, parsed in parsed_schedules.items()
    }


def extract_values_at_frame(interpolated_schedules: Dict[str, InterpolatedSchedule], 
                           frame: int) -> Dict[str, Union[float, str]]:
    """Pure function: interpolated schedules + frame -> values at that frame"""
    return {
        key: schedule.values[frame] if frame < len(schedule.values) else schedule.values[-1]
        for key, schedule in interpolated_schedules.items()
    }


# Performance optimization: batch processing
def batch_interpolate_frames(parsed_schedule: ParsedSchedule, 
                           frame_indices: Tuple[int, ...],
                           seed: int = -1) -> Tuple[Union[float, str], ...]:
    """
    Pure function: parsed schedule + frame indices -> interpolated values for those frames
    Optimized for when you only need specific frame values
    """
    keyframes = parsed_schedule.keyframes
    max_frames = parsed_schedule.max_frames
    
    return tuple(
        interpolate_for_frame(keyframes, frame, max_frames, seed)
        for frame in frame_indices
    )


# Validation functions
def validate_schedule_syntax(schedule_str: str) -> Tuple[bool, Optional[str]]:
    """Pure function: schedule string -> (is_valid, error_message)"""
    try:
        parsed = parse_schedule_string(schedule_str, 100)
        if not parsed.keyframes:
            return False, "No valid keyframes found in schedule"
        return True, None
    except Exception as e:
        return False, str(e)


def get_schedule_statistics(interpolated_schedule: InterpolatedSchedule) -> Dict[str, Any]:
    """Pure function: interpolated schedule -> statistics dict"""
    values = interpolated_schedule.values
    numeric_values = tuple(filter(lambda v: isinstance(v, (int, float)), values))
    
    if not numeric_values:
        return {
            "type": "string",
            "unique_values": len(set(values)),
            "first_value": values[0] if values else None,
            "last_value": values[-1] if values else None
        }
    
    return {
        "type": "numeric",
        "min": min(numeric_values),
        "max": max(numeric_values),
        "mean": sum(numeric_values) / len(numeric_values),
        "range": max(numeric_values) - min(numeric_values),
        "first_value": values[0],
        "last_value": values[-1],
        "total_frames": len(values)
    } 