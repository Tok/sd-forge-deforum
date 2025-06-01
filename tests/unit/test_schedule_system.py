#!/usr/bin/env python3

"""
Unit tests for schedule system functionality.
Tests the functional schedule parsing and evaluation system.
"""

import pytest
import unittest
from typing import Dict, Any, List
import math

# Import from the new location
from deforum.animation.schedules import (
    ScheduleSystem,
    parse_schedule_string,
    evaluate_schedule_at_frame,
    validate_schedule_syntax
)

import numpy as np
from typing import Tuple, Dict

from scripts.deforum_helpers.schedule_system import (
    # Data structures
    ScheduleKeyframe, ParsedSchedule, InterpolatedSchedule, InterpolationMethod,
    
    # Pure utility functions
    sanitize_value, check_is_number, safe_numexpr_evaluate,
    
    # Core parsing functions
    tokenize_schedule, parse_keyframe_token, parse_schedule_tokens, parse_schedule_string,
    
    # Interpolation functions
    interpolate_linear, find_surrounding_keyframes, interpolate_for_frame, 
    evaluate_keyframe_value, interpolate_schedule, fill_string_values,
    
    # High-level functions
    parse_and_interpolate_schedule, create_schedule_series,
    
    # Multi-schedule functions
    parse_multiple_schedules, interpolate_multiple_schedules, extract_values_at_frame,
    
    # Performance and validation
    batch_interpolate_frames, validate_schedule_syntax, get_schedule_statistics
)


class TestDataStructures:
    """Test immutable data structures"""
    
    def test_schedule_keyframe_immutability(self):
        """Test that ScheduleKeyframe is immutable"""
        kf = ScheduleKeyframe(frame=10, value=5.0, is_numeric=True)
        
        with pytest.raises(AttributeError):
            kf.frame = 20  # Should be immutable
        
        assert kf.frame == 10
        assert kf.value == 5.0
        assert kf.is_numeric is True
    
    def test_parsed_schedule_immutability(self):
        """Test that ParsedSchedule is immutable"""
        kf = ScheduleKeyframe(frame=0, value=1.0, is_numeric=True)
        schedule = ParsedSchedule(
            keyframes=(kf,),
            raw_schedule="0:(1.0)",
            max_frames=100
        )
        
        with pytest.raises(AttributeError):
            schedule.max_frames = 200
        
        assert len(schedule.keyframes) == 1
        assert schedule.max_frames == 100
    
    def test_interpolated_schedule_immutability(self):
        """Test that InterpolatedSchedule is immutable"""
        kf = ScheduleKeyframe(frame=0, value=1.0, is_numeric=True)
        parsed = ParsedSchedule(keyframes=(kf,), raw_schedule="0:(1.0)", max_frames=10)
        interpolated = InterpolatedSchedule(
            values=(1.0, 1.0, 1.0),
            method=InterpolationMethod.LINEAR,
            original_schedule=parsed
        )
        
        with pytest.raises(AttributeError):
            interpolated.method = InterpolationMethod.CUBIC
        
        assert len(interpolated.values) == 3
        assert interpolated.method == InterpolationMethod.LINEAR


class TestUtilityFunctions:
    """Test pure utility functions"""
    
    def test_sanitize_value(self):
        """Test value sanitization"""
        assert sanitize_value("(1.5)") == "1.5"
        assert sanitize_value("'test'") == "test"
        assert sanitize_value('"quoted"') == "quoted"
        assert sanitize_value("(sin(t))") == "sint"  # Note: removes all parentheses
    
    def test_check_is_number(self):
        """Test numeric checking"""
        assert check_is_number("1.5") is True
        assert check_is_number("-10") is True
        assert check_is_number("0") is True
        assert check_is_number("1e-5") is True
        assert check_is_number("sin(t)") is False
        assert check_is_number("test") is False
        assert check_is_number("") is False
    
    def test_safe_numexpr_evaluate(self):
        """Test safe expression evaluation"""
        # Simple numeric expression
        result = safe_numexpr_evaluate("2 + 3")
        assert result == 5.0
        
        # Expression with variables
        result = safe_numexpr_evaluate("t * 2", frame=5)
        assert result == 10.0
        
        # Expression with max_f
        result = safe_numexpr_evaluate("max_f / 2", max_frames=100)
        assert result == 49.5  # (100-1) / 2
        
        # Invalid expression should return original
        result = safe_numexpr_evaluate("invalid_expression")
        assert result == "invalid_expression"


class TestParsingFunctions:
    """Test schedule parsing functions"""
    
    def test_tokenize_schedule(self):
        """Test schedule tokenization"""
        # Normal schedule
        tokens = tokenize_schedule("0:(1.0), 30:(2.0), 60:(1.5)")
        assert tokens == ("0:(1.0)", "30:(2.0)", "60:(1.5)")
        
        # Empty schedule
        tokens = tokenize_schedule("")
        assert tokens == ()
        
        # Single token
        tokens = tokenize_schedule("0:(1.0)")
        assert tokens == ("0:(1.0)",)
        
        # With extra spaces
        tokens = tokenize_schedule(" 0:(1.0) ,  30:(2.0)  ")
        assert tokens == ("0:(1.0)", "30:(2.0)")
    
    def test_parse_keyframe_token(self):
        """Test individual keyframe token parsing"""
        # Simple numeric frame and value
        result = parse_keyframe_token("0:(1.5)")
        assert result == (0, "(1.5)")
        
        # Frame with quotes
        result = parse_keyframe_token('"max_f-1":(0.5)')
        assert result is not None
        assert result[1] == "(0.5)"
        
        # Mathematical expression in frame
        result = parse_keyframe_token("30:(sin(t))")
        assert result == (30, "(sin(t))")
        
        # Invalid token
        result = parse_keyframe_token("invalid")
        assert result is None
        
        # Missing colon
        result = parse_keyframe_token("0(1.0)")
        assert result is None
    
    def test_parse_schedule_tokens(self):
        """Test conversion of tokens to keyframes"""
        tokens = ("0:(1.0)", "30:(2.5)", "60:(sin(t))")
        keyframes = parse_schedule_tokens(tokens)
        
        assert len(keyframes) == 3
        
        # First keyframe - numeric
        assert keyframes[0].frame == 0
        assert keyframes[0].value == 1.0
        assert keyframes[0].is_numeric is True
        
        # Second keyframe - numeric
        assert keyframes[1].frame == 30
        assert keyframes[1].value == 2.5
        assert keyframes[1].is_numeric is True
        
        # Third keyframe - expression
        assert keyframes[2].frame == 60
        assert keyframes[2].value == "sin(t)"
        assert keyframes[2].is_numeric is False
        
        # Should be sorted by frame
        assert keyframes[0].frame <= keyframes[1].frame <= keyframes[2].frame
    
    def test_parse_schedule_string_simple(self):
        """Test simple schedule string parsing"""
        schedule = parse_schedule_string("0:(1.0), 30:(2.0)", max_frames=100)
        
        assert len(schedule.keyframes) == 2
        assert schedule.raw_schedule == "0:(1.0), 30:(2.0)"
        assert schedule.max_frames == 100
        assert schedule.is_single_string is False
        
        # Check keyframes
        assert schedule.keyframes[0].frame == 0
        assert schedule.keyframes[0].value == 1.0
        assert schedule.keyframes[1].frame == 30
        assert schedule.keyframes[1].value == 2.0
    
    def test_parse_schedule_string_empty(self):
        """Test empty schedule string handling"""
        schedule = parse_schedule_string("", max_frames=100)
        
        assert len(schedule.keyframes) == 1
        assert schedule.keyframes[0].frame == 0
        assert schedule.keyframes[0].value == 0.0
        assert schedule.keyframes[0].is_numeric is True
    
    def test_parse_schedule_string_single_value(self):
        """Test single value parsing"""
        schedule = parse_schedule_string("1.5", max_frames=100)
        
        assert len(schedule.keyframes) == 1
        assert schedule.keyframes[0].frame == 0
        assert schedule.keyframes[0].value == 1.5
        assert schedule.keyframes[0].is_numeric is True
    
    def test_parse_schedule_string_complex(self):
        """Test complex schedule with expressions"""
        schedule_str = '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)'
        schedule = parse_schedule_string(schedule_str, max_frames=100)
        
        assert len(schedule.keyframes) >= 2  # At least the numeric ones
        
        # Check that string expressions are preserved (parentheses removed)
        for kf in schedule.keyframes:
            if not kf.is_numeric:
                assert kf.value in ["s", "-1"]


class TestInterpolationFunctions:
    """Test interpolation functions"""
    
    def test_interpolate_linear_numeric(self):
        """Test linear interpolation between numeric keyframes"""
        kf1 = ScheduleKeyframe(frame=0, value=0.0, is_numeric=True)
        kf2 = ScheduleKeyframe(frame=10, value=10.0, is_numeric=True)
        
        # Test midpoint
        result = interpolate_linear(kf1, kf2, 5)
        assert result == 5.0
        
        # Test quarter point
        result = interpolate_linear(kf1, kf2, 2)
        assert result == 2.0
        
        # Test exact keyframe
        result = interpolate_linear(kf1, kf2, 0)
        assert result == 0.0
        
        result = interpolate_linear(kf1, kf2, 10)
        assert result == 10.0
    
    def test_interpolate_linear_string(self):
        """Test linear interpolation with string values"""
        kf1 = ScheduleKeyframe(frame=0, value="first", is_numeric=False)
        kf2 = ScheduleKeyframe(frame=10, value="second", is_numeric=False)
        
        # Should use nearest keyframe for strings
        result = interpolate_linear(kf1, kf2, 3)
        assert result == "first"  # Closer to kf1
        
        result = interpolate_linear(kf1, kf2, 7)
        assert result == "second"  # Closer to kf2
    
    def test_find_surrounding_keyframes(self):
        """Test finding surrounding keyframes"""
        keyframes = (
            ScheduleKeyframe(frame=0, value=0.0, is_numeric=True),
            ScheduleKeyframe(frame=10, value=5.0, is_numeric=True),
            ScheduleKeyframe(frame=20, value=10.0, is_numeric=True)
        )
        
        # Frame within range
        prev, next_kf = find_surrounding_keyframes(keyframes, 5)
        assert prev.frame == 0
        assert next_kf.frame == 10
        
        # Frame at keyframe
        prev, next_kf = find_surrounding_keyframes(keyframes, 10)
        assert prev.frame == 10
        assert next_kf.frame == 10
        
        # Frame before first keyframe
        prev, next_kf = find_surrounding_keyframes(keyframes, -5)
        assert prev is None
        assert next_kf.frame == 0
        
        # Frame after last keyframe
        prev, next_kf = find_surrounding_keyframes(keyframes, 25)
        assert prev.frame == 20
        assert next_kf is None
    
    def test_evaluate_keyframe_value(self):
        """Test keyframe value evaluation"""
        # Numeric keyframe
        kf_numeric = ScheduleKeyframe(frame=0, value=5.0, is_numeric=True)
        result = evaluate_keyframe_value(kf_numeric, 0, 100, -1)
        assert result == 5.0
        
        # Expression keyframe
        kf_expr = ScheduleKeyframe(frame=0, value="t * 2", is_numeric=False)
        result = evaluate_keyframe_value(kf_expr, 5, 100, -1)
        assert result == 10.0  # 5 * 2
    
    def test_interpolate_for_frame(self):
        """Test interpolation for specific frame"""
        keyframes = (
            ScheduleKeyframe(frame=0, value=0.0, is_numeric=True),
            ScheduleKeyframe(frame=10, value=10.0, is_numeric=True)
        )
        
        # Test interpolation within range
        result = interpolate_for_frame(keyframes, 5, 100)
        assert result == 5.0
        
        # Test before first keyframe
        result = interpolate_for_frame(keyframes, -5, 100)
        assert result == 0.0
        
        # Test after last keyframe
        result = interpolate_for_frame(keyframes, 15, 100)
        assert result == 10.0
    
    def test_fill_string_values(self):
        """Test filling gaps in string values"""
        values = ("first", None, "", "second", None)
        filled = fill_string_values(values)
        
        assert filled[0] == "first"
        assert filled[1] == "first"  # Should propagate last valid
        assert filled[2] == "first"  # Empty string replaced
        assert filled[3] == "second"
        assert filled[4] == "second"  # Should propagate last valid
    
    def test_interpolate_schedule(self):
        """Test full schedule interpolation"""
        kf1 = ScheduleKeyframe(frame=0, value=0.0, is_numeric=True)
        kf2 = ScheduleKeyframe(frame=9, value=9.0, is_numeric=True)
        
        parsed = ParsedSchedule(
            keyframes=(kf1, kf2),
            raw_schedule="0:(0.0), 9:(9.0)",
            max_frames=10
        )
        
        interpolated = interpolate_schedule(parsed)
        
        assert len(interpolated.values) == 10
        assert interpolated.method == InterpolationMethod.LINEAR
        assert interpolated.original_schedule == parsed
        
        # Check interpolated values
        assert interpolated.values[0] == 0.0
        assert interpolated.values[5] == 5.0
        assert interpolated.values[9] == 9.0


class TestHighLevelFunctions:
    """Test high-level convenience functions"""
    
    def test_parse_and_interpolate_schedule(self):
        """Test combined parse and interpolate"""
        result = parse_and_interpolate_schedule("0:(0), 9:(9)", max_frames=10)
        
        assert isinstance(result, InterpolatedSchedule)
        assert len(result.values) == 10
        assert result.values[0] == 0.0
        assert result.values[9] == 9.0
    
    def test_create_schedule_series(self):
        """Test legacy-compatible schedule series creation"""
        values = create_schedule_series("0:(0), 9:(9)", max_frames=10)
        
        assert isinstance(values, tuple)
        assert len(values) == 10
        assert values[0] == 0.0
        assert values[9] == 9.0
    
    def test_parse_multiple_schedules(self):
        """Test parsing multiple schedules at once"""
        schedules = {
            "translation_x": "0:(0), 30:(10)",
            "translation_y": "0:(0), 30:(5)",
            "zoom": "0:(1.0), 30:(1.2)"
        }
        
        parsed = parse_multiple_schedules(schedules, max_frames=100)
        
        assert len(parsed) == 3
        assert "translation_x" in parsed
        assert len(parsed["translation_x"].keyframes) == 2
        assert parsed["translation_y"].keyframes[1].value == 5.0
    
    def test_interpolate_multiple_schedules(self):
        """Test interpolating multiple parsed schedules"""
        kf1 = ScheduleKeyframe(frame=0, value=0.0, is_numeric=True)
        kf2 = ScheduleKeyframe(frame=9, value=9.0, is_numeric=True)
        
        parsed_schedules = {
            "schedule1": ParsedSchedule(keyframes=(kf1, kf2), raw_schedule="0:(0), 9:(9)", max_frames=10),
            "schedule2": ParsedSchedule(keyframes=(kf1, kf2), raw_schedule="0:(0), 9:(9)", max_frames=10)
        }
        
        interpolated = interpolate_multiple_schedules(parsed_schedules)
        
        assert len(interpolated) == 2
        assert len(interpolated["schedule1"].values) == 10
        assert interpolated["schedule2"].values[5] == 5.0
    
    def test_extract_values_at_frame(self):
        """Test extracting values at specific frame"""
        values1 = tuple(range(10))  # 0, 1, 2, ..., 9
        values2 = tuple(x * 2 for x in range(10))  # 0, 2, 4, ..., 18
        
        kf = ScheduleKeyframe(frame=0, value=0.0, is_numeric=True)
        parsed = ParsedSchedule(keyframes=(kf,), raw_schedule="", max_frames=10)
        
        interpolated_schedules = {
            "sched1": InterpolatedSchedule(values=values1, method=InterpolationMethod.LINEAR, original_schedule=parsed),
            "sched2": InterpolatedSchedule(values=values2, method=InterpolationMethod.LINEAR, original_schedule=parsed)
        }
        
        frame_values = extract_values_at_frame(interpolated_schedules, 5)
        
        assert frame_values["sched1"] == 5
        assert frame_values["sched2"] == 10
        
        # Test out of bounds
        frame_values = extract_values_at_frame(interpolated_schedules, 15)
        assert frame_values["sched1"] == 9  # Should use last value
        assert frame_values["sched2"] == 18


class TestPerformanceAndValidation:
    """Test performance optimization and validation functions"""
    
    def test_batch_interpolate_frames(self):
        """Test batch interpolation for specific frame indices"""
        kf1 = ScheduleKeyframe(frame=0, value=0.0, is_numeric=True)
        kf2 = ScheduleKeyframe(frame=10, value=10.0, is_numeric=True)
        
        parsed = ParsedSchedule(
            keyframes=(kf1, kf2),
            raw_schedule="0:(0), 10:(10)",
            max_frames=20
        )
        
        frame_indices = (2, 5, 8)
        values = batch_interpolate_frames(parsed, frame_indices)
        
        assert len(values) == 3
        assert values[0] == 2.0  # Frame 2
        assert values[1] == 5.0  # Frame 5
        assert values[2] == 8.0  # Frame 8
    
    def test_validate_schedule_syntax(self):
        """Test schedule syntax validation"""
        # Valid schedules
        valid, error = validate_schedule_syntax("0:(1.0), 30:(2.0)")
        assert valid is True
        assert error is None
        
        valid, error = validate_schedule_syntax("0:(sin(t))")
        assert valid is True
        assert error is None
        
        # Empty schedule (valid - creates default)
        valid, error = validate_schedule_syntax("")
        assert valid is True
        
        # Single value (valid)
        valid, error = validate_schedule_syntax("1.5")
        assert valid is True
    
    def test_get_schedule_statistics_numeric(self):
        """Test statistics for numeric schedules"""
        values = (0.0, 2.5, 5.0, 7.5, 10.0)
        kf = ScheduleKeyframe(frame=0, value=0.0, is_numeric=True)
        parsed = ParsedSchedule(keyframes=(kf,), raw_schedule="", max_frames=5)
        
        interpolated = InterpolatedSchedule(
            values=values,
            method=InterpolationMethod.LINEAR,
            original_schedule=parsed
        )
        
        stats = get_schedule_statistics(interpolated)
        
        assert stats["type"] == "numeric"
        assert stats["min"] == 0.0
        assert stats["max"] == 10.0
        assert stats["mean"] == 5.0
        assert stats["range"] == 10.0
        assert stats["first_value"] == 0.0
        assert stats["last_value"] == 10.0
        assert stats["total_frames"] == 5
    
    def test_get_schedule_statistics_string(self):
        """Test statistics for string schedules"""
        values = ("first", "second", "first", "third")
        kf = ScheduleKeyframe(frame=0, value="first", is_numeric=False)
        parsed = ParsedSchedule(keyframes=(kf,), raw_schedule="", max_frames=4)
        
        interpolated = InterpolatedSchedule(
            values=values,
            method=InterpolationMethod.LINEAR,
            original_schedule=parsed
        )
        
        stats = get_schedule_statistics(interpolated)
        
        assert stats["type"] == "string"
        assert stats["unique_values"] == 3  # first, second, third
        assert stats["first_value"] == "first"
        assert stats["last_value"] == "third"


class TestIntegrationScenarios:
    """Test real-world integration scenarios"""
    
    def test_deforum_style_schedules(self):
        """Test actual Deforum-style schedule strings"""
        # Translation schedule
        translation_x = "0:(0), 30:(10), 60:(0), 90:(-10), 120:(0)"
        interpolated = parse_and_interpolate_schedule(translation_x, max_frames=121)
        
        assert interpolated.values[0] == 0.0
        assert interpolated.values[30] == 10.0
        assert interpolated.values[60] == 0.0
        assert interpolated.values[90] == -10.0
        assert interpolated.values[120] == 0.0
        
        # Check interpolation
        assert interpolated.values[15] == 5.0  # Halfway between 0 and 10
    
    def test_mathematical_expressions(self):
        """Test schedules with mathematical expressions"""
        # Simple mathematical expression that can be evaluated
        simple_math = "0:(2 + 3)"
        interpolated = parse_and_interpolate_schedule(simple_math, max_frames=10, seed=42)
        
        # Should have values for all frames
        assert len(interpolated.values) == 10
        
        # Values should be numeric (expressions evaluated)
        assert all(isinstance(v, (int, float)) for v in interpolated.values)
        assert interpolated.values[0] == 5.0  # 2 + 3
        
        # Test with frame-dependent expression
        frame_expr = "0:(t * 2)"
        interpolated = parse_and_interpolate_schedule(frame_expr, max_frames=5, seed=42)
        
        # Should have frame-dependent values
        assert len(interpolated.values) == 5
        assert interpolated.values[0] == 0.0  # t=0, 0*2=0
        assert interpolated.values[2] == 4.0  # t=2, 2*2=4
    
    def test_seed_schedule_compatibility(self):
        """Test complex seed scheduling like Deforum uses"""
        seed_schedule = '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)'
        interpolated = parse_and_interpolate_schedule(seed_schedule, max_frames=100, seed=12345)
        
        # Should handle the schedule without errors
        assert len(interpolated.values) == 100
        
        # Should have some values (though they may be expressions)
        assert interpolated.values[0] is not None
    
    def test_large_schedule_performance(self):
        """Test performance with large schedules"""
        # Create a schedule with many keyframes
        keyframe_parts = [f"{i*10}:({i})" for i in range(50)]  # 50 keyframes
        large_schedule = ", ".join(keyframe_parts)
        
        # Should handle large schedules efficiently
        interpolated = parse_and_interpolate_schedule(large_schedule, max_frames=1000)
        
        assert len(interpolated.values) == 1000
        assert interpolated.values[0] == 0.0
        assert interpolated.values[490] == 49.0  # Frame 490 should be value 49
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Schedule with gaps
        gap_schedule = "0:(0), 100:(10)"
        interpolated = parse_and_interpolate_schedule(gap_schedule, max_frames=101)
        
        assert interpolated.values[0] == 0.0
        assert interpolated.values[50] == 5.0  # Interpolated middle
        assert interpolated.values[100] == 10.0
        
        # Schedule with single keyframe
        single_schedule = "50:(7.5)"
        interpolated = parse_and_interpolate_schedule(single_schedule, max_frames=100)
        
        # All frames should have the same value
        assert all(v == 7.5 for v in interpolated.values)


class TestFunctionalProgrammingPrinciples:
    """Test that functional programming principles are followed"""
    
    def test_pure_functions_no_side_effects(self):
        """Test that pure functions don't modify input arguments"""
        original_schedule = "0:(0), 30:(10)"
        max_frames = 100
        
        # Parse multiple times - should get identical results
        result1 = parse_schedule_string(original_schedule, max_frames)
        result2 = parse_schedule_string(original_schedule, max_frames)
        
        # Results should be equal (same parsing)
        assert result1.raw_schedule == result2.raw_schedule
        assert len(result1.keyframes) == len(result2.keyframes)
        assert result1.max_frames == result2.max_frames
        
        # Original string should be unchanged
        assert original_schedule == "0:(0), 30:(10)"
    
    def test_immutability_preserved(self):
        """Test that data structures remain immutable throughout processing"""
        schedule_str = "0:(0), 30:(10)"
        parsed = parse_schedule_string(schedule_str, 100)
        interpolated = interpolate_schedule(parsed)
        
        # Original parsed schedule should be unchanged
        assert parsed.raw_schedule == "0:(0), 30:(10)"
        assert len(parsed.keyframes) == 2
        
        # Interpolated schedule should reference original
        assert interpolated.original_schedule is parsed
        
        # Cannot modify any part
        with pytest.raises(AttributeError):
            parsed.max_frames = 200
        
        with pytest.raises(AttributeError):
            interpolated.values = (1, 2, 3)
    
    def test_functional_composition(self):
        """Test that functions compose well together"""
        # Create a pipeline using function composition
        schedule_str = "0:(0), 30:(10), 60:(0)"
        
        # Compose functions: parse -> interpolate -> extract statistics
        parsed = parse_schedule_string(schedule_str, 100)
        interpolated = interpolate_schedule(parsed)
        stats = get_schedule_statistics(interpolated)
        
        # Should work seamlessly
        assert stats["type"] == "numeric"
        assert stats["min"] == 0.0
        assert stats["max"] == 10.0
        
        # Can also use convenience function
        direct_interpolated = parse_and_interpolate_schedule(schedule_str, 100)
        direct_stats = get_schedule_statistics(direct_interpolated)
        
        # Should get same results
        assert stats == direct_stats
    
    def test_map_filter_usage(self):
        """Test that functional operators like map and filter are used correctly"""
        # This tests the internal use of functional operators
        tokens = ("0:(1.0)", "invalid", "30:(2.0)", "")
        keyframes = parse_schedule_tokens(tokens)
        
        # Should filter out invalid tokens and map valid ones
        assert len(keyframes) == 2  # Only 2 valid keyframes
        assert keyframes[0].frame == 0
        assert keyframes[1].frame == 30
        
        # Test that None values are filtered out properly
        assert all(kf is not None for kf in keyframes) 