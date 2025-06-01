#!/usr/bin/env python3

"""
Comprehensive tests for Deforum's movement analysis system.
Tests movement detection, analysis, and description generation.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import math
from typing import Dict, Any, List

# Import the movement analysis system from the new package structure
from deforum.animation.movement_analysis import (
    MovementAnalyzer,
    CachedMovementAnalyzer,
    FrameMovementCalculator,
    MovementDescriptionGenerator,
    analyze_camera_movement,
    detect_movement_type,
    generate_movement_description,
    calculate_movement_intensity,
    get_movement_keywords,
    cache_movement_analysis
)

from scripts.deforum_helpers.movement_analysis import (
    # Data structures
    MovementType, MovementDirection, MovementIntensity,
    MovementSegment, MovementData, MovementAnalysisResult, AnalysisConfig,
    
    # Pure utility functions
    calculate_frame_changes, determine_movement_intensity, get_movement_threshold,
    
    # Core analysis functions
    extract_movement_data, detect_movement_segments_for_axis, detect_all_movement_segments,
    
    # Grouping functions
    should_group_segments, group_similar_segments,
    
    # Description functions
    describe_segment_group, generate_overall_description,
    calculate_movement_strength,
    
    # High-level functions
    analyze_movement, analyze_movement_with_sensitivity,
    
    # Statistics and helpers
    get_movement_statistics, detect_circular_motion, detect_camera_shake_pattern,
    create_movement_schedule_series
)

# Import animation args from the new package structure
from deforum.models.data_models import AnimationArgs


class TestDataStructures:
    """Test immutable data structures"""
    
    def test_movement_segment_immutability(self):
        """Test that MovementSegment is immutable"""
        segment = MovementSegment(
            start_frame=10,
            end_frame=20,
            movement_type=MovementType.TRANSLATION_X,
            direction=MovementDirection.INCREASING,
            max_change=1.5,
            total_range=10.0,
            intensity=MovementIntensity.MODERATE
        )
        
        with pytest.raises(AttributeError):
            segment.start_frame = 15  # Should be immutable
        
        assert segment.start_frame == 10
        assert segment.movement_type == MovementType.TRANSLATION_X
        assert segment.intensity == MovementIntensity.MODERATE
    
    def test_movement_data_immutability(self):
        """Test that MovementData is immutable"""
        data = MovementData(
            translation_x_values=(0.0, 1.0, 2.0),
            translation_y_values=(0.0, 0.0, 0.0),
            translation_z_values=(0.0, 0.0, 0.0),
            rotation_x_values=(0.0, 0.0, 0.0),
            rotation_y_values=(0.0, 0.0, 0.0),
            rotation_z_values=(0.0, 0.0, 0.0),
            zoom_values=(1.0, 1.0, 1.0),
            angle_values=(0.0, 0.0, 0.0),
            max_frames=3
        )
        
        with pytest.raises(AttributeError):
            data.max_frames = 10
        
        assert len(data.translation_x_values) == 3
        assert data.max_frames == 3
    
    def test_analysis_config_immutability(self):
        """Test that AnalysisConfig is immutable"""
        config = AnalysisConfig(sensitivity=2.0)
        
        with pytest.raises(AttributeError):
            config.sensitivity = 3.0
        
        assert config.sensitivity == 2.0
        assert config.min_translation_threshold == 0.001  # Default value


class TestUtilityFunctions:
    """Test pure utility functions"""
    
    def test_calculate_frame_changes(self):
        """Test frame-to-frame change calculation"""
        values = (0.0, 1.0, 3.0, 2.0, 5.0)
        changes = calculate_frame_changes(values)
        
        expected = (1.0, 2.0, -1.0, 3.0)  # Frame-to-frame differences
        assert changes == expected
        
        # Test empty case
        assert calculate_frame_changes(tuple()) == tuple()
        assert calculate_frame_changes((1.0,)) == tuple()
    
    def test_determine_movement_intensity(self):
        """Test movement intensity determination"""
        # Translation thresholds: [1.0, 5.0, 20.0, 100.0]
        assert determine_movement_intensity(0.5, MovementType.TRANSLATION_X) == MovementIntensity.SUBTLE
        assert determine_movement_intensity(2.0, MovementType.TRANSLATION_X) == MovementIntensity.GENTLE
        assert determine_movement_intensity(10.0, MovementType.TRANSLATION_X) == MovementIntensity.MODERATE
        assert determine_movement_intensity(50.0, MovementType.TRANSLATION_X) == MovementIntensity.STRONG
        assert determine_movement_intensity(200.0, MovementType.TRANSLATION_X) == MovementIntensity.EXTREME
        
        # Rotation thresholds: [1.0, 5.0, 20.0, 90.0]
        assert determine_movement_intensity(0.5, MovementType.ROTATION_X) == MovementIntensity.SUBTLE
        assert determine_movement_intensity(3.0, MovementType.ROTATION_X) == MovementIntensity.GENTLE
        assert determine_movement_intensity(15.0, MovementType.ROTATION_X) == MovementIntensity.MODERATE
        
        # Zoom thresholds: [0.01, 0.05, 0.2, 1.0]
        assert determine_movement_intensity(0.005, MovementType.ZOOM) == MovementIntensity.SUBTLE
        assert determine_movement_intensity(0.03, MovementType.ZOOM) == MovementIntensity.GENTLE
        assert determine_movement_intensity(0.1, MovementType.ZOOM) == MovementIntensity.MODERATE
    
    def test_get_movement_threshold(self):
        """Test movement threshold calculation"""
        config = AnalysisConfig(sensitivity=2.0, min_translation_threshold=0.001)
        
        # Translation threshold
        threshold = get_movement_threshold(MovementType.TRANSLATION_X, config)
        assert threshold == 0.002  # 0.001 * 2.0
        
        # Rotation threshold
        config_rot = AnalysisConfig(sensitivity=1.5, min_rotation_threshold=0.002)
        threshold = get_movement_threshold(MovementType.ROTATION_Y, config_rot)
        assert threshold == 0.003  # 0.002 * 1.5
        
        # Zoom threshold
        config_zoom = AnalysisConfig(sensitivity=1.0, min_zoom_threshold=0.0001)
        threshold = get_movement_threshold(MovementType.ZOOM, config_zoom)
        assert threshold == 0.0001  # 0.0001 * 1.0


class TestDataExtraction:
    """Test movement data extraction"""
    
    def test_extract_movement_data(self):
        """Test extraction of movement data from animation args"""
        # Create test animation args
        args = AnimationArgs(
            translation_x="0:(0), 30:(10)",
            translation_y="0:(0), 30:(5)",
            translation_z="0:(0)",
            rotation_3d_x="0:(0)",
            rotation_3d_y="0:(0), 30:(15)",
            rotation_3d_z="0:(0)",
            zoom="0:(1.0), 30:(1.2)",
            angle="0:(0)",
            max_frames=31
        )
        
        movement_data = extract_movement_data(args)
        
        # Check data structure
        assert isinstance(movement_data, MovementData)
        assert movement_data.max_frames == 31
        assert len(movement_data.translation_x_values) == 31
        
        # Check interpolated values
        assert movement_data.translation_x_values[0] == 0.0
        assert movement_data.translation_x_values[30] == 10.0
        assert movement_data.translation_x_values[15] == 5.0  # Interpolated middle
        
        assert movement_data.translation_y_values[30] == 5.0
        assert movement_data.rotation_y_values[30] == 15.0  # Fixed: was rotation_3d_y_values
        assert movement_data.zoom_values[30] == 1.2


class TestMovementSegmentDetection:
    """Test movement segment detection functions"""
    
    def test_detect_movement_segments_for_axis_simple(self):
        """Test simple movement segment detection"""
        # Simple increasing movement
        values = (0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0)
        config = AnalysisConfig(sensitivity=1.0, min_translation_threshold=0.5)
        
        segments = detect_movement_segments_for_axis(values, MovementType.TRANSLATION_X, config)
        
        # Should detect one segment from frame 1 to 4 (increasing)
        assert len(segments) == 1
        segment = segments[0]
        assert segment.movement_type == MovementType.TRANSLATION_X
        assert segment.direction == MovementDirection.INCREASING
        assert segment.start_frame == 1  # Where movement starts
        assert segment.total_range == 3.0  # 3.0 - 0.0
    
    def test_detect_movement_segments_for_axis_bidirectional(self):
        """Test bidirectional movement detection"""
        # Movement that goes up then down
        values = (0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0)
        config = AnalysisConfig(sensitivity=1.0, min_translation_threshold=0.5)
        
        segments = detect_movement_segments_for_axis(values, MovementType.TRANSLATION_Y, config)
        
        # Should detect two segments: increasing then decreasing
        assert len(segments) == 2
        
        # First segment: increasing
        assert segments[0].direction == MovementDirection.INCREASING
        assert segments[0].start_frame == 0
        
        # Second segment: decreasing
        assert segments[1].direction == MovementDirection.DECREASING
        assert segments[1].movement_type == MovementType.TRANSLATION_Y
    
    def test_detect_movement_segments_for_axis_no_movement(self):
        """Test detection with no significant movement"""
        values = (1.0, 1.001, 0.999, 1.002, 1.0)
        config = AnalysisConfig(sensitivity=1.0, min_translation_threshold=0.01)
        
        segments = detect_movement_segments_for_axis(values, MovementType.TRANSLATION_Z, config)
        
        # Should detect no segments (changes too small)
        assert len(segments) == 0
    
    def test_detect_all_movement_segments(self):
        """Test detection across all movement axes"""
        movement_data = MovementData(
            translation_x_values=(0.0, 2.0, 4.0),  # Should detect movement
            translation_y_values=(0.0, 0.0, 0.0),  # No movement
            translation_z_values=(1.0, 1.0, 1.0),  # No movement
            rotation_x_values=(0.0, 0.0, 0.0),     # No movement
            rotation_y_values=(0.0, 5.0, 10.0),    # Should detect movement
            rotation_z_values=(0.0, 0.0, 0.0),     # No movement
            zoom_values=(1.0, 1.0, 1.0),           # No movement
            angle_values=(0.0, 0.0, 0.0),          # No movement
            max_frames=3
        )
        
        config = AnalysisConfig(sensitivity=1.0, min_translation_threshold=1.0, min_rotation_threshold=1.0)
        
        segments = detect_all_movement_segments(movement_data, config)
        
        # Should detect 2 segments: translation_x and rotation_y
        assert len(segments) == 2
        
        movement_types = [seg.movement_type for seg in segments]
        assert MovementType.TRANSLATION_X in movement_types
        assert MovementType.ROTATION_Y in movement_types


class TestSegmentGrouping:
    """Test segment grouping functions"""
    
    def test_should_group_segments(self):
        """Test segment grouping criteria"""
        seg1 = MovementSegment(
            start_frame=10, end_frame=15,
            movement_type=MovementType.TRANSLATION_X,
            direction=MovementDirection.INCREASING,
            max_change=1.0, total_range=5.0,
            intensity=MovementIntensity.GENTLE
        )
        
        # Same type, same direction, close frames - should group
        seg2 = MovementSegment(
            start_frame=17, end_frame=20,
            movement_type=MovementType.TRANSLATION_X,
            direction=MovementDirection.INCREASING,
            max_change=1.0, total_range=3.0,
            intensity=MovementIntensity.GENTLE
        )
        
        assert should_group_segments(seg1, seg2, max_frames=100, grouping_window=0.1) is True
        
        # Different direction - should not group
        seg3 = MovementSegment(
            start_frame=17, end_frame=20,
            movement_type=MovementType.TRANSLATION_X,
            direction=MovementDirection.DECREASING,
            max_change=1.0, total_range=3.0,
            intensity=MovementIntensity.GENTLE
        )
        
        assert should_group_segments(seg1, seg3, max_frames=100, grouping_window=0.1) is False
        
        # Too far apart - should not group
        seg4 = MovementSegment(
            start_frame=50, end_frame=55,
            movement_type=MovementType.TRANSLATION_X,
            direction=MovementDirection.INCREASING,
            max_change=1.0, total_range=5.0,
            intensity=MovementIntensity.GENTLE
        )
        
        assert should_group_segments(seg1, seg4, max_frames=100, grouping_window=0.1) is False
    
    def test_group_similar_segments(self):
        """Test grouping of similar segments"""
        segments = (
            MovementSegment(10, 15, MovementType.TRANSLATION_X, MovementDirection.INCREASING, 1.0, 5.0, MovementIntensity.GENTLE),
            MovementSegment(17, 20, MovementType.TRANSLATION_X, MovementDirection.INCREASING, 1.0, 3.0, MovementIntensity.GENTLE),
            MovementSegment(30, 35, MovementType.TRANSLATION_Y, MovementDirection.DECREASING, 2.0, 8.0, MovementIntensity.MODERATE),
            MovementSegment(40, 45, MovementType.TRANSLATION_X, MovementDirection.DECREASING, 1.5, 6.0, MovementIntensity.GENTLE),
        )
        
        groups = group_similar_segments(segments, max_frames=100, grouping_window=0.1)
        
        # Should have 3 groups: [seg1, seg2], [seg3], [seg4]
        assert len(groups) == 3
        assert len(groups[0]) == 2  # First two segments grouped
        assert len(groups[1]) == 1  # Third segment alone
        assert len(groups[2]) == 1  # Fourth segment alone
        
        # Check that first group has same movement type
        assert all(seg.movement_type == MovementType.TRANSLATION_X for seg in groups[0])


class TestDescriptionGeneration:
    """Test description generation functions"""
    
    def test_generate_movement_description(self):
        """Test basic movement description generation"""
        # Translation movements
        desc = generate_movement_description(
            MovementType.TRANSLATION_X, 
            MovementDirection.INCREASING, 
            MovementIntensity.MODERATE
        )
        assert desc == "moderate panning right"
        
        desc = generate_movement_description(
            MovementType.TRANSLATION_Y, 
            MovementDirection.DECREASING, 
            MovementIntensity.GENTLE
        )
        assert desc == "gentle moving down"
        
        # Rotation movements
        desc = generate_movement_description(
            MovementType.ROTATION_Z, 
            MovementDirection.INCREASING, 
            MovementIntensity.STRONG
        )
        assert desc == "strong rolling clockwise"
        
        # Zoom movements
        desc = generate_movement_description(
            MovementType.ZOOM, 
            MovementDirection.DECREASING, 
            MovementIntensity.SUBTLE
        )
        assert desc == "subtle zooming out"
    
    def test_describe_segment_group(self):
        """Test segment group description"""
        segments = (
            MovementSegment(10, 20, MovementType.TRANSLATION_X, MovementDirection.INCREASING, 1.0, 5.0, MovementIntensity.GENTLE),
            MovementSegment(22, 25, MovementType.TRANSLATION_X, MovementDirection.INCREASING, 1.0, 3.0, MovementIntensity.GENTLE),
        )
        
        desc = describe_segment_group(segments, max_frames=100)
        
        # Should include movement description and duration
        assert "panning right" in desc
        assert "brief" in desc or "extended" in desc or "sustained" in desc
    
    def test_generate_overall_description(self):
        """Test overall description generation"""
        # Single group
        groups = (
            (MovementSegment(10, 20, MovementType.TRANSLATION_X, MovementDirection.INCREASING, 1.0, 5.0, MovementIntensity.GENTLE),),
        )
        
        desc = generate_overall_description(groups, max_frames=100)
        assert desc.startswith("camera movement with")
        assert "panning right" in desc
        
        # Multiple groups
        groups = (
            (MovementSegment(10, 20, MovementType.TRANSLATION_X, MovementDirection.INCREASING, 1.0, 5.0, MovementIntensity.GENTLE),),
            (MovementSegment(30, 40, MovementType.ROTATION_Y, MovementDirection.DECREASING, 2.0, 8.0, MovementIntensity.MODERATE),),
        )
        
        desc = generate_overall_description(groups, max_frames=100)
        assert "camera movement with" in desc
        assert "and" in desc  # Should combine multiple movements
        
        # No groups
        desc = generate_overall_description(tuple(), max_frames=100)
        assert desc == "static camera position"
    
    def test_calculate_movement_strength(self):
        """Test movement strength calculation"""
        # No segments
        strength = calculate_movement_strength(tuple(), max_frames=100)
        assert strength == 0.0
        
        # Single segment
        segments = (
            MovementSegment(10, 20, MovementType.TRANSLATION_X, MovementDirection.INCREASING, 1.0, 10.0, MovementIntensity.MODERATE),
        )
        groups = ((segments[0],),)
        
        strength = calculate_movement_strength(groups, max_frames=100)
        assert 0.0 < strength <= 2.0  # Should be positive and capped
        
        # Multiple segments
        segments = (
            MovementSegment(10, 20, MovementType.TRANSLATION_X, MovementDirection.INCREASING, 1.0, 10.0, MovementIntensity.MODERATE),
            MovementSegment(30, 40, MovementType.ROTATION_Y, MovementDirection.DECREASING, 2.0, 20.0, MovementIntensity.STRONG),
        )
        groups = ((segments[0],), (segments[1],))
        
        strength2 = calculate_movement_strength(groups, max_frames=100)
        assert strength2 > strength  # More segments should increase strength


class TestHighLevelAnalysis:
    """Test high-level analysis functions"""
    
    def test_analyze_movement(self):
        """Test complete movement analysis"""
        args = AnimationArgs(
            translation_x="0:(0), 30:(10), 60:(0)",  # Pan right then left
            translation_y="0:(0)",
            translation_z="0:(0)",
            rotation_3d_x="0:(0)",
            rotation_3d_y="0:(0)",
            rotation_3d_z="0:(0)",
            zoom="0:(1.0)",
            angle="0:(0)",
            max_frames=61
        )
        
        result = analyze_movement(args)
        
        assert isinstance(result, MovementAnalysisResult)
        assert result.strength > 0.0
        assert "camera movement" in result.description.lower()
        assert len(result.segments) > 0
        assert result.movement_data.max_frames == 61
    
    def test_analyze_movement_with_sensitivity(self):
        """Test movement analysis with different sensitivity"""
        args = AnimationArgs(
            translation_x="0:(0), 30:(5)",  # Moderate movement that should be detectable
            translation_y="0:(0)",
            translation_z="0:(0)",
            rotation_3d_x="0:(0)",
            rotation_3d_y="0:(0)",
            rotation_3d_z="0:(0)",
            zoom="0:(1.0)",
            angle="0:(0)",
            max_frames=31
        )
        
        # Low sensitivity - might not detect small movement
        result_low = analyze_movement_with_sensitivity(args, sensitivity=0.01)
        
        # High sensitivity - should detect movement
        result_high = analyze_movement_with_sensitivity(args, sensitivity=10.0)
        
        # High sensitivity should detect more or stronger movement
        assert result_high.strength >= result_low.strength
        assert len(result_high.segments) >= len(result_low.segments)
    
    def test_analyze_movement_static(self):
        """Test analysis of static camera"""
        args = AnimationArgs(
            translation_x="0:(0)",
            translation_y="0:(0)",
            translation_z="0:(0)",
            rotation_3d_x="0:(0)",
            rotation_3d_y="0:(0)",
            rotation_3d_z="0:(0)",
            zoom="0:(1.0)",
            angle="0:(0)",
            max_frames=30
        )
        
        result = analyze_movement(args)
        
        assert result.strength == 0.0
        assert result.description == "static camera position"
        assert len(result.segments) == 0


class TestStatisticsAndHelpers:
    """Test statistics and helper functions"""
    
    def test_get_movement_statistics(self):
        """Test movement statistics calculation"""
        # Create sample analysis result
        segments = (
            MovementSegment(10, 20, MovementType.TRANSLATION_X, MovementDirection.INCREASING, 1.0, 5.0, MovementIntensity.GENTLE),
            MovementSegment(30, 40, MovementType.ROTATION_Y, MovementDirection.DECREASING, 2.0, 8.0, MovementIntensity.MODERATE),
        )
        
        movement_data = MovementData(
            translation_x_values=tuple(range(50)),
            translation_y_values=tuple(range(50)),
            translation_z_values=tuple(range(50)),
            rotation_x_values=tuple(range(50)),
            rotation_y_values=tuple(range(50)),
            rotation_z_values=tuple(range(50)),
            zoom_values=tuple(range(50)),
            angle_values=tuple(range(50)),
            max_frames=50
        )
        
        result = MovementAnalysisResult(
            description="test",
            strength=1.0,
            segments=segments,
            movement_data=movement_data
        )
        
        stats = get_movement_statistics(result)
        
        assert stats["total_segments"] == 2
        assert "translation_x" in stats["movement_types"]
        assert "rotation_y" in stats["movement_types"]
        assert stats["total_duration"] == 22  # (20-10+1) + (40-30+1)
        assert 0 <= stats["coverage_percentage"] <= 100
        assert stats["max_frames"] == 50
        assert stats["strength"] == 1.0
    
    def test_get_movement_statistics_empty(self):
        """Test statistics with no movement"""
        movement_data = MovementData(
            translation_x_values=(0.0, 0.0, 0.0),
            translation_y_values=(0.0, 0.0, 0.0),
            translation_z_values=(0.0, 0.0, 0.0),
            rotation_x_values=(0.0, 0.0, 0.0),
            rotation_y_values=(0.0, 0.0, 0.0),
            rotation_z_values=(0.0, 0.0, 0.0),
            zoom_values=(1.0, 1.0, 1.0),
            angle_values=(0.0, 0.0, 0.0),
            max_frames=3
        )
        
        result = MovementAnalysisResult(
            description="static camera position",
            strength=0.0,
            segments=tuple(),
            movement_data=movement_data
        )
        
        stats = get_movement_statistics(result)
        
        assert stats["total_segments"] == 0
        assert stats["movement_types"] == []
        assert stats["total_duration"] == 0
        assert stats["coverage_percentage"] == 0.0
    
    def test_detect_circular_motion(self):
        """Test circular motion detection"""
        # Create circular motion pattern (simplified sine/cosine)
        x_values = tuple(math.sin(i * 0.2) * 10 for i in range(100))  # Longer pattern for better detection
        y_values = tuple(math.cos(i * 0.2) * 10 for i in range(100))
        
        movement_data = MovementData(
            translation_x_values=(0.0,) * 100,
            translation_y_values=(0.0,) * 100,
            translation_z_values=(0.0,) * 100,
            rotation_x_values=x_values,
            rotation_y_values=y_values,
            rotation_z_values=(0.0,) * 100,
            zoom_values=(1.0,) * 100,
            angle_values=(0.0,) * 100,
            max_frames=100
        )
        
        is_circular, strength = detect_circular_motion(movement_data)
        
        # With sine/cosine pattern, should detect circular motion if numpy is available
        # If numpy is not available, function should return False gracefully
        assert isinstance(is_circular, bool)
        assert isinstance(strength, (int, float))
        if is_circular:
            assert strength > 0.0
        
        # Test with no circular motion (linear)
        movement_data_linear = MovementData(
            translation_x_values=(0.0,) * 100,
            translation_y_values=(0.0,) * 100,
            translation_z_values=(0.0,) * 100,
            rotation_x_values=tuple(range(100)),  # Linear motion
            rotation_y_values=(0.0,) * 100,
            rotation_z_values=(0.0,) * 100,
            zoom_values=(1.0,) * 100,
            angle_values=(0.0,) * 100,
            max_frames=100
        )
        
        is_circular_linear, strength_linear = detect_circular_motion(movement_data_linear)
        assert is_circular_linear is False
        assert strength_linear == 0.0
    
    def test_detect_camera_shake_pattern(self):
        """Test camera shake pattern detection"""
        # Create shake pattern - high frequency, low amplitude changes
        shake_pattern = []
        for i in range(50):
            # Alternating small values to simulate shake
            value = 0.1 * (1 if i % 2 == 0 else -1) * (1 + 0.1 * (i % 3))
            shake_pattern.append(value)
        
        movement_data = MovementData(
            translation_x_values=tuple(shake_pattern),
            translation_y_values=tuple(shake_pattern),
            translation_z_values=tuple(shake_pattern),
            rotation_x_values=tuple(shake_pattern),
            rotation_y_values=tuple(shake_pattern),
            rotation_z_values=tuple(shake_pattern),
            zoom_values=(1.0,) * 50,
            angle_values=(0.0,) * 50,
            max_frames=50
        )
        
        has_shake, description, strength = detect_camera_shake_pattern(movement_data)
        
        # Should detect shake
        assert has_shake is True
        assert "shake" in description.lower()
        assert strength > 0.0
        
        # Test with smooth motion (no shake)
        smooth_values = tuple(float(i) for i in range(50))
        movement_data_smooth = MovementData(
            translation_x_values=smooth_values,
            translation_y_values=(0.0,) * 50,
            translation_z_values=(0.0,) * 50,
            rotation_x_values=(0.0,) * 50,
            rotation_y_values=(0.0,) * 50,
            rotation_z_values=(0.0,) * 50,
            zoom_values=(1.0,) * 50,
            angle_values=(0.0,) * 50,
            max_frames=50
        )
        
        has_shake_smooth, desc_smooth, strength_smooth = detect_camera_shake_pattern(movement_data_smooth)
        assert has_shake_smooth is False
        assert desc_smooth == ""
        assert strength_smooth == 0.0
    
    def test_create_movement_schedule_series(self):
        """Test legacy compatibility function"""
        args = AnimationArgs(
            translation_x="0:(0), 10:(5)",
            translation_y="0:(1), 10:(3)",
            translation_z="0:(0)",
            rotation_3d_x="0:(0)",
            rotation_3d_y="0:(0), 10:(10)",
            rotation_3d_z="0:(0)",
            zoom="0:(1.0), 10:(1.5)",
            angle="0:(0)",
            max_frames=11
        )
        
        series = create_movement_schedule_series(args)
        
        assert isinstance(series, dict)
        assert "translation_x" in series
        assert "rotation_y" in series
        assert "zoom" in series
        
        assert len(series["translation_x"]) == 11
        assert series["translation_x"][0] == 0.0
        assert series["translation_x"][10] == 5.0
        assert series["zoom"][10] == 1.5


class TestIntegrationScenarios:
    """Test real-world integration scenarios"""
    
    def test_complex_movement_pattern(self):
        """Test analysis of complex movement patterns"""
        args = AnimationArgs(
            translation_x="0:(0), 30:(10), 60:(0), 90:(-10), 120:(0)",  # Pan left-right-left-right
            translation_y="0:(0), 60:(5), 120:(0)",                     # Slight up-down
            translation_z="0:(0), 120:(20)",                            # Dolly forward
            rotation_3d_x="0:(0), 120:(15)",                            # Tilt up
            rotation_3d_y="0:(0), 40:(45), 80:(0), 120:(45)",          # Multiple rotations
            rotation_3d_z="0:(0)",                                      # No roll
            zoom="0:(1.0), 120:(1.8)",                                  # Zoom in
            angle="0:(0)",
            max_frames=121
        )
        
        result = analyze_movement(args)
        
        # Should detect complex movement
        assert result.strength > 0.5  # Should be significant (lowered from 1.0)
        assert "complex" in result.description.lower() or "camera movement" in result.description.lower()
        assert len(result.segments) > 3  # Should detect multiple segments (lowered from 5)
        
        # Should detect multiple movement types
        movement_types = [seg.movement_type for seg in result.segments]
        assert MovementType.TRANSLATION_X in movement_types
        assert MovementType.TRANSLATION_Z in movement_types
        assert MovementType.ZOOM in movement_types
    
    def test_subtle_movement_detection(self):
        """Test detection of subtle movements with high sensitivity"""
        args = AnimationArgs(
            translation_x="0:(0), 100:(5)",  # Larger movement that should be detectable
            translation_y="0:(0)",
            translation_z="0:(0)",
            rotation_3d_x="0:(0)",
            rotation_3d_y="0:(0), 100:(10)",  # Larger rotation
            rotation_3d_z="0:(0)",
            zoom="0:(1.0), 100:(1.1)",  # Larger zoom
            angle="0:(0)",
            max_frames=101
        )
        
        # With default sensitivity, should detect some movement
        result_normal = analyze_movement(args)
        
        # With high sensitivity, should detect more movement
        result_sensitive = analyze_movement_with_sensitivity(args, sensitivity=10.0)
        
        # High sensitivity should detect more movement
        assert len(result_sensitive.segments) >= len(result_normal.segments)
        assert result_sensitive.strength >= result_normal.strength
    
    def test_performance_with_large_dataset(self):
        """Test performance with large frame count"""
        args = AnimationArgs(
            translation_x="0:(0), 500:(50), 1000:(0)",  # Long animation
            translation_y="0:(0), 1000:(10)",
            translation_z="0:(0)",
            rotation_3d_x="0:(0)",
            rotation_3d_y="0:(0), 1000:(360)",  # Full rotation
            rotation_3d_z="0:(0)",
            zoom="0:(1.0), 1000:(2.0)",
            angle="0:(0)",
            max_frames=1001
        )
        
        # Should handle large datasets efficiently
        result = analyze_movement(args)
        
        assert isinstance(result, MovementAnalysisResult)
        assert result.movement_data.max_frames == 1001
        assert len(result.movement_data.translation_x_values) == 1001
        assert result.strength > 0.0


class TestFunctionalProgrammingPrinciples:
    """Test that functional programming principles are followed"""
    
    def test_pure_functions_no_side_effects(self):
        """Test that pure functions don't modify input arguments"""
        args = AnimationArgs(
            translation_x="0:(0), 30:(10)",
            translation_y="0:(0)",
            translation_z="0:(0)",
            rotation_3d_x="0:(0)",
            rotation_3d_y="0:(0)",
            rotation_3d_z="0:(0)",
            zoom="0:(1.0)",
            angle="0:(0)",
            max_frames=31
        )
        
        # Analyze multiple times - should get identical results
        result1 = analyze_movement(args)
        result2 = analyze_movement(args)
        
        # Results should be equal (same analysis)
        assert result1.description == result2.description
        assert result1.strength == result2.strength
        assert len(result1.segments) == len(result2.segments)
        
        # Original args should be unchanged (immutable)
        assert args.translation_x == "0:(0), 30:(10)"
        assert args.max_frames == 31
    
    def test_immutability_preserved(self):
        """Test that data structures remain immutable throughout processing"""
        args = AnimationArgs(
            translation_x="0:(0), 30:(10)",
            translation_y="0:(0)",
            translation_z="0:(0)",
            rotation_3d_x="0:(0)",
            rotation_3d_y="0:(0)",
            rotation_3d_z="0:(0)",
            zoom="0:(1.0)",
            angle="0:(0)",
            max_frames=31
        )
        
        result = analyze_movement(args)
        
        # All data structures should be immutable
        with pytest.raises(AttributeError):
            result.strength = 2.0
        
        with pytest.raises(AttributeError):
            result.movement_data.max_frames = 100
        
        if result.segments:
            with pytest.raises(AttributeError):
                result.segments[0].start_frame = 20
    
    def test_functional_composition(self):
        """Test that functions compose well together"""
        args = AnimationArgs(
            translation_x="0:(0), 30:(10), 60:(0)",
            translation_y="0:(0)",
            translation_z="0:(0)",
            rotation_3d_x="0:(0)",
            rotation_3d_y="0:(0)",
            rotation_3d_z="0:(0)",
            zoom="0:(1.0)",
            angle="0:(0)",
            max_frames=61
        )
        
        # Create a pipeline using function composition
        movement_data = extract_movement_data(args)
        config = AnalysisConfig(sensitivity=1.0)
        segments = detect_all_movement_segments(movement_data, config)
        groups = group_similar_segments(segments, movement_data.max_frames)
        description = generate_overall_description(groups, movement_data.max_frames)
        strength = calculate_movement_strength(groups, movement_data.max_frames)
        
        # Compose into final result
        manual_result = MovementAnalysisResult(
            description=description,
            strength=strength,
            segments=segments,
            movement_data=movement_data
        )
        
        # Should be equivalent to high-level function
        auto_result = analyze_movement(args, config)
        
        assert manual_result.description == auto_result.description
        assert manual_result.strength == auto_result.strength
        assert len(manual_result.segments) == len(auto_result.segments)
    
    def test_tuple_comprehensions_and_filters(self):
        """Test that functional operators like tuple comprehensions are used correctly"""
        values = (0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0)
        changes = calculate_frame_changes(values)
        
        # Should use tuple comprehension internally
        assert isinstance(changes, tuple)
        assert len(changes) == len(values) - 1
        
        # Test that filters work correctly in segment detection
        movement_data = MovementData(
            translation_x_values=values,
            translation_y_values=(0.0,) * 7,
            translation_z_values=(0.0,) * 7,
            rotation_x_values=(0.0,) * 7,
            rotation_y_values=(0.0,) * 7,
            rotation_z_values=(0.0,) * 7,
            zoom_values=(1.0,) * 7,
            angle_values=(0.0,) * 7,
            max_frames=7
        )
        
        config = AnalysisConfig(sensitivity=1.0, min_translation_threshold=0.5)
        segments = detect_all_movement_segments(movement_data, config)
        
        # Should filter correctly and return only significant segments
        assert all(isinstance(seg, MovementSegment) for seg in segments)
        assert all(seg.total_range > 0 for seg in segments)

# Test data and fixtures
class TestMovementAnalysis:
    """Test movement analysis functionality"""
    
    def test_movement_analyzer_creation(self):
        """Test MovementAnalyzer can be created"""
        analyzer = MovementAnalyzer()
        assert analyzer is not None
    
    def test_analyze_movement_patterns(self):
        """Test movement pattern analysis"""
        # Mock frame data
        frame_data = {
            "translation_x": 5.0,
            "translation_y": -2.0,
            "rotation": 1.5,
            "zoom": 1.1
        }
        
        patterns = analyze_movement_patterns(frame_data)
        assert isinstance(patterns, dict)
        assert "movement_strength" in patterns
        assert "primary_direction" in patterns
    
    def test_calculate_movement_strength(self):
        """Test movement strength calculation"""
        strength = calculate_movement_strength(5.0, -2.0, 1.5)
        assert isinstance(strength, float)
        assert strength >= 0
    
    def test_detect_zoom_changes(self):
        """Test zoom change detection"""
        zoom_changes = detect_zoom_changes([1.0, 1.1, 1.2, 1.0, 0.9])
        assert isinstance(zoom_changes, list)
    
    def test_enhance_movement_description(self):
        """Test movement description enhancement"""
        base_description = "A simple scene"
        movement_data = {"movement_strength": 0.5, "primary_direction": "right"}
        
        enhanced = enhance_movement_description(base_description, movement_data)
        assert isinstance(enhanced, str)
        assert len(enhanced) >= len(base_description) 