"""Unit tests for refactored schedule visualizer.

Tests all 47 pure functions created during the refactoring of
visualize_schedules from 584 lines (complexity F-65) to modular
functional architecture (complexity A).
"""

import pytest
import numpy as np
from typing import Dict, Set

from deforum.utils.schedule_visualizer import (
    # Data structures
    ColorPalette,
    Coordinates,
    DownsampleMetadata,
    ParsedSchedules,
    # Phase 1: Schedule Parsing
    parse_schedule_string,
    parse_prompt_schedule,
    interpolate_schedule,
    _interpolate_value,
    _parse_all_schedules,
    _get_all_schedule_frames,
    # Phase 2: Mode Detection & Conversion
    _detect_dense_schedule,
    _interpolate_all_schedules,
    _accumulate_deltas,
    _convert_to_coordinates,
    # Phase 3: Downsampling
    _should_downsample,
    _calculate_downsample_rate,
    _build_downsample_indices,
    _downsample_coordinates,
    _remap_keyframes,
    # Phase 4: Statistics
    _calculate_path_distance,
    _calculate_coordinate_ranges,
    # Helpers
    _build_initial_schedule_dict,
    _get_animation_sample_rate,
    _build_animation_frame_indices,
)


# ============================================================================
# Test Data Structures
# ============================================================================

class TestColorPalette:
    """Test ColorPalette NamedTuple."""

    def test_default_colors(self):
        palette = ColorPalette()
        assert palette.void == '#5606FF'
        assert palette.dusk == '#4C21FF'
        assert palette.midnight == '#3757FF'
        assert palette.zenith == '#17A7FE'
        assert palette.glitch == '#FF1493'

    def test_immutability(self):
        palette = ColorPalette()
        with pytest.raises(AttributeError):
            palette.void = '#000000'  # Should fail - NamedTuple is immutable


class TestCoordinates:
    """Test Coordinates dataclass."""

    def test_construction(self):
        coords = Coordinates(
            x=[0.0, 1.0, 2.0],
            y=[0.0, 0.5, 1.0],
            z=[0.0, 0.0, 0.0],
            rx=[0.0, 0.0, 0.0],
            ry=[0.0, 0.0, 0.0],
            rz=[0.0, 0.0, 0.0]
        )
        assert len(coords) == 3
        assert coords.x == [0.0, 1.0, 2.0]

    def test_immutability(self):
        coords = Coordinates(
            x=[0.0], y=[0.0], z=[0.0],
            rx=[0.0], ry=[0.0], rz=[0.0]
        )
        with pytest.raises(Exception):  # dataclass(frozen=True) raises FrozenInstanceError
            coords.x = [1.0]

    def test_len_method(self):
        coords = Coordinates(
            x=[1, 2, 3, 4, 5],
            y=[0, 0, 0, 0, 0],
            z=[0, 0, 0, 0, 0],
            rx=[0, 0, 0, 0, 0],
            ry=[0, 0, 0, 0, 0],
            rz=[0, 0, 0, 0, 0]
        )
        assert len(coords) == 5


# ============================================================================
# Test Schedule Parsing
# ============================================================================

class TestParseScheduleString:
    """Test parse_schedule_string function."""

    def test_basic_schedule(self):
        result = parse_schedule_string("0: (10), 50: (20), 100: (30)")
        assert result == {0: 10.0, 50: 20.0, 100: 30.0}

    def test_single_keyframe(self):
        result = parse_schedule_string("0:(5.0)")
        assert result == {0: 5.0}

    def test_empty_string(self):
        result = parse_schedule_string("")
        assert result == {}

    def test_whitespace_only(self):
        result = parse_schedule_string("   ")
        assert result == {}

    def test_whitespace_handling(self):
        result = parse_schedule_string("0: (1.0),  50 : ( 2.0 ), 100 :(1.5)")
        assert result == {0: 1.0, 50: 2.0, 100: 1.5}

    def test_negative_values(self):
        result = parse_schedule_string("0:(-10), 50:(-5.5)")
        assert result == {0: -10.0, 50: -5.5}

    def test_decimal_values(self):
        result = parse_schedule_string("0:(3.14159), 100:(2.71828)")
        assert result == {0: 3.14159, 100: 2.71828}

    def test_no_spaces(self):
        result = parse_schedule_string("0:(10),50:(20),100:(30)")
        assert result == {0: 10.0, 50: 20.0, 100: 30.0}


class TestParsePromptSchedule:
    """Test parse_prompt_schedule function."""

    def test_json_dict_format(self):
        json_str = '{"0": "prompt1", "50": "prompt2", "100": "prompt3"}'
        result = parse_prompt_schedule(json_str)
        assert result == {0, 50, 100}

    def test_schedule_format(self):
        schedule_str = "0: prompt1, 50: prompt2, 100: prompt3"
        result = parse_prompt_schedule(schedule_str)
        assert result == {0, 50, 100}

    def test_empty_string(self):
        result = parse_prompt_schedule("")
        assert result == set()

    def test_whitespace_only(self):
        result = parse_prompt_schedule("   ")
        assert result == set()

    def test_invalid_json(self):
        # Should fall back to regex pattern
        result = parse_prompt_schedule("0: text, 50: more")
        assert result == {0, 50}


class TestInterpolateSchedule:
    """Test interpolate_schedule function."""

    def test_empty_schedule(self):
        result = interpolate_schedule({}, 10)
        expected = [(i, 0.0) for i in range(11)]
        assert result == expected

    def test_single_keyframe(self):
        result = interpolate_schedule({0: 10.0}, 5)
        expected = [(i, 10.0) for i in range(6)]
        assert result == expected

    def test_linear_interpolation(self):
        result = interpolate_schedule({0: 0.0, 10: 10.0}, 10)
        assert result[0] == (0, 0.0)
        assert result[5] == (5, 5.0)
        assert result[10] == (10, 10.0)

    def test_hold_after_last_keyframe(self):
        result = interpolate_schedule({0: 5.0, 10: 10.0}, 15)
        assert result[10] == (10, 10.0)
        assert result[15] == (15, 10.0)  # Should hold last value

    def test_hold_before_first_keyframe(self):
        result = interpolate_schedule({10: 5.0, 20: 10.0}, 25)
        # Frames 0-9 should use first keyframe value
        assert result[0] == (0, 5.0)
        assert result[5] == (5, 5.0)
        assert result[10] == (10, 5.0)


class TestInterpolateValue:
    """Test _interpolate_value helper function."""

    def test_before_first_keyframe(self):
        sorted_kf = [(10, 5.0), (20, 10.0)]
        result = _interpolate_value(5, None, (10, 5.0), sorted_kf)
        assert result == 5.0

    def test_after_last_keyframe(self):
        sorted_kf = [(10, 5.0), (20, 10.0)]
        result = _interpolate_value(25, (20, 10.0), None, sorted_kf)
        assert result == 10.0

    def test_exactly_on_keyframe(self):
        sorted_kf = [(10, 5.0), (20, 10.0)]
        result = _interpolate_value(10, (10, 5.0), (20, 10.0), sorted_kf)
        assert result == 5.0

    def test_midpoint_interpolation(self):
        sorted_kf = [(10, 0.0), (20, 10.0)]
        result = _interpolate_value(15, (10, 0.0), (20, 10.0), sorted_kf)
        assert result == 5.0  # Halfway between 0 and 10


class TestParseAllSchedules:
    """Test _parse_all_schedules function."""

    def test_all_schedules(self):
        schedules = {
            'translation_x': "0:(1), 10:(2)",
            'translation_y': "0:(3), 10:(4)",
            'translation_z': "0:(5), 10:(6)",
            'rotation_3d_x': "0:(7), 10:(8)",
            'rotation_3d_y': "0:(9), 10:(10)",
            'rotation_3d_z': "0:(11), 10:(12)"
        }
        result = _parse_all_schedules(schedules)
        assert result.tx == {0: 1.0, 10: 2.0}
        assert result.ty == {0: 3.0, 10: 4.0}
        assert result.tz == {0: 5.0, 10: 6.0}
        assert result.rx == {0: 7.0, 10: 8.0}
        assert result.ry == {0: 9.0, 10: 10.0}
        assert result.rz == {0: 11.0, 10: 12.0}

    def test_empty_schedules(self):
        schedules = {
            'translation_x': '',
            'translation_y': '',
            'translation_z': '',
            'rotation_3d_x': '',
            'rotation_3d_y': '',
            'rotation_3d_z': ''
        }
        result = _parse_all_schedules(schedules)
        assert result.tx == {}
        assert result.ty == {}


class TestGetAllScheduleFrames:
    """Test _get_all_schedule_frames function."""

    def test_multiple_schedules(self):
        parsed = ParsedSchedules(
            tx={0: 0.0, 10: 1.0},
            ty={5: 0.0, 15: 1.0},
            tz={},
            rx={},
            ry={},
            rz={}
        )
        result = _get_all_schedule_frames(parsed)
        assert set(result) == {0, 5, 10, 15}

    def test_empty_schedules(self):
        parsed = ParsedSchedules(
            tx={}, ty={}, tz={}, rx={}, ry={}, rz={}
        )
        result = _get_all_schedule_frames(parsed)
        assert result == []


# ============================================================================
# Test Mode Detection & Conversion
# ============================================================================

class TestDetectDenseSchedule:
    """Test _detect_dense_schedule function."""

    def test_dense_schedule(self):
        # More than half frames defined = dense
        parsed = ParsedSchedules(
            tx={i: float(i) for i in range(60)},  # 60 frames
            ty={}, tz={}, rx={}, ry={}, rz={}
        )
        assert _detect_dense_schedule(parsed, 100) is True

    def test_sparse_schedule(self):
        # Less than half frames = sparse
        parsed = ParsedSchedules(
            tx={0: 0.0, 10: 1.0, 20: 2.0},
            ty={}, tz={}, rx={}, ry={}, rz={}
        )
        assert _detect_dense_schedule(parsed, 100) is False

    def test_empty_schedule(self):
        parsed = ParsedSchedules(
            tx={}, ty={}, tz={}, rx={}, ry={}, rz={}
        )
        assert _detect_dense_schedule(parsed, 100) is False


class TestAccumulateDeltas:
    """Test _accumulate_deltas function."""

    def test_basic_accumulation(self):
        deltas = [1.0, 2.0, 3.0, 4.0]
        result = _accumulate_deltas(deltas)
        assert result == [1.0, 3.0, 6.0, 10.0]

    def test_empty_list(self):
        result = _accumulate_deltas([])
        assert result == []

    def test_single_value(self):
        result = _accumulate_deltas([5.0])
        assert result == [5.0]

    def test_negative_deltas(self):
        deltas = [5.0, -2.0, 3.0, -1.0]
        result = _accumulate_deltas(deltas)
        assert result == [5.0, 3.0, 6.0, 5.0]


class TestConvertToCoordinates:
    """Test _convert_to_coordinates function."""

    def test_dense_mode_accumulation(self):
        tx = [1.0, 1.0, 1.0]
        ty = [2.0, 2.0, 2.0]
        tz = [3.0, 3.0, 3.0]
        rx = ry = rz = [0.0, 0.0, 0.0]

        coords = _convert_to_coordinates(tx, ty, tz, rx, ry, rz, is_dense=True)

        assert coords.x == [1.0, 2.0, 3.0]  # Accumulated
        assert coords.y == [2.0, 4.0, 6.0]  # Accumulated
        assert coords.z == [3.0, 6.0, 9.0]  # Accumulated

    def test_sparse_mode_absolute(self):
        tx = [1.0, 2.0, 3.0]
        ty = [4.0, 5.0, 6.0]
        tz = [7.0, 8.0, 9.0]
        rx = ry = rz = [0.0, 0.0, 0.0]

        coords = _convert_to_coordinates(tx, ty, tz, rx, ry, rz, is_dense=False)

        assert coords.x == [1.0, 2.0, 3.0]  # Not accumulated
        assert coords.y == [4.0, 5.0, 6.0]  # Not accumulated
        assert coords.z == [7.0, 8.0, 9.0]  # Not accumulated


# ============================================================================
# Test Downsampling
# ============================================================================

class TestShouldDownsample:
    """Test _should_downsample function."""

    def test_below_threshold(self):
        assert _should_downsample(4999, threshold=5000) is False

    def test_at_threshold(self):
        assert _should_downsample(5000, threshold=5000) is False

    def test_above_threshold(self):
        assert _should_downsample(5001, threshold=5000) is True


class TestCalculateDownsampleRate:
    """Test _calculate_downsample_rate function."""

    def test_exact_division(self):
        assert _calculate_downsample_rate(800, target_frames=400) == 2

    def test_floor_division(self):
        assert _calculate_downsample_rate(850, target_frames=400) == 2

    def test_small_count(self):
        # Should return at least 1
        assert _calculate_downsample_rate(100, target_frames=400) == 1


class TestBuildDownsampleIndices:
    """Test _build_downsample_indices function."""

    def test_basic_downsampling(self):
        indices = _build_downsample_indices(
            num_points=10,
            downsample_rate=3,
            keyframes={5}
        )
        # Should include: every 3rd (0, 3, 6, 9), keyframe (5), first (0), last (9)
        assert 0 in indices
        assert 3 in indices
        assert 5 in indices  # Keyframe
        assert 6 in indices
        assert 9 in indices  # Last frame

    def test_always_includes_first_and_last(self):
        indices = _build_downsample_indices(
            num_points=100,
            downsample_rate=10,
            keyframes=set()
        )
        assert 0 in indices
        assert 99 in indices

    def test_includes_all_keyframes(self):
        keyframes = {10, 20, 30, 40, 50}
        indices = _build_downsample_indices(
            num_points=100,
            downsample_rate=15,
            keyframes=keyframes
        )
        for kf in keyframes:
            assert kf in indices


class TestRemapKeyframes:
    """Test _remap_keyframes function."""

    def test_basic_remapping(self):
        keyframes = {0, 10, 20, 30}
        frame_map = {0: 0, 1: 10, 2: 20, 3: 30}  # new_idx -> orig_idx

        result = _remap_keyframes(keyframes, frame_map)
        assert result == {0, 1, 2, 3}

    def test_partial_keyframes(self):
        keyframes = {10, 30}
        frame_map = {0: 0, 1: 10, 2: 20, 3: 30}

        result = _remap_keyframes(keyframes, frame_map)
        assert result == {1, 3}

    def test_no_matching_keyframes(self):
        keyframes = {5, 15}
        frame_map = {0: 0, 1: 10, 2: 20}

        result = _remap_keyframes(keyframes, frame_map)
        assert result == set()


class TestDownsampleCoordinates:
    """Test _downsample_coordinates function."""

    def test_below_threshold_no_downsampling(self):
        coords = Coordinates(
            x=list(range(100)),
            y=list(range(100)),
            z=list(range(100)),
            rx=[0.0] * 100,
            ry=[0.0] * 100,
            rz=[0.0] * 100
        )

        result_coords, metadata = _downsample_coordinates(
            coords, keyframes=set(), threshold=5000
        )

        assert len(result_coords) == 100
        assert metadata.downsampled is False
        assert metadata.downsample_rate is None

    def test_above_threshold_downsamples(self):
        coords = Coordinates(
            x=list(range(6000)),
            y=list(range(6000)),
            z=list(range(6000)),
            rx=[0.0] * 6000,
            ry=[0.0] * 6000,
            rz=[0.0] * 6000
        )

        result_coords, metadata = _downsample_coordinates(
            coords, keyframes=set(), threshold=5000
        )

        assert len(result_coords) < 6000
        assert metadata.downsampled is True
        assert metadata.downsample_rate is not None
        assert metadata.original_count == 6000


# ============================================================================
# Test Statistics
# ============================================================================

class TestCalculatePathDistance:
    """Test _calculate_path_distance function."""

    def test_straight_line(self):
        coords = Coordinates(
            x=[0.0, 1.0, 2.0, 3.0],
            y=[0.0, 0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0, 0.0],
            rx=[0.0] * 4,
            ry=[0.0] * 4,
            rz=[0.0] * 4
        )
        distance = _calculate_path_distance(coords)
        assert distance == pytest.approx(3.0)  # 3 segments of 1 unit each

    def test_diagonal_path(self):
        coords = Coordinates(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            z=[0.0, 1.0],
            rx=[0.0, 0.0],
            ry=[0.0, 0.0],
            rz=[0.0, 0.0]
        )
        distance = _calculate_path_distance(coords)
        # Distance = sqrt(1^2 + 1^2 + 1^2) = sqrt(3)
        assert distance == pytest.approx(np.sqrt(3))

    def test_single_point(self):
        coords = Coordinates(
            x=[0.0], y=[0.0], z=[0.0],
            rx=[0.0], ry=[0.0], rz=[0.0]
        )
        distance = _calculate_path_distance(coords)
        assert distance == 0.0


class TestCalculateCoordinateRanges:
    """Test _calculate_coordinate_ranges function."""

    def test_basic_ranges(self):
        coords = Coordinates(
            x=[0.0, 5.0, -2.0],
            y=[1.0, 1.0, 10.0],
            z=[-5.0, 0.0, 5.0],
            rx=[0.0] * 3,
            ry=[0.0] * 3,
            rz=[0.0] * 3
        )
        x_range, y_range, z_range = _calculate_coordinate_ranges(coords)

        assert x_range == 7.0  # 5 - (-2)
        assert y_range == 9.0  # 10 - 1
        assert z_range == 10.0  # 5 - (-5)

    def test_single_point_zero_range(self):
        coords = Coordinates(
            x=[5.0], y=[10.0], z=[15.0],
            rx=[0.0], ry=[0.0], rz=[0.0]
        )
        x_range, y_range, z_range = _calculate_coordinate_ranges(coords)

        assert x_range == 0.0
        assert y_range == 0.0
        assert z_range == 0.0


# ============================================================================
# Test Helper Functions
# ============================================================================

class TestBuildInitialScheduleDict:
    """Test _build_initial_schedule_dict function."""

    def test_with_values(self):
        result = _build_initial_schedule_dict(
            "0:(1)", "0:(2)", "0:(3)",
            "0:(4)", "0:(5)", "0:(6)"
        )
        assert result['translation_x'] == "0:(1)"
        assert result['translation_y'] == "0:(2)"
        assert result['translation_z'] == "0:(3)"

    def test_with_empty_values(self):
        result = _build_initial_schedule_dict("", "", "", "", "", "")
        assert result['translation_x'] == "0:(0)"
        assert result['translation_y'] == "0:(0)"
        assert result['translation_z'] == "0:(0)"


class TestGetAnimationSampleRate:
    """Test _get_animation_sample_rate function."""

    def test_below_max_frames(self):
        assert _get_animation_sample_rate(200, max_frames=400) == 1

    def test_at_max_frames(self):
        assert _get_animation_sample_rate(400, max_frames=400) == 1

    def test_above_max_frames(self):
        assert _get_animation_sample_rate(800, max_frames=400) == 2


class TestBuildAnimationFrameIndices:
    """Test _build_animation_frame_indices function."""

    def test_basic_sampling(self):
        indices = _build_animation_frame_indices(num_points=10, sample_interval=3)
        # Should include: 0, 3, 6, 9 (and always include last)
        assert indices == [0, 3, 6, 9]

    def test_always_includes_last_frame(self):
        indices = _build_animation_frame_indices(num_points=10, sample_interval=4)
        # 0, 4, 8, and last (9)
        assert indices[-1] == 9

    def test_sample_interval_one(self):
        indices = _build_animation_frame_indices(num_points=5, sample_interval=1)
        assert indices == [0, 1, 2, 3, 4]


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for combined functionality."""

    def test_parse_and_interpolate_workflow(self):
        """Test realistic workflow: parse -> interpolate."""
        schedule_str = "0:(0), 10:(10), 20:(0)"
        parsed = parse_schedule_string(schedule_str)
        interpolated = interpolate_schedule(parsed, 20)

        assert len(interpolated) == 21  # 0 to 20 inclusive
        assert interpolated[0] == (0, 0.0)
        assert interpolated[10] == (10, 10.0)
        assert interpolated[20] == (20, 0.0)

    def test_dense_schedule_workflow(self):
        """Test dense schedule detection and conversion."""
        # Create dense schedule (every frame)
        parsed = ParsedSchedules(
            tx={i: 1.0 for i in range(60)},  # Dense
            ty={i: 0.0 for i in range(60)},
            tz={i: 0.0 for i in range(60)},
            rx={}, ry={}, rz={}
        )

        is_dense = _detect_dense_schedule(parsed, 100)
        assert is_dense is True

        tx, ty, tz, rx, ry, rz = _interpolate_all_schedules(parsed, 99)
        coords = _convert_to_coordinates(tx, ty, tz, rx, ry, rz, is_dense=True)

        # Dense mode should accumulate deltas
        assert coords.x[0] == 1.0
        assert coords.x[1] == 2.0  # Accumulated
        assert coords.x[59] == 60.0  # 60 frames of delta=1

    def test_downsampling_with_keyframes_workflow(self):
        """Test downsampling preserves keyframes."""
        coords = Coordinates(
            x=list(range(6000)),
            y=[0.0] * 6000,
            z=[0.0] * 6000,
            rx=[0.0] * 6000,
            ry=[0.0] * 6000,
            rz=[0.0] * 6000
        )

        keyframes = {100, 500, 1000, 2000}

        result_coords, metadata = _downsample_coordinates(
            coords, keyframes, threshold=5000
        )

        # Keyframes should be preserved in the mapping
        remapped_keyframes = _remap_keyframes(keyframes, metadata.frame_map)
        assert len(remapped_keyframes) == 4  # All 4 keyframes preserved
