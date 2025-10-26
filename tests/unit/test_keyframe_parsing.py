"""Unit tests for deforum.utils.parsing.keyframes module.

Tests pure functions for keyframe analysis and type assignment.
All functions here are side-effect free and easily testable.
"""

from deforum.utils.parsing.keyframes import (
    extract_frame_numbers,
    calculate_distance_threshold,
    suggest_keyframe_type,
    build_keyframe_type_schedule,
    format_keyframe_schedule,
    auto_assign_keyframe_types,
    THRESHOLD_RATIO,
    DEFAULT_TYPE,
    SHORT_SECTION_TYPE,
)


# ============================================================================
# Frame Number Extraction
# ============================================================================


class TestExtractFrameNumbers:
    """Test extraction of frame numbers from prompts."""

    def test_simple_numeric_keys(self):
        """Test extraction from simple numeric keys."""
        prompts = {"0": "start", "60": "middle", "120": "end"}
        frames = extract_frame_numbers(prompts)

        assert frames == [0, 60, 120]

    def test_sorts_frame_numbers(self):
        """Test that frame numbers are sorted."""
        prompts = {"120": "end", "0": "start", "60": "middle"}
        frames = extract_frame_numbers(prompts)

        assert frames == [0, 60, 120]

    def test_filters_non_numeric_keys(self):
        """Test that non-numeric keys are filtered out."""
        prompts = {"0": "start", "metadata": "test", "60": "middle", "max_f-2": "almost end"}
        frames = extract_frame_numbers(prompts)

        assert frames == [0, 60]

    def test_empty_prompts(self):
        """Test empty prompts dictionary."""
        frames = extract_frame_numbers({})

        assert frames == []

    def test_single_frame(self):
        """Test single frame."""
        prompts = {"100": "solo"}
        frames = extract_frame_numbers(prompts)

        assert frames == [100]

    def test_handles_string_numbers(self):
        """Test that string numeric keys work."""
        prompts = {"0": "a", "10": "b", "20": "c"}
        frames = extract_frame_numbers(prompts)

        assert frames == [0, 10, 20]


# ============================================================================
# Distance Threshold Calculation
# ============================================================================


class TestCalculateDistanceThreshold:
    """Test distance threshold calculation."""

    def test_standard_chunk_size(self):
        """Test 80% of 100 = 80."""
        threshold = calculate_distance_threshold(100)

        assert threshold == 80

    def test_small_chunk_size(self):
        """Test 80% of 10 = 8."""
        threshold = calculate_distance_threshold(10)

        assert threshold == 8

    def test_large_chunk_size(self):
        """Test 80% of 1000 = 800."""
        threshold = calculate_distance_threshold(1000)

        assert threshold == 800

    def test_fractional_result_truncated(self):
        """Test that result is integer (truncated)."""
        threshold = calculate_distance_threshold(17)

        # 17 * 0.8 = 13.6, int() truncates to 13
        assert threshold == 13

    def test_uses_correct_ratio(self):
        """Test that THRESHOLD_RATIO (0.8) is used."""
        threshold = calculate_distance_threshold(50)

        assert threshold == int(50 * THRESHOLD_RATIO)


# ============================================================================
# Keyframe Type Suggestion
# ============================================================================


class TestSuggestKeyframeType:
    """Test keyframe type suggestion based on distance."""

    def test_short_distance_suggests_flf2v(self):
        """Test distance < threshold suggests flf2v."""
        kf_type = suggest_keyframe_type(40, 80)

        assert kf_type == SHORT_SECTION_TYPE

    def test_at_threshold_suggests_flf2v(self):
        """Test distance == threshold suggests flf2v."""
        kf_type = suggest_keyframe_type(80, 80)

        assert kf_type == SHORT_SECTION_TYPE

    def test_long_distance_suggests_tween(self):
        """Test distance > threshold suggests tween."""
        kf_type = suggest_keyframe_type(90, 80)

        assert kf_type == DEFAULT_TYPE

    def test_zero_distance_suggests_flf2v(self):
        """Test zero distance suggests flf2v."""
        kf_type = suggest_keyframe_type(0, 80)

        assert kf_type == SHORT_SECTION_TYPE

    def test_exact_threshold_boundary(self):
        """Test boundary case at exact threshold."""
        threshold = 50

        # At threshold: flf2v
        assert suggest_keyframe_type(50, threshold) == SHORT_SECTION_TYPE

        # One above threshold: tween
        assert suggest_keyframe_type(51, threshold) == DEFAULT_TYPE


# ============================================================================
# Schedule Building
# ============================================================================


class TestBuildKeyframeTypeSchedule:
    """Test building of keyframe type schedule."""

    def test_empty_frames(self):
        """Test empty frame list returns default."""
        schedule = build_keyframe_type_schedule([], 80)

        assert schedule == [(0, DEFAULT_TYPE)]

    def test_single_frame(self):
        """Test single frame gets default type."""
        schedule = build_keyframe_type_schedule([0], 80)

        assert schedule == [(0, DEFAULT_TYPE)]

    def test_two_frames_short_distance(self):
        """Test two frames with short distance."""
        schedule = build_keyframe_type_schedule([0, 40], 80)

        assert schedule == [(0, DEFAULT_TYPE), (40, SHORT_SECTION_TYPE)]  # Distance 40 <= 80

    def test_two_frames_long_distance(self):
        """Test two frames with long distance."""
        schedule = build_keyframe_type_schedule([0, 100], 80)

        assert schedule == [(0, DEFAULT_TYPE), (100, DEFAULT_TYPE)]  # Distance 100 > 80

    def test_multiple_frames_mixed(self):
        """Test multiple frames with mixed distances."""
        # Threshold: 80
        # 0 → 40 (40 frames, <= 80) = flf2v
        # 40 → 140 (100 frames, > 80) = tween
        # 140 → 200 (60 frames, <= 80) = flf2v
        schedule = build_keyframe_type_schedule([0, 40, 140, 200], 80)

        assert schedule == [
            (0, DEFAULT_TYPE),
            (40, SHORT_SECTION_TYPE),
            (140, DEFAULT_TYPE),
            (200, SHORT_SECTION_TYPE),
        ]

    def test_first_frame_always_default(self):
        """Test first frame always gets default type regardless of threshold."""
        schedule = build_keyframe_type_schedule([100, 120, 130], 5)

        # First frame is always DEFAULT_TYPE
        assert schedule[0] == (100, DEFAULT_TYPE)


# ============================================================================
# Schedule Formatting
# ============================================================================


class TestFormatKeyframeSchedule:
    """Test formatting of keyframe schedule to string."""

    def test_single_keyframe(self):
        """Test single keyframe formatting."""
        schedule = [(0, "tween")]
        formatted = format_keyframe_schedule(schedule)

        assert formatted == "0:(tween)"

    def test_multiple_keyframes(self):
        """Test multiple keyframes formatting."""
        schedule = [(0, "tween"), (60, "flf2v"), (120, "tween")]
        formatted = format_keyframe_schedule(schedule)

        assert formatted == "0:(tween), 60:(flf2v), 120:(tween)"

    def test_preserves_order(self):
        """Test that order is preserved."""
        schedule = [(0, "tween"), (30, "flf2v"), (60, "flf2v"), (120, "tween")]
        formatted = format_keyframe_schedule(schedule)

        assert formatted == "0:(tween), 30:(flf2v), 60:(flf2v), 120:(tween)"

    def test_empty_schedule(self):
        """Test empty schedule."""
        formatted = format_keyframe_schedule([])

        assert formatted == ""


# ============================================================================
# Complete Workflow
# ============================================================================


class TestAutoAssignKeyframeTypes:
    """Test complete auto-assignment workflow."""

    def test_basic_workflow(self):
        """Test basic workflow with mixed distances."""
        prompts = {
            "0": "start",
            "40": "short section",
            "140": "long section",
            "200": "another short",
        }

        formatted, schedule = auto_assign_keyframe_types(prompts, 100)

        # Threshold: 80
        # 0 → always tween
        # 40 (40 frames, <= 80) → flf2v
        # 140 (100 frames, > 80) → tween
        # 200 (60 frames, <= 80) → flf2v
        assert formatted == "0:(tween), 40:(flf2v), 140:(tween), 200:(flf2v)"
        assert len(schedule) == 4

    def test_empty_prompts(self):
        """Test with empty prompts."""
        formatted, schedule = auto_assign_keyframe_types({}, 100)

        assert formatted == "0:(tween)"
        assert schedule == [(0, DEFAULT_TYPE)]

    def test_filters_non_numeric_keys(self):
        """Test that non-numeric keys are ignored."""
        prompts = {"0": "start", "metadata": "ignore me", "60": "end"}

        formatted, schedule = auto_assign_keyframe_types(prompts, 100)

        assert formatted == "0:(tween), 60:(flf2v)"

    def test_all_short_sections(self):
        """Test all short sections get flf2v (except first)."""
        prompts = {"0": "a", "10": "b", "20": "c", "30": "d"}

        formatted, schedule = auto_assign_keyframe_types(prompts, 100)

        # All distances are 10, threshold is 80, so all are flf2v except first
        assert formatted == "0:(tween), 10:(flf2v), 20:(flf2v), 30:(flf2v)"

    def test_all_long_sections(self):
        """Test all long sections get tween."""
        prompts = {"0": "a", "100": "b", "200": "c"}

        formatted, schedule = auto_assign_keyframe_types(prompts, 50)

        # Threshold: 40, all distances are 100, so all are tween
        assert formatted == "0:(tween), 100:(tween), 200:(tween)"

    def test_chunk_size_affects_threshold(self):
        """Test that different chunk sizes affect assignments."""
        prompts = {"0": "start", "50": "end"}

        # Small chunk size (threshold: 40)
        formatted_small, _ = auto_assign_keyframe_types(prompts, 50)
        # Distance 50 > 40, so tween
        assert formatted_small == "0:(tween), 50:(tween)"

        # Large chunk size (threshold: 80)
        formatted_large, _ = auto_assign_keyframe_types(prompts, 100)
        # Distance 50 <= 80, so flf2v
        assert formatted_large == "0:(tween), 50:(flf2v)"


# ============================================================================
# Integration Tests
# ============================================================================


class TestKeyframeParsingIntegration:
    """Integration tests for complete keyframe parsing workflow."""

    def test_realistic_animation_24fps(self):
        """Test realistic 24 FPS animation (1 keyframe per second)."""
        # 5 seconds at 24 FPS
        prompts = {
            "0": "scene start",
            "24": "1 second",
            "48": "2 seconds",
            "72": "3 seconds",
            "96": "4 seconds",
            "120": "5 seconds",
        }

        formatted, schedule = auto_assign_keyframe_types(prompts, 100)

        # All distances are 24 frames, threshold is 80
        # So all should be flf2v except first
        assert schedule[0][1] == DEFAULT_TYPE
        assert all(s[1] == SHORT_SECTION_TYPE for s in schedule[1:])

    def test_realistic_animation_60fps(self):
        """Test realistic 60 FPS animation."""
        # 3 seconds at 60 FPS with keyframes every second
        prompts = {"0": "start", "60": "1 second", "120": "2 seconds", "180": "3 seconds"}

        formatted, schedule = auto_assign_keyframe_types(prompts, 100)

        # All distances are 60 frames, threshold is 80
        # So all should be flf2v except first
        assert formatted == "0:(tween), 60:(flf2v), 120:(flf2v), 180:(flf2v)"

    def test_mixed_pacing_animation(self):
        """Test animation with mixed pacing (slow and fast cuts)."""
        prompts = {
            "0": "slow intro",
            "10": "quick cut",
            "20": "quick cut 2",
            "30": "quick cut 3",
            "150": "long scene",
        }

        formatted, schedule = auto_assign_keyframe_types(prompts, 50)

        # Threshold: 40
        # All quick cuts (10 frames) should be flf2v
        # Long scene (120 frames) should be tween
        assert schedule == [
            (0, DEFAULT_TYPE),
            (10, SHORT_SECTION_TYPE),
            (20, SHORT_SECTION_TYPE),
            (30, SHORT_SECTION_TYPE),
            (150, DEFAULT_TYPE),
        ]
