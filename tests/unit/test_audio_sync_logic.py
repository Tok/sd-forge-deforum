"""Unit tests for deforum.utils.audio.sync module.

Tests pure business logic functions for audio synchronization.
All functions here are side-effect free and easily testable.
"""

import pytest
from deforum.utils.audio.sync import (
    calculate_keyframes_per_beat,
    calculate_bpm_based_target,
    apply_keyframe_adjustment,
    calculate_spacing_multiplier,
    calculate_adjusted_min_spacing,
    calculate_compensation_target,
    build_keyframe_visualization,
    build_status_message,
    resolve_keyframe_target,
)


# ============================================================================
# BPM and Keyframe Calculations
# ============================================================================


class TestCalculateKeyframesPerBeat:
    """Test keyframes per beat calculation based on BPM ranges."""

    def test_slow_music_60_bpm(self):
        assert calculate_keyframes_per_beat(60) == 1.0

    def test_slow_music_89_bpm(self):
        assert calculate_keyframes_per_beat(89) == 1.0

    def test_medium_music_90_bpm(self):
        assert calculate_keyframes_per_beat(90) == 0.5

    def test_medium_music_139_bpm(self):
        assert calculate_keyframes_per_beat(139) == 0.5

    def test_fast_music_140_bpm(self):
        assert calculate_keyframes_per_beat(140) == 0.25

    def test_fast_music_180_bpm(self):
        assert calculate_keyframes_per_beat(180) == 0.25


class TestCalculateBpmBasedTarget:
    """Test BPM-based target keyframe calculation."""

    def test_basic_calculation(self):
        # 10 second audio at 120 BPM with 0.5 keyframes/beat
        # 120 BPM = 2 beats/second
        # 10s * 2 beats/s * 0.5 kf/beat = 10 keyframes
        result = calculate_bpm_based_target(10.0, 120.0, 0.5)
        assert result == 10

    def test_short_audio(self):
        # 2 second audio at 60 BPM with 1.0 keyframes/beat
        # 60 BPM = 1 beat/second
        # 2s * 1 beat/s * 1.0 kf/beat = 2 keyframes
        result = calculate_bpm_based_target(2.0, 60.0, 1.0)
        assert result == 2

    def test_fast_music(self):
        # 5 second audio at 180 BPM with 0.25 keyframes/beat
        # 180 BPM = 3 beats/second
        # 5s * 3 beats/s * 0.25 kf/beat = 3.75 → 3 keyframes
        result = calculate_bpm_based_target(5.0, 180.0, 0.25)
        assert result == 3


class TestApplyKeyframeAdjustment:
    """Test percentage adjustment to keyframe count."""

    def test_positive_adjustment(self):
        result = apply_keyframe_adjustment(10, 20)
        assert result == 12  # 10 * 1.2 = 12

    def test_negative_adjustment(self):
        result = apply_keyframe_adjustment(10, -20)
        assert result == 8  # 10 * 0.8 = 8

    def test_zero_adjustment(self):
        result = apply_keyframe_adjustment(10, 0)
        assert result == 10

    def test_minimum_enforcement(self):
        # Even with -90% adjustment, minimum is 2
        result = apply_keyframe_adjustment(3, -90)
        assert result == 2

    def test_small_positive_adjustment(self):
        result = apply_keyframe_adjustment(11, 5)
        # 11 * 1.05 = 11.55 → would round to 11, but guaranteed +1 minimum
        assert result == 12


class TestCalculateSpacingMultiplier:
    """Test spacing multiplier calculation from adjustment percentage."""

    def test_positive_adjustment_reduces_spacing(self):
        # +5% keyframes → -5% spacing → 0.95 multiplier
        result = calculate_spacing_multiplier(5)
        assert result == pytest.approx(0.95)

    def test_negative_adjustment_increases_spacing(self):
        # -5% keyframes → +5% spacing → 1.05 multiplier
        result = calculate_spacing_multiplier(-5)
        assert result == pytest.approx(1.05)

    def test_zero_adjustment(self):
        result = calculate_spacing_multiplier(0)
        assert result == 1.0


class TestCalculateAdjustedMinSpacing:
    """Test min_spacing calculation with multiplier."""

    def test_reduce_spacing(self):
        result = calculate_adjusted_min_spacing(12, 0.95)
        assert result == 11  # 12 * 0.95 = 11.4 → 11

    def test_increase_spacing(self):
        result = calculate_adjusted_min_spacing(12, 1.05)
        assert result == 12  # 12 * 1.05 = 12.6 → 12

    def test_minimum_enforcement(self):
        # Even with very low multiplier, minimum is 1
        result = calculate_adjusted_min_spacing(2, 0.1)
        assert result == 1


class TestCalculateCompensationTarget:
    """Test compensation factor for min_spacing filtering."""

    def test_default_compensation(self):
        result = calculate_compensation_target(10)
        assert result == 12  # 10 * 1.25 = 12.5 → 12

    def test_custom_compensation(self):
        result = calculate_compensation_target(10, 1.5)
        assert result == 15

    def test_low_target(self):
        result = calculate_compensation_target(4)
        assert result == 5  # 4 * 1.25 = 5


# ============================================================================
# Visualization and Display
# ============================================================================


class TestBuildKeyframeVisualization:
    """Test ASCII visualization generation."""

    def test_simple_case(self):
        keyframes = [{"frame": 0}, {"frame": 50}, {"frame": 100}]
        viz_str, spacing_str = build_keyframe_visualization(keyframes, 100, viz_width=10)

        # First and last should have markers
        assert viz_str[0] == "|"
        assert viz_str[-1] == "|"

        # Middle should have marker around position 5
        assert "|" in viz_str[4:6]

        # Spacing should show 0 and 99
        assert spacing_str.startswith("0")
        assert spacing_str.endswith("99")

    def test_single_keyframe(self):
        keyframes = [{"frame": 0}]
        viz_str, spacing_str = build_keyframe_visualization(keyframes, 100, viz_width=10)

        assert viz_str[0] == "|"
        assert viz_str.count("|") == 1

    def test_width_parameter(self):
        keyframes = [{"frame": 0}, {"frame": 100}]
        viz_str, spacing_str = build_keyframe_visualization(keyframes, 100, viz_width=20)

        assert len(viz_str) == 20


class TestBuildStatusMessage:
    """Test status message formatting."""

    def test_basic_message(self):
        msg = build_status_message(
            duration=5.5,
            fps=60,
            total_frames=330,
            bpm=120.0,
            events_detected=15,
            keyframes_created=12,
            prompts_used=5,
            distribution_mode="sequential",
            viz_str="|____|",
            spacing_str="0   100",
        )

        assert "✓ Successfully synchronized!" in msg
        assert "5.5s" in msg
        assert "60 FPS" in msg
        assert "120.0" in msg
        assert "15" in msg  # events
        assert "12" in msg  # keyframes
        assert "5" in msg  # prompts
        assert "sequential" in msg
        assert "|____|" in msg

    def test_average_spacing_calculation(self):
        msg = build_status_message(
            duration=10.0,
            fps=30,
            total_frames=300,
            bpm=90.0,
            events_detected=10,
            keyframes_created=10,
            prompts_used=3,
            distribution_mode="cycle",
            viz_str="|__|",
            spacing_str="0  300",
        )

        # Average spacing should be 300/10 = 30 frames = 1.00s at 30 FPS
        assert "30.0 frames" in msg
        assert "1.00s" in msg


# ============================================================================
# Target Resolution
# ============================================================================


class TestResolveKeyframeTarget:
    """Test keyframe target resolution logic."""

    def test_user_target_no_adjustment(self):
        final, desc = resolve_keyframe_target(
            user_target=15, bpm_based_target=10, keyframe_adjustment=0
        )
        assert final == 15
        assert "user-specified" in desc

    def test_user_target_with_positive_adjustment(self):
        final, desc = resolve_keyframe_target(
            user_target=10, bpm_based_target=8, keyframe_adjustment=20
        )
        assert final == 12  # 10 * 1.2 = 12
        assert "10 → 12" in desc
        assert "+20%" in desc

    def test_user_target_with_negative_adjustment(self):
        final, desc = resolve_keyframe_target(
            user_target=10, bpm_based_target=8, keyframe_adjustment=-20
        )
        assert final == 8  # 10 * 0.8 = 8
        assert "10 → 8" in desc
        assert "-20%" in desc

    def test_bpm_based_no_adjustment(self):
        final, desc = resolve_keyframe_target(
            user_target=0, bpm_based_target=12, keyframe_adjustment=0
        )
        assert final == 12
        assert "BPM-based" in desc

    def test_bpm_based_with_adjustment(self):
        final, desc = resolve_keyframe_target(
            user_target=0, bpm_based_target=10, keyframe_adjustment=5
        )
        # 10 * 1.05 = 10.5 → would round to 10, but guaranteed +1 minimum
        assert final == 11
        assert "10 → 11" in desc
        assert "+5%" in desc
