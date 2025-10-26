"""Unit tests for deforum.utils.conversion.fps module.

Tests pure functions for FPS conversion of prompt frame numbers.
All functions here are side-effect free and easily testable.
"""

import pytest
from deforum.utils.conversion.fps import (
    calculate_fps_ratio,
    convert_frame_number,
    convert_prompts_dict,
    validate_fps_values,
    build_conversion_status,
)


# ============================================================================
# FPS Ratio Calculation
# ============================================================================


class TestCalculateFpsRatio:
    """Test FPS ratio calculation."""

    def test_double_fps(self):
        """Test doubling FPS (24 → 48)."""
        ratio = calculate_fps_ratio(24, 48)
        assert ratio == 2.0

    def test_half_fps(self):
        """Test halving FPS (60 → 30)."""
        ratio = calculate_fps_ratio(60, 30)
        assert ratio == 0.5

    def test_common_conversion_24_to_60(self):
        """Test 24 FPS to 60 FPS conversion."""
        ratio = calculate_fps_ratio(24, 60)
        assert ratio == pytest.approx(2.5)

    def test_same_fps(self):
        """Test same FPS (30 → 30)."""
        ratio = calculate_fps_ratio(30, 30)
        assert ratio == 1.0

    def test_fractional_fps(self):
        """Test fractional FPS values."""
        ratio = calculate_fps_ratio(23.976, 29.97)
        assert ratio == pytest.approx(1.25, abs=0.01)


# ============================================================================
# Frame Number Conversion
# ============================================================================


class TestConvertFrameNumber:
    """Test single frame number conversion."""

    def test_double_fps_frame_0(self):
        """Test frame 0 always stays 0."""
        new_frame = convert_frame_number(0, 2.0)
        assert new_frame == 0

    def test_double_fps_frame_100(self):
        """Test doubling: frame 100 → 200."""
        new_frame = convert_frame_number(100, 2.0)
        assert new_frame == 200

    def test_half_fps_frame_100(self):
        """Test halving: frame 100 → 50."""
        new_frame = convert_frame_number(100, 0.5)
        assert new_frame == 50

    def test_ratio_2_5_frame_24(self):
        """Test 2.5x ratio: frame 24 → 60."""
        new_frame = convert_frame_number(24, 2.5)
        assert new_frame == 60

    def test_rounding_down(self):
        """Test rounding to int: frame 10 * 1.3 = 13.0 → 13."""
        new_frame = convert_frame_number(10, 1.3)
        assert new_frame == 13

    def test_rounding_truncation(self):
        """Test int() truncates: frame 10 * 1.19 = 11.9 → 11."""
        new_frame = convert_frame_number(10, 1.19)
        assert new_frame == 11


# ============================================================================
# Prompts Dictionary Conversion
# ============================================================================


class TestConvertPromptsDict:
    """Test conversion of prompts dictionary."""

    def test_simple_conversion(self):
        """Test basic frame conversion."""
        prompts = {"0": "start", "24": "middle", "48": "end"}

        converted, log = convert_prompts_dict(prompts, 2.0)

        assert converted == {"0": "start", "48": "middle", "96": "end"}
        assert log == ["Frame 0 → 0", "Frame 24 → 48", "Frame 48 → 96"]

    def test_preserves_non_numeric_keys(self):
        """Test that non-numeric keys are preserved."""
        prompts = {"0": "start", "metadata": "test data", "30": "end"}

        converted, log = convert_prompts_dict(prompts, 2.0)

        assert converted["metadata"] == "test data"
        assert "metadata" not in [entry for entry in log]

    def test_empty_prompts(self):
        """Test conversion of empty dict."""
        converted, log = convert_prompts_dict({}, 1.5)

        assert converted == {}
        assert log == []

    def test_single_prompt(self):
        """Test conversion of single prompt."""
        prompts = {"100": "single"}

        converted, log = convert_prompts_dict(prompts, 0.5)

        assert converted == {"50": "single"}
        assert log == ["Frame 100 → 50"]

    def test_preserves_prompt_text(self):
        """Test that prompt text is unchanged."""
        prompts = {"0": "a beautiful landscape, detailed", "60": "sunset with vibrant colors"}

        converted, log = convert_prompts_dict(prompts, 2.0)

        assert converted["0"] == "a beautiful landscape, detailed"
        assert converted["120"] == "sunset with vibrant colors"


# ============================================================================
# FPS Validation
# ============================================================================


class TestValidateFpsValues:
    """Test FPS value validation."""

    def test_valid_fps_values(self):
        """Test valid FPS values."""
        is_valid, error = validate_fps_values(24, 60)

        assert is_valid is True
        assert error == ""

    def test_zero_source_fps(self):
        """Test zero source FPS is invalid."""
        is_valid, error = validate_fps_values(0, 60)

        assert is_valid is False
        assert "positive" in error

    def test_negative_source_fps(self):
        """Test negative source FPS is invalid."""
        is_valid, error = validate_fps_values(-24, 60)

        assert is_valid is False
        assert "positive" in error

    def test_zero_target_fps(self):
        """Test zero target FPS is invalid."""
        is_valid, error = validate_fps_values(24, 0)

        assert is_valid is False
        assert "positive" in error

    def test_negative_target_fps(self):
        """Test negative target FPS is invalid."""
        is_valid, error = validate_fps_values(24, -60)

        assert is_valid is False
        assert "positive" in error

    def test_same_fps_values(self):
        """Test same FPS is invalid (no conversion needed)."""
        is_valid, error = validate_fps_values(30, 30)

        assert is_valid is False
        assert "same" in error.lower()


# ============================================================================
# Status Message Building
# ============================================================================


class TestBuildConversionStatus:
    """Test HTML status message generation."""

    def test_basic_status_message(self):
        """Test basic status message contains key info."""
        log = ["Frame 0 → 0", "Frame 24 → 48"]
        status = build_conversion_status(24, 48, 2.0, log, False)

        assert "24" in status  # Source FPS
        assert "48" in status  # Target FPS
        assert "2.0" in status  # Ratio
        assert "2" in status  # Number of prompts

    def test_preview_mode_indicator(self):
        """Test preview mode shows correct indicator."""
        log = ["Frame 0 → 0"]
        status = build_conversion_status(24, 48, 2.0, log, True)

        assert "PREVIEW MODE" in status
        assert "not updated" in status.lower()

    def test_update_mode_indicator(self):
        """Test update mode shows correct indicator."""
        log = ["Frame 0 → 0"]
        status = build_conversion_status(24, 48, 2.0, log, False)

        assert "Updated" in status

    def test_shows_conversion_log(self):
        """Test conversion log is shown."""
        log = ["Frame 0 → 0", "Frame 24 → 48", "Frame 48 → 96"]
        status = build_conversion_status(24, 48, 2.0, log, False)

        assert "Frame 0 → 0" in status
        assert "Frame 24 → 48" in status
        assert "Frame 48 → 96" in status

    def test_limits_log_entries(self):
        """Test log is limited to max_entries."""
        log = [f"Frame {i} → {i*2}" for i in range(20)]
        status = build_conversion_status(24, 48, 2.0, log, False, max_entries=5)

        # First 5 should be shown
        assert "Frame 0 → 0" in status
        assert "Frame 4 → 8" in status

        # Later ones should be truncated
        assert "and 15 more" in status

    def test_formula_explanation(self):
        """Test formula explanation is included."""
        log = ["Frame 0 → 0"]
        status = build_conversion_status(24, 60, 2.5, log, False)

        assert "Formula" in status
        assert "new_frame = old_frame" in status


# ============================================================================
# Integration Tests
# ============================================================================


class TestFpsConversionIntegration:
    """Integration tests for complete FPS conversion workflow."""

    def test_complete_workflow_24_to_60(self):
        """Test complete workflow: 24 FPS → 60 FPS."""
        # Setup
        prompts = {"0": "start scene", "24": "1 second mark", "48": "2 second mark"}

        # Validate
        is_valid, error = validate_fps_values(24, 60)
        assert is_valid

        # Calculate ratio
        ratio = calculate_fps_ratio(24, 60)
        assert ratio == 2.5

        # Convert
        converted, log = convert_prompts_dict(prompts, ratio)

        assert converted == {"0": "start scene", "60": "1 second mark", "120": "2 second mark"}

        # Build status
        status = build_conversion_status(24, 60, ratio, log, False)
        assert "24" in status
        assert "60" in status

    def test_workflow_with_invalid_fps(self):
        """Test workflow catches invalid FPS early."""
        is_valid, error = validate_fps_values(0, 60)

        assert is_valid is False
        # Should not proceed to conversion

    def test_workflow_preserves_timing(self):
        """Test that real-world timing is preserved."""
        # At 24 FPS: frame 24 = 1 second
        # At 60 FPS: frame 60 = 1 second
        prompts = {"24": "1 second mark"}

        ratio = calculate_fps_ratio(24, 60)
        converted, _ = convert_prompts_dict(prompts, ratio)

        # Frame 60 at 60 FPS should be same time as frame 24 at 24 FPS
        assert "60" in converted
        assert converted["60"] == "1 second mark"
