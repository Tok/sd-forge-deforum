"""Unit tests for deforum.utils.schedule_manipulation_utils module."""

import pytest

from deforum.utils.parsing.schedule_manipulation import (
    apply_shakify_to_schedule,
    combine_schedules,
    scale_schedule,
    offset_schedule,
)


class TestApplyShakifyToSchedule:
    """Tests for apply_shakify_to_schedule function."""

    def test_empty_shake_values_returns_original(self):
        """Should return original schedule if shake values are empty."""
        schedule = "0:(0), 100:(10)"
        result = apply_shakify_to_schedule(schedule, [], 100)
        assert result == schedule

    def test_constant_shake_added_to_constant_schedule(self):
        """Should add constant shake to constant schedule."""
        schedule = "0:(5)"
        shake_values = [0.1] * 100
        result = apply_shakify_to_schedule(schedule, shake_values, 100)

        # Parse result to check values
        assert "0:(5.100000)" in result
        # All sampled values should be 5.1 (5.0 + 0.1)
        assert all("5.100000" in kf for kf in result.split(", "))

    def test_varying_shake_added_to_linear_schedule(self):
        """Should add varying shake to linear schedule."""
        schedule = "0:(0), 100:(100)"  # Linear from 0 to 100
        shake_values = [1.0] * 100  # Constant shake of 1.0
        result = apply_shakify_to_schedule(schedule, shake_values, 100)

        # First frame should be 0 + 1 = 1
        assert result.startswith("0:(1.000000)")

    def test_shake_values_shorter_than_max_frames(self):
        """Should handle shake values shorter than max_frames."""
        schedule = "0:(0), 100:(10)"
        shake_values = [0.5] * 50  # Only 50 frames of shake
        result = apply_shakify_to_schedule(schedule, shake_values, 100)

        # Should only combine up to min(base_values, shake_values)
        assert "0:(0.500000)" in result

    def test_negative_shake_values(self):
        """Should handle negative shake values."""
        schedule = "0:(10)"
        shake_values = [-0.5] * 100
        result = apply_shakify_to_schedule(schedule, shake_values, 100)

        # 10 + (-0.5) = 9.5
        assert all("9.500000" in kf for kf in result.split(", "))

    def test_zero_shake_returns_original_values(self):
        """Should return original schedule values if shake is all zeros."""
        schedule = "0:(5), 50:(10), 100:(15)"
        shake_values = [0.0] * 100
        result = apply_shakify_to_schedule(schedule, shake_values, 100)

        # Values should match original schedule (within floating point precision)
        assert "0:(5.000000)" in result

    def test_sampling_creates_max_20_keyframes(self):
        """Should sample at intervals to create max 20 keyframes."""
        schedule = "0:(0)"
        shake_values = [0.1] * 1000  # 1000 frames
        # Use preserve_dense=False to enable sampling behavior
        result = apply_shakify_to_schedule(schedule, shake_values, 1000, preserve_dense=False)

        # Should have ~20 keyframes + last frame
        keyframes = result.split(", ")
        assert len(keyframes) <= 21  # 20 samples + last frame

    def test_always_includes_last_frame(self):
        """Should always include the last frame as a keyframe."""
        schedule = "0:(0), 100:(10)"
        shake_values = [0.0] * 100
        result = apply_shakify_to_schedule(schedule, shake_values, 100)

        # Should end with frame 99 (0-indexed, so 100 frames = 0-99)
        # Linear interpolation: frame 99 = 0 + (99/100) * 10 = 9.9
        assert result.endswith("99:(9.900000)")


class TestCombineSchedules:
    """Tests for combine_schedules function."""

    def test_add_two_constant_schedules(self):
        """Should add two constant schedules."""
        result = combine_schedules("0:(5)", "0:(3)", 10, "add")
        # 5 + 3 = 8
        assert all("8.000000" in kf for kf in result.split(", "))

    def test_subtract_two_constant_schedules(self):
        """Should subtract second schedule from first."""
        result = combine_schedules("0:(10)", "0:(3)", 10, "subtract")
        # 10 - 3 = 7
        assert all("7.000000" in kf for kf in result.split(", "))

    def test_multiply_two_constant_schedules(self):
        """Should multiply two schedules."""
        result = combine_schedules("0:(4)", "0:(3)", 10, "multiply")
        # 4 * 3 = 12
        assert all("12.000000" in kf for kf in result.split(", "))

    def test_average_two_constant_schedules(self):
        """Should average two schedules."""
        result = combine_schedules("0:(10)", "0:(6)", 10, "average")
        # (10 + 6) / 2 = 8
        assert all("8.000000" in kf for kf in result.split(", "))

    def test_add_linear_schedules(self):
        """Should add two linear schedules."""
        result = combine_schedules("0:(0), 10:(10)", "0:(5), 10:(15)", 10, "add")
        # At frame 0: 0 + 5 = 5
        assert result.startswith("0:(5.000000)")
        # At frame 9: (9/10)*10 + (5 + (9/10)*10) = 9 + 14 = 23
        assert "9:(23.000000)" in result

    def test_unsupported_operation_raises_error(self):
        """Should raise ValueError for unsupported operation."""
        with pytest.raises(ValueError, match="Unsupported operation"):
            combine_schedules("0:(1)", "0:(2)", 10, "invalid_op")

    def test_multiply_by_zero(self):
        """Should handle multiplication by zero."""
        result = combine_schedules("0:(100)", "0:(0)", 10, "multiply")
        # 100 * 0 = 0
        assert all("0.000000" in kf for kf in result.split(", "))

    def test_negative_values_in_subtraction(self):
        """Should handle negative results from subtraction."""
        result = combine_schedules("0:(5)", "0:(10)", 10, "subtract")
        # 5 - 10 = -5
        assert all("-5.000000" in kf for kf in result.split(", "))


class TestScaleSchedule:
    """Tests for scale_schedule function."""

    def test_scale_constant_schedule_by_two(self):
        """Should scale constant schedule by factor of 2."""
        result = scale_schedule("0:(10)", 10, 2.0)
        # 10 * 2 = 20
        assert all("20.000000" in kf for kf in result.split(", "))

    def test_scale_constant_schedule_by_half(self):
        """Should scale constant schedule by factor of 0.5."""
        result = scale_schedule("0:(20)", 10, 0.5)
        # 20 * 0.5 = 10
        assert all("10.000000" in kf for kf in result.split(", "))

    def test_scale_by_zero(self):
        """Should scale all values to zero."""
        result = scale_schedule("0:(100), 10:(200)", 10, 0.0)
        # All values * 0 = 0
        assert all("0.000000" in kf for kf in result.split(", "))

    def test_scale_by_negative_factor(self):
        """Should handle negative scale factors."""
        result = scale_schedule("0:(10)", 10, -2.0)
        # 10 * -2 = -20
        assert all("-20.000000" in kf for kf in result.split(", "))

    def test_scale_linear_schedule(self):
        """Should scale linear schedule proportionally."""
        result = scale_schedule("0:(0), 10:(10)", 10, 2.0)
        # At frame 0: 0 * 2 = 0
        assert result.startswith("0:(0.000000)")
        # At frame 9: ((9/10)*10) * 2 = 9 * 2 = 18
        assert "9:(18.000000)" in result

    def test_scale_by_one_unchanged(self):
        """Should return unchanged values when scaling by 1."""
        result = scale_schedule("0:(5), 10:(15)", 10, 1.0)
        # Values should match original
        assert result.startswith("0:(5.000000)")


class TestOffsetSchedule:
    """Tests for offset_schedule function."""

    def test_offset_constant_schedule_positive(self):
        """Should add positive offset to constant schedule."""
        result = offset_schedule("0:(10)", 10, 5.0)
        # 10 + 5 = 15
        assert all("15.000000" in kf for kf in result.split(", "))

    def test_offset_constant_schedule_negative(self):
        """Should add negative offset to constant schedule."""
        result = offset_schedule("0:(10)", 10, -3.0)
        # 10 + (-3) = 7
        assert all("7.000000" in kf for kf in result.split(", "))

    def test_offset_by_zero_unchanged(self):
        """Should return unchanged values with zero offset."""
        result = offset_schedule("0:(5)", 10, 0.0)
        # 5 + 0 = 5
        assert all("5.000000" in kf for kf in result.split(", "))

    def test_offset_linear_schedule(self):
        """Should add offset to all values in linear schedule."""
        result = offset_schedule("0:(0), 10:(10)", 10, 5.0)
        # At frame 0: 0 + 5 = 5
        assert result.startswith("0:(5.000000)")
        # At frame 9: ((9/10)*10) + 5 = 9 + 5 = 14
        assert "9:(14.000000)" in result

    def test_offset_creates_negative_values(self):
        """Should handle offsets that create negative values."""
        result = offset_schedule("0:(5)", 10, -10.0)
        # 5 + (-10) = -5
        assert all("-5.000000" in kf for kf in result.split(", "))

    def test_offset_large_values(self):
        """Should handle large offset values."""
        result = offset_schedule("0:(1)", 10, 1000.0)
        # 1 + 1000 = 1001
        assert all("1001.000000" in kf for kf in result.split(", "))


class TestScheduleManipulationIntegration:
    """Integration tests combining multiple schedule operations."""

    def test_apply_shake_then_scale(self):
        """Should correctly combine shake application and scaling."""
        schedule = "0:(10)"
        shake_values = [1.0] * 100  # Add 1 to all frames

        # Apply shake: 10 + 1 = 11
        shaken = apply_shakify_to_schedule(schedule, shake_values, 100)
        assert "11.000000" in shaken

        # Scale by 2: 11 * 2 = 22
        scaled = scale_schedule(shaken, 100, 2.0)
        assert "22.000000" in scaled

    def test_combine_then_offset(self):
        """Should correctly combine schedules then apply offset."""
        # Add two schedules: 5 + 3 = 8
        combined = combine_schedules("0:(5)", "0:(3)", 10, "add")
        assert "8.000000" in combined

        # Offset by 2: 8 + 2 = 10
        offset = offset_schedule(combined, 10, 2.0)
        assert "10.000000" in offset

    def test_scale_then_combine(self):
        """Should correctly scale then combine schedules."""
        # Scale first schedule: 10 * 2 = 20
        scaled = scale_schedule("0:(10)", 10, 2.0)
        assert "20.000000" in scaled

        # Add to second schedule: 20 + 5 = 25
        combined = combine_schedules(scaled, "0:(5)", 10, "add")
        assert "25.000000" in combined

    def test_complex_pipeline(self):
        """Should handle complex multi-step schedule manipulation."""
        schedule = "0:(1), 10:(5)"  # Linear from 1 to 5

        # Step 1: Scale by 2
        step1 = scale_schedule(schedule, 10, 2.0)  # 2 to 10

        # Step 2: Offset by 3
        step2 = offset_schedule(step1, 10, 3.0)  # 5 to 13

        # Step 3: Combine with constant schedule using average
        step3 = combine_schedules(step2, "0:(10)", 10, "average")

        # At frame 0: (5 + 10) / 2 = 7.5
        assert "0:(7.500000)" in step3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
