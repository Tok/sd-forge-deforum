"""Unit tests for schedule truncation utilities."""

import pytest
from deforum.utils.schedule_truncation import (
    truncate_schedule_for_display,
    truncate_schedules_dict,
    should_truncate_schedules,
    get_truncation_info_message
)


def test_truncate_schedule_no_truncation_needed():
    """Test that small schedules are not truncated."""
    schedule = "0: (0.00), 50: (1.50), 100: (3.00)"
    result = truncate_schedule_for_display(schedule, max_frames=1000)
    assert result == schedule
    assert "[truncated" not in result


def test_truncate_schedule_with_truncation():
    """Test that large schedules are properly truncated."""
    # Create schedule with 2000 frames
    entries = [f"{i}: ({i * 0.5:.2f})" for i in range(0, 2000, 50)]
    schedule = ', '.join(entries)

    result = truncate_schedule_for_display(schedule, max_frames=1000)

    # Should contain entries up to frame 1000
    assert "0: (0.00)" in result
    assert "1000: (500.00)" in result

    # Should NOT contain entries beyond 1000
    assert "1050: (525.00)" not in result
    assert "1950: (975.00)" not in result

    # Should have truncation indicator
    assert "[truncated at frame 1000" in result


def test_truncate_schedule_empty():
    """Test that empty schedules are handled gracefully."""
    result = truncate_schedule_for_display("", max_frames=1000)
    assert result == ""

    result = truncate_schedule_for_display("   ", max_frames=1000)
    assert result == "   "


def test_truncate_schedules_dict():
    """Test truncating a dictionary of schedules."""
    schedules = {
        'translation_x': ', '.join([f"{i}: ({i:.2f})" for i in range(0, 2000, 10)]),
        'translation_y': ', '.join([f"{i}: ({i * 2:.2f})" for i in range(0, 2000, 10)]),
        'rotation_3d_z': ', '.join([f"{i}: ({i * 0.5:.2f})" for i in range(0, 2000, 10)])
    }

    result = truncate_schedules_dict(schedules, max_frames=500)

    # All schedules should be truncated
    for key, value in result.items():
        assert "[truncated at frame" in value
        # Should not contain frames beyond 500
        assert "600:" not in value
        assert "1000:" not in value


def test_should_truncate_schedules():
    """Test truncation threshold check."""
    assert should_truncate_schedules(100, threshold=1000) is False
    assert should_truncate_schedules(1000, threshold=1000) is False
    assert should_truncate_schedules(1001, threshold=1000) is True
    assert should_truncate_schedules(30000, threshold=1000) is True


def test_get_truncation_info_message():
    """Test truncation info message generation."""
    msg = get_truncation_info_message(30000, 1000)

    assert "30,000" in msg
    assert "1,000" in msg
    assert "settings.json" in msg.lower()
    assert "browser freeze" in msg.lower()


def test_truncate_schedule_malformed():
    """Test handling of malformed schedule entries."""
    # Missing colon
    schedule = "0 (0.00), 50: (1.50), 100: (3.00)"
    result = truncate_schedule_for_display(schedule, max_frames=1000)
    # Should include valid entries
    assert "50: (1.50)" in result
    assert "100: (3.00)" in result


def test_truncate_schedule_boundary():
    """Test truncation at exact boundary."""
    # Create schedule where last frame is exactly at threshold
    schedule = "0: (0.00), 500: (1.00), 1000: (2.00)"
    result = truncate_schedule_for_display(schedule, max_frames=1000)

    # Should include frame 1000 (at boundary)
    assert "1000: (2.00)" in result
    # Should NOT be truncated
    assert "[truncated" not in result


def test_truncate_schedule_just_over_boundary():
    """Test truncation just over boundary."""
    schedule = "0: (0.00), 500: (1.00), 1000: (2.00), 1001: (2.01)"
    result = truncate_schedule_for_display(schedule, max_frames=1000)

    # Should include up to frame 1000
    assert "1000: (2.00)" in result
    # Should NOT include frame 1001
    assert "1001: (2.01)" not in result
    # Should have truncation indicator
    assert "[truncated at frame 1000" in result
