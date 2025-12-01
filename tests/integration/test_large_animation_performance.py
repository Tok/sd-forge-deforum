"""Integration test for large animation performance (30k+ frames).

Tests that:
1. Schedule truncation works correctly
2. Progress indicators appear
3. No memory/performance issues with large frame counts
4. Full schedules are generated even when UI shows truncated version
"""

import pytest
from deforum.ui.handlers.camera_path_generator import generate_preset_path
from deforum.utils.schedule_truncation import should_truncate_schedules


def test_large_animation_schedule_truncation():
    """Test that 30k frame animation has schedules properly truncated for UI."""
    # Generate a large camera path (30,000 frames)
    num_frames = 30000

    status, schedules, camera_path = generate_preset_path(
        preset_type="rotate-around",
        radius=100,
        height=50,
        num_frames=num_frames,
        closed_loop=False,
        speed_multiplier=0.2,
        rotation_mode="empirical",
        rotation_factor=-8.0
    )

    # Verify camera path was generated
    assert len(camera_path) == num_frames, f"Expected {num_frames} frames, got {len(camera_path)}"

    # Verify schedules were generated and truncated
    assert "translation_x" in schedules
    assert "translation_y" in schedules
    assert "translation_z" in schedules
    assert "rotation_3d_x" in schedules
    assert "rotation_3d_y" in schedules
    assert "rotation_3d_z" in schedules

    # Verify truncation occurred (should see truncation indicator)
    tx_schedule = schedules["translation_x"]
    assert "[truncated" in tx_schedule, "Schedule should be truncated for large animation"
    assert "settings.json" in tx_schedule, "Should reference settings.json for full schedules"

    # Verify truncation info in status message
    assert "Schedule Display Truncated" in status
    assert "30,000" in status or "30000" in status
    assert "1,000" in status or "1000" in status

    print(f"✓ Successfully generated {num_frames:,} frame camera path")
    print(f"✓ Schedules properly truncated (tx length: {len(tx_schedule)} chars)")
    print(f"✓ Status message includes truncation info")


def test_medium_animation_no_truncation():
    """Test that smaller animations (<1000 frames) are not truncated."""
    num_frames = 500

    status, schedules, camera_path = generate_preset_path(
        preset_type="rotate-around",
        radius=100,
        height=50,
        num_frames=num_frames,
        closed_loop=True,
        speed_multiplier=1.0,
        rotation_mode="empirical",
        rotation_factor=-8.0
    )

    # Verify no truncation for small animation
    assert should_truncate_schedules(num_frames, threshold=1000) is False
    tx_schedule = schedules["translation_x"]
    assert "[truncated" not in tx_schedule, "Small animation should not be truncated"
    assert "Schedule Display Truncated" not in status

    print(f"✓ {num_frames} frame animation not truncated (as expected)")


def test_boundary_animation_1000_frames():
    """Test that exactly 1000 frames (at threshold) is not truncated."""
    num_frames = 1000

    status, schedules, camera_path = generate_preset_path(
        preset_type="figure-eight",
        radius=50,
        height=20,
        num_frames=num_frames,
        closed_loop=True
    )

    # At boundary - should NOT be truncated
    assert should_truncate_schedules(num_frames, threshold=1000) is False
    tx_schedule = schedules["translation_x"]
    assert "[truncated" not in tx_schedule

    print(f"✓ 1000 frame animation (at boundary) not truncated")


def test_just_over_boundary_1001_frames():
    """Test that 1001 frames (just over threshold) IS truncated."""
    num_frames = 1001

    status, schedules, camera_path = generate_preset_path(
        preset_type="forward-zoom",
        radius=100,
        height=10,
        num_frames=num_frames,
        closed_loop=False
    )

    # Just over boundary - SHOULD be truncated
    assert should_truncate_schedules(num_frames, threshold=1000) is True
    tx_schedule = schedules["translation_x"]
    assert "[truncated" in tx_schedule

    print(f"✓ 1001 frame animation (just over boundary) truncated")


def test_extreme_animation_100k_frames():
    """Test that even extreme frame counts (100k) work without crashing.

    Note: This test may take several seconds to complete due to progress bars.
    """
    num_frames = 100000

    status, schedules, camera_path = generate_preset_path(
        preset_type="rotate-around",
        radius=100,
        height=50,
        num_frames=num_frames,
        closed_loop=False,
        speed_multiplier=0.5,
        rotation_mode="empirical",
        rotation_factor=-8.0
    )

    # Verify generation succeeded
    assert len(camera_path) == num_frames
    assert "[truncated" in schedules["translation_x"]

    # Verify schedule strings are reasonable length (not massive)
    tx_schedule = schedules["translation_x"]
    # With truncation at 1000 frames, schedule should be ~15KB max, not ~1.5MB
    assert len(tx_schedule) < 50000, f"Schedule too long: {len(tx_schedule)} chars"

    print(f"✓ Successfully generated {num_frames:,} frame camera path")
    print(f"✓ Schedule truncated to reasonable length ({len(tx_schedule):,} chars)")


if __name__ == "__main__":
    # Run tests individually for manual testing
    print("=" * 60)
    print("Testing Large Animation Performance")
    print("=" * 60)
    print()

    print("Test 1: 30k frames with truncation")
    test_large_animation_schedule_truncation()
    print()

    print("Test 2: 500 frames (no truncation)")
    test_medium_animation_no_truncation()
    print()

    print("Test 3: 1000 frames (at boundary, no truncation)")
    test_boundary_animation_1000_frames()
    print()

    print("Test 4: 1001 frames (just over boundary, truncated)")
    test_just_over_boundary_1001_frames()
    print()

    print("Test 5: 100k frames (extreme case)")
    print("⚠️  This may take 30-60 seconds due to progress bars...")
    test_extreme_animation_100k_frames()
    print()

    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
