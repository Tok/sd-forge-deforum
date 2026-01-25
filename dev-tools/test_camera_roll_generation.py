#!/usr/bin/env python3
"""Test script to verify camera roll and look-at-center generation.

Run this to check if rolls are being properly generated in schedules.
"""

import sys
sys.path.insert(0, '/home/zirteq/workspace/forge-neo')

from deforum.utils.spline_camera_path import generate_rotate_around_path, camera_path_to_schedules
import re


def test_rotate_around_with_quaternion_center():
    """Test rotate-around with quaternion mode and look-at center."""
    print("=" * 80)
    print("TEST: Rotate-around with quaternion look-at center mode")
    print("=" * 80)

    # Generate path
    path = generate_rotate_around_path(
        num_frames=10,
        radius=50.0,
        height=10.0,
        center_x=0.0,
        center_y=0.0,
        center_z=0.0,
        stabilize_camera=True,
        rotation_mode='quaternion',
        look_at_mode='center'
    )

    print(f"\n1. Generated {len(path)} camera points")
    print("\nFirst 3 points (raw):")
    for i, point in enumerate(path[:3]):
        print(f"  Frame {point.frame}: pos=({point.x:.1f}, {point.y:.1f}, {point.z:.1f}) "
              f"rot=({point.rot_x:.2f}, {point.rot_y:.2f}, {point.rot_z:.2f})")

    # Convert to schedules with look_at_mode="center"
    schedules = camera_path_to_schedules(
        path,
        speed_multiplier=1.0,
        look_at_mode='center',  # This should recalculate rotations
        stabilize_camera=True
    )

    print("\n2. Generated schedules:")
    for key in ['rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z']:
        schedule = schedules[key]
        # Parse first few values
        pattern = r'(\d+)\s*:\s*\(([^)]+)\)'
        matches = re.findall(pattern, schedule)[:5]
        values = [(int(frame), float(val)) for frame, val in matches]

        print(f"\n  {key}:")
        for frame, val in values:
            print(f"    Frame {frame}: {val:.2f}")

    # Check if rot_z has any non-zero values
    rot_z_schedule = schedules['rotation_3d_z']
    matches = re.findall(r'(\d+)\s*:\s*\(([^)]+)\)', rot_z_schedule)
    rot_z_values = [float(val) for frame, val in matches]

    has_nonzero_roll = any(abs(v) > 0.01 for v in rot_z_values)
    max_roll = max(abs(v) for v in rot_z_values)

    print(f"\n3. Roll (rot_z) analysis:")
    print(f"   Has non-zero values: {has_nonzero_roll}")
    print(f"   Max absolute value: {max_roll:.4f}")
    print(f"   Expected: Near 0 with stabilize_camera=True (should be < 5.0 degrees)")

    if max_roll < 5.0:
        print(f"   ✅ PASS: Roll is properly stabilized")
    else:
        print(f"   ⚠️  WARNING: Roll may be too large")

    return schedules


def test_rotate_around_without_stabilization():
    """Test rotate-around without stabilization to see natural roll."""
    print("\n" + "=" * 80)
    print("TEST: Rotate-around WITHOUT stabilization (natural roll)")
    print("=" * 80)

    # Generate path
    path = generate_rotate_around_path(
        num_frames=10,
        radius=50.0,
        height=20.0,  # More height = more potential roll
        center_x=0.0,
        center_y=0.0,
        center_z=0.0,
        stabilize_camera=False,  # Allow natural roll
        rotation_mode='quaternion',
        look_at_mode='center'
    )

    print(f"\n1. Generated {len(path)} camera points")
    print("\nFirst 3 points (raw):")
    for i, point in enumerate(path[:3]):
        print(f"  Frame {point.frame}: pos=({point.x:.1f}, {point.y:.1f}, {point.z:.1f}) "
              f"rot=({point.rot_x:.2f}, {point.rot_y:.2f}, {point.rot_z:.2f})")

    # Convert to schedules
    schedules = camera_path_to_schedules(
        path,
        speed_multiplier=1.0,
        look_at_mode='center',
        stabilize_camera=False  # Natural roll
    )

    # Check rot_z
    rot_z_schedule = schedules['rotation_3d_z']
    matches = re.findall(r'(\d+)\s*:\s*\(([^)]+)\)', rot_z_schedule)
    rot_z_values = [float(val) for frame, val in matches]

    has_nonzero_roll = any(abs(v) > 0.01 for v in rot_z_values)
    max_roll = max(abs(v) for v in rot_z_values)

    print(f"\n2. Roll (rot_z) analysis:")
    print(f"   Has non-zero values: {has_nonzero_roll}")
    print(f"   Max absolute value: {max_roll:.4f}")
    print(f"   Expected: May have larger values without stabilization")

    if has_nonzero_roll:
        print(f"   ✅ PASS: Roll is being calculated (not hardcoded to 0)")
    else:
        print(f"   ❌ FAIL: Roll is still zero (hardcoded?)")

    return schedules


def main():
    """Run all tests."""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Camera Roll Generation Test" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")

    try:
        schedules1 = test_rotate_around_with_quaternion_center()
        schedules2 = test_rotate_around_without_stabilization()

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("\nIf you see non-zero roll values in the unstabilized test,")
        print("the roll calculation is working correctly!")
        print("\nThe stabilized test should have roll values close to 0.")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
