"""Test wormtrail visualization with rotate-around path to verify movement is visible."""

import sys
import re
sys.path.insert(0, '/home/zirteq/workspace/forge-neo')

from deforum.utils.spline_camera_path import generate_rotate_around_path, camera_path_to_schedules


def parse_schedule_string(schedule_str: str) -> dict:
    """Parse schedule string (same logic as visualizer)."""
    if not schedule_str or not schedule_str.strip():
        return {}
    pattern = r'(\d+)\s*:\s*\(\s*(-?\d+\.?\d*)\s*\)'
    matches = re.findall(pattern, schedule_str)
    schedule_dict = {}
    for frame_str, value_str in matches:
        frame = int(frame_str)
        value = float(value_str)
        schedule_dict[frame] = value
    return schedule_dict


def test_wormtrail_visualization():
    """Test that wormtrail visualization shows both movement and rotation correctly."""
    print("\n=== Testing Wormtrail Visualization ===\n")

    # Generate rotate-around path (333 frames, closed loop)
    camera_path = generate_rotate_around_path(
        num_frames=333,
        radius=100.0,
        height=0.0,
        closed_loop=True,
        rotation_mode="quaternion",
        look_at_mode="center",
        look_at_blend=0.3
    )

    print(f"Generated {len(camera_path)} camera points")

    # Convert to schedules
    schedules = camera_path_to_schedules(
        camera_path,
        speed_multiplier=1.0,
        speed_randomization=0.0,
        random_seed=0,
        look_at_mode="center"
    )

    print("Schedules generated successfully")

    # Parse schedules (same as visualization does)
    tx_dict = parse_schedule_string(schedules['translation_x'])
    ty_dict = parse_schedule_string(schedules['translation_y'])
    tz_dict = parse_schedule_string(schedules['translation_z'])
    rx_dict = parse_schedule_string(schedules['rotation_3d_x'])
    ry_dict = parse_schedule_string(schedules['rotation_3d_y'])
    rz_dict = parse_schedule_string(schedules['rotation_3d_z'])

    print(f"\nParsed schedules:")
    print(f"  Translation X: {len(tx_dict)} keyframes")
    print(f"  Translation Y: {len(ty_dict)} keyframes")
    print(f"  Translation Z: {len(tz_dict)} keyframes")
    print(f"  Rotation X: {len(rx_dict)} keyframes")
    print(f"  Rotation Y: {len(ry_dict)} keyframes")
    print(f"  Rotation Z: {len(rz_dict)} keyframes")

    # Simulate what visualization does (accumulate translations, use rotations directly)
    actual_max_frame = 333
    is_dense = len(tx_dict) > (actual_max_frame / 2)
    print(f"\nDense schedule: {is_dense}")

    if is_dense:
        # Get translation deltas
        tx_values = [tx_dict.get(i, 0.0) for i in range(333)]
        ty_values = [ty_dict.get(i, 0.0) for i in range(333)]
        tz_values = [tz_dict.get(i, 0.0) for i in range(333)]

        # Accumulate translations
        cum_x, cum_y, cum_z = 0.0, 0.0, 0.0
        positions = []
        for i in range(333):
            cum_x += tx_values[i]
            cum_y += ty_values[i]
            cum_z += tz_values[i]
            positions.append((cum_x, cum_y, cum_z))

        # Get rotation values directly (NOT accumulated)
        rx_values = [rx_dict.get(i, 0.0) for i in range(333)]
        ry_values = [ry_dict.get(i, 0.0) for i in range(333)]
        rz_values = [rz_dict.get(i, 0.0) for i in range(333)]

        print("\n=== First 10 Positions (Accumulated Translation Deltas) ===")
        for i in range(10):
            x, y, z = positions[i]
            dist = (x**2 + y**2 + z**2)**0.5
            print(f"  Frame {i}: ({x:7.2f}, {y:7.2f}, {z:7.2f}) dist={dist:7.2f}")

        print("\n=== First 10 Rotations (Absolute Angles) ===")
        for i in range(10):
            rx, ry, rz = rx_values[i], ry_values[i], rz_values[i]
            print(f"  Frame {i}: rot_x={rx:7.2f}°, rot_y={ry:7.2f}°, rot_z={rz:7.2f}°")

        # Check full path movement
        distances = [(x**2 + y**2 + z**2)**0.5 for x, y, z in positions]
        print(f"\n=== Full Path Stats ===")
        print(f"  Min distance from origin: {min(distances):.2f}")
        print(f"  Max distance from origin: {max(distances):.2f}")
        print(f"  Avg distance from origin: {sum(distances)/len(distances):.2f}")

        # Check rotation variation
        print(f"\n=== Rotation Stats ===")
        print(f"  Rotation X range: [{min(rx_values):.2f}°, {max(rx_values):.2f}°]")
        print(f"  Rotation Y range: [{min(ry_values):.2f}°, {max(ry_values):.2f}°]")
        print(f"  Rotation Z range: [{min(rz_values):.2f}°, {max(rz_values):.2f}°]")

        # Verify results
        movement_ok = max(distances) > 50
        # For center look-at mode, rotations stay relatively constant
        # (camera always looks at center, so local Euler angles don't change much)
        rotation_ok = abs(max(rx_values) - min(rx_values)) > 5  # Should see pitch variation

        if movement_ok:
            print("\n✓ Camera shows significant MOVEMENT (max distance > 50)")
        else:
            print(f"\n✗ Camera barely moves! (max={max(distances):.2f})")

        if rotation_ok:
            print(f"✓ Camera rotation varies correctly (X rotation range > 5°, actual: {abs(max(rx_values) - min(rx_values)):.2f}°)")
        else:
            print(f"✗ Camera rotation doesn't vary! (X range={abs(max(rx_values) - min(rx_values)):.2f}°)")

        # Verify the fix resolves the issue
        print("\n=== VERIFICATION ===")
        if movement_ok and rotation_ok:
            print("\n✓✓✓ WORMTRAIL VISUALIZATION FIX WORKING! ✓✓✓")
            print("\nThe fix correctly:")
            print("  - Accumulates translation deltas → Camera moves in full orbit (200 unit radius)")
            print("  - Uses rotation angles directly → Camera faces center throughout orbit")
            print("\nBefore fix: Rotations were accumulated, causing camera to spiral incorrectly")
            print("After fix: Rotations used as absolute angles, camera stays centered on target")
        else:
            print("\n✗✗✗ SOMETHING IS STILL WRONG! ✗✗✗")


if __name__ == "__main__":
    test_wormtrail_visualization()
