"""Detailed wormtrail diagnostic - compare actual camera path vs visualization interpretation."""

import sys
import re
sys.path.insert(0, '/home/zirteq/workspace/forge-neo')

from deforum.utils.spline_camera_path import generate_rotate_around_path, camera_path_to_schedules


def parse_schedule_string(schedule_str: str) -> dict:
    """Parse schedule string."""
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


def test_wormtrail_detailed():
    """Compare actual camera path vs what visualization sees."""
    print("\n=== DETAILED WORMTRAIL DIAGNOSTIC ===\n")

    # Generate small 20-frame path for easy inspection
    camera_path = generate_rotate_around_path(
        num_frames=20,
        radius=100.0,
        height=0.0,
        closed_loop=True,
        rotation_mode="quaternion",
        look_at_mode="center",
        look_at_blend=0.3
    )

    print("=== ORIGINAL CAMERA PATH (before schedule conversion) ===")
    print("\nFirst 5 frames:")
    for i in range(min(5, len(camera_path))):
        p = camera_path[i]
        print(f"  Frame {i}: pos=({p.x:7.2f}, {p.y:7.2f}, {p.z:7.2f}), rot=({p.rot_x:7.2f}°, {p.rot_y:7.2f}°, {p.rot_z:7.2f}°)")

    # Convert to schedules
    schedules = camera_path_to_schedules(
        camera_path,
        speed_multiplier=1.0,
        speed_randomization=0.0,
        random_seed=0,
        look_at_mode="center"  # Passing this parameter
    )

    print("\n=== SCHEDULES (deltas) ===")

    # Parse schedules
    tx_dict = parse_schedule_string(schedules['translation_x'])
    ty_dict = parse_schedule_string(schedules['translation_y'])
    tz_dict = parse_schedule_string(schedules['translation_z'])
    rx_dict = parse_schedule_string(schedules['rotation_3d_x'])
    ry_dict = parse_schedule_string(schedules['rotation_3d_y'])
    rz_dict = parse_schedule_string(schedules['rotation_3d_z'])

    print("\nFirst 5 translation X deltas:")
    for i in range(5):
        print(f"  Frame {i}: {tx_dict.get(i, 0.0):7.2f}")

    print("\nFirst 5 rotation Y deltas:")
    for i in range(5):
        print(f"  Frame {i}: {ry_dict.get(i, 0.0):7.2f}°")

    # Simulate visualization accumulation
    print("\n=== VISUALIZATION INTERPRETATION (accumulated) ===")

    tx_values = [tx_dict.get(i, 0.0) for i in range(20)]
    ty_values = [ty_dict.get(i, 0.0) for i in range(20)]
    tz_values = [tz_dict.get(i, 0.0) for i in range(20)]
    rx_values = [rx_dict.get(i, 0.0) for i in range(20)]
    ry_values = [ry_dict.get(i, 0.0) for i in range(20)]
    rz_values = [rz_dict.get(i, 0.0) for i in range(20)]

    # Accumulate
    cum_x, cum_y, cum_z = 0.0, 0.0, 0.0
    cum_rx, cum_ry, cum_rz = 0.0, 0.0, 0.0

    print("\nFirst 5 frames (accumulated from deltas):")
    for i in range(5):
        cum_x += tx_values[i]
        cum_y += ty_values[i]
        cum_z += tz_values[i]
        cum_rx += rx_values[i]
        cum_ry += ry_values[i]
        cum_rz += rz_values[i]

        dist = (cum_x**2 + cum_y**2 + cum_z**2)**0.5
        print(f"  Frame {i}: pos=({cum_x:7.2f}, {cum_y:7.2f}, {cum_z:7.2f}), rot=({cum_rx:7.2f}°, {cum_ry:7.2f}°, {cum_rz:7.2f}°), dist={dist:7.2f}")

    # Compare with original camera path
    print("\n=== COMPARISON ===")
    print("\nOriginal vs Accumulated (first 5 frames):")

    cum_x, cum_y, cum_z = 0.0, 0.0, 0.0
    cum_rx, cum_ry, cum_rz = 0.0, 0.0, 0.0

    for i in range(5):
        cum_x += tx_values[i]
        cum_y += ty_values[i]
        cum_z += tz_values[i]
        cum_rx += rx_values[i]
        cum_ry += ry_values[i]
        cum_rz += rz_values[i]

        p = camera_path[i]
        # Original path positions (before normalization)
        orig_x, orig_y, orig_z = p.x, p.y, p.z
        # Normalized (first frame to origin)
        offset_x = camera_path[0].x
        offset_y = camera_path[0].y
        offset_z = camera_path[0].z
        norm_x = orig_x - offset_x
        norm_y = orig_y - offset_y
        norm_z = orig_z - offset_z

        print(f"\nFrame {i}:")
        print(f"  Original path (normalized): ({norm_x:7.2f}, {norm_y:7.2f}, {norm_z:7.2f})")
        print(f"  Accumulated from deltas:    ({cum_x:7.2f}, {cum_y:7.2f}, {cum_z:7.2f})")

        pos_match = abs(norm_x - cum_x) < 0.01 and abs(norm_y - cum_y) < 0.01 and abs(norm_z - cum_z) < 0.01
        print(f"  Position match: {'✓' if pos_match else '✗ MISMATCH!'}")

    # Full path check
    print("\n=== FULL PATH ACCUMULATION ===")
    cum_x, cum_y, cum_z = 0.0, 0.0, 0.0
    cum_rx, cum_ry, cum_rz = 0.0, 0.0, 0.0

    positions = []
    rotations = []

    for i in range(20):
        cum_x += tx_values[i]
        cum_y += ty_values[i]
        cum_z += tz_values[i]
        cum_rx += rx_values[i]
        cum_ry += ry_values[i]
        cum_rz += rz_values[i]

        positions.append((cum_x, cum_y, cum_z))
        rotations.append((cum_rx, cum_ry, cum_rz))

    distances = [(x**2 + y**2 + z**2)**0.5 for x, y, z in positions]

    print(f"Position stats:")
    print(f"  Min distance: {min(distances):.2f}")
    print(f"  Max distance: {max(distances):.2f}")
    print(f"  Avg distance: {sum(distances)/len(distances):.2f}")

    ry_angles = [r[1] for r in rotations]
    print(f"\nRotation Y stats:")
    print(f"  Min: {min(ry_angles):.2f}°")
    print(f"  Max: {max(ry_angles):.2f}°")
    print(f"  Range: {max(ry_angles) - min(ry_angles):.2f}°")

    # Check if it's actually orbiting
    if max(distances) > 50:
        print("\n✓ Positions show orbital movement")
    else:
        print(f"\n✗ Positions barely move! (max={max(distances):.2f})")

    if abs(max(ry_angles) - min(ry_angles)) > 180:
        print("✓ Rotation Y shows full orbit (>180° range)")
    else:
        print(f"✗ Rotation Y limited range: {abs(max(ry_angles) - min(ry_angles)):.2f}°")


if __name__ == "__main__":
    test_wormtrail_detailed()
