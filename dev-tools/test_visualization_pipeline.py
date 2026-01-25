"""Test the full pipeline from camera path to visualization."""

import sys
sys.path.insert(0, '/home/zirteq/workspace/forge-neo')

from deforum.utils.spline_camera_path import generate_rotate_around_path, camera_path_to_schedules


def test_full_pipeline():
    """Test generating preset and converting to schedules."""
    print("\n=== Testing Full Pipeline ===\n")

    # Generate rotate-around path (same as UI would do)
    camera_path = generate_rotate_around_path(
        num_frames=333,
        radius=100.0,
        height=0.0,
        closed_loop=True,
        rotation_mode="quaternion",
        look_at_mode="center",
        look_at_blend=0.3
    )

    print("Camera path length:", len(camera_path))

    # Convert to schedules (same as handler would do)
    schedules = camera_path_to_schedules(
        camera_path,
        speed_multiplier=1.0,
        speed_randomization=0.0,
        random_seed=0,
        look_at_mode="center"
    )

    print("Schedules generated successfully")

    # Check schedules
    print("\n=== Schedules Generated ===")
    for key in ['translation_x', 'translation_y', 'translation_z']:
        schedule_str = schedules[key]
        # Count keyframes
        import re
        pattern = r'(\d+)\s*:\s*\(\s*(-?\d+\.?\d*)\s*\)'
        matches = re.findall(pattern, schedule_str)
        values = [float(v) for _, v in matches]
        print(f"{key}: {len(matches)} keyframes, values range [{min(values):.2f}, {max(values):.2f}]")

    # Simulate what visualization would do
    print("\n=== Simulating Visualization ===")

    # Parse schedules
    import re
    pattern = r'(\d+)\s*:\s*\(\s*(-?\d+\.?\d*)\s*\)'

    tx_matches = re.findall(pattern, schedules['translation_x'])
    ty_matches = re.findall(pattern, schedules['translation_y'])
    tz_matches = re.findall(pattern, schedules['translation_z'])

    tx_values = [float(v) for _, v in tx_matches]
    ty_values = [float(v) for _, v in ty_matches]
    tz_values = [float(v) for _, v in tz_matches]

    # Check if dense
    actual_max_frame = 333
    is_dense = len(tx_matches) > (actual_max_frame / 2)
    print(f"Is dense schedule: {is_dense} (len={len(tx_matches)} > {actual_max_frame/2})")

    if is_dense:
        # Accumulate deltas
        positions = []
        cum_x, cum_y, cum_z = 0.0, 0.0, 0.0
        for i in range(min(len(tx_values), 10)):  # First 10
            cum_x += tx_values[i]
            cum_y += ty_values[i]
            cum_z += tz_values[i]
            positions.append((cum_x, cum_y, cum_z))

        print("\nFirst 10 accumulated positions:")
        for i, (x, y, z) in enumerate(positions):
            dist = (x**2 + y**2 + z**2)**0.5
            print(f"  Frame {i}: ({x:7.2f}, {y:7.2f}, {z:7.2f}) dist={dist:7.2f}")

        # Check full path
        cum_x, cum_y, cum_z = 0.0, 0.0, 0.0
        all_dists = []
        for i in range(len(tx_values)):
            cum_x += tx_values[i]
            cum_y += ty_values[i]
            cum_z += tz_values[i]
            dist = (cum_x**2 + cum_y**2 + cum_z**2)**0.5
            all_dists.append(dist)

        print(f"\nFull path stats:")
        print(f"  Min distance: {min(all_dists):.2f}")
        print(f"  Max distance: {max(all_dists):.2f}")
        print(f"  Avg distance: {sum(all_dists)/len(all_dists):.2f}")

        if max(all_dists) > 50:
            print("\n✓ Camera shows significant movement (max > 50)")
        else:
            print(f"\n✗ Camera barely moves! (max={max(all_dists):.2f})")
    else:
        print("Schedule is sparse, would use values directly (not accumulate)")


if __name__ == "__main__":
    test_full_pipeline()
