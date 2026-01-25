"""Diagnostic to check what schedules are being generated and how they're visualized."""

import sys
sys.path.insert(0, '/home/zirteq/workspace/forge-neo')

from deforum.utils.spline_camera_path import generate_rotate_around_path, camera_path_to_schedules


def diagnose_schedules():
    """Check what schedules look like for a simple rotate-around path."""
    print("\n=== Diagnostic: Schedule Generation ===")

    # Generate simple rotate-around path with CENTER mode
    camera_path = generate_rotate_around_path(
        num_frames=20,  # Small for easy inspection
        radius=100.0,
        closed_loop=True,
        rotation_mode="quaternion",
        look_at_mode="center",  # Test center mode specifically
        look_at_blend=0.3
    )

    print(f"\nGenerated {len(camera_path)} camera points")
    print("\nFirst 3 camera points:")
    for i in range(min(3, len(camera_path))):
        p = camera_path[i]
        print(f"  Frame {i}: pos=({p.x:.2f}, {p.y:.2f}, {p.z:.2f}), rot=({p.rot_x:.2f}, {p.rot_y:.2f}, {p.rot_z:.2f})")

    # Convert to schedules (pass look_at_mode for center mode)
    schedules = camera_path_to_schedules(camera_path, look_at_mode="center")

    print("\n=== Generated Schedules ===")
    for key, value in schedules.items():
        print(f"\n{key}:")
        # Show first 200 chars
        print(f"  {value[:200]}...")

        # Parse to check values
        import re
        pattern = r'(\d+)\s*:\s*\(\s*(-?\d+\.?\d*)\s*\)'
        matches = re.findall(pattern, value)
        if len(matches) > 0:
            values = [float(v) for _, v in matches[:5]]
            print(f"  First 5 values: {values}")
            print(f"  Total keyframes: {len(matches)}")

    # Check if translation schedules are actually moving
    import re
    pattern = r'(\d+)\s*:\s*\(\s*(-?\d+\.?\d*)\s*\)'
    tx_matches = re.findall(pattern, schedules['translation_x'])
    ty_matches = re.findall(pattern, schedules['translation_y'])
    tz_matches = re.findall(pattern, schedules['translation_z'])

    tx_values = [float(v) for _, v in tx_matches]
    ty_values = [float(v) for _, v in ty_matches]
    tz_values = [float(v) for _, v in tz_matches]

    print("\n=== Delta Analysis ===")
    print(f"Translation X deltas: min={min(tx_values):.2f}, max={max(tx_values):.2f}, non-zero={sum(1 for v in tx_values if abs(v) > 0.01)}/{len(tx_values)}")
    print(f"Translation Y deltas: min={min(ty_values):.2f}, max={max(ty_values):.2f}, non-zero={sum(1 for v in ty_values if abs(v) > 0.01)}/{len(ty_values)}")
    print(f"Translation Z deltas: min={min(tz_values):.2f}, max={max(tz_values):.2f}, non-zero={sum(1 for v in tz_values if abs(v) > 0.01)}/{len(tz_values)}")

    # Accumulate to see positions
    cum_x, cum_y, cum_z = 0.0, 0.0, 0.0
    positions = []
    for i in range(len(tx_values)):
        cum_x += tx_values[i]
        cum_y += ty_values[i]
        cum_z += tz_values[i]
        positions.append((cum_x, cum_y, cum_z))

    print("\n=== Accumulated Positions (first 5) ===")
    for i in range(min(5, len(positions))):
        x, y, z = positions[i]
        print(f"  Frame {i}: ({x:.2f}, {y:.2f}, {z:.2f})")

    # Check if positions are actually moving away from origin
    distances = [(x**2 + y**2 + z**2)**0.5 for x, y, z in positions]
    print(f"\nDistance from origin: min={min(distances):.2f}, max={max(distances):.2f}, avg={sum(distances)/len(distances):.2f}")

    if max(distances) < 10.0:
        print("\n⚠️  WARNING: Camera barely moves! Positions are all near origin!")
        print("   This would explain why wormtrail looks like rotation only.")
    else:
        print("\n✓ Camera positions show movement (max distance > 10)")


if __name__ == "__main__":
    diagnose_schedules()
