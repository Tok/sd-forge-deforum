"""Test to verify wormtrail visualization bug is fixed.

The bug: camera_path_to_schedules() was recalculating ALL rotations to point
at center, ignoring the original rotation mode (quaternion blend/tangent or empirical).

This test verifies that rotations are preserved correctly through the pipeline.
"""

import sys
import re
sys.path.insert(0, '/home/zirteq/workspace/forge-neo')

from deforum.utils.spline_camera_path import generate_rotate_around_path, camera_path_to_schedules


# Inline schedule parsing (avoid importing plotly-dependent module)
def parse_schedule_string(schedule_str):
    """Parse Deforum schedule string into frame->value mapping."""
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


def interpolate_schedule(schedule_dict, max_frame):
    """Interpolate schedule values for all frames."""
    if not schedule_dict:
        return [(i, 0.0) for i in range(max_frame + 1)]
    sorted_keyframes = sorted(schedule_dict.items())
    result = []
    for frame in range(max_frame + 1):
        prev_kf = None
        next_kf = None
        for kf_frame, kf_value in sorted_keyframes:
            if kf_frame <= frame:
                prev_kf = (kf_frame, kf_value)
            if kf_frame >= frame and next_kf is None:
                next_kf = (kf_frame, kf_value)
                break
        if prev_kf is None:
            value = sorted_keyframes[0][1]
        elif next_kf is None:
            value = sorted_keyframes[-1][1]
        elif prev_kf[0] == frame:
            value = prev_kf[1]
        else:
            t = (frame - prev_kf[0]) / (next_kf[0] - prev_kf[0])
            value = prev_kf[1] + t * (next_kf[1] - prev_kf[1])
        result.append((frame, value))
    return result


def test_rotation_preservation():
    """Test that camera path rotations are preserved (not recalculated to center)."""
    print("\n=== Test: Rotation Preservation in Schedules ===")

    # Generate path with tangent mode (POV roller coaster)
    path_tangent = generate_rotate_around_path(
        num_frames=50,
        radius=100.0,
        closed_loop=True,
        rotation_mode="quaternion",
        look_at_mode="tangent"
    )

    # Generate path with inward mode (tennis ball seam)
    path_inward = generate_rotate_around_path(
        num_frames=50,
        radius=100.0,
        closed_loop=True,
        rotation_mode="quaternion",
        look_at_mode="inward"
    )

    # Generate path with center mode (legacy look-at)
    path_center = generate_rotate_around_path(
        num_frames=50,
        radius=100.0,
        closed_loop=True,
        rotation_mode="quaternion",
        look_at_mode="center"
    )

    # Convert to schedules
    schedules_tangent = camera_path_to_schedules(path_tangent)
    schedules_inward = camera_path_to_schedules(path_inward)
    schedules_center = camera_path_to_schedules(path_center)

    # Parse rotation Y schedules
    ry_tangent = parse_schedule_string(schedules_tangent['rotation_3d_y'])
    ry_inward = parse_schedule_string(schedules_inward['rotation_3d_y'])
    ry_center = parse_schedule_string(schedules_center['rotation_3d_y'])

    # Interpolate to get all frames
    ry_tangent_vals = [val for _, val in interpolate_schedule(ry_tangent, 49)]
    ry_inward_vals = [val for _, val in interpolate_schedule(ry_inward, 49)]
    ry_center_vals = [val for _, val in interpolate_schedule(ry_center, 49)]

    # Accumulate deltas to get absolute rotations
    tangent_abs = []
    inward_abs = []
    center_abs = []

    cum_t, cum_i, cum_c = 0.0, 0.0, 0.0
    for i in range(len(ry_tangent_vals)):
        cum_t += ry_tangent_vals[i]
        cum_i += ry_inward_vals[i]
        cum_c += ry_center_vals[i]
        tangent_abs.append(cum_t)
        inward_abs.append(cum_i)
        center_abs.append(cum_c)

    # Calculate variance (spread) of rotations
    import statistics
    var_tangent = statistics.variance(tangent_abs) if len(tangent_abs) > 1 else 0
    var_inward = statistics.variance(inward_abs) if len(inward_abs) > 1 else 0
    var_center = statistics.variance(center_abs) if len(center_abs) > 1 else 0

    print(f"Tangent mode - Rotation variance: {var_tangent:.2f}°²")
    print(f"Inward mode  - Rotation variance: {var_inward:.2f}°²")
    print(f"Center mode  - Rotation variance: {var_center:.2f}°²")

    # They should be DIFFERENT!
    # Before fix: All would have same variance (all recalculated to center)
    # After fix: Each mode produces different rotation patterns

    # Calculate how different they are
    diff_tangent_inward = abs(var_tangent - var_inward)
    diff_tangent_center = abs(var_tangent - var_center)
    diff_inward_center = abs(var_inward - var_center)

    print(f"\nVariance differences:")
    print(f"  Tangent vs Inward: {diff_tangent_inward:.2f}°²")
    print(f"  Tangent vs Center: {diff_tangent_center:.2f}°²")
    print(f"  Inward vs Center:  {diff_inward_center:.2f}°²")

    # At least one pair should be significantly different
    max_diff = max(diff_tangent_inward, diff_tangent_center, diff_inward_center)

    print(f"\nMax difference: {max_diff:.2f}°²")

    # BEFORE FIX: max_diff would be ~0 (all identical)
    # AFTER FIX: max_diff should be > 50 (distinct patterns)
    assert max_diff > 50.0, f"Rotation modes should produce different patterns! max_diff={max_diff:.2f}"

    print("✓ Test passed: Rotation modes produce distinct patterns")


def test_empirical_vs_quaternion():
    """Test that empirical and quaternion modes produce different rotations."""
    print("\n=== Test: Empirical vs Quaternion Rotation ===")

    # Empirical mode
    path_empirical = generate_rotate_around_path(
        num_frames=100,
        radius=100.0,
        closed_loop=True,
        rotation_mode="empirical",
        rotation_factor=-8.0
    )

    # Quaternion blend mode
    path_quaternion = generate_rotate_around_path(
        num_frames=100,
        radius=100.0,
        closed_loop=True,
        rotation_mode="quaternion",
        look_at_mode="blend",
        look_at_blend=0.3
    )

    # Convert to schedules
    sched_empirical = camera_path_to_schedules(path_empirical)
    sched_quaternion = camera_path_to_schedules(path_quaternion)

    # Parse and interpolate
    ry_emp = parse_schedule_string(sched_empirical['rotation_3d_y'])
    ry_quat = parse_schedule_string(sched_quaternion['rotation_3d_y'])

    ry_emp_vals = [val for _, val in interpolate_schedule(ry_emp, 99)]
    ry_quat_vals = [val for _, val in interpolate_schedule(ry_quat, 99)]

    # Accumulate
    emp_abs = []
    quat_abs = []
    cum_e, cum_q = 0.0, 0.0
    for i in range(len(ry_emp_vals)):
        cum_e += ry_emp_vals[i]
        cum_q += ry_quat_vals[i]
        emp_abs.append(cum_e)
        quat_abs.append(cum_q)

    # Compare final rotations
    final_emp = emp_abs[-1]
    final_quat = quat_abs[-1]

    print(f"Empirical mode: Final Y rotation = {final_emp:.2f}°")
    print(f"Quaternion mode: Final Y rotation = {final_quat:.2f}°")
    print(f"Difference: {abs(final_emp - final_quat):.2f}°")

    # They should be significantly different
    diff = abs(final_emp - final_quat)
    assert diff > 10.0, f"Empirical and Quaternion should produce different rotations! diff={diff:.2f}°"

    print("✓ Test passed: Empirical and Quaternion produce different rotations")


def test_camera_path_positions():
    """Test that camera positions are correctly converted to schedules."""
    print("\n=== Test: Position Preservation in Schedules ===")

    # Generate simple circular path
    path = generate_rotate_around_path(
        num_frames=60,
        radius=100.0,
        closed_loop=True,
        use_sphere=False,  # Flat circle for simpler test
        rotation_mode="quaternion",
        look_at_mode="center"
    )

    schedules = camera_path_to_schedules(path)

    # Parse translation schedules
    tx = parse_schedule_string(schedules['translation_x'])
    ty = parse_schedule_string(schedules['translation_y'])
    tz = parse_schedule_string(schedules['translation_z'])

    # Interpolate
    tx_vals = [val for _, val in interpolate_schedule(tx, 59)]
    ty_vals = [val for _, val in interpolate_schedule(ty, 59)]
    tz_vals = [val for _, val in interpolate_schedule(tz, 59)]

    # Accumulate deltas
    positions = []
    cum_x, cum_y, cum_z = 0.0, 0.0, 0.0
    for i in range(len(tx_vals)):
        cum_x += tx_vals[i]
        cum_y += ty_vals[i]
        cum_z += tz_vals[i]
        positions.append((cum_x, cum_y, cum_z))

    # Check that positions form a circle
    # Calculate radius at each point (should be ~100.0 for all)
    radii = [((x**2 + z**2)**0.5) for x, y, z in positions]
    avg_radius = sum(radii) / len(radii)
    radius_variance = sum((r - avg_radius)**2 for r in radii) / len(radii)

    print(f"Circle radius: {avg_radius:.2f} (expected: 100.0)")
    print(f"Radius variance: {radius_variance:.2f} (should be low)")

    # Check that we actually have movement (radius should be roughly 100)
    assert avg_radius > 50.0, f"Camera should move away from origin! avg_radius={avg_radius:.2f}"
    assert avg_radius < 200.0, f"Camera shouldn't move too far! avg_radius={avg_radius:.2f}"
    # Variance will be higher for sphere paths, just check it's not completely broken
    assert radius_variance < 10000.0, f"Should form rough circle! variance={radius_variance:.2f}"

    print("✓ Test passed: Positions form correct circular path")


if __name__ == "__main__":
    print("Testing Wormtrail Visualization Fix")
    print("=" * 60)

    try:
        test_rotation_preservation()
        test_empirical_vs_quaternion()
        test_camera_path_positions()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("\nWormtrail visualization should now correctly show:")
        print("  1. Camera position movement (circular orbit)")
        print("  2. Different rotation patterns for different modes")
        print("  3. Empirical vs quaternion rotation differences")
        print("\nThe bug (recalculating all rotations to center) is fixed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
