"""Test what visualization actually sees from schedule strings."""

import sys
import re
sys.path.insert(0, '/home/zirteq/workspace/forge-neo')

from deforum.utils.spline_camera_path import generate_rotate_around_path, camera_path_to_schedules


def parse_schedule_string(schedule_str: str):
    """Same parsing logic as visualization."""
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


# Generate camera path
camera_path = generate_rotate_around_path(
    num_frames=20,
    radius=100.0,
    height=0.0,
    closed_loop=True,
    rotation_mode="quaternion",
    look_at_mode="center",
    look_at_blend=0.3
)

# Convert to schedules
schedules = camera_path_to_schedules(
    camera_path,
    speed_multiplier=1.0,
    speed_randomization=0.0,
    random_seed=0,
    look_at_mode="center"
)

print("=== SCHEDULE STRINGS (raw) ===\n")
print(f"translation_x: {schedules['translation_x'][:200]}...")
print(f"translation_y: {schedules['translation_y'][:200]}...")
print(f"rotation_3d_y: {schedules['rotation_3d_y'][:200]}...")

print("\n=== PARSED SCHEDULES ===\n")

tx_dict = parse_schedule_string(schedules['translation_x'])
ty_dict = parse_schedule_string(schedules['translation_y'])
tz_dict = parse_schedule_string(schedules['translation_z'])

print(f"Translation X: {len(tx_dict)} entries")
print(f"Translation Y: {len(ty_dict)} entries")
print(f"Translation Z: {len(tz_dict)} entries")

if len(tx_dict) == 0:
    print("\n❌ TRANSLATION SCHEDULES ARE EMPTY!")
    print("This would cause wormtrail to show no movement!")
else:
    print(f"\nFirst 5 translation X values: {[tx_dict.get(i, 0.0) for i in range(5)]}")
    print(f"First 5 translation Y values: {[ty_dict.get(i, 0.0) for i in range(5)]}")

    # Check if they're all zeros
    tx_nonzero = sum(1 for v in tx_dict.values() if abs(v) > 0.01)
    ty_nonzero = sum(1 for v in ty_dict.values() if abs(v) > 0.01)

    if tx_nonzero == 0 and ty_nonzero == 0:
        print("\n❌ ALL TRANSLATION VALUES ARE ZERO!")
        print("This would cause wormtrail to show no movement!")
    else:
        print(f"\n✓ Translation X has {tx_nonzero}/{len(tx_dict)} non-zero values")
        print(f"✓ Translation Y has {ty_nonzero}/{len(ty_dict)} non-zero values")
