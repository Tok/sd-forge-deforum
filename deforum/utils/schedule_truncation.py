"""Schedule Truncation Utilities

Functions to downsample/truncate long schedule strings for UI display while preserving
full schedules in settings.json.
"""

from typing import Dict, Set
import re


def downsample_schedule_for_display(
    schedule_str: str,
    target_max_frames: int = 1000,
    total_frames: int = None
) -> str:
    """Downsample schedule string by keeping every Nth frame to target max frames.

    Instead of truncating at N frames, this intelligently samples across the full
    timeline to give a representative view of the entire animation.

    Args:
        schedule_str: Full schedule string
        target_max_frames: Target number of frames to display
        total_frames: Total frames in animation (auto-detected if None)

    Returns:
        Downsampled schedule string with note about downsampling
    """
    if not schedule_str or not schedule_str.strip():
        return schedule_str

    # Parse all frame entries
    pattern = r'(\d+)\s*:\s*\(([^)]+)\)'
    matches = re.findall(pattern, schedule_str)

    if not matches or len(matches) <= target_max_frames:
        # No downsampling needed
        return schedule_str

    # Get all frame numbers
    frame_numbers = [int(frame) for frame, _ in matches]
    max_frame = max(frame_numbers)

    if total_frames is None:
        total_frames = max_frame + 1

    # Calculate downsample rate to target ~target_max_frames
    downsample_rate = max(1, len(matches) // target_max_frames)

    # Build downsampled set: every Nth frame + first + last
    downsampled_indices = set()
    for i in range(0, len(matches), downsample_rate):
        downsampled_indices.add(i)
    downsampled_indices.add(0)  # Always keep first
    downsampled_indices.add(len(matches) - 1)  # Always keep last

    # Extract downsampled entries
    downsampled_entries = []
    for idx in sorted(downsampled_indices):
        frame, value = matches[idx]
        downsampled_entries.append(f"{frame}: ({value})")

    result = ', '.join(downsampled_entries)
    result += f" ... [downsampled: showing {len(downsampled_entries)} of {len(matches)} keyframes, full schedule in settings.json]"

    return result


def truncate_schedule_for_display(
    schedule_str: str,
    max_frames: int = 1000
) -> str:
    """Truncate schedule string to display only first N frames.

    Parses Deforum schedule format like "0: (10), 50: (20), 100: (30)"
    and keeps only entries where frame number <= max_frames.

    Args:
        schedule_str: Full schedule string
        max_frames: Maximum frame number to display

    Returns:
        Truncated schedule string with ellipsis indicator if truncated
    """
    if not schedule_str or not schedule_str.strip():
        return schedule_str

    # Split by commas to get individual frame entries
    entries = [entry.strip() for entry in schedule_str.split(',')]

    # Filter entries where frame number <= max_frames
    truncated_entries = []
    last_frame = 0
    was_truncated = False

    for entry in entries:
        # Parse frame number (format: "frame: (value)")
        if ':' not in entry:
            continue

        try:
            frame_str = entry.split(':')[0].strip()
            frame_num = int(frame_str)

            if frame_num <= max_frames:
                truncated_entries.append(entry)
                last_frame = frame_num
            else:
                was_truncated = True
                break
        except (ValueError, IndexError):
            # If parsing fails, include entry as-is
            truncated_entries.append(entry)

    # Reconstruct schedule string
    result = ', '.join(truncated_entries)

    # Add ellipsis indicator if truncated
    if was_truncated:
        result += f" ... [truncated at frame {last_frame}, full schedule in settings.json]"

    return result


def truncate_schedules_dict(
    schedules: Dict[str, str],
    max_frames: int = 1000
) -> Dict[str, str]:
    """Truncate all schedules in dictionary for display.

    Args:
        schedules: Dict of schedule strings (tx, ty, tz, rx, ry, rz)
        max_frames: Maximum frame number to display

    Returns:
        Dict with truncated schedule strings
    """
    return {
        key: truncate_schedule_for_display(value, max_frames)
        for key, value in schedules.items()
    }


def should_truncate_schedules(num_frames: int, threshold: int = 1000) -> bool:
    """Check if schedules should be truncated based on frame count.

    Args:
        num_frames: Total number of frames
        threshold: Truncation threshold

    Returns:
        True if num_frames exceeds threshold
    """
    return num_frames > threshold


def get_truncation_info_message(num_frames: int, max_display: int) -> str:
    """Generate informative message about schedule truncation.

    Args:
        num_frames: Total number of frames
        max_display: Maximum frames displayed

    Returns:
        Formatted info message
    """
    return (
        f"⚠️ Schedule Display Truncated\n\n"
        f"Total frames: {num_frames:,}\n"
        f"Displayed: {max_display:,} frames (first portion only)\n\n"
        f"✅ Full schedules are generated and will be saved to settings.json\n"
        f"⚠️ UI textboxes show truncated preview to prevent browser freeze\n\n"
        f"To adjust truncation threshold, see Settings → Deforum → Max Schedule Display Frames"
    )
