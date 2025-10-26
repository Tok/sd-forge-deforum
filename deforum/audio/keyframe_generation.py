"""Convert audio events to animation keyframes with spacing constraints."""

from typing import List, Dict, Tuple
import numpy as np


def filter_events_by_intensity(
    event_times: np.ndarray,
    event_intensities: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter events by intensity threshold.

    Args:
        event_times: Event times in seconds
        event_intensities: Event intensities (0-1)
        threshold: Minimum intensity (0-1)

    Returns:
        Tuple of (filtered_times, filtered_intensities)
    """
    mask = event_intensities >= threshold
    return event_times[mask], event_intensities[mask]


def cluster_nearby_events(
    event_times: np.ndarray,
    event_intensities: np.ndarray,
    min_spacing_seconds: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster nearby events to enforce minimum spacing.

    When multiple events occur within min_spacing, keep only the strongest.

    Args:
        event_times: Event times in seconds
        event_intensities: Event intensities (0-1)
        min_spacing_seconds: Minimum time between events

    Returns:
        Tuple of (clustered_times, clustered_intensities)
    """
    if len(event_times) == 0:
        return event_times, event_intensities

    # Sort by time
    sort_idx = np.argsort(event_times)
    times_sorted = event_times[sort_idx]
    intensities_sorted = event_intensities[sort_idx]

    # Keep first event
    kept_times = [times_sorted[0]]
    kept_intensities = [intensities_sorted[0]]

    for time, intensity in zip(times_sorted[1:], intensities_sorted[1:]):
        # Check spacing from last kept event
        if time - kept_times[-1] >= min_spacing_seconds:
            kept_times.append(time)
            kept_intensities.append(intensity)
        else:
            # Replace with stronger event if current is stronger
            if intensity > kept_intensities[-1]:
                kept_times[-1] = time
                kept_intensities[-1] = intensity

    return np.array(kept_times), np.array(kept_intensities)


def generate_keyframes_from_events(
    event_times: np.ndarray,
    event_intensities: np.ndarray,
    fps: float = 24.0,
    max_frames: int = 240,
    min_spacing_frames: int = 12,
    intensity_threshold: float = 0.5
) -> List[Dict]:
    """Generate animation keyframes from audio events.

    Args:
        event_times: Event times in seconds
        event_intensities: Event intensities (0-1)
        fps: Animation frames per second
        max_frames: Maximum frame number for animation
        min_spacing_frames: Minimum frames between keyframes
        intensity_threshold: Minimum event intensity to generate keyframe

    Returns:
        List of keyframe dicts with 'frame' and 'intensity' keys

    Example:
        >>> keyframes = generate_keyframes_from_events(
        ...     times, intensities, fps=24, max_frames=240
        ... )
        >>> for kf in keyframes[:3]:
        ...     print(f"Frame {kf['frame']}: intensity={kf['intensity']:.2f}")
    """
    # Filter by intensity
    times, intensities = filter_events_by_intensity(
        event_times, event_intensities, intensity_threshold
    )

    # Enforce minimum spacing in seconds
    min_spacing_seconds = min_spacing_frames / fps
    times, intensities = cluster_nearby_events(
        times, intensities, min_spacing_seconds
    )

    # Convert to frames
    keyframes = []
    for time, intensity in zip(times, intensities):
        frame = int(round(time * fps))

        # Skip if at or beyond max_frames (max_frames is the total count, frames are 0-indexed)
        if frame >= max_frames:
            continue

        keyframes.append({
            'frame': frame,
            'intensity': float(intensity),
            'time_seconds': float(time)
        })

    # Sort by frame number
    keyframes.sort(key=lambda x: x['frame'])

    # Ensure frame 0 is included
    if not keyframes or keyframes[0]['frame'] != 0:
        keyframes.insert(0, {
            'frame': 0,
            'intensity': 1.0,
            'time_seconds': 0.0
        })

    return keyframes


def keyframes_to_schedule_string(
    keyframes: List[Dict],
    base_prompt: str = "scene"
) -> str:
    """Convert keyframes to Deforum animation_prompts format.

    Args:
        keyframes: List of keyframe dicts from generate_keyframes_from_events
        base_prompt: Base prompt text to use for all keyframes

    Returns:
        JSON string in Deforum format: {"0": "prompt", "frame": "prompt", ...}

    Example:
        >>> schedule = keyframes_to_schedule_string(
        ...     keyframes, base_prompt="cyberpunk city"
        ... )
        >>> print(schedule)
        '{"0": "cyberpunk city", "37": "cyberpunk city", "65": "cyberpunk city"}'
    """
    import json

    schedule = {}
    for kf in keyframes:
        schedule[str(kf['frame'])] = base_prompt

    return json.dumps(schedule)
