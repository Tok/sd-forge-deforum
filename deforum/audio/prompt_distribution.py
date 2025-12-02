"""Distribute user prompts across audio-detected keyframes."""

from typing import List, Dict
import json


def distribute_prompts_across_keyframes(
    keyframes: List[Dict],
    user_prompts: List[str],
    mode: str = "cycle"
) -> str:
    """Distribute user prompts across detected audio keyframes.

    Args:
        keyframes: List of keyframe dicts from generate_keyframes_from_events()
                  Each has 'frame', 'intensity', 'time_seconds' keys
        user_prompts: List of prompt strings without frame numbers
                     e.g., ["bunny in forest", "bunny hopping", "bunny sitting"]
        mode: Distribution mode:
              - "cycle": Cycle through prompts repeatedly
              - "random": Random prompt assignment
              - "intensity": Higher intensity frames get earlier prompts
              - "sequential": Divide keyframes into N sections, one prompt per section

    Returns:
        JSON string in Deforum animation_prompts format:
        '{"0": "bunny in forest", "24": "bunny hopping", ...}'

    Example:
        >>> keyframes = [
        ...     {'frame': 0, 'intensity': 1.0, 'time_seconds': 0.0},
        ...     {'frame': 24, 'intensity': 0.8, 'time_seconds': 1.0},
        ...     {'frame': 48, 'intensity': 0.9, 'time_seconds': 2.0}
        ... ]
        >>> prompts = ["bunny in forest", "bunny hopping"]
        >>> schedule = distribute_prompts_across_keyframes(keyframes, prompts)
        >>> print(schedule)
        '{"0": "bunny in forest", "24": "bunny hopping", "48": "bunny in forest"}'
    """
    if not keyframes or not user_prompts:
        return "{}"

    schedule = {}
    num_prompts = len(user_prompts)
    num_keyframes = len(keyframes)

    # Ensure frame 0 always has a prompt (critical for generation start)
    has_frame_zero = any(kf['frame'] == 0 for kf in keyframes)
    if not has_frame_zero:
        # Add frame 0 with first prompt
        schedule["0"] = user_prompts[0]

    if mode == "cycle":
        # Cycle through prompts repeatedly
        for i, kf in enumerate(keyframes):
            prompt_idx = i % num_prompts
            schedule[str(kf['frame'])] = user_prompts[prompt_idx]

    elif mode == "sequential":
        # Divide keyframes into N equal sections, one prompt per section
        keyframes_per_prompt = max(1, num_keyframes // num_prompts)
        for i, kf in enumerate(keyframes):
            prompt_idx = min(i // keyframes_per_prompt, num_prompts - 1)
            schedule[str(kf['frame'])] = user_prompts[prompt_idx]

    elif mode == "intensity":
        # Sort keyframes by intensity (descending)
        # Assign earlier prompts to higher intensity keyframes
        sorted_kf = sorted(keyframes, key=lambda x: x['intensity'], reverse=True)
        kf_to_prompt = {}
        for i, kf in enumerate(sorted_kf):
            prompt_idx = min(i % num_prompts, num_prompts - 1)
            kf_to_prompt[kf['frame']] = user_prompts[prompt_idx]

        # Build schedule in frame order
        for kf in keyframes:
            schedule[str(kf['frame'])] = kf_to_prompt[kf['frame']]

    elif mode == "random":
        import random
        for kf in keyframes:
            schedule[str(kf['frame'])] = random.choice(user_prompts)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return json.dumps(schedule, indent=2)


def suggest_keyframe_count_from_audio(
    duration_seconds: float,
    fps: float,
    desired_prompts: int = None,
    min_spacing_seconds: float = 0.5,
    max_keyframes: int = 50
) -> int:
    """Suggest number of keyframes based on audio duration and user preferences.

    Args:
        duration_seconds: Audio duration in seconds
        fps: Animation frames per second
        desired_prompts: Number of unique prompts user wants to use
        min_spacing_seconds: Minimum time between keyframes
        max_keyframes: Maximum number of keyframes to generate

    Returns:
        Suggested number of keyframes

    Example:
        >>> suggest_keyframe_count_from_audio(180.0, 60, desired_prompts=5)
        20  # 20 keyframes = 4 repetitions of 5 prompts
    """
    # Calculate maximum possible keyframes given spacing constraint
    max_possible = int(duration_seconds / min_spacing_seconds)

    if desired_prompts:
        # Aim for multiple repetitions of the prompt set
        # Typically 3-5 repetitions works well
        target_repetitions = 4
        suggested = desired_prompts * target_repetitions
        suggested = min(suggested, max_possible, max_keyframes)
    else:
        # No preference - use a reasonable density
        # Aim for ~1 keyframe every 2-3 seconds
        suggested = int(duration_seconds / 2.5)
        suggested = min(suggested, max_possible, max_keyframes)

    return max(1, suggested)


def parse_prompt_list(prompt_text: str) -> List[str]:
    """Parse newline or comma-separated prompt list into list of strings.

    Args:
        prompt_text: Multi-line or comma-separated prompt text

    Returns:
        List of cleaned prompt strings

    Example:
        >>> parse_prompt_list("bunny in forest\\nbunny hopping\\nbunny sitting")
        ['bunny in forest', 'bunny hopping', 'bunny sitting']
        >>> parse_prompt_list("bunny in forest, bunny hopping, bunny sitting")
        ['bunny in forest', 'bunny hopping', 'bunny sitting']
    """
    # Try newline-separated first
    if '\n' in prompt_text:
        prompts = [p.strip() for p in prompt_text.split('\n') if p.strip()]
    else:
        # Fall back to comma-separated
        prompts = [p.strip() for p in prompt_text.split(',') if p.strip()]

    return prompts
