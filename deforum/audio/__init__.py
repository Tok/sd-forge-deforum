"""Audio synchronization module for Deforum.

This module provides audio event detection and keyframe generation
for syncing animations to music beats and bass kicks.

Main components:
- analysis: Event detection, beat tracking, onset detection
- processing: Signal processing (lowpass filter, distortion, normalization)
- keyframe_generation: Convert audio events to animation keyframes
- visualization: Plot waveforms and event markers
"""

from .analysis import (
    detect_events,
    detect_onsets,
    detect_beats,
    extract_bass_energy,
    detect_events_bpm_aware,
    get_n_strongest_events,
)

from .processing import (
    apply_lowpass_filter,
    apply_distortion,
    normalize_audio,
    process_audio_for_detection,
)

from .keyframe_generation import (
    generate_keyframes_from_events,
    cluster_nearby_events,
    filter_events_by_intensity,
)

from .prompt_distribution import (
    distribute_prompts_across_keyframes,
    suggest_keyframe_count_from_audio,
    parse_prompt_list,
)

__all__ = [
    # Analysis
    "detect_events",
    "detect_onsets",
    "detect_beats",
    "extract_bass_energy",
    "detect_events_bpm_aware",
    "get_n_strongest_events",
    # Processing
    "apply_lowpass_filter",
    "apply_distortion",
    "normalize_audio",
    "process_audio_for_detection",
    # Keyframe generation
    "generate_keyframes_from_events",
    "cluster_nearby_events",
    "filter_events_by_intensity",
    # Prompt distribution
    "distribute_prompts_across_keyframes",
    "suggest_keyframe_count_from_audio",
    "parse_prompt_list",
]
