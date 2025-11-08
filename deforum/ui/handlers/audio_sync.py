"""Audio synchronization handler for Deforum.

Detects audio events and distributes prompts across keyframes.
Extracted from ui_left.py to reduce complexity.
"""

import gradio as gr
from deforum.utils.system.logging import get_logger, emoji_if_enabled, emoji as emoji_utils

# Initialize logger
logger = get_logger()


def create_empty_timeline_plot(message: str = "No data"):
    """Create empty timeline plot for error/empty states.

    Args:
        message: Message to display in empty plot

    Returns:
        Empty Plotly Figure with message

    Raises:
        ImportError: If plotly is not installed
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for timeline visualization. Install with: pip install plotly")

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        font=dict(color='#CBD5E1'),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color='#94A3B8')
            )
        ],
        margin=dict(l=40, r=20, t=10, b=40),
        height=200
    )
    return fig


# ============================================================================
# PURE FUNCTIONS - Audio Sync Helpers
# ============================================================================


def _create_error_response(message: str) -> tuple:
    """Create standardized error response tuple.

    Args:
        message: Error message to display

    Returns:
        Tuple of (gr.update(), gr.update(), gr.update(value=empty_plot), error_message)
    """
    empty_plot = create_empty_timeline_plot("Error")
    cross = emoji_utils.maybe_cross()
    return gr.update(), gr.update(), gr.update(value=empty_plot), f"{cross} {message}"


def _validate_and_parse_inputs(
    soundtrack_path: str, prompts_text: str
) -> tuple:
    """Validate inputs and parse prompts.

    Args:
        soundtrack_path: Path or URL to audio file
        prompts_text: Prompts text input

    Returns:
        Tuple of (success: bool, prompts_or_error: list|tuple)
        If success is False, second value is error response tuple
        If success is True, second value is parsed prompts list

    Raises:
        None - returns error responses instead
    """
    from deforum.audio import parse_prompt_list

    # Validate soundtrack path
    if not soundtrack_path or not soundtrack_path.strip():
        return False, _create_error_response("Error: Please provide a soundtrack path or URL")

    # Parse and validate prompts
    prompts = parse_prompt_list(prompts_text)
    if not prompts:
        return False, _create_error_response("Error: Please enter at least one prompt")

    logger.info(f"{emoji_if_enabled('✅')} Parsed {len(prompts)} prompts from input")
    return True, prompts


def _load_and_process_audio(soundtrack_path: str, frequency_band: str):
    """Load audio file and process for event detection.

    Args:
        soundtrack_path: Path or URL to audio file
        frequency_band: Frequency band to analyze

    Returns:
        Tuple of (processed_audio, sample_rate, duration, local_path)

    Raises:
        Exception: If audio loading or processing fails
    """
    import librosa
    from deforum.media.video_audio_utilities import download_audio
    from deforum.audio import process_audio_for_detection

    local_audio_path = download_audio(soundtrack_path)
    y, sr = librosa.load(local_audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    y_processed = process_audio_for_detection(
        y, sr,
        frequency_band=frequency_band,
        lowpass_cutoff=4000,
        distortion_gain=10.0
    )

    logger.info(f"{emoji_if_enabled('✅')} Loaded audio: {duration:.2f}s at {sr}Hz")
    return y_processed, sr, duration, local_audio_path


def _estimate_bpm_from_events(event_times: list) -> float:
    """Estimate BPM from event intervals.

    Args:
        event_times: List of event timestamps in seconds

    Returns:
        Estimated BPM (defaults to 120.0 if insufficient data)
    """
    if len(event_times) <= 1:
        return 120.0

    import numpy as np
    event_intervals = np.diff(event_times)
    median_interval = np.median(event_intervals)
    return 60.0 / median_interval if median_interval > 0 else 120.0


def _generate_keyframes_intensity_based(
    event_times: list,
    event_intensities: list,
    target_count: int,
    fps: int
) -> list:
    """Generate keyframes by selecting N strongest events.

    Args:
        event_times: Event timestamps in seconds
        event_intensities: Event intensity values
        target_count: Number of keyframes to generate
        fps: Frames per second

    Returns:
        List of keyframe dicts with 'frame', 'intensity', 'time_seconds'
    """
    from deforum.audio import get_n_strongest_events

    logger.info(f"Using intensity-based selection for exact count: {target_count}")

    selected_times, selected_intensities = get_n_strongest_events(
        event_times=event_times,
        event_intensities=event_intensities,
        n=target_count
    )

    keyframes = [
        {
            'frame': int(t * fps),
            'intensity': float(i),
            'time_seconds': float(t)
        }
        for t, i in zip(selected_times, selected_intensities)
    ]

    logger.info(f"{emoji_if_enabled('✅')} Selected {len(keyframes)} strongest events")
    return keyframes


def _generate_keyframes_spacing_based(
    event_times: list,
    event_intensities: list,
    fps: int,
    min_spacing_frames: int,
    total_frames: int,
    target_count: int,
    keyframe_adjustment: int
) -> tuple:
    """Generate keyframes using spacing-based filtering with iterative adjustment.

    Args:
        event_times: Event timestamps in seconds
        event_intensities: Event intensity values
        fps: Frames per second
        min_spacing_frames: Minimum spacing between keyframes
        total_frames: Total frames in animation
        target_count: Target number of keyframes
        keyframe_adjustment: Keyframe count adjustment percentage

    Returns:
        Tuple of (keyframes, best_spacing_used)
    """
    from deforum.audio import generate_keyframes_from_events
    from deforum.utils.audio.sync import (
        calculate_spacing_multiplier,
        calculate_adjusted_min_spacing
    )

    spacing_multiplier = calculate_spacing_multiplier(keyframe_adjustment)
    adjusted_min_spacing = calculate_adjusted_min_spacing(min_spacing_frames, spacing_multiplier)

    max_attempts = 5
    best_keyframes = None
    best_spacing = adjusted_min_spacing

    for attempt in range(max_attempts):
        keyframes = generate_keyframes_from_events(
            event_times=event_times,
            event_intensities=event_intensities,
            fps=fps,
            min_spacing_frames=adjusted_min_spacing,
            max_frames=total_frames
        )

        if not keyframes:
            adjusted_min_spacing = max(1, int(adjusted_min_spacing * 0.7))
            continue

        best_keyframes = keyframes
        best_spacing = adjusted_min_spacing

        # Check if we hit target (within 10% tolerance)
        if len(keyframes) >= target_count * 0.9:
            break

        # If significantly under target, reduce spacing
        if len(keyframes) < target_count * 0.9:
            ratio = len(keyframes) / target_count
            adjusted_min_spacing = max(1, int(adjusted_min_spacing * ratio * 0.9))
            logger.debug(
                f"Attempt {attempt + 1}: {len(keyframes)} < {target_count}, "
                f"reducing spacing to {adjusted_min_spacing}"
            )
        else:
            break

    if best_keyframes:
        logger.info(
            f"{emoji_if_enabled('✅')} Generated {len(best_keyframes)} keyframes "
            f"with spacing ≥{best_spacing} frames"
        )

    return best_keyframes, best_spacing


def _generate_keyframes(
    keyframe_adjustment: int,
    user_target: int,
    resolved_target: int,
    event_times: list,
    event_intensities: list,
    current_fps: int,
    min_spacing_frames: int,
    total_frames: int
) -> tuple:
    """Generate keyframes using appropriate method based on user preferences.

    Args:
        keyframe_adjustment: Keyframe count adjustment percentage
        user_target: User-specified target count
        resolved_target: Resolved target count
        event_times: Event timestamps
        event_intensities: Event intensities
        current_fps: Frames per second
        min_spacing_frames: Minimum spacing
        total_frames: Total frames

    Returns:
        Tuple of (success: bool, keyframes_or_error: list|tuple)
        If success is False, second value is error response tuple
        If success is True, second value is keyframes list
    """
    # Choose generation method based on user preferences
    use_intensity_based = keyframe_adjustment != 0 or (user_target and user_target > 0)

    if use_intensity_based:
        # USER WANTS EXACT COUNT - select top N strongest events
        keyframes = _generate_keyframes_intensity_based(
            event_times, event_intensities, resolved_target, current_fps
        )
        if not keyframes:
            return False, _create_error_response(
                "Error: No keyframes generated. Try adjusting detection settings."
            )
    else:
        # DEFAULT - use spacing-based filtering with iterative adjustment
        keyframes, best_spacing = _generate_keyframes_spacing_based(
            event_times,
            event_intensities,
            current_fps,
            min_spacing_frames,
            total_frames,
            resolved_target,
            keyframe_adjustment
        )
        if not keyframes:
            return False, _create_error_response(
                "Error: No keyframes generated after filtering. Try reducing min spacing."
            )

    return True, keyframes


def _build_success_status(
    duration: float,
    fps: int,
    total_frames: int,
    estimated_bpm: float,
    num_events: int,
    num_keyframes: int,
    num_prompts: int,
    distribution_mode: str
) -> str:
    """Build success status message for audio sync.

    Args:
        duration: Audio duration in seconds
        fps: Frames per second
        total_frames: Total frames
        estimated_bpm: Estimated BPM
        num_events: Number of events detected
        num_keyframes: Number of keyframes created
        num_prompts: Number of prompts used
        distribution_mode: Prompt distribution mode

    Returns:
        Formatted status message string
    """
    avg_spacing = total_frames / num_keyframes if num_keyframes else 0
    return (
        f"✓ Successfully synchronized!\n"
        f"• Audio: {duration:.1f}s @ {fps} FPS ({total_frames} frames)\n"
        f"• Detected BPM: {estimated_bpm:.1f}\n"
        f"• Events detected: {num_events}\n"
        f"• Keyframes created: {num_keyframes}\n"
        f"• Average spacing: {avg_spacing:.1f} frames (~{avg_spacing/fps:.2f}s)\n"
        f"• Prompts used: {num_prompts} (mode: {distribution_mode})\n\n"
        f"See interactive timeline below for keyframe placement."
    )



def synchronize_prompts_to_audio(
    soundtrack_path_val,
    audio_sync_prompts_val,
    distribution_mode,
    target_count,
    detection_method,
    frequency_band,
    sensitivity,
    intensity_threshold,
    min_spacing_frames,
    current_fps,
    keyframe_adjustment=0  # ±20% adjustment for more/fewer keyframes
):
    """Detect audio events and distribute prompts across them.

    Args:
        soundtrack_path_val: Path or URL to audio file
        audio_sync_prompts_val: Prompts text input
        distribution_mode: How to distribute prompts
        target_count: Target keyframe count
        detection_method: Audio event detection method
        frequency_band: Frequency band for analysis
        sensitivity: Detection sensitivity
        intensity_threshold: Intensity threshold
        min_spacing_frames: Minimum spacing between keyframes
        current_fps: Frames per second
        keyframe_adjustment: ±% adjustment for keyframe count (used by +/- buttons)

    Returns:
        Tuple of (formatted_schedule, keyframe_count, status_msg, timeline_plot)
    """
    logger.info("="*80)
    logger.info("AUDIO SYNC FUNCTION CALLED", emoji='sound')
    logger.info(f"   Soundtrack: {soundtrack_path_val}")
    logger.info(f"   Prompts: {audio_sync_prompts_val[:100]}...")
    logger.info(f"   Detection: {detection_method}, Sensitivity: {sensitivity}")
    if keyframe_adjustment != 0:
        logger.info(f"   Keyframe adjustment: {keyframe_adjustment:+d}%")
    logger.info("="*80)

    try:
        # 1. VALIDATE AND PARSE INPUTS
        success, prompts_or_error = _validate_and_parse_inputs(
            soundtrack_path_val, audio_sync_prompts_val
        )
        if not success:
            return prompts_or_error
        prompts = prompts_or_error

        # 3. LOAD AUDIO: Download (if URL) and load audio file for analysis
        try:
            y_processed, sr, duration, local_audio_path = _load_and_process_audio(
                soundtrack_path_val, frequency_band
            )
        except Exception as e:
            return _create_error_response(f"Error loading audio: {str(e)}")

        # 4. DETECT EVENTS: Detect beats/onsets in audio
        from deforum.audio import detect_events

        event_times, event_intensities = detect_events(
            audio=y_processed,
            sample_rate=sr,
            method=detection_method,
            sensitivity=sensitivity
        )

        if len(event_times) == 0:
            return _create_error_response("Error: No audio events detected. Check your audio file.")

        logger.info(f"{emoji_if_enabled('✅')} Detected {len(event_times)} events using {detection_method} method")

        # 5. CALCULATE TARGET: Determine how many keyframes to generate
        from deforum.utils.audio.sync import (
            calculate_keyframes_per_beat,
            calculate_bpm_based_target,
            resolve_keyframe_target
        )

        total_frames = int(duration * current_fps)
        estimated_bpm = _estimate_bpm_from_events(event_times)

        keyframes_per_beat = calculate_keyframes_per_beat(estimated_bpm)
        bpm_based_target = calculate_bpm_based_target(duration, estimated_bpm, keyframes_per_beat)

        logger.info(
            f"Estimated BPM: {estimated_bpm:.1f}, keyframes_per_beat: {keyframes_per_beat}, "
            f"bpm_target: {bpm_based_target}"
        )

        user_target = int(target_count) if target_count else 0
        resolved_target, target_desc = resolve_keyframe_target(
            user_target, bpm_based_target, keyframe_adjustment
        )
        logger.info(f"Target keyframes: {target_desc}", emoji='target')

        # 6. GENERATE KEYFRAMES
        success, keyframes_or_error = _generate_keyframes(
            keyframe_adjustment,
            user_target,
            resolved_target,
            event_times,
            event_intensities,
            current_fps,
            min_spacing_frames,
            total_frames
        )
        if not success:
            return keyframes_or_error
        keyframes = keyframes_or_error

        # 7. DISTRIBUTE PROMPTS: Assign prompts to keyframes
        from deforum.audio import distribute_prompts_across_keyframes

        formatted_schedule = distribute_prompts_across_keyframes(
            keyframes=keyframes,
            user_prompts=prompts,
            mode=distribution_mode
        )

        logger.info(f"{emoji_if_enabled('✅')} Distributed {len(prompts)} prompts across {len(keyframes)} keyframes")

        # 8. BUILD VISUALIZATIONS: Create both text status and interactive plot
        from deforum.utils.audio.sync import create_keyframe_timeline_plot

        timeline_plot = create_keyframe_timeline_plot(
            keyframes=keyframes,
            total_frames=total_frames,
            duration=duration,
            fps=current_fps,
            prompts=prompts
        )

        status_msg = _build_success_status(
            duration=duration,
            fps=current_fps,
            total_frames=total_frames,
            estimated_bpm=estimated_bpm,
            num_events=len(event_times),
            num_keyframes=len(keyframes),
            num_prompts=len(prompts),
            distribution_mode=distribution_mode
        )

        logger.info("="*80)
        logger.info(f"{emoji_if_enabled('✅')} AUDIO SYNC COMPLETE")
        logger.info(f"   Keyframes: {len(keyframes)}")
        logger.info(f"   Prompts: {len(prompts)}")
        logger.info(f"   Total frames: {total_frames}")
        logger.info("="*80)

        logger.debug(f"{emoji_if_enabled('🔍')} DEBUG synchronize_prompts_to_audio return:")
        logger.debug(f"   formatted_schedule type: {type(formatted_schedule)}, length: {len(formatted_schedule)}")
        logger.debug(f"   formatted_schedule preview: {formatted_schedule[:100]}...")
        logger.debug(f"   target_count: {len(keyframes)}")

        return (
            gr.update(value=formatted_schedule),
            gr.update(value=len(keyframes)),
            gr.update(value=timeline_plot),  # 3rd: timeline plot for audio_sync_timeline
            gr.update(value=status_msg)      # 4th: status message for audio_sync_status
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _create_error_response(f"Error: {str(e)}")
