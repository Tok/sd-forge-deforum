"""Audio synchronization handler for Deforum.

Detects audio events and distributes prompts across keyframes.
Extracted from ui_left.py to reduce complexity.
"""

import gradio as gr
import plotly.graph_objects as go
from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()


def create_empty_timeline_plot(message: str = "No data") -> go.Figure:
    """Create empty timeline plot for error/empty states.

    Args:
        message: Message to display in empty plot

    Returns:
        Empty Plotly Figure with message
    """
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
        keyframe_adjustment: Positive = more keyframes, negative = fewer keyframes
                           Used by +/- buttons to adjust target count
    """
    # Consolidated imports
    from pathlib import Path
    import json
    import librosa
    from deforum.audio import (
        parse_prompt_list,
        process_audio_for_detection,
        detect_events,
        generate_keyframes_from_events,
        distribute_prompts_across_keyframes
    )
    from deforum.utils.audio.sync import (
        calculate_keyframes_per_beat,
        calculate_bpm_based_target,
        resolve_keyframe_target,
        calculate_spacing_multiplier,
        calculate_adjusted_min_spacing,
        calculate_compensation_target,
        build_keyframe_visualization,
        build_status_message
    )

    logger.info("="*80)
    logger.info("AUDIO SYNC FUNCTION CALLED", emoji='sound')
    logger.info(f"   Soundtrack: {soundtrack_path_val}")
    logger.info(f"   Prompts: {audio_sync_prompts_val[:100]}...")
    logger.info(f"   Detection: {detection_method}, Sensitivity: {sensitivity}")
    if keyframe_adjustment != 0:
        logger.info(f"   Keyframe adjustment: {keyframe_adjustment:+d}%")
    logger.info("="*80)

    try:
        # 1. VALIDATION: Check soundtrack path
        if not soundtrack_path_val or not soundtrack_path_val.strip():
            empty_plot = create_empty_timeline_plot("Please provide a soundtrack path")
            return gr.update(), gr.update(), empty_plot, "✗ Error: Please provide a soundtrack path or URL"

        # 2. PARSE PROMPTS: Extract prompts from prompt list
        prompts = parse_prompt_list(audio_sync_prompts_val)
        if not prompts:
            empty_plot = create_empty_timeline_plot("Please enter prompts")
            return gr.update(), gr.update(), empty_plot, "✗ Error: Please enter at least one prompt"

        logger.info(f"{emoji_if_enabled('✅')} Parsed {len(prompts)} prompts from input")

        # 3. LOAD AUDIO: Download (if URL) and load audio file for analysis
        try:
            import librosa
            from deforum.media.video_audio_utilities import download_audio

            # Download audio if it's a URL (or pass through if local path)
            # This ensures we have a valid local file path for librosa
            local_audio_path = download_audio(soundtrack_path_val)

            # Load audio with librosa
            y, sr = librosa.load(local_audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            # Process audio for detection
            y_processed = process_audio_for_detection(
                y, sr,
                frequency_band=frequency_band,
                lowpass_cutoff=4000,
                distortion_gain=10.0
            )

            # Store audio data for event detection
            audio_data = {
                'audio': y_processed,
                'sr': sr,
                'duration': duration
            }

            logger.info(f"{emoji_if_enabled('✅')} Loaded audio: {duration:.2f}s at {sr}Hz")
        except Exception as e:
            empty_plot = create_empty_timeline_plot("Error loading audio")
            return gr.update(), gr.update(), empty_plot, f"✗ Error loading audio: {str(e)}"

        # 4. DETECT EVENTS: Detect beats/onsets in audio
        event_times, event_intensities = detect_events(
            audio=y_processed,
            sample_rate=sr,
            method=detection_method,
            sensitivity=sensitivity
        )

        if len(event_times) == 0:
            empty_plot = create_empty_timeline_plot("No audio events detected")
            return gr.update(), gr.update(), empty_plot, f"✗ Error: No audio events detected. Check your audio file."

        logger.info(f"{emoji_if_enabled('✅')} Detected {len(event_times)} events using {detection_method} method")

        # 5. CALCULATE TARGET: Determine how many keyframes to generate
        # (taking into account keyframe adjustment from +/- buttons)
        total_frames = int(audio_data['duration'] * current_fps)

        # Estimate BPM from event intervals
        if len(event_times) > 1:
            import numpy as np
            event_intervals = np.diff(event_times)
            median_interval = np.median(event_intervals)
            estimated_bpm = 60.0 / median_interval if median_interval > 0 else 120.0
        else:
            estimated_bpm = 120.0  # Default fallback

        # BPM-based target calculation
        keyframes_per_beat = calculate_keyframes_per_beat(estimated_bpm)
        bpm_based_target = calculate_bpm_based_target(audio_data['duration'], estimated_bpm, keyframes_per_beat)

        logger.info(f"Estimated BPM: {estimated_bpm:.1f}, keyframes_per_beat: {keyframes_per_beat}, bpm_target: {bpm_based_target}")

        # Convert target_count to int (comes from UI as string)
        user_target = int(target_count) if target_count else 0

        # Resolve target (use explicit target or BPM-based, with adjustment applied)
        resolved_target, target_desc = resolve_keyframe_target(user_target, bpm_based_target, keyframe_adjustment)
        logger.info(f"Target keyframes: {target_desc}", emoji='target')

        # 6. GENERATE KEYFRAMES: Two approaches based on whether user wants exact count
        # When user has adjusted target (via +/- buttons or explicit count), use get_n_strongest_events
        # Otherwise, use spacing-based filtering
        if keyframe_adjustment != 0 or (user_target and user_target > 0):
            # USER WANTS EXACT COUNT - select top N strongest events
            from deforum.audio import get_n_strongest_events

            logger.info(f"Using intensity-based selection for exact count: {resolved_target}")

            # Get top N strongest events (bypasses spacing restrictions)
            selected_times, selected_intensities = get_n_strongest_events(
                event_times=event_times,
                event_intensities=event_intensities,
                n=resolved_target
            )

            # Convert to keyframes (dict format matching generate_keyframes_from_events)
            keyframes = [
                {
                    'frame': int(t * current_fps),
                    'intensity': float(i),
                    'time_seconds': float(t)
                }
                for t, i in zip(selected_times, selected_intensities)
            ]

            logger.info(f"{emoji_if_enabled('✅')} Selected {len(keyframes)} strongest events")

        else:
            # DEFAULT - use spacing-based filtering (original behavior)
            spacing_multiplier = calculate_spacing_multiplier(keyframe_adjustment)
            adjusted_min_spacing = calculate_adjusted_min_spacing(min_spacing_frames, spacing_multiplier)

            # Try to hit the target by iteratively reducing spacing if needed
            max_attempts = 5
            best_keyframes = None
            best_spacing = adjusted_min_spacing

            for attempt in range(max_attempts):
                keyframes = generate_keyframes_from_events(
                    event_times=event_times,
                    event_intensities=event_intensities,
                    fps=current_fps,
                    min_spacing_frames=adjusted_min_spacing,
                    max_frames=total_frames
                )

                if not keyframes:
                    # Reduce spacing and try again
                    adjusted_min_spacing = max(1, int(adjusted_min_spacing * 0.7))
                    continue

                best_keyframes = keyframes
                best_spacing = adjusted_min_spacing

                # Check if we hit the target (within 10% tolerance)
                if len(keyframes) >= resolved_target * 0.9:
                    break

                # If we're significantly under target, reduce spacing
                if len(keyframes) < resolved_target * 0.9:
                    # Calculate how much we need to reduce spacing
                    ratio = len(keyframes) / resolved_target
                    adjusted_min_spacing = max(1, int(adjusted_min_spacing * ratio * 0.9))
                    logger.debug(f"Attempt {attempt + 1}: {len(keyframes)} < {resolved_target}, reducing spacing to {adjusted_min_spacing}")
                else:
                    # Close enough
                    break

            keyframes = best_keyframes
            if not keyframes:
                empty_plot = create_empty_timeline_plot("No keyframes generated")
                return gr.update(), gr.update(), empty_plot, "✗ Error: No keyframes generated after filtering. Try reducing min spacing."

            logger.info(f"{emoji_if_enabled('✅')} Generated {len(keyframes)} keyframes with spacing ≥{best_spacing} frames")

        # Final validation
        if not keyframes:
            empty_plot = create_empty_timeline_plot("No keyframes generated")
            return gr.update(), gr.update(), empty_plot, "✗ Error: No keyframes generated. Try adjusting detection settings."

        # 8. DISTRIBUTE PROMPTS: Assign prompts to keyframes
        # Returns JSON string ready for Deforum animation_prompts format
        formatted_schedule = distribute_prompts_across_keyframes(
            keyframes=keyframes,
            user_prompts=prompts,
            mode=distribution_mode
        )

        logger.info(f"{emoji_if_enabled('✅')} Distributed {len(prompts)} prompts across {len(keyframes)} keyframes")

        # 9. BUILD VISUALIZATIONS: Create both text status and interactive plot
        from deforum.utils.audio.sync import create_keyframe_timeline_plot

        # Create interactive Plotly timeline
        timeline_plot = create_keyframe_timeline_plot(
            keyframes=keyframes,
            total_frames=total_frames,
            duration=audio_data['duration'],
            fps=current_fps,
            prompts=prompts
        )

        # Build text status (simplified since we have the plot)
        avg_spacing = total_frames / len(keyframes) if keyframes else 0
        status_msg = (
            f"✓ Successfully synchronized!\n"
            f"• Audio: {audio_data['duration']:.1f}s @ {current_fps} FPS ({total_frames} frames)\n"
            f"• Detected BPM: {estimated_bpm:.1f}\n"
            f"• Events detected: {len(event_times)}\n"
            f"• Keyframes created: {len(keyframes)}\n"
            f"• Average spacing: {avg_spacing:.1f} frames (~{avg_spacing/current_fps:.2f}s)\n"
            f"• Prompts used: {len(prompts)} (mode: {distribution_mode})\n\n"
            f"See interactive timeline below for keyframe placement."
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
        logger.debug(f"   Returning timeline plot and status")

        return (
            gr.update(value=formatted_schedule),
            gr.update(value=len(keyframes)),
            gr.update(value=status_msg),
            gr.update(value=timeline_plot)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        empty_plot = create_empty_timeline_plot("Error during sync")
        return gr.update(), gr.update(), empty_plot, f"✗ Error: {str(e)}"
