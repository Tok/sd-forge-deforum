"""Audio synchronization handler for Deforum.

Detects audio events and distributes prompts across keyframes.
Extracted from ui_left.py to reduce complexity.
"""

import gradio as gr
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()



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
            return gr.update(), gr.update(), "✗ Error: Please provide a soundtrack path or URL"

        # 2. PARSE PROMPTS: Extract prompts from prompt list
        prompts = parse_prompt_list(audio_sync_prompts_val)
        if not prompts:
            return gr.update(), gr.update(), "✗ Error: Please enter at least one prompt"

        logger.info(f"✅ Parsed {len(prompts)} prompts from input")

        # 3. LOAD AUDIO: Process audio file for analysis
        try:
            audio_data = process_audio_for_detection(soundtrack_path_val)
            logger.info(f"✅ Loaded audio: {audio_data['duration']:.2f}s at {audio_data['sr']}Hz")
        except Exception as e:
            return gr.update(), gr.update(), f"✗ Error loading audio: {str(e)}"

        # 4. DETECT EVENTS: Detect beats/onsets in audio
        events = detect_events(
            audio_data=audio_data,
            method=detection_method,
            frequency_band=frequency_band,
            sensitivity=sensitivity,
            intensity_threshold=intensity_threshold
        )

        if not events:
            return gr.update(), gr.update(), f"✗ Error: No audio events detected. Check your audio file."

        logger.info(f"✅ Detected {len(events)} events using {detection_method} method")

        # 5. CALCULATE TARGET: Determine how many keyframes to generate
        # (taking into account keyframe adjustment from +/- buttons)
        total_frames = int(audio_data['duration'] * current_fps)

        # BPM-based target calculation (if applicable)
        keyframes_per_beat = calculate_keyframes_per_beat(current_fps, events, audio_data['duration'])
        bpm_based_target = calculate_bpm_based_target(events, audio_data['duration'], keyframes_per_beat)

        # Resolve target (use explicit target or BPM-based)
        resolved_target = resolve_keyframe_target(distribution_mode, target_count, bpm_based_target)

        # Apply adjustment from +/- buttons (±5%)
        if keyframe_adjustment != 0:
            adjusted_target = int(resolved_target * (1 + keyframe_adjustment / 100))
            adjusted_target = max(2, min(adjusted_target, len(events)))  # Clamp to valid range
            logger.info(f"Adjusted target: {resolved_target} → {adjusted_target} ({keyframe_adjustment:+d}%)", emoji='wrench')
            resolved_target = adjusted_target

        # 6. GENERATE KEYFRAMES: Convert events to keyframes with spacing
        spacing_multiplier = calculate_spacing_multiplier(resolved_target, len(events))
        adjusted_min_spacing = calculate_adjusted_min_spacing(min_spacing_frames, spacing_multiplier)

        keyframes = generate_keyframes_from_events(
            events=events,
            fps=current_fps,
            min_spacing=adjusted_min_spacing,
            target_count=resolved_target
        )

        if not keyframes:
            return gr.update(), gr.update(), "✗ Error: No keyframes generated after filtering. Try reducing min spacing."

        logger.info(f"✅ Generated {len(keyframes)} keyframes with spacing ≥{adjusted_min_spacing} frames")

        # 7. COMPENSATE FOR LOST KEYFRAMES: If we lost too many keyframes due to spacing,
        #    try again with reduced spacing
        if len(keyframes) < resolved_target * 0.7:  # Lost >30% of keyframes
            compensation_target = calculate_compensation_target(resolved_target, len(keyframes))
            compensated_spacing = int(adjusted_min_spacing * 0.5)  # Reduce spacing by 50%

            logger.warning(f"⚠️ Compensation triggered: {len(keyframes)} < {resolved_target * 0.7:.0f}")
            logger.info(f"   Retrying with target={compensation_target}, spacing={compensated_spacing}")

            keyframes = generate_keyframes_from_events(
                events=events,
                fps=current_fps,
                min_spacing=compensated_spacing,
                target_count=compensation_target
            )

            if keyframes:
                logger.info(f"✅ Compensation successful: {len(keyframes)} keyframes generated")

        # 8. DISTRIBUTE PROMPTS: Assign prompts to keyframes
        prompt_assignments = distribute_prompts_across_keyframes(
            keyframes=keyframes,
            prompts=prompts,
            distribution_mode=distribution_mode
        )

        logger.info(f"✅ Distributed {len(prompts)} prompts across {len(keyframes)} keyframes")

        # 9. FORMAT OUTPUT: Convert to Deforum schedule format
        schedule_dict = {kf: prompt for kf, prompt in prompt_assignments.items()}
        formatted_schedule = json.dumps(schedule_dict, indent=None)

        # 10. BUILD STATUS MESSAGE: Create visualization and status
        visualization = build_keyframe_visualization(keyframes, total_frames)
        status_msg = build_status_message(
            keyframes=keyframes,
            prompts=prompts,
            distribution_mode=distribution_mode,
            detection_method=detection_method,
            total_frames=total_frames,
            visualization=visualization
        )

        logger.info("="*80)
        logger.info("✅ AUDIO SYNC COMPLETE")
        logger.info(f"   Keyframes: {len(keyframes)}")
        logger.info(f"   Prompts: {len(prompts)}")
        logger.info(f"   Total frames: {total_frames}")
        logger.info("="*80)

        logger.info(f"🔍 DEBUG synchronize_prompts_to_audio return:")
        logger.info(f"   formatted_schedule type: {type(formatted_schedule)}, length: {len(formatted_schedule)}")
        logger.info(f"   formatted_schedule preview: {formatted_schedule[:100]}...")
        logger.info(f"   target_count: {len(keyframes)}")
        logger.info(f"   status_msg length: {len(status_msg)} chars")
        logger.info(f"   status_msg first line: {status_msg.split(chr(10))[0]}")

        return (
            gr.update(value=formatted_schedule),
            gr.update(value=len(keyframes)),
            gr.update(value=status_msg)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return gr.update(), gr.update(), f"✗ Error: {str(e)}"
