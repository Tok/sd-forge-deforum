"""Init tab for Deforum UI.

Contains audio sync, Parseq integration, image init, and video init settings.
Includes audio upload, event detection, automatic prompt synchronization, and AI prompt generation.
"""

import gradio as gr
from types import SimpleNamespace
from modules.ui_components import FormRow, FormColumn
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.ui.builders import create_gr_elem, create_row
from deforum.config.defaults import get_gradio_html
from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()



def get_tab_init(d, da, dp, dau, dv=None):
    """Create the Init tab with audio, Parseq, and image init settings.

    Args:
        d: DeforumArgs namespace
        da: DeforumAnimArgs namespace
        dp: ParseqArgs namespace
        dau: DeforumAudioArgs namespace
        dv: DeforumOutputArgs namespace (optional)

    Returns:
        dict: Component dictionary for event binding
    """
    logger.info("========== get_tab_init() CALLED ==========")
    # Import dv if not provided
    if dv is None:
        from deforum.config.args import DeforumOutputArgs
        dv = SimpleNamespace(**DeforumOutputArgs())

    logger.info("About to create Init TabItem")
    with gr.TabItem('Init'):
        logger.info("Inside Init TabItem")
        with gr.Tabs() as init_subtabs:  # First tab (Audio Sync) will be selected by default
            # AUDIO SYNC INNER-TAB - First tab (selected by default)
            with gr.Tab("Audio Sync") as audio_sync_subtab:
                gr.HTML(
                    value="<p>Audio event detection for prompt synchronization and video soundtrack. Upload audio file or enter path/URL below. Disabled when Parseq is active.</p>"
                )

                # Audio upload section
                audio_upload = gr.Audio(
                    label="🎵 Upload Audio File",
                    type="filepath",
                    sources=["upload"],
                    info="Upload MP3, WAV, FLAC, etc. File will be saved to output directory and path auto-filled below."
                )

                # Soundtrack controls (moved from Prompts tab)
                with FormRow():
                    add_soundtrack = create_gr_elem(dv.add_soundtrack)
                    soundtrack_path = create_gr_elem(dv.soundtrack_path)

                # Display for calculated audio info
                audio_info_display = gr.Textbox(
                    label="Audio Info",
                    value="",
                    interactive=False,
                    info="Audio duration and suggested max_frames (updates when you upload audio above)"
                )

                # Wire up audio upload to save file and update path
                def handle_audio_upload(audio_filepath, current_fps):
                    """Save uploaded audio to output directory and calculate suggested max_frames."""
                    if audio_filepath is None:
                        return None, "File", ""

                    import os
                    import shutil
                    from pathlib import Path

                    # Create output/audio directory
                    output_dir = Path("output/audio")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Get filename from uploaded file
                    filename = Path(audio_filepath).name
                    dest_path = output_dir / filename

                    # Copy uploaded file to output directory
                    shutil.copy2(audio_filepath, dest_path)
                    abs_path = str(dest_path.absolute())

                    # Calculate audio duration using librosa
                    try:
                        import librosa
                        import soundfile as sf

                        # Get audio duration (faster than loading full audio)
                        duration = librosa.get_duration(path=abs_path)

                        # Calculate suggested max_frames
                        # Use current_fps if provided, otherwise default to 24
                        fps = current_fps if current_fps and current_fps > 0 else 24
                        suggested_max_frames = int(duration * fps)

                        info_text = f"Duration: {duration:.2f}s | Suggested max_frames @ {fps} FPS: {suggested_max_frames}"
                    except Exception as e:
                        info_text = f"Could not calculate duration: {str(e)}"

                    # Return absolute path, set add_soundtrack to "File", and info text
                    return abs_path, "File", info_text

                # Note: fps component not accessible here - will be wired up in ui_left.py
                audio_upload.upload(
                    fn=handle_audio_upload,
                    inputs=[audio_upload, gr.Number(value=24, visible=False)],  # Placeholder for FPS
                    outputs=[soundtrack_path, add_soundtrack, audio_info_display]
                )

                gr.Markdown("---")
                gr.Markdown("### Event Detection Settings")

                # Row 1: Main toggles
                with FormRow():
                    enable_audio_sync = create_gr_elem(dau.enable_audio_sync)
                    audio_apply_to_prompts = create_gr_elem(dau.audio_apply_to_prompts)
                    audio_min_spacing_frames = create_gr_elem(dau.audio_min_spacing_frames)

                # Row 2: Detection method and processing
                with FormRow():
                    audio_detection_method = create_gr_elem(dau.audio_detection_method)
                    audio_frequency_band = create_gr_elem(dau.audio_frequency_band)
                    audio_distortion_type = create_gr_elem(dau.audio_distortion_type)

                # Row 3: Tuning parameters
                with FormRow():
                    audio_lowpass_cutoff = create_gr_elem(dau.audio_lowpass_cutoff)
                    audio_distortion_gain = create_gr_elem(dau.audio_distortion_gain)
                    audio_sensitivity = create_gr_elem(dau.audio_sensitivity)
                    audio_intensity_threshold = create_gr_elem(dau.audio_intensity_threshold)

                gr.Markdown("---")
                gr.Markdown("### 🎯 Automatic Prompt Synchronization")
                gr.Markdown(
                    "Enter your prompts below (one per line or comma-separated). Click **Synchronize** to **detect audio events and populate the Prompts tab** with your prompts distributed across detected keyframes."
                )

                # AI prompt generation controls - Subject first, then style
                with FormRow():
                    audio_ai_prompt_theme = gr.Textbox(
                        label="Subject/Theme",
                        value="bunny",
                        placeholder="e.g., bunny, dragon, landscape",
                        info="Main subject for the animation"
                    )
                    audio_ai_style = gr.Textbox(
                        label="Style (optional)",
                        value="synthwave",
                        placeholder="e.g., synthwave, cyberpunk, fantasy",
                        info="Optional visual style to apply to all prompts"
                    )
                    audio_ai_prompt_count = gr.Number(
                        label="Number of Prompts",
                        value=5,
                        precision=0,
                        minimum=1,
                        maximum=20,
                        info="How many prompts to generate"
                    )

                # Generation mode and intensity row
                with FormRow():
                    audio_ai_generation_mode = gr.Dropdown(
                        label="Generation Mode",
                        choices=[
                            "",
                            "escalating",
                            "start-to-end",
                            "varied",
                            "thematic",
                            "narrative",
                            "cyclical",
                            "random-walk",
                        ],
                        value="escalating",
                        allow_custom_value=True,
                        info="Leave empty or type custom. escalating=build intensity, start-to-end=interpolate, varied=random mix, thematic=variations, narrative=story, cyclical=loops, random-walk=related changes"
                    )
                    audio_ai_intensity = gr.Dropdown(
                        label="Intensity",
                        choices=["", "subtle", "normal", "crazy", "extreme", "chaotic", "surreal"],
                        value="crazy",
                        allow_custom_value=True,
                        info="Leave empty or type custom. subtle=minimal, normal=realistic, crazy=over-the-top, extreme=bonkers, chaotic=unpredictable, surreal=dream-like"
                    )

                # Start/End prompts (visible only in start-to-end mode)
                audio_ai_start_prompt = gr.Textbox(
                    label="Start Prompt (for start-to-end mode)",
                    value="cute bunny hopping on grass",
                    placeholder="First prompt in sequence",
                    info="Starting point for interpolation",
                    visible=False
                )
                audio_ai_end_prompt = gr.Textbox(
                    label="End Prompt (for start-to-end mode)",
                    value="crazy synthwave bunny on a motorcycle",
                    placeholder="Final prompt in sequence",
                    info="Ending point for interpolation",
                    visible=False
                )

                # Generate button with elem_classes for purple styling
                audio_ai_generate_button = gr.Button(
                    f"{emoji_utils.bulb()} Generate Prompts with local Qwen",
                    variant="primary",
                    elem_id="audio_ai_generate_button",
                    elem_classes=["slopcore-button"]
                )

                # Prompt input for auto-sync
                audio_sync_prompts = gr.Textbox(
                    label="Prompts for Synchronization",
                    lines=5,
                    value="bunny in forest\nbunny hopping\nbunny sitting\nbunny looking around",
                    placeholder="Enter prompts (one per line or comma-separated)",
                    info="These will be distributed across audio events when you click Synchronize"
                )

                # Distribution settings
                with FormRow():
                    audio_prompt_distribution_mode = gr.Dropdown(
                        label="Distribution Mode",
                        choices=["cycle", "sequential", "intensity", "random"],
                        value="sequential",
                        info="How to distribute prompts: sequential=divide evenly, cycle=repeat pattern, intensity=assign by beat strength"
                    )
                    audio_target_keyframe_count = gr.Number(
                        label="Target Keyframes (optional)",
                        value=0,
                        precision=0,
                        info="Leave at 0 for auto-detect based on audio. Or specify desired count."
                    )

                # Synchronize buttons with purple slopecore gradient styling
                gr.Markdown("**Click to detect audio events and populate the Prompts tab:**")
                with FormRow():
                    audio_sync_fewer_button = gr.Button(
                        "➖ -5% Keyframes",
                        variant="primary",
                        elem_id="audio_sync_fewer_button",
                        elem_classes=["slopcore-button"],
                        scale=1
                    )
                    audio_sync_button = gr.Button(
                        f"{emoji_utils.music()} Synchronize Audio to Keyframe Prompts",
                        variant="primary",
                        elem_id="audio_sync_button",
                        elem_classes=["slopcore-button"],
                        scale=2
                    )
                    audio_sync_more_button = gr.Button(
                        "➕ +5% Keyframes",
                        variant="primary",
                        elem_id="audio_sync_more_button",
                        elem_classes=["slopcore-button"],
                        scale=1
                    )

                # Status output
                audio_sync_status = gr.Textbox(
                    label="Sync Status",
                    value="",
                    interactive=False,
                    lines=12,
                    info="Status messages will appear here"
                )

            # PARSEQ INNER-TAB
            with gr.Tab(f"{emoji_utils.numbers()} Parseq"):
                gr.HTML(value=get_gradio_html('parseq'))
                parseq_manifest = create_row(dp.parseq_manifest)
                parseq_non_schedule_overrides = create_row(dp.parseq_non_schedule_overrides)
                parseq_use_deltas = create_row(dp.parseq_use_deltas)

            # IMAGE INIT INNER-TAB
            with gr.Tab('Image Init'):
                with FormRow():
                    with gr.Column(min_width=150):
                        use_init = create_gr_elem(d.use_init)
                    with gr.Column(min_width=150):
                        strength_0_no_init = create_gr_elem(d.strength_0_no_init)
                    with gr.Column(min_width=170):
                        strength = create_gr_elem(d.strength)  # TODO rename to init_strength
                init_image = create_row(d.init_image)
                init_image_box = create_row(d.init_image_box)

            # VIDEO INIT INNER-TAB - Hidden (deprecated - use Image Init or Parseq instead)
            with gr.Tab('Video Init', visible=False):
                video_init_path = create_row(da.video_init_path)
                with FormRow():
                    extract_from_frame = create_gr_elem(da.extract_from_frame)
                    extract_to_frame = create_gr_elem(da.extract_to_frame)
                    extract_nth_frame = create_gr_elem(da.extract_nth_frame)
                    overwrite_extracted_frames = create_gr_elem(da.overwrite_extracted_frames)
                # NOTE: use_mask_video and video_mask_path moved to dedicated Masking tab
            # NOTE: Mask Init tab moved to dedicated Masking tab

            # ZERO-HITL INNER-TAB - Last tab for one-click generation
            from .tab_zero_hitl import get_tab_zero_hitl
            zero_hitl_tab_emoji = emoji_if_enabled(emoji_utils.dice())
            zero_hitl_title = f"{zero_hitl_tab_emoji} Zero-HITL" if zero_hitl_tab_emoji else "Zero-HITL"
            logger.debug(f"Creating Zero-HITL subtab with title: {zero_hitl_title}")
            with gr.Tab(zero_hitl_title) as zero_hitl_subtab:
                logger.debug("Inside Zero-HITL gr.Tab context")
                zero_hitl_params = get_tab_zero_hitl(skip_tabitem=True)
                logger.debug(f"Zero-HITL params returned: {list(zero_hitl_params.keys()) if zero_hitl_params else 'None'}")

    # Build result dict from locals/vars
    result = {k: v for k, v in {**locals(), **vars()}.items()}

    # Merge Zero-HITL components into result
    if 'zero_hitl_params' in locals() and zero_hitl_params:
        result.update(zero_hitl_params)

    # DEBUG: Check what audio components are in locals()
    audio_component_names = [
        'audio_ai_generation_mode',
        'audio_ai_intensity',
        'audio_ai_style',
        'audio_ai_prompt_theme',
        'audio_ai_prompt_count',
        'audio_ai_start_prompt',
        'audio_ai_end_prompt',
        'audio_sync_prompts'
    ]

    local_scope = locals()
    found_components = [name for name in audio_component_names if name in local_scope]
    missing_components = [name for name in audio_component_names if name not in local_scope]

    logger.debug(f"{emoji_if_enabled('🔍')} DEBUG get_tab_init() return:")
    logger.debug(f"   Found in locals(): {found_components}")
    logger.debug(f"   Missing from locals(): {missing_components}")

    # Add found components to result
    for comp_name in found_components:
        result[comp_name] = local_scope[comp_name]

    return result
