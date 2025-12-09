"""Keyframes tab for Deforum UI.

Contains keyframe settings, guided images, motion schedules, CFG/seed/steps scheduling,
noise settings, coherence controls, and anti-blur settings.
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn, ToolButton
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.ui.builders import create_gr_elem, create_row
from deforum.config.defaults import get_gradio_html


def get_tab_keyframes(d, da, dloopArgs):
    """Create the Keyframes tab with all keyframe-related settings.

    Args:
        d: DeforumArgs namespace
        da: DeforumAnimArgs namespace
        dloopArgs: DeforumLoopArgs namespace

    Returns:
        dict: Component dictionary for event binding
    """
    components = {}

    with gr.TabItem(f"{emoji_utils.key()} Keyframes"):
        # NOTE: animation_mode, cadence, strength_schedule, keyframe_strength_schedule moved to top-level in ui_left.py
        # NOTE: max_frames moved to top-level in ui_left.py (with FPS and duration display)
        with FormRow():
            border = create_gr_elem(da.border)

        # GUIDED IMAGES ACCORD
        with gr.Accordion('Guided Images', open=False, elem_id='guided_images_accord') as guided_images_accord:
            # GUIDED IMAGES INFO ACCORD
            with gr.Accordion('*READ ME before you use this mode!*', open=False):
                gr.HTML(value=get_gradio_html('guided_imgs'))

            use_looper = create_row(dloopArgs.use_looper)
            init_images = create_row(dloopArgs.init_images)

            # GUIDED IMAGES SCHEDULES ACCORD
            with gr.Accordion('Guided images schedules', open=False):
                image_strength_schedule = create_row(dloopArgs.image_strength_schedule)
                image_keyframe_strength_schedule = create_row(dloopArgs.image_keyframe_strength_schedule)
                blendFactorMax = create_row(dloopArgs.blendFactorMax)
                blendFactorSlope = create_row(dloopArgs.blendFactorSlope)
                tweening_frames_schedule = create_row(dloopArgs.tweening_frames_schedule)
                color_correction_factor = create_row(dloopArgs.color_correction_factor)

        # KEYFRAME SCHEDULES - Single unified tab level
        # NOTE: Distribution promoted to main tab level - see get_tab_distribution()
        # NOTE: Shakify and Depth Warping promoted to main tab level - see get_tab_shakify() and get_tab_depth_warping()
        # NOTE: Motion moved to Camera Path tab - see get_tab_camera_path()
        with gr.Tabs():
            # SCHEDULE TABS
            # NOTE: Strength moved to main level (after animation_mode) for better visibility

            with gr.TabItem(f"{emoji_utils.scale()} CFG"):
                cfg_scale_schedule = create_row(da.cfg_scale_schedule)
                distilled_cfg_scale_schedule = create_row(da.distilled_cfg_scale_schedule)
                enable_clipskip_scheduling = create_row(da.enable_clipskip_scheduling)
                clipskip_schedule = create_row(da.clipskip_schedule)

            with gr.TabItem(f"{emoji_utils.seed()} Seed & SubSeed") as subseed_sch_tab:
                seed_behavior = create_row(d.seed_behavior)
                with FormRow() as seed_iter_N_row:
                    seed_iter_N = create_row(d.seed_iter_N)
                with FormRow(visible=False) as seed_schedule_row:
                    seed_schedule = create_gr_elem(da.seed_schedule)
                enable_subseed_scheduling, subseed_schedule, subseed_strength_schedule = create_row(
                    da, 'enable_subseed_scheduling', 'subseed_schedule', 'subseed_strength_schedule'
                )
                seed_resize_from_w, seed_resize_from_h = create_row(
                    d, 'seed_resize_from_w', 'seed_resize_from_h'
                )

            with gr.TabItem('Step'):
                enable_steps_scheduling = create_row(da.enable_steps_scheduling)
                steps_schedule = create_row(da.steps_schedule)

            with gr.TabItem('Sampler'):
                enable_sampler_scheduling = create_row(da.enable_sampler_scheduling)
                sampler_schedule = create_row(da.sampler_schedule)

            with gr.TabItem('Scheduler'):
                enable_scheduler_scheduling = create_row(da.enable_scheduler_scheduling)
                scheduler_schedule = create_row(da.scheduler_schedule)

            with gr.TabItem('Checkpoint'):
                enable_checkpoint_scheduling = create_row(da.enable_checkpoint_scheduling)
                checkpoint_schedule = create_row(da.checkpoint_schedule)

            # NOISE TAB
            with gr.TabItem(f"{emoji_utils.wave()} Noise"):
                with FormColumn() as noise_tab_column:
                    noise_type = create_row(da.noise_type)
                    noise_schedule = create_row(da.noise_schedule)
                    with FormRow() as perlin_row:
                        with FormColumn(min_width=220):
                            perlin_octaves = create_gr_elem(da.perlin_octaves)
                        with FormColumn(min_width=220):
                            perlin_persistence = create_gr_elem(da.perlin_persistence)
                            # following two params are INVISIBLE IN UI as of 21-05-23
                            perlin_w = create_gr_elem(da.perlin_w)
                            perlin_h = create_gr_elem(da.perlin_h)
                    enable_noise_multiplier_scheduling = create_row(da.enable_noise_multiplier_scheduling)
                    noise_multiplier_schedule = create_row(da.noise_multiplier_schedule)

            # COHERENCE INNER TAB
            with gr.TabItem(f"{emoji_utils.palette()} Coherence", open=False) as coherence_accord:
                gr.Markdown("""
                ## Vibrancy Preservation
                Prevents brownout and maintains brightness/saturation from frame 0.

                **How it works:**
                - Locks **brightness** and **color saturation** to frame 0
                - Allows **hue to change freely** with prompts
                - Perfect for: white bunny → red apple (both stay vibrant!)

                **Prevents:**
                - Cumulative darkening (brownout)
                - Color desaturation (everything turns gray/brown)
                - Requires zero tuning - works automatically
                """)

                enable_vibrancy_preservation = gr.Checkbox(
                    label="Enable Vibrancy Preservation",
                    value=True,
                    info="ON by default. Prevents cumulative darkening and desaturation."
                )

                vibrancy_preservation_strength = gr.Slider(
                    label="Correction Strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.7,
                    info="How aggressively to correct drift (0=off, 0.7=recommended, 1.0=maximum)"
                )

                # Blank frame handling
                gr.Markdown("---")
                gr.Markdown("### Blank Frame Handling")
                reroll_blank_frames, reroll_patience = create_row(
                    d, 'reroll_blank_frames', 'reroll_patience'
                )

                # Hidden legacy components for backwards compatibility
                color_coherence = gr.Dropdown(
                    label="Color coherence (deprecated)",
                    choices=['None', 'HSV', 'LAB', 'RGB', 'Image'],
                    value="None",
                    visible=False
                )
                color_force_grayscale = gr.Checkbox(value=False, visible=False)
                legacy_colormatch = gr.Checkbox(value=False, visible=False)
                color_coherence_image_path = gr.Textbox(value="", visible=False)
                color_coherence_video_every_N_frames = gr.Number(value=1, visible=False)
                contrast_schedule = gr.Textbox(value=da.contrast_schedule, visible=False)
                diffusion_redo = gr.Slider(value=da.diffusion_redo, visible=False)

            # ANTI BLUR TAB
            with gr.TabItem(f"{emoji_utils.broom()} Anti Blur", elem_id='anti_blur_accord') as anti_blur_tab:
                gr.Markdown("""
                **Anti-Blur (Unsharp Masking)** counteracts blur introduced by image transformations.

                **When to use:**
                - 3D mode with high `translation_z` (zoom in/out)
                - Optical flow enabled (RAFT)
                - Color coherence enabled
                - Depth warping with large movements

                **Quick settings:**
                - **Light sharpening:** amount=0.1, kernel=5, sigma=1.0
                - **Medium sharpening:** amount=0.2, kernel=5, sigma=1.0
                - **Strong sharpening:** amount=0.3, kernel=7, sigma=1.5

                *Leave amount at 0 (disabled) if you don't see blur issues.*
                """)
                amount_schedule = create_row(da.amount_schedule)
                kernel_schedule = create_row(da.kernel_schedule)
                sigma_schedule = create_row(da.sigma_schedule)
                threshold_schedule = create_row(da.threshold_schedule)

    return {k: v for k, v in {**locals(), **vars()}.items()}
