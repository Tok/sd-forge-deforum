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
        with FormRow():
            border = create_gr_elem(da.border)
            max_frames = create_gr_elem(da.max_frames)

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
                color_coherence, color_force_grayscale = create_row(
                    da, 'color_coherence', 'color_force_grayscale'
                )
                gr.Markdown("""
                **Color Coherence Modes:**
                - **None:** No color matching (default)
                - **HSV:** Good for preserving vibrant colors and saturation
                - **LAB:** Most perceptually accurate for human vision
                - **RGB:** Simple channel-by-channel matching
                - **Image:** Match colors to a reference image (requires image path below)

                *Use when frames drift in color/tone. Works best with 3D mode.*
                """)
                legacy_colormatch = create_row(da.legacy_colormatch)
                with FormRow(visible=False) as color_coherence_image_path_row:
                    color_coherence_image_path = create_gr_elem(da.color_coherence_image_path)
                with FormRow(visible=False) as color_coherence_video_every_N_frames_row:
                    color_coherence_video_every_N_frames = create_gr_elem(da.color_coherence_video_every_N_frames)
                # NOTE: Optical flow settings moved to 3D Depth tab
                with FormRow():
                    contrast_schedule = gr.Textbox(
                        label="Contrast schedule",
                        lines=1,
                        value=da.contrast_schedule,
                        interactive=True,
                        info="""adjusts the overall contrast per frame
                            [neutral at 1.0, recommended to *not* play with this param]"""
                    )
                    diffusion_redo = gr.Slider(
                        label="Redo generation",
                        minimum=0,
                        maximum=50,
                        step=1,
                        value=da.diffusion_redo,
                        interactive=True,
                        info="""this option renders N times before the final render.
                            it is suggested to lower your steps if you up your redo.
                            seed is randomized during redo generations and restored afterwards"""
                    )

                # what to do with blank frames (they may result from glitches or the NSFW filter being turned on):
                # reroll with +1 seed, interrupt the animation generation, or do nothing
                reroll_blank_frames, reroll_patience = create_row(
                    d, 'reroll_blank_frames', 'reroll_patience'
                )

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
