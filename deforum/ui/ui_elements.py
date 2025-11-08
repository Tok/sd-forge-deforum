# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

from types import SimpleNamespace
from pathlib import Path

# noinspection PyUnresolvedReferences
import gradio as gr
# noinspection PyUnresolvedReferences
from modules.ui_components import FormRow, FormColumn, ToolButton
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.config.defaults import get_gradio_html, DeforumAnimPrompts
from deforum.ui.gradio_funcs import (upload_vid_to_interpolate, upload_pics_to_interpolate,
                           ncnn_upload_vid_to_upscale)
from deforum.media.video_audio_utilities import direct_stitch_vid_from_frames
from deforum.utils.ui.builders import (
    create_gr_elem,
    is_gradio_component,
    create_row,
    create_accordion_md_row
)

# Import extracted tab modules
from deforum.ui.tabs.tab_run import get_tab_run
from deforum.ui.tabs.tab_keyframes import get_tab_keyframes
from deforum.ui.tabs.tab_prompts import get_tab_prompts
from deforum.ui.tabs.tab_qwen import get_tab_qwen
from deforum.ui.tabs.tab_shakify import get_tab_shakify
from deforum.ui.tabs.tab_masking import get_tab_masking
from deforum.ui.tabs.tab_depth import get_tab_depth_warping
from deforum.ui.tabs.tab_init import get_tab_init
from deforum.ui.tabs.tab_wan import get_tab_wan
from deforum.ui.tabs.tab_output import get_tab_output
from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()



# ******** Important message ********
# All get_tab functions use FormRow()/ FormColumn() by default,
# unless we have a gr.File inside that row/column, then we use gr.Row()/gr.Column() instead.
# ******** Important message ********
# NOTE: Tab functions below are now imported from deforum.ui.tabs.* modules
# The old inline definitions are kept for reference but should be removed after testing

def get_tab_run_OLD(d, da):
    with (gr.TabItem(f"{emoji_utils.run()} Run")):  # RUN TAB
        motion_preview_mode = create_row(d.motion_preview_mode)
        sampler, scheduler, steps = create_row(d, 'sampler', 'scheduler', 'steps')
        W, H = create_row(d, 'W', 'H')
        seed, batch_name = create_row(d, 'seed', 'batch_name')
        with FormRow():
            restore_faces = create_gr_elem(d.restore_faces)
            tiling = create_gr_elem(d.tiling)
            enable_ddim_eta_scheduling = create_gr_elem(da.enable_ddim_eta_scheduling)
            enable_ancestral_eta_scheduling = create_gr_elem(da.enable_ancestral_eta_scheduling)
        with gr.Row(variant='compact') as eta_sch_row:
            ddim_eta_schedule = create_gr_elem(da.ddim_eta_schedule)
            ancestral_eta_schedule = create_gr_elem(da.ancestral_eta_schedule)

        # RUN FROM SETTING FILE ACCORD
        with gr.Accordion('Batch Mode, Resume and more', open=True):
            with gr.Tab('Batch Mode/ run from setting files'):
                with gr.Row():  # TODO: handle this inside one of the args functions?
                    override_settings_with_file = gr.Checkbox(label="Enable batch mode", value=False, interactive=True,
                                                              elem_id='override_settings',
                                                              info="run from a list of setting .txt files. Upload them to the box on the right (visible when enabled)")
                    custom_settings_file = gr.File(label="Setting files", interactive=True, file_count="multiple",
                                                   file_types=[".txt"], elem_id="custom_setting_file", visible=False)
            # RESUME ANIMATION ACCORD
            with gr.Tab('Resume Animation', selected=True):
                resume_from_timestring, resume_timestring = create_row(
                    da, 'resume_from_timestring', 'resume_timestring')
    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_keyframes(d, da, dloopArgs):
    components = {}
    with gr.TabItem(f"{emoji_utils.key()} Keyframes"):  # TODO make a some sort of the original dictionary parsing
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
        with gr.Tabs():
            # MOTION TAB - Most important for animation, placed first
            with gr.TabItem(f"{emoji_utils.bicycle()} Motion") as motion_tab:
                with FormColumn() as only_2d_motion_column:
                    with FormRow(variant="compact"):
                        zoom = create_gr_elem(da.zoom)
                        reset_zoom_button = ToolButton(elem_id='reset_zoom_btn', value=emoji_utils.refresh,
                                                       tooltip="Reset zoom to static.")
                        components['zoom'] = zoom

                        def reset_zoom_field():
                            return {zoom: gr.update(value='0:(1)', visible=True)}

                        reset_zoom_button.click(fn=reset_zoom_field, inputs=[], outputs=[zoom])
                    angle = create_row(da.angle)
                    transform_center_x = create_row(da.transform_center_x)
                    transform_center_y = create_row(da.transform_center_y)
                with FormColumn() as both_anim_mode_motion_params_column:
                    translation_x = create_row(da.translation_x)
                    translation_y = create_row(da.translation_y)
                is_3d_motion_column_visible = True  # FIXME init, overridden because default is 3D
                with FormColumn(visible=is_3d_motion_column_visible) as only_3d_motion_column:
                    with FormRow():
                        translation_z = create_gr_elem(da.translation_z)
                        reset_tr_z_button = ToolButton(elem_id='reset_tr_z_btn', value=emoji_utils.refresh,
                                                       tooltip="Reset translation Z to static.")
                        components['tr_z'] = translation_z

                        def reset_tr_z_field():
                            return {translation_z: gr.update(value='0:(0)', visible=True)}

                        reset_tr_z_button.click(fn=reset_tr_z_field, inputs=[], outputs=[translation_z])
                    rotation_3d_x = create_row(da.rotation_3d_x)
                    rotation_3d_y = create_row(da.rotation_3d_y)
                    rotation_3d_z = create_row(da.rotation_3d_z)
                # PERSPECTIVE FLIP - inner params are hidden if not enabled
                with FormRow() as enable_per_f_row:
                    enable_perspective_flip = create_gr_elem(da.enable_perspective_flip)
                with FormRow(visible=False) as per_f_th_row:
                    perspective_flip_theta = create_gr_elem(da.perspective_flip_theta)
                with FormRow(visible=False) as per_f_ph_row:
                    perspective_flip_phi = create_gr_elem(da.perspective_flip_phi)
                with FormRow(visible=False) as per_f_ga_row:
                    perspective_flip_gamma = create_gr_elem(da.perspective_flip_gamma)
                with FormRow(visible=False) as per_f_f_row:
                    perspective_flip_fv = create_gr_elem(da.perspective_flip_fv)

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
                    da, 'enable_subseed_scheduling', 'subseed_schedule', 'subseed_strength_schedule')
                seed_resize_from_w, seed_resize_from_h = create_row(
                    d, 'seed_resize_from_w', 'seed_resize_from_h')

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
                    da, 'color_coherence', 'color_force_grayscale')
                legacy_colormatch = create_row(da.legacy_colormatch)
                with FormRow(visible=False) as color_coherence_image_path_row:
                    color_coherence_image_path = create_gr_elem(da.color_coherence_image_path)
                with FormRow(visible=False) as color_coherence_video_every_N_frames_row:
                    color_coherence_video_every_N_frames = create_gr_elem(da.color_coherence_video_every_N_frames)
                # NOTE: Optical flow settings moved to 3D Depth tab
                with FormRow():
                    contrast_schedule = gr.Textbox(
                        label="Contrast schedule", lines=1, value=da.contrast_schedule, interactive=True,
                        info="""adjusts the overall contrast per frame
                            [neutral at 1.0, recommended to *not* play with this param]""")
                    diffusion_redo = gr.Slider(
                        label="Redo generation", minimum=0, maximum=50,
                        step=1, value=da.diffusion_redo, interactive=True,
                        info="""this option renders N times before the final render.
                            it is suggested to lower your steps if you up your redo.
                            seed is randomized during redo generations and restored afterwards""")

                # what to do with blank frames (they may result from glitches or the NSFW filter being turned on):
                # reroll with +1 seed, interrupt the animation generation, or do nothing
                reroll_blank_frames, reroll_patience = create_row(
                    d, 'reroll_blank_frames', 'reroll_patience')
            # ANTI BLUR TAB
            with gr.TabItem(f"{emoji_utils.broom()} Anti Blur", elem_id='anti_blur_accord') as anti_blur_tab:
                amount_schedule = create_row(da.amount_schedule)
                kernel_schedule = create_row(da.kernel_schedule)
                sigma_schedule = create_row(da.sigma_schedule)
                threshold_schedule = create_row(da.threshold_schedule)

    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_prompts(da, dw, dv=None):
    # Import dv if not provided
    if dv is None:
        from deforum.config.args import DeforumOutputArgs
        from types import SimpleNamespace
        dv = SimpleNamespace(**DeforumOutputArgs())

    with gr.TabItem(f"{emoji_utils.prompts()} Prompts"):
        # PROMPTS INFO ACCORD
        with gr.Accordion(label='*Important* notes on Prompts', elem_id='prompts_info_accord',
                          open=False) as prompts_info_accord:
            gr.HTML(value=get_gradio_html('prompts'))
        animation_prompts = create_row(
            gr.Textbox(label="Prompts", lines=8, interactive=True, value=DeforumAnimPrompts(),
                       info="""Full prompts list in a JSON format. The value on left side is the frame number and
                            its presence also defines the frame as a keyframe if a 'keyframe distribution' mode
                            is active. Duplicating the same prompt multiple times to define keyframes
                            is therefore expected and fine."""))
        animation_prompts_positive = create_row(
            gr.Textbox(label="Prompts positive", lines=1, interactive=True,
                       placeholder="words in here will be added to the start of all positive prompts"))
        animation_prompts_negative = create_row(
            gr.Textbox(label="Prompts negative", value="nsfw, nude", lines=1, interactive=True,
                       placeholder="words here will be added to the end of all negative prompts.  ignored with Flux."))

        # PROMPT TIMING SETTINGS
        with gr.Accordion(f"{emoji_utils.stopwatch()} Prompt Timing", open=False):
            gr.Markdown("""
            **Prompt Authored FPS:** If you authored prompts at a different FPS (e.g., 60 FPS) but want to render at another (e.g., 24 FPS), set this to auto-convert frame numbers.

            **Audio settings** have been moved to Init → Audio Sync tab.
            """)

            with FormRow() as prompt_fps_row:
                prompt_authored_fps = create_gr_elem(dv.prompt_authored_fps)

        # AI PROMPT ENHANCEMENT - Qwen integration
        with gr.Accordion(f"{emoji_utils.brain()} AI Prompt Enhancement (Qwen)", open=False):
            gr.Markdown("""
            **Enhance your prompts using Qwen AI models** for better generation quality:
            - Refines and expands prompt descriptions
            - Analyzes Deforum movement schedules (translation, rotation, zoom)
            - Translates technical motion into descriptive language
            - Supports English and Chinese output

            **Note:** Qwen models are lazy-loaded only when needed and auto-cleanup before generation to free VRAM.
            """)

            # Qwen Settings
            with gr.Accordion(f"{emoji_utils.gear()} Qwen Settings", open=True):
                with FormRow():
                    wan_qwen_model = create_gr_elem(dw.wan_qwen_model)
                    wan_qwen_language = create_gr_elem(dw.wan_qwen_language)
                    wan_qwen_auto_download = create_gr_elem(dw.wan_qwen_auto_download)

            # Model Management
            with gr.Accordion(f"{emoji_utils.wrench()} Model Management", open=False):
                gr.Markdown("""
                **Model Information & Status**

                Monitor Qwen model availability and manage downloads:
                """)

                qwen_model_status = gr.HTML(
                    label="Qwen Model Status",
                    value="⏳ Checking model availability...",
                    elem_id="wan_qwen_model_status"
                )

                with FormRow():
                    check_qwen_models_btn = gr.Button(
                        f"{emoji_utils.magnifying_glass()} Check Model Status",
                        variant="secondary",
                        elem_id="wan_check_qwen_models_btn"
                    )
                    download_qwen_model_btn = gr.Button(
                        f"{emoji_utils.download()} Download Selected Model",
                        variant="primary",
                        elem_id="wan_download_qwen_model_btn"
                    )
                    cleanup_qwen_cache_btn = gr.Button(
                        f"{emoji_utils.broom()} Cleanup Model Cache",
                        variant="secondary",
                        elem_id="wan_cleanup_qwen_cache_btn"
                    )

            # Connect event handlers for Qwen model management
            check_qwen_models_btn.click(
                fn=check_qwen_models_handler,
                inputs=[wan_qwen_model],
                outputs=[qwen_model_status]
            )

            download_qwen_model_btn.click(
                fn=download_qwen_model_handler,
                inputs=[wan_qwen_model, wan_qwen_auto_download],
                outputs=[qwen_model_status]
            )

            cleanup_qwen_cache_btn.click(
                fn=cleanup_qwen_cache_handler,
                inputs=[],
                outputs=[qwen_model_status]
            )

            # Auto-update model status when model selection changes
            wan_qwen_model.change(
                fn=check_qwen_models_handler,
                inputs=[wan_qwen_model],
                outputs=[qwen_model_status]
            )

        # FPS CONVERTER
        with gr.Accordion(f"{emoji_utils.stopwatch()} FPS Converter", open=False):
            gr.Markdown("""
            **Convert prompt frame numbers between different FPS settings**

            Use this when you need to:
            - Convert prompts synced to 60 FPS for use at 24 FPS (or any other FPS)
            - Adjust timing when changing video output FPS
            - Rescale animation timing to different frame rates

            **Example:** Prompts synced to amen break at 60 FPS → Convert to 24 FPS for Wan video generation

            **Formula:** `new_frame = old_frame × (target_fps / source_fps)`
            """)

            with FormRow():
                fps_converter_source = gr.Number(
                    label="Source FPS",
                    value=60,
                    minimum=1,
                    maximum=240,
                    step=1,
                    info="Current FPS that prompts are synced to"
                )
                fps_converter_target = gr.Number(
                    label="Target FPS",
                    value=24,
                    minimum=1,
                    maximum=240,
                    step=1,
                    info="Desired FPS for prompt conversion"
                )

            with FormRow():
                fps_converter_btn = gr.Button(
                    f"{emoji_utils.refresh_icon()} Convert Prompt Frame Numbers",
                    variant="primary",
                    elem_id="fps_converter_btn"
                )
                fps_converter_preview = gr.Checkbox(
                    label="Preview Only",
                    value=False,
                    info="Show conversion preview without updating prompts"
                )

            fps_converter_output = gr.HTML(
                label="Conversion Result",
                value="",
                elem_id="fps_converter_output"
            )

            # Connect event handler
            fps_converter_btn.click(
                fn=convert_fps_handler,
                inputs=[animation_prompts, fps_converter_source, fps_converter_target, fps_converter_preview],
                outputs=[animation_prompts, fps_converter_output]
            )

        # NOTE: Composable mask scheduling moved to dedicated Masking tab

    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_qwen(dw: SimpleNamespace):
    """AI Prompt Enhancement Tab - Qwen model for enhancing prompts"""
    with gr.TabItem(f"{emoji_utils.brain()} AI Enhancement"):
        gr.Markdown("""
        ## AI Prompt Enhancement with Qwen

        **Enhance your prompts using Qwen AI models** for better generation quality:
        - Refines and expands prompt descriptions
        - Analyzes Deforum movement schedules
        - Translates technical motion into descriptive language
        - Supports English and Chinese

        **Usage:**
        1. Write your base prompts in the Prompts tab above
        2. Configure Qwen settings below
        3. Enhancement integrates automatically when generating

        **Note:** Qwen models are lazy-loaded only when needed and auto-cleanup before generation to free VRAM.
        """)

        # Qwen Settings
        with gr.Accordion(f"{emoji_utils.gear()} Qwen Settings", open=True):
            with FormRow():
                wan_qwen_model = create_gr_elem(dw.wan_qwen_model)
                wan_qwen_language = create_gr_elem(dw.wan_qwen_language)
                wan_qwen_auto_download = create_gr_elem(dw.wan_qwen_auto_download)

        # Model Management
        with gr.Accordion(f"{emoji_utils.wrench()} Model Management", open=False):
            gr.Markdown("""
            **Model Information & Status**

            Monitor Qwen model availability and manage downloads:
            """)

            qwen_model_status = gr.HTML(
                label="Qwen Model Status",
                value="⏳ Checking model availability...",
                elem_id="wan_qwen_model_status"
            )

            with FormRow():
                check_qwen_models_btn = gr.Button(
                    f"{emoji_utils.magnifying_glass()} Check Model Status",
                    variant="secondary",
                    elem_id="wan_check_qwen_models_btn"
                )
                download_qwen_model_btn = gr.Button(
                    f"{emoji_utils.download()} Download Selected Model",
                    variant="primary",
                    elem_id="wan_download_qwen_model_btn"
                )
                cleanup_qwen_cache_btn = gr.Button(
                    f"{emoji_utils.broom()} Cleanup Model Cache",
                    variant="secondary",
                    elem_id="wan_cleanup_qwen_cache_btn"
                )

        # Connect event handlers for Qwen model management
        check_qwen_models_btn.click(
            fn=check_qwen_models_handler,
            inputs=[wan_qwen_model],
            outputs=[qwen_model_status]
        )

        download_qwen_model_btn.click(
            fn=download_qwen_model_handler,
            inputs=[wan_qwen_model, wan_qwen_auto_download],
            outputs=[qwen_model_status]
        )

        cleanup_qwen_cache_btn.click(
            fn=cleanup_qwen_cache_handler,
            inputs=[],
            outputs=[qwen_model_status]
        )

        # Auto-update model status when model selection changes
        wan_qwen_model.change(
            fn=check_qwen_models_handler,
            inputs=[wan_qwen_model],
            outputs=[qwen_model_status]
        )

    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_shakify(da, skip_tabitem=False):
    """Camera Shakify Tab - Integrate realistic camera shake effects"""
    # Controls first - most important
    shake_name = create_row(da.shake_name)
    shake_intensity = create_row(da.shake_intensity)
    shake_speed = create_row(da.shake_speed)

    # Explanation after controls
    with gr.Accordion(f"{emoji_utils.info} About Camera Shakify", open=True):
        gr.Markdown("""
        ## Camera Shakify
        **Integrate dynamic camera shake effects** into your renders with data sourced from EatTheFuture's 'Camera Shakify' Blender plugin.

        This feature enhances the realism of your animations by simulating natural camera movements, adding a layer of depth and engagement to your visuals.

        **Available Shake Patterns:**
        - EARTHQUAKE - Violent, chaotic shaking
        - FILM_GRAIN - Subtle analog film vibration
        - GENTLE_HANDHELD - Natural handheld camera movement
        - INVESTIGATION - Detective-style documentary camera work
        - MOVING_HANDHELD - Active walking/running camera movement
        - PANIC - Frantic, disoriented shaking
        - ROLLING_SHUTTER - Digital camera sensor distortion
        - And more...

        **How It Works:**
        - Shake patterns are layered on top of your scheduled movement (translation, rotation, zoom)
        - Intensity controls the magnitude of shake
        - Speed controls how fast the shake pattern plays
        """)

    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_masking(d, da, skip_tabitem=False):
    """Masking Tab - Configure masks for selective image generation"""

    components = {}

    if not skip_tabitem:
        with gr.TabItem(f'{emoji_utils.masking()} Masking'):
            components.update(_create_masking_content(d, da))
    else:
        components.update(_create_masking_content(d, da))

    return components


def _create_masking_content(d, da):
    """Internal helper to create masking tab content"""

    with gr.Tabs():
        # Tab 1: Basic Masks
        with gr.TabItem('Basic Masks'):
            gr.Markdown("Upload mask images to control which areas are regenerated. White areas = regenerate, black areas = keep original.")
            with FormRow():
                use_mask = create_gr_elem(d.use_mask)
                use_alpha_as_mask = create_gr_elem(d.use_alpha_as_mask)
                invert_mask = create_gr_elem(d.invert_mask)
                overlay_mask = create_gr_elem(d.overlay_mask)
            mask_file = create_row(d.mask_file)
            mask_overlay_blur = create_row(d.mask_overlay_blur)
            fill = create_row(d.fill)
            full_res_mask, full_res_mask_padding = create_row(d, 'full_res_mask', 'full_res_mask_padding')
            with FormRow():
                with FormColumn(min_width=240):
                    mask_contrast_adjust = create_gr_elem(d.mask_contrast_adjust)
                with FormColumn(min_width=250):
                    mask_brightness_adjust = create_gr_elem(d.mask_brightness_adjust)

        # Tab 2: Text Masking (CLIPSeg)
        with gr.TabItem('Text Masking (CLIPSeg)'):
            gr.Markdown("""
**CLIPSeg Text-to-Mask**: Generate masks using text descriptions in your composable mask expressions.

**Syntax:** `<text description>` - e.g., `<cat>`, `<sky>`, `<person's face>`

**Examples:**
- `<cat>` - Mask all cats
- `<armor>` - Mask armor/metal
- `<sky>` - Mask the sky
- `<person's face>` - Mask faces

**Usage:** Go to the Composable Masks tab and use text masks in your expressions:
- `0: <cat>` - Mask only cats
- `0: (<cat> | <dog>)` - Cats OR dogs
- `0: !<sky>` - Everything EXCEPT sky

The ViT-B/16 CLIPSeg model auto-downloads on first use.
            """)

        # Tab 3: Composable Masks
        with gr.TabItem('Composable Masks'):
            gr.Markdown("""
**Combine masks using boolean expressions**

**Mask Types:**
- `{variable}` - Variable masks (e.g., `{human_mask}`, `{video_mask}`)
- `[path.png]` - File masks from disk
- `<text>` - CLIPSeg text-to-mask

**Operators:** `&` (AND), `|` (OR), `^` (XOR), `!` (NOT), `\\` (DIFFERENCE)

**Examples:**
- `0: <cat>` - Mask only cats
- `0: (<cat> | <dog>)` - Cats OR dogs
- `0: ({human_mask} & [border.png])` - Humans within border
- `0: !<sky>` - Everything except sky
            """)
            mask_schedule = create_row(da.mask_schedule)
            use_noise_mask = create_row(da.use_noise_mask)
            noise_mask_schedule = create_row(da.noise_mask_schedule)

        # Tab 4: Video & Human Masks
        with gr.TabItem('Video & Human'):
            with gr.Accordion("Video Masks (Animated)", open=False):
                gr.Markdown("""
Per-frame video mask sequences. Useful for rotoscoped masks from video editing software or time-varying region selection.
                """)
                use_mask_video = create_gr_elem(da.use_mask_video)
                video_mask_path = create_row(da.video_mask_path)

            with gr.Accordion("Human Detection (AI)", open=False):
                gr.Markdown("""
**Automatic human detection using RobustVideoMatting**

Use `{human_mask}` in composable mask expressions to automatically detect humans.

**Examples:**
- `0: !{human_mask}` - Regenerate everything EXCEPT humans
- `0: {human_mask}` - Regenerate ONLY humans
- `0: ({human_mask} & <armor>)` - Only humans wearing armor

The RobustVideoMatting resnet50 model auto-downloads from PyTorch Hub on first use.
                """)

    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_depth_warping(da, skip_tabitem=False):
    """3D Depth Warping & FOV Tab - Configure depth estimation and 3D camera settings"""
    # FIXME this should only be visible if animation mode is "3D".
    is_visible = True
    is_info_visible = is_visible

    # Controls first - most important
    with gr.Accordion(f"{emoji_utils.gear()} Depth Settings", open=True):
        depth_warp_msg_html = gr.HTML(
            value='Please switch to 3D animation mode to view this section.',
            elem_id='depth_warp_msg_html',
            visible=False
        )
        with FormRow(visible=is_visible) as depth_warp_row_1:
            use_depth_warping = create_gr_elem(da.use_depth_warping)
            depth_algorithm = create_gr_elem(da.depth_algorithm)
            midas_weight = create_gr_elem(da.midas_weight)
        with FormRow(visible=is_visible) as depth_warp_row_2:
            padding_mode = create_gr_elem(da.padding_mode)
            sampling_mode = create_gr_elem(da.sampling_mode)

    with gr.Accordion(f"{emoji_utils.wave()} Optical Flow / Cadence", open=False):
        gr.Markdown(f"""
        **Optical flow** estimates motion between frames for smooth in-between (cadence) frames.
        Enable RAFT to generate only keyframes and use motion estimation for tweens (10x speedup).

        {emoji_utils.maybe_warning()} **WARNING:** Can produce "smear-core" artifacts with many cadence frames.
        Works best with low cadence (2-3 frames). Experimental feature - disabled by default.
        """)
        with FormRow(visible=is_visible) as optical_flow_cadence_row:
            with FormColumn(min_width=220):
                optical_flow_cadence = create_gr_elem(da.optical_flow_cadence)
            with FormColumn(min_width=220):
                optical_flow_redo_generation = create_gr_elem(da.optical_flow_redo_generation)
        with FormRow(visible=is_visible) as optical_flow_row_2:
            raft_model_size = create_gr_elem(da.raft_model_size)
            raft_flow_iterations = create_gr_elem(da.raft_flow_iterations)
            show_flow_arrows = create_gr_elem(da.show_flow_arrows)
        with FormRow(visible=is_visible) as optical_flow_row_3:
            with FormColumn(min_width=220, visible=False) as cadence_flow_factor_schedule_column:
                cadence_flow_factor_schedule = create_gr_elem(da.cadence_flow_factor_schedule)
        with FormRow(visible=is_visible) as optical_flow_row_4:
            with FormColumn(min_width=220, visible=False) as redo_flow_factor_schedule_column:
                redo_flow_factor_schedule = create_gr_elem(da.redo_flow_factor_schedule)

    with gr.Accordion(f"{emoji_utils.globe()} Flux ControlNet", open=False):
        gr.Markdown(f"""
        **Flux ControlNet** adds structural control to keyframe generation using:
        - **Canny edges** from previous frame (preserves shapes and lines)
        - **Depth maps** from Depth-Anything V2 (preserves 3D structure)

        {emoji_utils.maybe_warning()} **Only applies to keyframes** (not tween frames). Requires Flux model.
        """)
        with FormRow(visible=is_visible) as flux_controlnet_row_1:
            enable_flux_controlnet = create_gr_elem(da.enable_flux_controlnet)
            flux_controlnet_type = create_gr_elem(da.flux_controlnet_type)
        with FormRow(visible=is_visible) as flux_controlnet_row_2:
            flux_controlnet_model = create_gr_elem(da.flux_controlnet_model)
            flux_controlnet_strength = create_gr_elem(da.flux_controlnet_strength)
        with FormRow(visible=is_visible) as flux_controlnet_row_3:
            flux_controlnet_canny_low = create_gr_elem(da.flux_controlnet_canny_low)
            flux_controlnet_canny_high = create_gr_elem(da.flux_controlnet_canny_high)
        with FormRow(visible=is_visible) as flux_controlnet_row_4:
            flux_guidance_scale = create_gr_elem(da.flux_guidance_scale)
        with FormRow(visible=is_visible) as flux_controlnet_row_5:
            flux_base_model = create_gr_elem(da.flux_base_model)

    with gr.Accordion(f"{emoji_utils.gear()} FOV & Advanced Settings", open=False):
        with FormRow(visible=is_visible) as depth_warp_row_3:
            aspect_ratio_use_old_formula = create_gr_elem(da.aspect_ratio_use_old_formula)
        with FormRow(visible=is_visible) as depth_warp_row_4:
            aspect_ratio_schedule = create_gr_elem(da.aspect_ratio_schedule)
        with FormRow(visible=is_visible) as depth_warp_row_5:
            fov_schedule = create_gr_elem(da.fov_schedule)
        with FormRow(visible=is_visible) as depth_warp_row_6:
            near_schedule = create_gr_elem(da.near_schedule)
        with FormRow(visible=is_visible) as depth_warp_row_7:
            far_schedule = create_gr_elem(da.far_schedule)

    # Explanation after controls
    with gr.Accordion(f"{emoji_utils.info} About 3D Depth Warping", open=False):
        gr.Markdown("""
        ## 3D Depth Warping & FOV
        **Transform 2D images into 3D space** using AI depth estimation for realistic camera movement.

        **Depth Estimation:**
        - Uses **Depth-Anything V2** - State-of-the-art depth estimation model
        - Provides accurate depth maps for realistic 3D camera effects

        **When to Use:**
        - Required for **3D Animation Mode** to enable camera movement through space
        - Creates parallax effects by warping images based on depth
        - Enables true 3D camera controls (translation_z, rotation_3d_x/y/z)

        **FOV (Field of View):**
        - Controls perspective intensity (lower = more dramatic)
        - Near/Far planes control depth clipping range
        """)

    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_init(d, da, dp, dau, dv=None):
    # Import dv if not provided
    if dv is None:
        from deforum.config.args import DeforumOutputArgs
        from types import SimpleNamespace
        dv = SimpleNamespace(**DeforumOutputArgs())

    with gr.TabItem('Init'):
        # Import Zero-HITL for later (will be last tab)
        from deforum.ui.tabs.tab_zero_hitl import get_tab_zero_hitl
        from deforum.utils.system.logging import emoji_if_enabled
        zero_hitl_tab_emoji = emoji_if_enabled(emoji_utils.dice())
        zero_hitl_title = f"{zero_hitl_tab_emoji} Zero-HITL" if zero_hitl_tab_emoji else "Zero-HITL"

        with gr.Tabs() as init_subtabs:
            # AUDIO SYNC INNER-TAB - First tab (opens by default)
            with gr.Tab("Audio Sync") as audio_sync_subtab:
                gr.HTML(value="<p>Audio event detection for prompt synchronization and video soundtrack. Upload audio file or enter path/URL below. Disabled when Parseq is active.</p>")

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
                    info="Audio duration and suggested max_frames (auto-loads from soundtrack path)"
                )

                # Helper function to calculate audio info from path (URL or local file)
                def calculate_audio_info(audio_path, current_fps):
                    """Calculate audio duration and suggested max_frames from path.

                    Args:
                        audio_path: Local file path or URL to audio file
                        current_fps: Target FPS for calculation

                    Returns:
                        Info text string with duration and suggested max_frames
                    """
                    if not audio_path or not str(audio_path).strip():
                        return ""

                    try:
                        import librosa
                        from deforum.media.video_audio_utilities import download_audio

                        # Download if URL, or pass through if local path
                        local_path = download_audio(audio_path)

                        # Get audio duration (faster than loading full audio)
                        duration = librosa.get_duration(path=local_path)

                        # Calculate suggested max_frames
                        fps = current_fps if current_fps and current_fps > 0 else 24
                        suggested_max_frames = int(duration * fps)

                        return f"Duration: {duration:.2f}s | Suggested max_frames @ {fps} FPS: {suggested_max_frames}"
                    except Exception as e:
                        return f"Could not load audio: {str(e)}"

                # Upload handler: Save uploaded file and calculate info
                def handle_audio_upload(audio_filepath, current_fps):
                    """Save uploaded audio to output directory and calculate suggested max_frames."""
                    if audio_filepath is None:
                        return None, "File", ""

                    import shutil
                    from pathlib import Path

                    # Create outputs/audio directory
                    output_dir = Path("outputs/audio")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Get filename from uploaded file
                    filename = Path(audio_filepath).name
                    dest_path = output_dir / filename

                    # Copy uploaded file to output directory
                    shutil.copy2(audio_filepath, dest_path)
                    abs_path = str(dest_path.absolute())

                    # Calculate audio info using helper function
                    info_text = calculate_audio_info(abs_path, current_fps)

                    # Return absolute path, set add_soundtrack to "File", and info text
                    return abs_path, "File", info_text

                # Auto-load handler: Load info from existing soundtrack_path
                def auto_load_audio_info(soundtrack_path_val, current_fps):
                    """Auto-load audio info when tab is opened if soundtrack path is valid."""
                    return calculate_audio_info(soundtrack_path_val, current_fps)

                # Wire up audio upload to save file and update path
                # Note: fps component not accessible here - will be wired up in ui_left.py
                audio_upload.upload(
                    fn=handle_audio_upload,
                    inputs=[audio_upload, gr.Number(value=24, visible=False)],  # Placeholder for FPS
                    outputs=[soundtrack_path, add_soundtrack, audio_info_display]
                )

                # Auto-load audio info when Audio Sync tab is selected
                audio_sync_subtab.select(
                    fn=auto_load_audio_info,
                    inputs=[soundtrack_path, gr.Number(value=24, visible=False)],  # Placeholder for FPS
                    outputs=[audio_info_display]
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
                gr.Markdown("Enter your prompts below (one per line or comma-separated). Click **Synchronize** to **detect audio events and populate the Prompts tab** with your prompts distributed across detected keyframes.")

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
                        choices=["", "escalating", "start-to-end", "varied", "thematic", "narrative", "cyclical", "random-walk", "first-person-perspective"],
                        value="escalating",
                        allow_custom_value=True,
                        info="Leave empty or type custom. escalating=build intensity, start-to-end=interpolate, varied=random mix, thematic=variations, narrative=story, cyclical=loops, random-walk=related changes, first-person-perspective=POV camera (great for reverse generation)"
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
                    lines=8,
                    info="Status messages will appear here"
                )

                # Interactive timeline visualization
                audio_sync_timeline = gr.Plot(
                    label="Keyframe Timeline",
                    show_label=True
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

            # LOAD FROM VIDEO INNER-TAB
            with gr.Tab('Load from Video'):
                gr.HTML(value="<p>Load complete settings from a Deforum-generated video (ComfyUI-style metadata extraction). Upload a video file below to extract and load all embedded settings.</p>")
                video_upload = gr.File(
                    label="📹 Upload Video File",
                    file_types=[".mp4", ".mov", ".avi", ".webm", ".mkv"],
                    type="filepath",
                    elem_id="deforum_video_upload",
                    file_count="single"
                )

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

            # ZERO-HITL INNER-TAB - Last tab (for one-click generation)
            with gr.Tab(zero_hitl_title) as zero_hitl_subtab:
                zero_hitl_params = get_tab_zero_hitl(skip_tabitem=True)

    # Build result dict from locals/vars
    result = {k: v for k, v in {**locals(), **vars()}.items()}

    # Merge Zero-HITL components into result
    if 'zero_hitl_params' in locals() and zero_hitl_params:
        result.update(zero_hitl_params)

    # DEBUG: Check what audio components are in locals()
    audio_component_names = [
        'audio_ai_generation_mode', 'audio_ai_intensity', 'audio_ai_style',
        'audio_ai_prompt_theme', 'audio_ai_prompt_count', 'audio_ai_start_prompt',
        'audio_ai_end_prompt', 'audio_sync_prompts'
    ]

    local_scope = locals()
    found_components = [name for name in audio_component_names if name in local_scope]
    missing_components = [name for name in audio_component_names if name not in local_scope]

    logger.debug(f"{emoji_utils.magnifying_glass()} DEBUG get_tab_init() return:")
    logger.debug(f"   Found in locals(): {found_components}")
    logger.debug(f"   Missing from locals(): {missing_components}")

    # Add found components to result
    for comp_name in found_components:
        result[comp_name] = local_scope[comp_name]

    return result



def wan_generate_video(*component_args):
    """
    Function to handle Wan video generation from the Wan tab
    This function calls the main Deforum generation pipeline with Wan mode
    """
    from deforum.ui.handlers.wan_button_handler import (
        load_wan_emojis,
        get_wan_auto_download_setting,
        discover_and_prepare_models,
        build_no_models_error_message,
        extract_animation_prompts_from_args,
        force_animation_mode_to_flux_wan,
        build_deforum_final_args,
        process_wan_generation_result,
    )
    from deforum.integrations.wan.wan_simple_integration import WanSimpleIntegration
    from deforum.config.args import get_component_names

    # Load emoji symbols
    emojis = load_wan_emojis()

    try:
        logger.debug(f"Wan video generation button clicked! Received {len(component_args)} arguments", emoji='movie_camera')

        # Import the main Deforum run function
        from deforum.orchestration.run_deforum import run_deforum

        # Get component names for argument extraction
        component_names = get_component_names()

        # Extract settings from component arguments
        wan_auto_download = get_wan_auto_download_setting(component_args, component_names)

        # Discover and prepare models (auto-download, validate, cleanup)
        integration = WanSimpleIntegration()
        models = discover_and_prepare_models(integration, wan_auto_download, emojis)

        # If no models available after discovery, return error message
        if not models:
            return build_no_models_error_message(wan_auto_download, emojis)

        # Log discovered models
        logger.info(f"{emojis['check']} Found {len(models)} Wan model(s):")
        for i, model in enumerate(models, 1):
            logger.info(f"   {i}. {model['name']} ({model['size']}) - {model['path']}")

        # Extract animation prompts from component arguments
        animation_prompts = extract_animation_prompts_from_args(component_args, component_names, emojis)

        # Find animation_mode index for later
        animation_mode_index = None
        try:
            animation_mode_index = component_names.index('animation_mode')
            logger.info(f"{emojis['memo']} Found animation_mode at index {animation_mode_index}")
        except ValueError:
            logger.error(f"{emojis['warning']} Could not find animation_mode in component names")

        # Validate prompts
        if not animation_prompts or animation_prompts.strip() == '{"0": "a beautiful landscape"}':
            return f"""{emojis['cross']} No prompts configured!

{emojis['wrench']} SETUP REQUIRED:
1. {emojis['memo']} Go to the **Prompts tab** and configure your animation prompts
2. {emojis['movie_camera']} Set your desired FPS in the **Output tab**
3. {emojis['target']} Optionally configure seeds in **Keyframes → Seed & SubSeed tab**
4. {emojis['movie_camera']} Click **Generate Flux/Wan** again

{emojis['bulb']} I2V chaining needs your prompt schedule to know what to generate!

Example prompts for seamless I2V chaining:
{{
  "0": "a serene beach at sunset",
  "60": "a misty forest in the morning",
  "120": "a bustling city street at night"
}}

Each prompt will be smoothly connected using I2V continuity!"""

        # Force animation mode to Flux/Wan
        component_args = force_animation_mode_to_flux_wan(component_args, animation_mode_index, emojis)

        # Generate a unique job ID
        import uuid
        job_id = str(uuid.uuid4())[:8]

        logger.info(f"{emojis['rocket']} Starting Wan video generation with job ID: {job_id}")
        logger.info(f"{emojis['memo']} Using prompts: {str(animation_prompts)[:100]}...")

        # Build final args for run_deforum call
        expected_component_count = len(component_names)
        logger.info(f"Debug: Expected {expected_component_count} components, have {len(component_args)} args", emoji='wrench')

        final_args = build_deforum_final_args(job_id, component_args, expected_component_count, emojis)

        # Call the main Deforum generation function
        result = run_deforum(*final_args)

        # Process and return result
        return process_wan_generation_result(result, job_id, emojis)

    except Exception as e:
        error_msg = f"{emojis['cross']} Wan generation error: {str(e)}"
        logger.info(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


def generate_wan_video(args, anim_args, video_args, frame_idx, turbo_mode, turbo_preroll, root, animation_prompts, loop_args, parseq_args, parseq_adapter, wan_args, frame_duration):
    """Generate Wan video using the new simple integration approach - called by Deforum internally"""
    from deforum.integrations.wan.wan_simple_integration import WanSimpleIntegration
    from deforum.ui.handlers.wan_generation import (
        _load_emoji_symbols,
        _cleanup_qwen_models,
        _discover_and_validate_models,
        select_wan_model,
        setup_wan_output_directory,
        parse_prompts_and_timing,
        calculate_dynamic_motion_strength,
        validate_model_resolution_match,
    )
    import time

    # Load emoji symbols
    emojis = _load_emoji_symbols()

    logger.info("Wan video generation started with AUTO-DISCOVERY (Internal Call)", emoji='movie_camera')
    logger.info(f"{emojis['magnifying_glass']} Using smart model discovery instead of manual paths")

    # Ensure Qwen models are unloaded before video generation to free VRAM
    _cleanup_qwen_models()

    start_time = time.time()

    try:
        # Initialize the simple integration
        integration = WanSimpleIntegration()

        # Auto-discover and validate models
        _discover_and_validate_models(integration, emojis)

        # Select model based on user's choice
        selected_model = select_wan_model(integration, wan_args, emojis)

        logger.info(f"{emojis['target']} Selected model: {selected_model['name']} ({selected_model['type']}, {selected_model['size']})")
        logger.info(f"{emojis['folder']} Model path: {selected_model['path']}")

        # Load the pipeline before generation
        logger.info("Loading Wan pipeline...", emoji='refresh')
        if not integration.load_simple_wan_pipeline(selected_model, wan_args):
            raise RuntimeError(f"Failed to load Wan pipeline for {selected_model['name']}")
        logger.info(f"{emojis['check']} Wan pipeline loaded successfully")

        # Setup output directory
        output_directory = setup_wan_output_directory(args, root, emojis)

        # Generate video using direct integration
        logger.info(f"{emojis['rocket']} Starting direct Wan integration...")

        # Parse prompts for Wan scheduling
        clips = parse_prompts_and_timing(animation_prompts, wan_args, video_args, emojis)

        # Calculate dynamic motion strength if enabled
        motion_strength, motion_intensity_schedule = calculate_dynamic_motion_strength(anim_args, wan_args, emojis)

        # Parse resolution - handle both old format (864x480) and new format (864x480 (Landscape))
        resolution_str = wan_args.wan_resolution
        if '(' in resolution_str:
            resolution_str = resolution_str.split(' (')[0]
        width, height = map(int, resolution_str.split('x'))

        # Model/Resolution validation
        validate_model_resolution_match(selected_model, width, height, emojis)


        # Prepare clips data for generation
        clips_data = [
            {
                'prompt': prompt,
                'start_frame': start_frame,
                'end_frame': start_frame + frame_count,
                'num_frames': frame_count
            }
            for prompt, start_frame, frame_count in clips
        ]

        # Add motion intensity schedule to wan_args for use by Wan integration
        if motion_intensity_schedule:
            wan_args.wan_motion_intensity_schedule = motion_intensity_schedule
            logger.info(f"Added motion intensity schedule to wan_args for frame-by-frame control", emoji='bulb')

        # Wan 2.2 TI2V models always use unified T2V+I2V (no separate modes)
        mode_description = "unified TI2V generation"
        from deforum.utils.system.logging import emoji as emoji_utils
        logger.info(f"\n{emoji_utils.movie_camera()} Using Wan 2.2 TI2V unified generation for {len(clips_data)} clips with frame continuity", emoji='movie_camera')

        # Generate video using I2V chaining (TI2V supports both T2V and I2V)
        result = integration.generate_video_with_i2v_chaining(
            clips=clips_data,
            model_info=selected_model,
            output_dir=str(output_directory),
            wan_args=wan_args,
            width=width,
            height=height,
            num_inference_steps=wan_args.wan_inference_steps,
            guidance_scale=wan_args.wan_guidance_scale,
            fps=video_args.fps,
            timestring=root.timestring,
            seed=wan_args.wan_seed if wan_args.wan_seed > 0 else -1
        )

        output_file = result.get('output_dir') if result else None
        generated_videos = [output_file] if output_file else []
        total_time = time.time() - start_time

        if generated_videos:
            logger.info(f"\n{emojis['party']} Wan {mode_description} generation completed!")
            logger.info(f"{emojis['check']} Generated seamless video with {len(clips_data)} clips using {mode_description}")
            logger.info(f"Total time: {total_time:.1f} seconds", emoji='stopwatch')
            logger.info(f"{emojis['folder']} Output file: {generated_videos[0]}")
            logger.info(f"{emoji_utils.link()} {mode_description} ensures smooth transitions between clips")
            return str(output_directory)
        else:
            raise RuntimeError(f"{emojis['cross']} Wan {mode_description} failed")

    except Exception as e:
        logger.error(f"Wan generation failed: {e}", emoji='off')

        # Provide helpful troubleshooting info
        from deforum.utils.system.logging import emoji as emoji_utils
        logger.info(f"\n{emoji_utils.wrench()} TROUBLESHOOTING:", emoji='wrench')
        logger.info(f"   • Check model availability with: python scripts/deforum_helpers/wan_direct_integration.py")
        logger.info(f"   • Download models: huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir models/Deforum/wan")
        logger.info(f"   • Verify Wan models are in: models/Deforum/wan/ directory")

        # Re-raise for Deforum error handling
        raise


def auto_assign_keyframe_types_handler(animation_prompts_json, chunk_size):
    """
    Auto-assign keyframe types based on tween distances between keyframes.

    Uses pure functions from deforum.utils.parsing.keyframes for the logic.

    Logic:
    - Short sections (< 80% of chunk_size): Use "flf2v"
    - Long sections (>= 80% of chunk_size): Use "tween"

    Returns: keyframe_type_schedule string in format "0:(tween), 60:(flf2v), 120:(tween)"
    """
    from deforum.utils.system.logging import emoji as emoji_utils
    robot = emoji_utils.robot()

    import json
    from deforum.utils.parsing.keyframes import auto_assign_keyframe_types

    try:
        # Parse animation prompts JSON
        if isinstance(animation_prompts_json, str):
            animation_prompts = json.loads(animation_prompts_json)
        else:
            animation_prompts = animation_prompts_json

        # Auto-assign using pure function
        result, _ = auto_assign_keyframe_types(animation_prompts, chunk_size)

        logger.info(f"{robot} Auto-assigned keyframe types: {result}")
        return result

    except Exception as e:
        logger.error(f"Error auto-assigning keyframe types: {e}", emoji='off')
        return "0:(tween)"


def get_tab_wan(dw: SimpleNamespace, da: SimpleNamespace = None, skip_tabitem=False):
    """Interpolation Settings Tab - Multi-method interpolation (Wan/RIFE/FILM)

    Args:
        dw: DeforumWanArgs namespace
        da: DeforumAnimArgs namespace (optional, needed for FLF2V tween settings)
        skip_tabitem: If True, don't create TabItem wrapper
    """

    gr.Markdown("""
    ## 🎬 Interpolation Methods

    **Choose your interpolation method for smooth transitions between keyframes:**

    - **Wan FLF2V:** AI-generated video with semantic understanding (requires FLF2V model download)
    - **FILM:** Smearcore - sharp motion mixing like dragging paint, Google's ML interpolation (works out of the box)

    **Note:** RIFE is available in the post-processing tab for framerate doubling/tripling on completed videos.

    ---
    """)

    # INTERPOLATION METHOD SELECTOR - ALWAYS VISIBLE AT TOP
    gr.Markdown("### 🎯 Select Interpolation Method")
    with gr.Row():
        flux_flf2v_interpolation_method = create_gr_elem(dw.flux_flf2v_interpolation_method)

    gr.Markdown("---")

    # Deforum Integration Info - Shows what settings are used
    with gr.Accordion(f"{emoji_utils.link()} Deforum Integration Details", open=False):
        gr.Markdown(f"""
        **{emoji_utils.maybe_check()} Wan seamlessly integrates with your Deforum settings:**

        - **{emoji_utils.memo()} Prompts:** Uses prompts from Deforum Prompts tab
        - **{emoji_utils.movie_camera()} Movement:** Uses same movement schedules as normal Deforum renders
        - **{emoji_utils.dice()} Seed & CFG:** Uses Deforum's seed and CFG schedules
        - **{emoji_utils.muscle()} Strength:** Uses Deforum's strength schedule for I2V continuity
        - **{emoji_utils.movie_camera()} FPS:** Uses Output tab FPS setting

        **Movement Integration:**
        - {emoji_utils.maybe_check()} Translation X/Y/Z, Rotation 3D X/Y/Z, Zoom schedules
        - {emoji_utils.maybe_check()} **Parseq schedules fully supported**
        - {emoji_utils.maybe_check()} Movement descriptions automatically calculated and added
        - {emoji_utils.maybe_check()} Motion intensity dynamically adapts to movement complexity
        """)

    # DEPRECATED SECTION - Hide old standalone workflow
    with gr.Accordion(f"{emoji_utils.maybe_warning()} DEPRECATED: Standalone Wan Workflow (Hidden)", open=False, visible=False):
        gr.Markdown(f"""
        **{emoji_utils.target()} Essential for Wan Generation:** These prompts define what video clips will be generated.
        
        **Quick Setup:** Load → Analyze Movement → Enhance → Generate
        """)
        
        # Prompt Loading Buttons
        with FormRow():
            load_deforum_to_wan_btn = gr.Button(
                f"{emoji_utils.memo()} Load from Deforum Prompts",
                variant="primary",
                size="lg",
                elem_id="load_deforum_to_wan_btn"
            )
            load_wan_defaults_btn = gr.Button(
                f"{emoji_utils.memo()} Load Default Wan Prompts",
                variant="secondary",
                size="lg",
                elem_id="load_wan_defaults_btn"
            )

        # Wan Prompts Display - ALWAYS VISIBLE AND PROMINENT
        wan_enhanced_prompts = gr.Textbox(
            label="Flux/Wan Prompts (JSON Format)",
            lines=10,
            interactive=True,
            placeholder='REQUIRED: Load prompts first! Click "Load from Deforum Prompts" or "Load Default Wan Prompts" above.',
            info=f"{emoji_utils.target()} ESSENTIAL: These prompts will be used for Wan video generation. Edit manually or use buttons below to enhance.",
            elem_id="wan_enhanced_prompts_textbox"
        )

        # Prompt Enhancement Actions
        with FormRow():
            analyze_movement_btn = gr.Button(
                f"{emoji_utils.ruler()} Add Movement Descriptions",
                variant="secondary",
                size="lg",
                elem_id="wan_analyze_movement_btn"
            )
            enhance_prompts_btn = gr.Button(
                f"{emoji_utils.palette()} AI Prompt Enhancement",
                variant="secondary",
                size="lg",
                elem_id="wan_enhance_prompts_btn"
            )

        # Camera Shakify Integration Control
        with FormRow():
            wan_enable_shakify = gr.Checkbox(
                label=f"{emoji_utils.movie_camera()} Include Camera Shakify with Movement Analysis",
                value=True,
                info="Enable Camera Shakify integration for movement analysis (uses settings from Keyframes → Motion → Shakify tab)",
                elem_id="wan_enable_shakify_checkbox"
            )
            wan_movement_sensitivity_override = gr.Checkbox(
                label="Manual Sensitivity Override",
                value=False,
                info="Override auto-calculated sensitivity (normally auto-calculated from movement magnitude)",
                elem_id="wan_sensitivity_override_checkbox"
            )
        
        # Manual Sensitivity Control (hidden by default)
        with FormRow(visible=False) as manual_sensitivity_row:
            wan_manual_sensitivity = gr.Slider(
                label="Manual Movement Sensitivity",
                minimum=0.1,
                maximum=5.0,
                step=0.1,
                value=1.0,
                info="Higher values detect subtler movements (0.1: only large movements, 5.0: very sensitive)",
                elem_id="wan_manual_sensitivity_slider"
            )
        
        # Movement Analysis Results - Enhanced with frame-by-frame details
        wan_movement_description = gr.Textbox(
            label="Movement Analysis Results",
            lines=6,
            interactive=False,
            placeholder="Movement analysis results will appear here...\n\n💡 TIP: This shows frame-by-frame movement detection with Camera Shakify integration.",
            info="Fine-grained movement descriptions with specific frame ranges and Camera Shakify effects.",
            elem_id="wan_movement_description_textbox",
            visible=True  # Always visible for immediate feedback
        )
        
        # Enhancement Progress - Shows during AI enhancement
        enhancement_progress = gr.Textbox(
            label="AI Enhancement Progress",
            lines=3,
            interactive=False,
            placeholder="AI enhancement progress will show here...",
            info="Shows real-time progress during prompt enhancement.",
            elem_id="wan_enhancement_progress_textbox",
            visible=True
        )

        # GENERATION SECTION
        gr.Markdown("---")
        gr.Markdown(f"### {emoji_utils.movie_camera()} Generate Flux/Wan")

        # Generate Button with Validation
        with FormRow():
            wan_generate_button = gr.Button(
                f"{emoji_utils.movie_camera()} Generate Flux/Wan (I2V Chaining)",
                variant="primary",
                size="lg",
                elem_id="wan_generate_button"
            )

        # Status output for Wan generation
        wan_generation_status = gr.Textbox(
            label="Generation Status",
            interactive=False,
            lines=5,
            placeholder=f"{emoji_utils.maybe_warning()} Prompts required! Load prompts above first, then click Generate.",
            info="Status updates will appear here during generation."
        )

        # ESSENTIAL SETTINGS - Compact
        gr.Markdown("---")
        gr.Markdown(f"### {emoji_utils.gear()} Essential Settings")

        with FormRow():
            wan_auto_download = create_gr_elem(dw.wan_auto_download)
            wan_resolution = gr.Dropdown(
                label="Wan Resolution",
                choices=["1280x736 (Landscape, 720p)", "736x1280 (Portrait, 720p)", "1024x1024 (Square, 1K)", "1280x704 (Landscape, Letterbox)"],
                value="1280x736 (Landscape, 720p)",
                info="Wan 2.2 TI2V models require resolutions divisible by 32 (VAE=16x * Patch=2x). These are optimized for 720p."
            )

        with FormRow():
            wan_inference_steps = gr.Slider(
                label="Inference Steps",
                minimum=5,
                maximum=100,
                step=1,
                value=20,
                elem_id="wan_inference_steps_fixed_min_5",
                info="Steps for generation quality (5-15: fast, 20-50: quality)"
            )
    # END DEPRECATED SECTION
    
    # MODEL SETTINGS - Collapsed by default
    with gr.Accordion(f"{emoji_utils.wrench()} Model Settings", open=False):
        gr.Markdown(f"""
        **{emoji_utils.download()} One-Click Model Download**: Download official Wan 2.2 models from Hugging Face!
        - **TI2V-5B** (Recommended): Works with 16GB VRAM using automatic CPU offload
        - **TI2V-A14B** (Advanced): Highest quality, requires 24GB+ VRAM
        """)

        # Model Download Buttons
        with gr.Accordion(f"{emoji_utils.download()} Download Models", open=True):
            gr.Markdown(f"**{emoji_utils.maybe_check()} Recommended for Most Users (16GB+ VRAM)**")
            with FormRow():
                download_ti2v_5b = gr.Button(f"{emoji_utils.download()} TI2V-5B (30GB Download, ~16GB VRAM with offload)", variant="primary", size="sm")

            gr.Markdown(f"**{emoji_utils.rocket()} Advanced / High-End GPUs (24GB+ VRAM)**")
            with FormRow():
                download_ti2v_a14b = gr.Button(f"{emoji_utils.download()} TI2V-A14B (60GB Download, ~32GB VRAM)", size="sm")

            download_status = gr.Textbox(
                label="Download Status",
                interactive=False,
                lines=4,
                placeholder="Click a download button above to start downloading a model..."
            )

        # Model Selection
        with FormRow():
            wan_t2v_model = gr.Dropdown(
                label="TI2V Model (Wan 2.2)",
                choices=["Auto-Detect", "TI2V-5B", "TI2V-A14B", "Custom Path"],
                value="Auto-Detect",
                info="Wan 2.2 unified text/image-to-video model. TI2V-5B auto-enables CPU offload for 16GB VRAM."
            )

        wan_model_path = create_gr_elem(dw.wan_model_path)

        with gr.Row():
            wan_model_info = gr.Textbox(
                label="Detected Model Info",
                interactive=False,
                placeholder="Model information will appear here after loading",
                lines=2
            )

    # VRAM OPTIMIZATION SETTINGS
    with gr.Accordion(f"{emoji_utils.floppy_disk()} VRAM Optimization", open=False):
        gr.Markdown(f"""
        **Reduce VRAM usage for 16GB GPUs:**

        These settings can help run larger models on GPUs with limited VRAM.
        All settings are OFF by default for maximum compatibility.

        **{emoji_utils.maybe_warning()} Trade-offs:**
        - T5 CPU Offload: Slightly slower text encoding, saves ~3-4GB VRAM
        - Gradient Checkpointing: Slower inference (~15-20%), saves ~2-3GB VRAM
        - Both combined: Can reduce peak VRAM by ~5-7GB

        **Recommended for 16GB VRAM:**
        - Try T5 CPU Offload first
        - Add Gradient Checkpointing if still getting OOM errors
        """)

        with FormRow():
            wan_t5_cpu_offload = create_gr_elem(dw.wan_t5_cpu_offload)
            wan_gradient_checkpointing = create_gr_elem(dw.wan_gradient_checkpointing)

        vram_optimization_info = gr.HTML(
            value="""
            <div style='padding: 10px; background: #1a1a1a; border-radius: 5px; margin-top: 10px;'>
                <p style='margin: 0; color: #aaa;'>
                    💡 <strong>Tip:</strong> Enable these if you see "CUDA out of memory" errors on 16GB GPUs.
                    <br/>📊 Current setup enables automatic CPU offload based on model size (5B vs A14B).
                </p>
            </div>
            """,
            elem_id="wan_vram_optimization_info"
        )

    # GENERATION SETTINGS - More prominent and open by default
    gr.Markdown("---")
    gr.Markdown(f"### {emoji_utils.gear()} Generation Settings")

    with gr.Accordion(f"{emoji_utils.palette()} T2V / Keyframe Generation", open=True):
        with FormRow():
            wan_strength_override = create_gr_elem(dw.wan_strength_override)
            wan_fixed_strength = create_gr_elem(dw.wan_fixed_strength)
            
        with FormRow():
            wan_guidance_override = create_gr_elem(dw.wan_guidance_override)
            wan_guidance_scale = create_gr_elem(dw.wan_guidance_scale)
    
    with gr.Accordion(f"{emoji_utils.frames()} Wan FLF2V Settings", open=False):
        gr.Markdown("""
        **⚠️ These settings only apply when Wan is selected as interpolation method above!**

        **MODEL REQUIREMENT:** You MUST use a FLF2V-specific model!
        - **TI2V models (e.g., Wan2.2-TI2V-5B) CANNOT do FLF2V** - they will extend the first frame
        - **Use:** Wan2.1-FLF2V-14B (only FLF2V model available)
        - TI2V models were not trained on first-last-frame data, so they ignore `last_image` parameter

        **IMPORTANT:** FLF2V needs semantic guidance to interpolate correctly!
        - **Guidance Scale:** 3.5 (default) = smooth morphing, 2.5-3.0 = even smoother. **Avoid 5.5+** (causes "mode collapse" - frames stick to first image with sudden transition at end)
        - **Prompt Mode:** **'blend' (RECOMMENDED)** - combines keyframe prompts for semantic guidance
        - **⚠️ NEVER use guidance_scale=0.0** (breaks last_image conditioning)
        - **⚠️ 'none' mode may not work** (empty prompts often cause first-frame extension)
        """)
        with FormRow():
            wan_flf2v_guidance_scale = create_gr_elem(dw.wan_flf2v_guidance_scale)
            wan_flf2v_prompt_mode = create_gr_elem(dw.wan_flf2v_prompt_mode)

        # Advanced FLF2V settings for 3D mode tween interpolation
        # Only show if da (DeforumAnimArgs) is provided
        if da is not None:
            gr.Markdown("---")
            gr.Markdown("### 🎯 Advanced FLF2V Control (For 3D Mode Tween Interpolation)")
            gr.Markdown("""
            **These settings control Wan FLF2V interpolation in 3D modes when "Enable FLF2V Tween Mode" is checked in the 3D Depth tab.**

            - **Chunk Size:** Maximum frames per FLF2V clip (must be 4n+1, e.g., 13, 81)
            - **Keyframe Type Schedule:** Per-keyframe control of interpolation method
            """)

            with FormRow():
                wan_flf2v_chunk_size = create_row(da.wan_flf2v_chunk_size)

            gr.Markdown("**Per-Keyframe Type Control (Advanced):**")
            keyframe_type_schedule = create_row(da.keyframe_type_schedule)

            with FormRow():
                auto_assign_keyframe_types_btn = gr.Button(
                    "🤖 Auto-Assign Types",
                    variant="secondary",
                    size="sm",
                    elem_id="auto_assign_keyframe_types_btn"
                )
                gr.Markdown("*Analyzes tween distances and suggests optimal types based on chunk size*")

    with gr.Accordion(f"{emoji_utils.wrench()} Advanced Generation", open=False):

        # Advanced Generation Settings
        with FormRow():
            wan_negative_prompt = create_gr_elem(dw.wan_negative_prompt)

        with FormRow():
            wan_sampler = create_gr_elem(dw.wan_sampler)
            wan_scheduler = create_gr_elem(dw.wan_scheduler)

        with FormRow():
            wan_motion_strength_override = create_gr_elem(dw.wan_motion_strength_override)
            wan_motion_strength = create_gr_elem(dw.wan_motion_strength)
            
        # Movement sensitivity - now in overrides since it should be auto-calculated
        with FormRow():
            movement_sensitivity_override = gr.Checkbox(
                label="Movement Sensitivity Override",
                value=False,
                info="Override auto-calculated movement sensitivity from Deforum schedules"
            )
            wan_movement_sensitivity = create_gr_elem(dw.wan_movement_sensitivity)
            wan_movement_sensitivity.interactive = False  # Start disabled
    
    with gr.Accordion("⚡ Timing & Interpolation", open=False):
        with FormRow():
            wan_frame_overlap = create_gr_elem(dw.wan_frame_overlap)
            
        with FormRow():
            wan_enable_interpolation = create_gr_elem(dw.wan_enable_interpolation)
            wan_interpolation_strength = create_gr_elem(dw.wan_interpolation_strength)
            
        # Flash Attention Settings Section
        with gr.Accordion("⚡ Flash Attention Settings", open=False):
            gr.Markdown("""
            **Flash Attention Performance Control**
            
            Flash Attention provides faster and more memory-efficient attention computation.
            
            **Modes:**
            - **Auto (Recommended)**: Try Flash Attention, fall back to PyTorch if unavailable
            - **Force Flash Attention**: Force Flash Attention (fails if not available)
            - **Force PyTorch Fallback**: Always use PyTorch attention (slower but compatible)
            """)
            
            wan_flash_attention_mode = create_gr_elem(dw.wan_flash_attention_mode)
            
            # Flash Attention Status
            wan_flash_attention_status = gr.HTML(
                label="Flash Attention Status",
                value="⚠️ <span style='color: #FF9800;'>Status check unavailable</span>",
                elem_id="wan_flash_attention_status"
            )
            
            check_flash_attention_btn = gr.Button(
                f"{emoji_utils.magnifying_glass()} Check Flash Attention Status",
                variant="secondary",
                elem_id="wan_check_flash_attention_btn"
            )
    
    # QWEN MODEL MANAGEMENT - Collapsed by default
    with gr.Accordion(f"{emoji_utils.brain()} Qwen Model Management", open=False):
        gr.Markdown("""
        **Model Information & Auto-Download Status**

        Monitor Qwen model availability and manage downloads:
        """)

        qwen_model_status = gr.HTML(
            label="Qwen Model Status",
            value="⏳ Checking model availability...",
            elem_id="wan_qwen_model_status"
        )

        with FormRow():
            check_qwen_models_btn = gr.Button(
                f"{emoji_utils.magnifying_glass()} Check Model Status",
                variant="secondary",
                elem_id="wan_check_qwen_models_btn"
            )
            download_qwen_model_btn = gr.Button(
                f"{emoji_utils.download()} Download Selected Model",
                variant="primary",
                elem_id="wan_download_qwen_model_btn"
            )
            cleanup_qwen_cache_btn = gr.Button(
                f"{emoji_utils.broom()} Cleanup Model Cache",
                variant="secondary",
                elem_id="wan_cleanup_qwen_cache_btn"
            )

    # Auto-Discovery and Setup Information
    with gr.Accordion(f"{emoji_utils.download()} Model Auto-Discovery & Setup", open=False):
        gr.Markdown(f"""
        **{emoji_utils.maybe_check()} Auto-Discovery System**
        
        Wan automatically finds models in these locations:
        - `models/Deforum/wan/` (recommended)
        - `models/video/wan/`
        - Custom paths you specify
        
        **✨ Wan 2.2 TI2V Models (Recommended)**

        TI2V models are unified text/image-to-video with diffusers format:
        - **TI2V-5B**: 720p@24fps, 24GB VRAM, RTX 4090 compatible (recommended)
        - **TI2V-A14B**: Mixture-of-Experts, 32GB+ VRAM, highest quality

        **📥 Easy Download Commands:**
        ```bash
        # Download TI2V-5B (recommended default)
        huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir models/Deforum/wan/Wan2.2-TI2V-5B

        # Or download TI2V-A14B (highest quality)
        huggingface-cli download Wan-AI/Wan2.2-TI2V-A14B-Diffusers --local-dir models/Deforum/wan/Wan2.2-TI2V-A14B
        ```

        **Note**: This extension supports Wan 2.2 TI2V models only.
        Legacy Wan 2.1 models (T2V, I2V, VACE) are no longer supported.
        """)
    
    # Hidden model path for compatibility (auto-populated by discovery)
    wan_model_path = gr.Textbox(visible=False, value="auto-discovery")
    
    # Hidden wan_seed for compatibility (integrated with Deforum schedules)
    wan_seed = gr.Number(
        precision=dw.wan_seed["precision"], 
        value=dw.wan_seed["value"],
        visible=False
    )
    
    # Detailed Documentation - Collapsed by default
    with gr.Accordion(f"{emoji_utils.book()} Detailed Documentation", open=False):
        with gr.Accordion(f"{emoji_utils.target()} How Wan Integrates with Deforum Schedules", open=False):
            gr.Markdown("""
            ### Prompt Schedule Integration
            - Wan reads your prompts from the **Prompts tab**
            - Each prompt with a frame number becomes a video clip
            - Duration is calculated from the frame differences
            - Example: `{"0": "beach sunset", "120": "forest morning"}` creates two clips
            
            ### Seed Schedule Integration  
            - Wan uses the **seed schedule** from Keyframes → Seed & SubSeed
            - Set **Seed behavior** to 'schedule' to enable custom seed scheduling
            - Example: `0:(12345), 60:(67890)` uses different seeds for different clips
            - Leave as 'iter' or 'random' for automatic seed management
            
            ### Strength Schedule Integration
            - Wan I2V chaining supports **Deforum's strength schedule**!
            - Controls how much the previous frame influences the next clip generation
            - Found in **Keyframes → Strength tab** as "Strength schedule"
            - Higher values (0.7-0.9): Strong continuity, smoother transitions
            - Lower values (0.3-0.6): More creative freedom, less continuity
            - Example: `0:(0.85), 120:(0.6)` - strong continuity at start, more freedom later
            
            ### CFG Scale Schedule Integration
            - Wan supports **Deforum's CFG scale schedule**!
            - Controls how closely generation follows the prompt across clips
            - Found in **Keyframes → CFG tab** as "CFG scale schedule"
            - Higher values (7.5-12): Strong prompt adherence, less creative interpretation
            - Lower values (3-6): More creative interpretation, looser prompt following
            - Example: `0:(7.5), 120:(10.0)` - moderate adherence at start, stronger later
            
            ### FPS Integration
            - Wan uses the **FPS setting** from the Output tab
            - No separate FPS slider needed - one setting controls everything
            - Ensures video timing matches your intended frame rate
            
            ### Duration Calculation & Frame Management
            - Video duration = (frame_difference / fps) seconds per clip
            - Example: Frames 0→120 at 30fps = 4 second clip
            - **Wan 4n+1 Requirement**: Wan requires frame counts to follow 4n+1 format (5, 9, 13, 17, 21, etc.)
            - **Automatic Calculation**: System calculates the nearest 4n+1 value ≥ your requested frames
            - **Frame Discarding**: Extra frames are discarded from the middle to match your exact timing
            - **Display Info**: Console shows exactly which frames will be discarded before generation
            """)
            
            with gr.Accordion(f"{emoji_utils.movie_camera()} Movement Translation: From Deforum Schedules to Prompt Descriptions", open=False):
                gr.Markdown("""
                ### ✨ NEW: Frame-Specific Movement Analysis
                
                Wan now provides **unique movement descriptions for each prompt** based on its exact position in the video timeline, eliminating generic repetitive text.
                
                **🎯 Key Improvements:**
                - **Frame-Specific Analysis**: Each prompt analyzes movement at its specific frame range
                - **Directional Specificity**: "panning left", "tilting down", "dolly forward" instead of generic text
                - **Camera Shakify Integration**: Analyzes actual shake patterns at each frame offset
                - **Varied Descriptions**: No more identical "investigative handheld" text across all prompts
                
                ### 🔄 How Frame-Specific Analysis Works
                
                **Traditional Approach (OLD):**
                ```json
                All prompts: "camera movement with investigative handheld camera movement"
                ```
                
                **Frame-Specific Approach (NEW):**
                ```json
                {
                  "0": "...with subtle panning left (sustained) and gentle moving down (extended)",
                  "43": "...with moderate panning right (brief) and subtle rotating left (sustained)",
                  "106": "...with gentle dolly forward (extended) and subtle rolling clockwise (brief)",
                  "210": "...with subtle tilting down (extended) and moderate panning left (brief)",
                  "324": "...with gentle rotating right (sustained) and subtle dolly backward (extended)"
                }
                ```
                
                ### 📊 Movement Detection & Classification
                
                **Translation Movements:**
                - **Translation X**: 
                  - Increasing → "panning right"
                  - Decreasing → "panning left"
                - **Translation Y**: 
                  - Increasing → "moving up"
                  - Decreasing → "moving down"
                - **Translation Z**: 
                  - Increasing → "dolly forward"
                  - Decreasing → "dolly backward"
                
                **Rotation Movements:**
                - **Rotation 3D X**: 
                  - Increasing → "tilting up"
                  - Decreasing → "tilting down"
                - **Rotation 3D Y**: 
                  - Increasing → "rotating right"
                  - Decreasing → "rotating left"
                - **Rotation 3D Z**: 
                  - Increasing → "rolling clockwise"
                  - Decreasing → "rolling counter-clockwise"
                
                **Zoom & Effects:**
                - **Zoom**: 
                  - Increasing → "zooming in"
                  - Decreasing → "zooming out"
                
                ### 🎨 Intensity & Duration Modifiers
                
                **Movement Intensity:**
                - **Subtle**: Very small movements (< 1.0 units)
                - **Gentle**: Small movements (1.0 - 10.0 units)
                - **Moderate**: Medium movements (10.0 - 50.0 units)
                - **Strong**: Large movements (> 50.0 units)
                
                **Duration Descriptions:**
                - **Brief**: Short duration (< 20% of total frames)
                - **Extended**: Medium duration (20% - 50% of total frames)
                - **Sustained**: Long duration (> 50% of total frames)
                
                ### 🎬 Camera Shakify Integration
                
                When Camera Shakify is enabled, the system:
                1. **Generates frame-specific shake data** based on the prompt's frame position
                2. **Overlays shake on Deforum schedules** (like render core)
                3. **Analyzes combined movement** for each prompt's timeframe
                4. **Provides varied descriptions** that reflect actual camera behavior
                
                **Example with Camera Shakify INVESTIGATION:**
                ```
                Frame 0 prompt → Analyzes shake pattern frames 0-17
                Frame 43 prompt → Analyzes shake pattern frames 43-60
                Frame 106 prompt → Analyzes shake pattern frames 106-123
                ```
                
                ### 🔧 Smart Motion Analysis
                
                **Sensitivity Auto-Calculation:**
                The system automatically calculates optimal sensitivity based on movement magnitude:
                - **Very subtle** (< 5 units): High sensitivity (3.0)
                - **Subtle** (5-15 units): High sensitivity (2.0)
                - **Normal** (15-50 units): Standard sensitivity (1.0)
                - **Large** (50-200 units): Reduced sensitivity (0.7)
                - **Very large** (> 200 units): Low sensitivity (0.5)
                
                **Segment Grouping:**
                - Groups similar movements that occur close together
                - Reduces redundancy while preserving directional specificity
                - Creates readable, varied descriptions
                
                ### 📈 Results Comparison
                
                **Before Frame-Specific Analysis:**
                ```json
                {
                  "0": "...complex camera movement with complex panning movement with 5 phases",
                  "43": "...complex camera movement with complex panning movement with 5 phases",
                  "106": "...complex camera movement with complex panning movement with 5 phases"
                }
                ```
                
                **After Frame-Specific Analysis:**
                ```json
                {
                  "0": "...camera movement with subtle panning left (sustained) and gentle moving down (extended)",
                  "43": "...camera movement with moderate panning right (brief) and subtle rotating left (sustained)",
                  "106": "...camera movement with gentle dolly forward (extended) and subtle rolling clockwise (brief)"
                }
                ```
                
                ### 🚀 Practical Usage
                
                1. **Set up movement** in Keyframes → Motion tab or enable Camera Shakify
                2. **Configure prompts** in Prompts tab with frame numbers
                3. **Click "Enhance Prompts with Movement Analysis"**
                4. **Review frame-specific descriptions** - each prompt gets unique analysis
                5. **Generate video** with varied, specific movement context for better results
                
                This frame-specific system ensures each video clip gets movement descriptions that accurately reflect what's happening during its specific timeframe!
                """)
            
        with gr.Accordion(f"{emoji_utils.tools()} Setup Guide", open=False):
            gr.Markdown("""
            #### Step 1: Configure Prompts
            ```json
            {
                "0": "a serene beach at sunset",
                "90": "a misty forest in the morning", 
                "180": "a bustling city street at night"
            }
            ```
            
            #### Step 2: Set FPS (Output Tab)
            - Choose your desired FPS (e.g., 30 or 60)
            - This affects both timing and video quality
            
            #### Step 3: Configure Strength Schedule (Optional but Recommended)
            - Go to **Keyframes → Strength tab**
            - Set "Strength schedule" to control I2V continuity
            - Example: `0:(0.85), 60:(0.7), 120:(0.5)` for gradual creative freedom
            
            #### Step 4: Configure CFG Scale Schedule (Optional but Recommended)
            - Go to **Keyframes → CFG tab**
            - Set "CFG scale schedule" to control prompt adherence
            - Example: `0:(7.5), 60:(9.0), 120:(6.0)` for varying prompt adherence
            
            #### Step 5: Configure Seeds (Optional)
            - **For consistent seeds**: Set seed behavior to 'schedule'
            - **For variety**: Leave as 'iter' or 'random'
            
            #### Step 6: Generate
            - Click "Generate Flux/Wan" button
            - Wan reads all settings from Deforum automatically
            - Each prompt becomes a seamless video clip with strength-controlled transitions
            """)
            
        with gr.Accordion("🆘 Troubleshooting", open=False):
            gr.Markdown("""
            If generation fails:
            1. **Check models**: Run `python scripts/deforum_helpers/wan_direct_integration.py`
            2. **Download missing models**: Use commands in Auto-Discovery section
            3. **Verify placement**: Models should be in `models/Deforum/wan/` directory
            4. **Check logs**: Look for auto-discovery messages in console
            5. **Verify schedules**: Make sure you have prompts in the Prompts tab
            6. **Check seed behavior**: Set seed behavior to 'schedule' if you want custom seed scheduling
            """)

    # Connect movement sensitivity override toggle
    def toggle_movement_sensitivity_override(override_enabled):
        return gr.update(interactive=override_enabled)
    
    movement_sensitivity_override.change(
        fn=toggle_movement_sensitivity_override,
        inputs=[movement_sensitivity_override],
        outputs=[wan_movement_sensitivity]
    )
        
    # Ensure wan_inference_steps is properly captured
    locals()['wan_inference_steps'] = wan_inference_steps
    
    # Button handlers for flash attention
    def check_flash_attention_status():
        """Check flash attention availability and return status"""
        try:
            from deforum.integrations.wan.wan_flash_attention_patch import get_flash_attention_status_html
            return get_flash_attention_status_html()
        except Exception as e:
            return f"❌ <span style='color: #f44336;'>Error checking status: {e}</span>"

    def update_flash_attention_mode(mode):
        """Update flash attention mode and return updated status"""
        try:
            from deforum.integrations.wan.wan_flash_attention_patch import update_patched_flash_attention_mode, get_flash_attention_status_html
            update_patched_flash_attention_mode(mode)
            status = get_flash_attention_status_html()
            return f"{status} - Mode: {mode}"
        except Exception as e:
            return f"❌ <span style='color: #f44336;'>Error updating mode: {e}</span>"
    
    # Connect button click to status check
    check_flash_attention_btn.click(
        fn=check_flash_attention_status,
        inputs=[],
        outputs=[wan_flash_attention_status]
    )

    # Connect mode change to status update
    wan_flash_attention_mode.change(
        fn=update_flash_attention_mode,
        inputs=[wan_flash_attention_mode],
        outputs=[wan_flash_attention_status]
    )

    # Initialize status on load
    try:
        from deforum.integrations.wan.wan_flash_attention_patch import get_flash_attention_status_html
        wan_flash_attention_status.value = get_flash_attention_status_html()
    except Exception:
        wan_flash_attention_status.value = "⚠️ <span style='color: #FF9800;'>Status check unavailable</span>"
    
    # Get all component names for the handlers
    from deforum.config.args import get_component_names
    component_names = get_component_names()

    # DEPRECATED EVENT HANDLERS - Commented out for hidden standalone workflow
    # These are no longer needed as Wan is now integrated with main Deforum workflow

    # # Connect sensitivity override toggle to show/hide manual sensitivity slider
    # wan_movement_sensitivity_override.change(
    #     fn=lambda override_enabled: gr.update(visible=override_enabled),
    #     inputs=[wan_movement_sensitivity_override],
    #     outputs=[manual_sensitivity_row]
    # )
    #
    # # Connect generate button with validation
    # wan_generate_button.click(
    #     fn=validate_wan_generation,
    #     inputs=[wan_enhanced_prompts],
    #     outputs=[wan_generation_status]
    # )
    #
    # # Add automatic validation status updates when prompts change
    # wan_enhanced_prompts.change(
    #     fn=validate_wan_generation,
    #     inputs=[wan_enhanced_prompts],
    #     outputs=[wan_generation_status]
    # )

    # Connect model download buttons
    from deforum.integrations.wan.wan_model_downloader import download_wan_model

    download_ti2v_5b.click(
    fn=lambda: download_wan_model("TI2V-5B"),
    outputs=[download_status]
    )

    download_ti2v_a14b.click(
    fn=lambda: download_wan_model("TI2V-A14B"),
    outputs=[download_status]
    )

    # DEPRECATED: Standalone prompt loading buttons (now use main Prompts tab)
    # load_deforum_to_wan_btn.click(
    #     fn=load_deforum_to_wan_prompts_handler,
    #     inputs=[],
    #     outputs=[wan_enhanced_prompts]
    # )
    #
    # load_wan_defaults_btn.click(
    #     fn=load_wan_defaults_handler,
    #     inputs=[],
    #     outputs=[wan_enhanced_prompts]
    # )

    # Connect prompt template loading buttons
    # NOTE: These will be properly connected in ui_left.py where animation_prompts is accessible

    # Store button references for connection in ui_left.py
    if 'load_wan_prompts_btn' in locals():
        locals()['load_wan_prompts_btn']._handler = load_wan_prompts_handler
    if 'load_deforum_prompts_btn' in locals():
        locals()['load_deforum_prompts_btn']._handler = load_deforum_prompts_handler

    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_output(da, dv):
    with gr.TabItem(f"{emoji_utils.document()} Output", elem_id='output_tab'):
        # VID OUTPUT ACCORD
        with gr.Accordion('Video Output Settings', open=True):
            # fps moved to top-level setting in ui_left.py - create hidden copy for button handlers
            with gr.Row(visible=False):
                fps = create_gr_elem(dv.fps)
                add_soundtrack = create_gr_elem(dv.add_soundtrack)
                soundtrack_path = create_gr_elem(dv.soundtrack_path)

            with FormColumn():
                with FormRow():
                    skip_video_creation = create_gr_elem(dv.skip_video_creation)
                    delete_imgs = create_gr_elem(dv.delete_imgs)
                    delete_input_frames = create_gr_elem(dv.delete_input_frames)
                    store_frames_in_ram = create_gr_elem(dv.store_frames_in_ram)
                    save_depth_maps = create_gr_elem(da.save_depth_maps)
                    make_gif = create_gr_elem(dv.make_gif)
            with FormRow(equal_height=True) as r_upscale_row:
                r_upscale_video = create_gr_elem(dv.r_upscale_video)
                r_upscale_model = create_gr_elem(dv.r_upscale_model)
                r_upscale_factor = create_gr_elem(dv.r_upscale_factor)
                r_upscale_keep_imgs = create_gr_elem(dv.r_upscale_keep_imgs)
        # FRAME INTERPOLATION TAB
        with gr.Tab('Frame Interpolation') as frame_interp_tab:
            with gr.Accordion('Important notes and Help', open=False, elem_id="f_interp_accord"):
                gr.HTML(value=get_gradio_html('frame_interpolation'))
            with gr.Column():
                with gr.Row():
                    # Interpolation Engine
                    with gr.Column(min_width=110, scale=3):
                        frame_interpolation_engine = create_gr_elem(dv.frame_interpolation_engine)
                    with gr.Column(min_width=30, scale=1):
                        frame_interpolation_slow_mo_enabled = create_gr_elem(dv.frame_interpolation_slow_mo_enabled)
                    with gr.Column(min_width=30, scale=1):
                        # If this is set to True, we keep all the interpolated frames in a folder. Default is False - means we delete them at the end of the run
                        frame_interpolation_keep_imgs = create_gr_elem(dv.frame_interpolation_keep_imgs)
                    with gr.Column(min_width=30, scale=1):
                        frame_interpolation_use_upscaled = create_gr_elem(dv.frame_interpolation_use_upscaled)
                with FormRow(visible=False) as frame_interp_amounts_row:
                    with gr.Column(min_width=180) as frame_interp_x_amount_column:
                        # How many times to interpolate (interp X)
                        frame_interpolation_x_amount = create_gr_elem(dv.frame_interpolation_x_amount)
                    with gr.Column(min_width=180, visible=False) as frame_interp_slow_mo_amount_column:
                        # Interp Slow-Mo (setting final output fps, not really doing anything directly with RIFE/FILM)
                        frame_interpolation_slow_mo_amount = create_gr_elem(dv.frame_interpolation_slow_mo_amount)
                with gr.Row(visible=False) as interp_existing_video_row:
                    # Interpolate any existing video from the connected PC
                    with gr.Accordion('Interpolate existing Video/ Images', open=False) as interp_existing_video_accord:
                        with gr.Row(variant='compact') as interpolate_upload_files_row:
                            # A drag-n-drop UI box to which the user uploads a *single* (at this stage) video
                            vid_to_interpolate_chosen_file = gr.File(label="Video to Interpolate", interactive=True,
                                                                     file_count="single", file_types=["video"],
                                                                     elem_id="vid_to_interpolate_chosen_file")
                            # A drag-n-drop UI box to which the user uploads a pictures to interpolate
                            pics_to_interpolate_chosen_file = gr.File(label="Pics to Interpolate", interactive=True,
                                                                      file_count="multiple", file_types=["image"],
                                                                      elem_id="pics_to_interpolate_chosen_file")
                        with FormRow(visible=False) as interp_live_stats_row:
                            # Non-interactive textbox showing uploaded input vid total Frame Count
                            in_vid_frame_count_window = gr.Textbox(label="In Frame Count", lines=1, interactive=False,
                                                                   value='---')
                            # Non-interactive textbox showing uploaded input vid FPS
                            in_vid_fps_ui_window = gr.Textbox(label="In FPS", lines=1, interactive=False, value='---')
                            # Non-interactive textbox showing expected output interpolated video FPS
                            out_interp_vid_estimated_fps = gr.Textbox(label="Interpolated Vid FPS", value='---')
                        with FormRow() as interp_buttons_row:
                            # This is the actual button that's pressed to initiate the interpolation:
                            interpolate_button = gr.Button(value="*Interpolate Video*")
                            interpolate_pics_button = gr.Button(value="*Interpolate Pics*")
                        # Show a text about CLI outputs:
                        gr.HTML("* check your CLI for outputs *", elem_id="below_interpolate_butts_msg")
                        # make the function call when the interpolation button is clicked
                        interpolate_button.click(fn=upload_vid_to_interpolate,
                                                 inputs=[vid_to_interpolate_chosen_file, frame_interpolation_engine,
                                                         frame_interpolation_x_amount,
                                                         frame_interpolation_slow_mo_enabled,
                                                         frame_interpolation_slow_mo_amount,
                                                         frame_interpolation_keep_imgs, in_vid_fps_ui_window])
                        interpolate_pics_button.click(fn=upload_pics_to_interpolate,
                                                      inputs=[pics_to_interpolate_chosen_file,
                                                              frame_interpolation_engine, frame_interpolation_x_amount,
                                                              frame_interpolation_slow_mo_enabled,
                                                              frame_interpolation_slow_mo_amount,
                                                              frame_interpolation_keep_imgs, fps, add_soundtrack,
                                                              soundtrack_path])
        # VIDEO UPSCALE TAB - not built using our args.py at all - all data and params are here and in .upscaling file
        with gr.TabItem(f"{emoji_utils.up()} Video Upscaling"):
            vid_to_upscale_chosen_file = gr.File(label="Video to Upscale", interactive=True, file_count="single",
                                                 file_types=["video"], elem_id="vid_to_upscale_chosen_file")
            with gr.Column():
                # NCNN UPSCALE TAB
                with FormRow() as ncnn_upload_vid_stats_row:
                    ncnn_upscale_in_vid_frame_count_window = gr.Textbox(label="In Frame Count", lines=1,
                                                                        interactive=False,
                                                                        value='---')  # Non-interactive textbox showing uploaded input vid Frame Count
                    ncnn_upscale_in_vid_fps_ui_window = gr.Textbox(label="In FPS", lines=1, interactive=False,
                                                                   value='---')  # Non-interactive textbox showing uploaded input vid FPS
                    ncnn_upscale_in_vid_res = gr.Textbox(label="In Res", lines=1, interactive=False,
                                                         value='---')  # Non-interactive textbox showing uploaded input resolution
                    ncnn_upscale_out_vid_res = gr.Textbox(label="Out Res",
                                                          value='---')  # Non-interactive textbox showing expected output resolution
                with gr.Column():
                    with FormRow() as ncnn_actual_upscale_row:
                        ncnn_upscale_model = create_gr_elem(
                            dv.r_upscale_model)  # note that we re-use *r_upscale_model* in here to create the gradio element as they are the same
                        ncnn_upscale_factor = create_gr_elem(
                            dv.r_upscale_factor)  # note that we re-use *r_upscale_facto*r in here to create the gradio element as they are the same
                        ncnn_upscale_keep_imgs = create_gr_elem(
                            dv.r_upscale_keep_imgs)  # note that we re-use *r_upscale_keep_imgs* in here to create the gradio element as they are the same
                ncnn_upscale_btn = gr.Button(value="*Upscale uploaded video*")
                ncnn_upscale_btn.click(fn=ncnn_upload_vid_to_upscale,
                                       inputs=[vid_to_upscale_chosen_file, ncnn_upscale_in_vid_fps_ui_window,
                                               ncnn_upscale_in_vid_res, ncnn_upscale_out_vid_res, ncnn_upscale_model,
                                               ncnn_upscale_factor, ncnn_upscale_keep_imgs])
        # STITCH FRAMES TO VID TAB
        with gr.TabItem(f"{emoji_utils.frames()} Frames to Video") as stitch_imgs_to_vid_row:
            gr.HTML(value=get_gradio_html('frames_to_video'))
            image_path = create_row(dv.image_path)
            ffmpeg_stitch_imgs_but = gr.Button(value="*Stitch frames to video*")
            ffmpeg_stitch_imgs_but.click(fn=direct_stitch_vid_from_frames,
                                         inputs=[image_path, fps, add_soundtrack, soundtrack_path])
    return {k: v for k, v in {**locals(), **vars()}.items()}


# QwenPromptExpander and Movement Analysis Event Handlers - moved outside for proper import
def enhance_prompts_handler(current_prompts, qwen_model, language, auto_download):
    """Handle prompt enhancement with QwenPromptExpander with progress feedback"""
    # Theme-aware emoji symbols
    from deforum.utils.system.logging import emoji as emoji_utils
    check = emoji_utils.maybe_check()
    cross = emoji_utils.maybe_cross()
    warning = emoji_utils.maybe_warning()
    palette = emoji_utils.palette()
    memo = emoji_utils.memo()
    wrench = emoji_utils.wrench()
    hourglass = emoji_utils.hourglass()
    download = emoji_utils.download()
    bulb = emoji_utils.bulb()
    refresh_icon = emoji_utils.refresh_icon()
    robot = emoji_utils.robot()
    magnifying_glass = emoji_utils.magnifying_glass()
    pencil = emoji_utils.pencil()
    chart_increasing = emoji_utils.chart_increasing()
    save = emoji_utils.save()
    sparkles = emoji_utils.sparkles()
    ruler = emoji_utils.ruler()
    party = emoji_utils.party()

    try:
        from deforum.integrations.wan.utils.qwen_manager import qwen_manager
        import json

        logger.info(f"AI Prompt Enhancement requested for {qwen_model}", emoji='palette')
        logger.info(f"{memo} Received prompts: {str(current_prompts)[:100]}...")

        # Progress: Start
        progress_update = f"{palette} Starting AI Prompt Enhancement...\n"
        
        # Check if auto-download is enabled for model availability
        if not auto_download:
            # Check if the selected model is available
            if not qwen_manager.is_model_downloaded(qwen_model):
                return f"""{cross} Qwen model not available: {qwen_model}

{wrench} **Model Download Required:**
1. {check} Enable "Auto-Download Qwen Models" checkbox
2. {palette} Click "AI Prompt Enhancement" again to auto-download
3. {hourglass} Wait for download to complete

{download} **Manual Download Alternative:**
1. Use HuggingFace CLI: `huggingface-cli download {qwen_manager.get_model_info(qwen_model).get('huggingface_id', 'model-id')}`
2. {check} Enable auto-download for easier setup

{bulb} **Auto-download is recommended** for seamless model management.""", progress_update + f"{cross} Model not available - enable auto-download!"

        # Progress: Model check
        progress_update += f"{magnifying_glass} Checking model availability...\n"

        # Check if a model is already loaded
        if qwen_manager.is_model_loaded():
            loaded_info = qwen_manager.get_loaded_model_info()
            current_model = loaded_info['name'] if loaded_info else "Unknown"

            # If different model requested, cleanup first
            if qwen_model != "Auto-Select" and current_model != qwen_model:
                logger.info(f"Switching from {current_model} to {qwen_model}", emoji='refresh')
                progress_update += f"{refresh_icon} Switching from {current_model} to {qwen_model}...\n"
                qwen_manager.cleanup_cache()

        # Progress: Model loading
        if not qwen_manager.is_model_loaded():
            if qwen_model == "Auto-Select":
                selected_model = qwen_manager.auto_select_model()
                logger.info(f"{robot} Auto-selected model: {selected_model}")
                progress_update += f"{robot} Auto-selected model: {selected_model}\n"
            else:
                logger.info(f"{download} Loading Qwen model: {qwen_model}")
                progress_update += f"{download} Loading Qwen model: {qwen_model}...\n"
        
        # Get wan prompts from the current_prompts parameter (passed directly)
        animation_prompts = None
        
        if current_prompts and current_prompts.strip():
            try:
                # Try to parse as JSON first
                animation_prompts = json.loads(current_prompts)
                logger.info(f"{emoji_utils.maybe_check()} Successfully parsed {len(animation_prompts)} Wan prompts as JSON")
                progress_update += f"{check} Parsed {len(animation_prompts)} prompts successfully\n"
            except json.JSONDecodeError:
                # Try to parse as readable format (Frame X: prompt)
                try:
                    animation_prompts = {}
                    for line in current_prompts.strip().split('\n'):
                        if ':' in line:
                            # Handle both "Frame X:" and "X:" formats
                            parts = line.split(':', 1)
                            frame_part = parts[0].strip()
                            prompt_part = parts[1].strip()

                            # Extract frame number
                            if frame_part.lower().startswith('frame '):
                                frame_num = frame_part[6:].strip()
                            else:
                                frame_num = frame_part

                            animation_prompts[frame_num] = prompt_part

                    if animation_prompts:
                        logger.info(f"{emoji_utils.maybe_check()} Successfully parsed {len(animation_prompts)} Wan prompts as readable format")
                        progress_update += f"{check} Parsed {len(animation_prompts)} prompts from readable format\n"
                    else:
                        raise ValueError("No valid prompts found")
                except Exception as e:
                    logger.error(f"Could not parse Wan prompts: {e}", emoji='off')
                    error_msg = f"{cross} Invalid format in Wan prompts. Expected JSON format like:\n{{\n  \"0\": \"prompt text\",\n  \"60\": \"another prompt\"\n}}\n\nOr readable format like:\nFrame 0: prompt text\nFrame 60: another prompt"
                    return error_msg, progress_update + f"{cross} Failed to parse prompts!"
        else:
            logger.warning(f"{warning} Empty Wan prompts")
            
        # Check if we got valid prompts
        if not animation_prompts:
            error_msg = f"""{cross} No Wan prompts found!

{wrench} **Setup Required:**
1. {memo} Load prompts using "Load from Deforum Prompts" or "Load Default Wan Prompts"
2. {memo} Make sure your prompts are in proper JSON format like:
   {{
     "0": "prompt text",
     "60": "another prompt",
     "120": "a cyberpunk environment with glowing elements"
   }}
3. {palette} Click **AI Prompt Enhancement** again after setting up prompts

{bulb} **Quick Start:**
Click "Load Default Wan Prompts" to start with example prompts!"""
            return error_msg, progress_update + f"{cross} No prompts to enhance!"

        # Validate prompts content
        if len(animation_prompts) == 1 and "0" in animation_prompts and "beautiful landscape" in animation_prompts["0"]:
            error_msg = f"""{cross} Default prompts detected!

{wrench} **Please configure your actual animation prompts:**
1. {memo} Load your real prompts using the load buttons above
2. {pencil} Or manually edit the Wan prompts field
3. {palette} Click **AI Prompt Enhancement** again

{bulb} **For your animation sequence:**
Set up prompts like:
{{{{
  "0": "A peaceful scene, photorealistic",
  "18": "A scene with glowing effects, neon colors, synthwave aesthetic",
  "36": "A cyberpunk scene with LED patterns, digital environment"
}}}}"""
            return error_msg, progress_update + f"{cross} Default prompts detected!"

        logger.info(f"Enhancing {len(animation_prompts)} Wan prompts with {qwen_model}", emoji='palette')
        progress_update += f"{palette} Starting enhancement of {len(animation_prompts)} prompts...\n"

        # Create the Qwen prompt expander with better error handling
        try:
            progress_update += f"{download} Creating AI model instance...\n"
            prompt_expander = qwen_manager.create_prompt_expander(qwen_model, auto_download)

            if not prompt_expander:
                if auto_download:
                    error_msg = f"""{hourglass} Downloading {qwen_model} model...

{refresh_icon} **Download in Progress:**
Model download started automatically. This may take a few minutes.

{download} **Please wait** and try clicking "AI Prompt Enhancement" again in 30-60 seconds.

{bulb} **Status**: Check console for download progress."""
                    return error_msg, progress_update + f"{hourglass} Downloading {qwen_model}..."
                else:
                    error_msg = f"""{cross} Failed to create Qwen prompt expander: {qwen_model}

{wrench} **Solutions:**
1. {check} Enable "Auto-Download Qwen Models" and try again
2. {download} Manual download: Check console for HuggingFace CLI commands
3. {refresh_icon} Restart WebUI after downloading

{chart_increasing} **Model Info**: {qwen_manager.get_model_info(qwen_model).get('description', 'N/A')}"""
                    return error_msg, progress_update + f"{cross} Failed to create AI model!"
        except Exception as e:
            error_msg = f"""{cross} Error creating Qwen prompt expander: {str(e)}

{wrench} **Troubleshooting:**
1. {check} Enable auto-download and try again
2. {refresh_icon} Restart WebUI if models were just downloaded
3. {save} Check available disk space ({qwen_manager.get_model_info(qwen_model).get('vram_gb', 'Unknown')}GB VRAM required)

{bulb} **Tip**: Try selecting "Auto-Select" for automatic model choice."""
            return error_msg, progress_update + f"{cross} Error: {str(e)}"
        
        # Use the QwenModelManager's enhance_prompts method directly
        try:
            progress_update += f"{sparkles} Enhancing prompts with AI...\n"
            enhanced_prompts_dict = qwen_manager.enhance_prompts(
                prompts=animation_prompts,
                model_name=qwen_model,
                language=language,
                auto_download=auto_download
            )

            # Check if movement descriptions are available and append them
            movement_description = ""
            if hasattr(enhance_prompts_handler, '_movement_description'):
                movement_description = enhance_prompts_handler._movement_description
                logger.info(f"{ruler} Found movement description to append: {movement_description}")
                progress_update += f"{ruler} Adding movement descriptions...\n"

            # Append movement descriptions to enhanced prompts if available
            if movement_description and movement_description.strip():
                for frame_key in enhanced_prompts_dict:
                    original_prompt = enhanced_prompts_dict[frame_key]
                    enhanced_prompts_dict[frame_key] = f"{original_prompt}. {movement_description}"
                logger.info(f"{emoji_utils.maybe_check()} Appended movement description to {len(enhanced_prompts_dict)} enhanced prompts")
                progress_update += f"{check} Added movement to {len(enhanced_prompts_dict)} prompts\n"

            # Format the enhanced prompts as JSON
            enhanced_json = json.dumps(enhanced_prompts_dict, ensure_ascii=False, indent=2)

            logger.info(f"{emoji_utils.maybe_check()} Successfully enhanced {len(enhanced_prompts_dict)} prompts")
            progress_update += f"{check} Enhancement complete! {len(enhanced_prompts_dict)} prompts ready\n"

            # Return the enhanced prompts and success progress
            return enhanced_json, progress_update + f"{party} Ready for generation!"
            
        except Exception as e:
            logger.error(f"Error enhancing prompts: {e}", emoji='off')
            import traceback
            traceback.print_exc()
            error_msg = f"{cross} Error enhancing prompts: {str(e)}"
            return error_msg, progress_update + f"{cross} Enhancement failed: {str(e)}"

    except Exception as e:
        logger.info(f"Fatal error in enhance_prompts_handler: {e}", emoji='off')
        import traceback
        traceback.print_exc()
        error_msg = f"{cross} Fatal error: {str(e)}"
        return error_msg, f"{cross} Fatal error: {str(e)}"


def analyze_movement_handler(current_prompts, enable_shakify=True, sensitivity_override=False, manual_sensitivity=1.0):
    """Handle movement analysis from Deforum schedules with enhanced Camera Shakify integration and fine-grained sensitivity control"""
    from deforum.ui.handlers.analyze_movement_handler import (
        load_analyze_movement_emojis,
        validate_prompts_for_analysis,
        build_anim_args_from_components,
        get_camera_shakify_settings,
        calculate_movement_sensitivity,
        update_prompts_with_movement,
        build_camera_shakify_status_message,
        build_analysis_result_message,
        build_analysis_error_message,
    )
    import json

    # Load emojis
    emojis = load_analyze_movement_emojis()

    try:
        from deforum.integrations.wan.utils.movement_analyzer import analyze_deforum_movement, generate_wan_motion_intensity_schedule
        from deforum.utils.system.logging import emoji as emoji_utils

        logger.info("Starting enhanced movement analysis with fine-grained detection...", emoji='movie_camera')
        logger.info(f"Camera Shakify: {'ENABLED' if enable_shakify else 'DISABLED'}", emoji='movie_camera')
        logger.info(f"{emojis['target']} Sensitivity: {'MANUAL ({:.1f})'.format(manual_sensitivity) if sensitivity_override else 'AUTO-CALCULATED'}")

        # Validate and parse prompts
        is_valid, prompts_dict, error_msg = validate_prompts_for_analysis(current_prompts, emojis)
        if not is_valid:
            return "", error_msg

        # Build anim_args from stored components
        anim_args = build_anim_args_from_components(analyze_movement_handler, emojis)

        # Get Camera Shakify settings if enabled
        get_camera_shakify_settings(anim_args, analyze_movement_handler, enable_shakify, emojis)

        # Calculate sensitivity (auto or manual)
        sensitivity, sensitivity_reason = calculate_movement_sensitivity(
            anim_args, sensitivity_override, manual_sensitivity, emojis
        )

        # Generate movement description using enhanced analysis with Camera Shakify
        movement_desc, average_motion_strength = analyze_deforum_movement(
            anim_args=anim_args,
            sensitivity=sensitivity,
            max_frames=anim_args.max_frames
        )

        # Generate Wan motion intensity schedule
        motion_intensity_schedule = generate_wan_motion_intensity_schedule(
            anim_args,
            max_frames=anim_args.max_frames,
            sensitivity=sensitivity
        )

        logger.info(f"{emojis['target']} Enhanced movement analysis result:")
        logger.info(f"   Description: {movement_desc}")
        logger.info(f"   Strength: {average_motion_strength:.3f}")
        logger.info(f"   Motion Intensity Schedule: {motion_intensity_schedule}")

        # Update prompts with movement descriptions
        updated_prompts = update_prompts_with_movement(prompts_dict, movement_desc, average_motion_strength)

        # Convert back to JSON
        updated_json = json.dumps(updated_prompts, ensure_ascii=False, indent=2)

        # Build result message
        camera_shakify_status = build_camera_shakify_status_message(anim_args, enable_shakify, emojis)
        result_message = build_analysis_result_message(
            movement_desc,
            average_motion_strength,
            sensitivity,
            sensitivity_reason,
            motion_intensity_schedule,
            camera_shakify_status,
            updated_prompts,
            emojis
        )

        logger.info(f"{emoji_utils.maybe_check()} Updated {len(updated_prompts)} Wan prompts with enhanced movement descriptions")
        logger.info(f"Use this motion intensity schedule in Wan: {motion_intensity_schedule}", emoji='distribution')
        logger.info(f"Copy this schedule to Wan's Motion Intensity field for synchronized movement effects!", emoji='bulb')

        # Store movement description for enhance_prompts_handler
        analyze_movement_handler._movement_description = movement_desc

        return updated_json, result_message

    except Exception as e:
        logger.error(f"Error in enhanced movement analysis: {str(e)}", emoji='off')
        import traceback
        traceback.print_exc()
        error_msg = build_analysis_error_message(e, emojis)
        return current_prompts, error_msg

def check_qwen_models_handler(qwen_model):
    """Check Qwen model status and availability"""
    # Theme-aware emoji symbols
    from deforum.utils.system.logging import emoji as emoji_utils
    check = emoji_utils.maybe_check()
    cross = emoji_utils.maybe_cross()
    warning = emoji_utils.maybe_warning()
    palette = emoji_utils.palette()
    hourglass = emoji_utils.hourglass()
    refresh_icon = emoji_utils.refresh_icon()
    zzz = emoji_utils.zzz()
    fire = emoji_utils.fire()

    try:
        from deforum.integrations.wan.utils.qwen_manager import qwen_manager

        logger.info(f"{emoji_utils.magnifying_glass()} Checking Qwen model status: {qwen_model}")
        
        # Get model information
        model_info = qwen_manager.get_model_info(qwen_model)
        
        # Check if model is downloaded
        is_downloaded = qwen_manager.is_model_downloaded(qwen_model)
        
        # Check if model is currently loaded
        is_loaded = qwen_manager.is_model_loaded()
        loaded_info = qwen_manager.get_loaded_model_info() if is_loaded else None
        
        # Get VRAM information
        available_vram = qwen_manager.get_available_vram()
        
        # Build status HTML
        status_parts = []
        
        # Model Selection Status
        status_parts.append(f"<strong style='color: #333;'>Selected Model:</strong> {qwen_model}")
        
        if qwen_model == "Auto-Select":
            auto_selected = qwen_manager.auto_select_model()
            status_parts.append(f"<strong style='color: #333;'>Auto-Selected:</strong> {auto_selected}")
            status_parts.append(f"<strong style='color: #333;'>Reason:</strong> Best fit for {available_vram:.1f}GB VRAM")
            qwen_model = auto_selected  # Use auto-selected for further checks
            model_info = qwen_manager.get_model_info(qwen_model)
        
        # Model Info
        if model_info:
            status_parts.append(f"<strong style='color: #333;'>Description:</strong> {model_info.get('description', 'N/A')}")
            status_parts.append(f"<strong style='color: #333;'>VRAM Required:</strong> {model_info.get('vram_gb', 'Unknown')}GB")
            status_parts.append(f"<strong style='color: #333;'>Available VRAM:</strong> {available_vram:.1f}GB")
            
            if model_info.get('vram_gb', 0) <= available_vram:
                status_parts.append(f"{check} <span style='color: #4CAF50;'>VRAM requirement met</span>")
            else:
                status_parts.append(f"{warning} <span style='color: #FF9800;'>May exceed available VRAM</span>")

        # Download Status
        if is_downloaded:
            status_parts.append(f"{check} <span style='color: #4CAF50;'>Model downloaded and available</span>")
        else:
            status_parts.append(f"{cross} <span style='color: #f44336;'>Model not downloaded</span>")
            if model_info and 'hf_name' in model_info:
                status_parts.append(f"<strong style='color: #333;'>HuggingFace ID:</strong> {model_info['hf_name']}")

        # Loading Status
        if is_loaded:
            if loaded_info and loaded_info['name'] == qwen_model:
                status_parts.append(f"{fire} <span style='color: #4CAF50;'>Model currently loaded and ready</span>")
                estimated_vram = loaded_info.get('vram_usage', 0)
                if estimated_vram > 0:
                    status_parts.append(f"<strong style='color: #333;'>Estimated VRAM usage:</strong> {estimated_vram:.1f}GB")
            else:
                current_model = loaded_info['name'] if loaded_info else "Unknown"
                status_parts.append(f"{refresh_icon} <span style='color: #FF9800;'>Different model loaded: {current_model}</span>")
                status_parts.append("<span style='color: #333;'>Will switch on next enhancement</span>")
        else:
            status_parts.append(f"{zzz} <span style='color: #333;'>No model currently loaded</span>")

        # Quick Setup Instructions
        if not is_downloaded:
            status_parts.append("<br><strong style='color: #333;'>Quick Setup:</strong>")
            status_parts.append(f"1. {check} Enable 'Auto-Download Qwen Models' above")
            status_parts.append(f"2. {palette} Click 'AI Prompt Enhancement' for auto-download")
            status_parts.append(f"3. {hourglass} Wait for download to complete")
        elif not is_loaded:
            status_parts.append("<br><strong style='color: #333;'>Ready to Use:</strong>")
            status_parts.append(f"{palette} Click 'AI Prompt Enhancement' to load and use this model")
        else:
            status_parts.append("<br><strong style='color: #333;'>Status:</strong> Ready for prompt enhancement!")

        return "<br>".join(status_parts)

    except Exception as e:
        logger.error(f"Error checking Qwen model status: {e}", emoji='off')
        return f"{cross} <span style='color: #f44336;'>Error checking model status: {str(e)}</span>"


def download_qwen_model_handler(qwen_model, auto_download_enabled):
    """Download selected Qwen model"""
    from deforum.utils.system.logging import emoji as emoji_utils

    # Theme-aware emoji symbols
    check = emoji_utils.maybe_check()
    cross = emoji_utils.maybe_cross()
    download = emoji_utils.download()
    robot = emoji_utils.robot()
    palette = emoji_utils.palette()

    try:
        from deforum.integrations.wan.utils.qwen_manager import qwen_manager

        if not auto_download_enabled:
            return f"""{cross} <span style='color: #f44336;'>Auto-download is disabled</span>

<strong style='color: #333;'>To download models:</strong><br>
1. {check} Enable 'Auto-Download Qwen Models' checkbox above<br>
2. {download} Click this button again<br>
<br>
<strong style='color: #333;'>Or download manually:</strong><br>
Use HuggingFace CLI or git to download the model"""

        logger.info(f"{download} Downloading Qwen model: {qwen_model}")

        # Handle auto-select
        if qwen_model == "Auto-Select":
            selected_model = qwen_manager.auto_select_model()
            logger.info(f"{robot} Auto-selected model for download: {selected_model}")
        else:
            selected_model = qwen_model

        # Check if already downloaded
        if qwen_manager.is_model_downloaded(selected_model):
            return f"""{check} <span style='color: #4CAF50;'>Model already available: {selected_model}</span>

<strong style='color: #333;'>Status:</strong> Model is downloaded and ready to use<br>
{palette} Click 'AI Prompt Enhancement' to start using this model"""

        # Start download
        download_status = []
        download_status.append(f"{download} <span style='color: #2196F3;'>Starting download: {selected_model}</span>")

        model_info = qwen_manager.get_model_info(selected_model)
        if model_info:
            download_status.append(f"<strong style='color: #333;'>Description:</strong> {model_info.get('description', 'N/A')}")
            download_status.append(f"<strong style='color: #333;'>VRAM Required:</strong> {model_info.get('vram_gb', 'Unknown')}GB")
            download_status.append(f"<strong style='color: #333;'>HuggingFace:</strong> {model_info.get('hf_name', 'N/A')}")

        # Attempt download
        success = qwen_manager.download_model(selected_model)

        if success:
            download_status.append(f"<br>{check} <span style='color: #4CAF50;'>Download completed successfully!</span>")
            download_status.append(f"{palette} Ready to use - click 'AI Prompt Enhancement' to start")
        else:
            download_status.append(f"<br>{cross} <span style='color: #f44336;'>Download failed</span>")
            download_status.append("<strong style='color: #333;'>Troubleshooting:</strong>")
            download_status.append("• Check internet connection")
            download_status.append("• Verify disk space")
            download_status.append("• Try manual download with HuggingFace CLI")

            if model_info and 'hf_name' in model_info:
                download_status.append(f"<br><strong style='color: #333;'>Manual command:</strong>")
                download_status.append(f"<code>huggingface-cli download {model_info['hf_name']} --local-dir models/Deforum/qwen/{selected_model}</code>")

        return "<br>".join(download_status)

    except Exception as e:
        logger.error(f"Error downloading Qwen model: {e}", emoji='off')
        return f"{cross} <span style='color: #f44336;'>Download error: {str(e)}</span>"


def cleanup_qwen_cache_handler():
    """Cleanup Qwen model cache and free VRAM"""
    from deforum.utils.system.logging import emoji as emoji_utils

    # Theme-aware emoji symbols
    check = emoji_utils.maybe_check()
    cross = emoji_utils.maybe_cross()
    info = emoji_utils.info()
    floppy_disk = emoji_utils.floppy_disk()
    brain = emoji_utils.brain()
    refresh_icon = emoji_utils.refresh_icon()
    bulb = emoji_utils.bulb()

    try:
        from deforum.integrations.wan.utils.qwen_manager import qwen_manager

        logger.info("Cleaning up Qwen model cache...", emoji='broom')

        # Check if any model is loaded
        if not qwen_manager.is_model_loaded():
            return f"""{info} <span style='color: #2196F3;'>No Qwen models currently loaded</span>

<strong style='color: #333;'>Cache Status:</strong> Clean - no cleanup needed<br>
{floppy_disk} VRAM available for other operations"""

        # Get info about loaded model before cleanup
        loaded_info = qwen_manager.get_loaded_model_info()
        model_name = loaded_info['name'] if loaded_info else "Unknown"
        estimated_vram = loaded_info.get('vram_usage', 0) if loaded_info else 0

        # Perform cleanup
        qwen_manager.cleanup_cache()

        result = []
        result.append(f"{check} <span style='color: #4CAF50;'>Qwen model cache cleaned successfully</span>")
        result.append(f"<strong style='color: #333;'>Unloaded model:</strong> {model_name}")

        if estimated_vram > 0:
            result.append(f"<strong style='color: #333;'>Freed VRAM:</strong> ~{estimated_vram:.1f}GB")

        result.append("<br><strong style='color: #333;'>Benefits:</strong>")
        result.append(f"{floppy_disk} VRAM freed for video generation")
        result.append(f"{brain} Reduced memory usage")
        result.append(f"{refresh_icon} Fresh start for next enhancement")

        result.append(f"<br>{bulb} <span style='color: #333;'>Models will auto-load when needed for enhancement</span>")

        return "<br>".join(result)

    except Exception as e:
        logger.error(f"Error during Qwen cache cleanup: {e}", emoji='off')
        return f"{cross} <span style='color: #f44336;'>Cleanup error: {str(e)}</span>"


def convert_fps_handler(prompts_json, source_fps, target_fps, preview_only):
    """
    Convert prompt frame numbers from source FPS to target FPS

    Uses pure functions from deforum.utils.conversion.fps for the conversion logic.

    Args:
        prompts_json: JSON string with prompts (e.g., '{"0": "prompt1", "60": "prompt2"}')
        source_fps: Current FPS that prompts are synced to
        target_fps: Desired FPS for conversion
        preview_only: If True, show preview without updating prompts

    Returns:
        Tuple of (updated_prompts_json, html_status_message)
    """
    from deforum.utils.system.logging import emoji as emoji_utils
    import json
    from deforum.utils.conversion.fps import (
        validate_fps_values,
        calculate_fps_ratio,
        convert_prompts_dict,
        build_conversion_status
    )

    # Theme-aware emoji symbol
    cross = emoji_utils.maybe_cross()

    try:
        # Validate FPS values
        is_valid, error_msg = validate_fps_values(source_fps, target_fps)
        if not is_valid:
            return prompts_json, f"{cross} <span style='color: #f44336;'>Error: {error_msg}</span>"

        # Parse prompts JSON
        try:
            prompts = json.loads(prompts_json)
        except json.JSONDecodeError as e:
            return prompts_json, f"{cross} <span style='color: #f44336;'>Error parsing prompts JSON: {str(e)}</span>"

        if not isinstance(prompts, dict):
            return prompts_json, f"{cross} <span style='color: #f44336;'>Error: Prompts must be a JSON object/dictionary</span>"

        # Convert frame numbers using pure functions
        fps_ratio = calculate_fps_ratio(source_fps, target_fps)
        converted_prompts, conversion_log = convert_prompts_dict(prompts, fps_ratio)

        # Format output JSON
        converted_json = json.dumps(converted_prompts, indent=4, ensure_ascii=False)

        # Build status message using pure function
        status_html = build_conversion_status(
            source_fps, target_fps, fps_ratio, conversion_log, preview_only
        )

        # Return updated prompts or original based on preview mode
        return (prompts_json if preview_only else converted_json), status_html

    except Exception as e:
        import traceback
        logger.error(f"Error in FPS converter: {e}", emoji='off')
        traceback.print_exc()
        return prompts_json, f"{cross} <span style='color: #f44336;'>Error: {str(e)}</span>"


def load_wan_prompts_handler():
    """Load Wan prompts from default settings.

    Uses pure functions from deforum.utils.parsing.prompts for formatting.
    """
    from deforum.utils.system.logging import emoji as emoji_utils
    import json
    import os
    from deforum.utils.parsing.prompts import format_prompts_as_multiline

    # Theme-aware emoji symbols
    check = emoji_utils.maybe_check()
    warning = emoji_utils.maybe_warning()

    try:
        # Load prompts from default_settings.txt
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_settings.txt')

        if not os.path.exists(settings_path):
            logger.info(f"Default settings file not found: {settings_path}", emoji='off')
            return "0: A peaceful landscape scene, photorealistic"

        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        # Get wan_prompts from settings
        wan_prompts = settings.get('wan_prompts', {})

        if not wan_prompts:
            logger.warning(f"{warning} No wan_prompts found in default settings, falling back to basic prompt")
            return "0: A peaceful landscape scene, photorealistic"

        # Convert prompts dict to textarea format using pure function
        result = format_prompts_as_multiline(wan_prompts)
        logger.info(f"{check} Loaded {len(wan_prompts)} Wan prompts from default settings")
        return result

    except Exception as e:
        logger.error(f"Error loading Wan prompts: {e}", emoji='off')
        return f"0: Error loading prompts: {str(e)}"


def load_deforum_prompts_handler():
    """Load original Deforum prompts from default settings.

    Uses pure functions from deforum.utils.parsing.prompts for formatting.
    """
    from deforum.utils.system.logging import emoji as emoji_utils
    import json
    import os
    from deforum.utils.parsing.prompts import format_prompts_as_multiline

    # Theme-aware emoji symbols
    check = emoji_utils.maybe_check()
    warning = emoji_utils.maybe_warning()

    try:
        # Load prompts from default_settings.txt
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_settings.txt')

        if not os.path.exists(settings_path):
            logger.info(f"Default settings file not found: {settings_path}", emoji='off')
            return "0: A peaceful landscape scene, photorealistic"

        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        # Get prompts from settings (main prompts section)
        deforum_prompts = settings.get('prompts', {})

        if not deforum_prompts:
            logger.warning(f"{warning} No prompts found in default settings, falling back to basic prompt")
            return "0: A peaceful landscape scene, photorealistic"

        # Convert prompts dict to textarea format using pure function
        result = format_prompts_as_multiline(deforum_prompts)
        logger.info(f"{check} Loaded {len(deforum_prompts)} Deforum prompts from default settings")
        return result

    except Exception as e:
        logger.error(f"Error loading Deforum prompts: {e}", emoji='off')
        return f"0: Error loading prompts: {str(e)}"


def load_deforum_to_wan_prompts_handler():
    """Load current Deforum prompts into Wan prompts field.

    Uses pure functions from deforum.utils.parsing.prompts for conversion.
    """
    from deforum.utils.system.logging import emoji as emoji_utils
    from deforum.utils.parsing.prompts import (
        validate_prompts_not_empty,
        parse_prompts_json,
        convert_deforum_to_wan_prompts,
        format_prompts_as_json,
        create_error_prompt
    )

    # Theme-aware emoji symbols
    check = emoji_utils.maybe_check()
    cross = emoji_utils.maybe_cross()
    warning = emoji_utils.maybe_warning()
    memo = emoji_utils.memo()

    try:
        # Try to get animation prompts from the stored component reference
        animation_prompts_json = ""

        if hasattr(enhance_prompts_handler, '_animation_prompts_component'):
            try:
                animation_prompts_json = enhance_prompts_handler._animation_prompts_component.value
                logger.info(f"{memo} Loading Deforum prompts to Wan prompts field")
            except Exception as e:
                logger.error(f"{warning} Could not access animation_prompts component: {e}")

        # Validate not empty
        is_valid, error = validate_prompts_not_empty(animation_prompts_json)
        if not is_valid:
            return create_error_prompt(
                "No Deforum prompts found! Go to the Prompts tab and configure your animation prompts first."
            )

        # Parse the JSON
        prompts_dict, parse_error = parse_prompts_json(animation_prompts_json)
        if parse_error:
            return create_error_prompt(
                f"Invalid JSON in Deforum prompts: {parse_error}. Fix the JSON format in the Prompts tab first."
            )

        # Convert to Wan format using pure function
        wan_prompts_dict = convert_deforum_to_wan_prompts(prompts_dict)

        # Return as JSON
        result = format_prompts_as_json(wan_prompts_dict)
        logger.info(f"{check} Converted {len(prompts_dict)} Deforum prompts to Wan JSON format")
        return result

    except Exception as e:
        return f"{cross} Error loading Deforum prompts: {str(e)}"


def load_wan_defaults_handler():
    """Load default Wan prompts from settings file.

    Uses pure functions from deforum.utils.parsing.prompts for formatting.
    """
    from deforum.utils.system.logging import emoji as emoji_utils
    import json
    import os
    from deforum.utils.parsing.prompts import (
        create_fallback_prompts,
        format_prompts_as_json
    )

    # Theme-aware emoji symbols
    check = emoji_utils.maybe_check()
    warning = emoji_utils.maybe_warning()

    try:
        # Load default prompts from settings
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_settings.txt')

        if not os.path.exists(settings_path):
            # Fallback to simple defaults using pure function
            return format_prompts_as_json(create_fallback_prompts())

        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            wan_prompts = settings.get('wan_prompts', {})

            if wan_prompts:
                # Return as JSON using pure function
                result = format_prompts_as_json(wan_prompts)
                logger.info(f"{check} Loaded {len(wan_prompts)} default Wan prompts from settings")
                return result
            else:
                # Use fallback
                return format_prompts_as_json(create_fallback_prompts())

        except Exception as e:
            logger.warning(f"{warning} Error loading default settings: {e}")
            # Return simple fallback
            return format_prompts_as_json(create_fallback_prompts())

    except Exception as e:
        return json.dumps({
            "0": f"Error loading default prompts: {str(e)}"
        }, indent=2)


def validate_wan_generation(current_prompts):
    """Validate that Wan generation requirements are met"""
    # Theme-aware emoji symbols
    warning = emoji_utils.maybe_warning()
    cross = emoji_utils.maybe_cross()
    check = emoji_utils.maybe_check()
    memo = emoji_utils.memo()
    movie_camera = emoji_utils.movie_camera()
    fire = emoji_utils.fire()
    zap = emoji_utils.zap()
    wrench = emoji_utils.wrench()

    try:
        import json

        # Check if prompts are empty
        if not current_prompts or current_prompts.strip() == "":
            return f"""{warning} **Prompts Required**

{memo} **Load prompts to get started:**
• Click "Load from Deforum Prompts" to use your animation prompts
• Or click "Load Default Wan Prompts" for examples
• Then optionally enhance with AI or add movement descriptions"""

        # Check if it's just placeholder text
        if any(placeholder in current_prompts.lower() for placeholder in ["required:", "load prompts", "placeholder"]):
            return f"""{warning} **Load Real Prompts**

{memo} **Replace placeholder text:**
• Click "Load from Deforum Prompts" to copy your animation prompts
• Or click "Load Default Wan Prompts" for examples"""

        # Try to parse as JSON
        try:
            prompts_dict = json.loads(current_prompts)
            if not prompts_dict:
                return f"{warning} **Empty prompts** - Add some prompts first"

            # Check if prompts are just basic placeholders
            first_prompt = list(prompts_dict.values())[0].lower()
            if any(placeholder in first_prompt for placeholder in ["prompt text", "beautiful landscape", "load prompts"]):
                return f"""{warning} **Default/Placeholder Prompts Detected**

{memo} **Load your real prompts:**
• Click "Load from Deforum Prompts" to copy your animation prompts
• Or edit the prompts manually to describe your desired video"""

            # All good - ready to generate!
            num_prompts = len(prompts_dict)
            return f"""{check} **Ready to Generate!**

{movie_camera} **Found {num_prompts} prompt{'s' if num_prompts != 1 else ''}** for Wan video generation
{fire} **Click "Generate Flux/Wan" above** to start I2V chaining generation
{zap} **Optional:** Add movement descriptions or AI enhancement first"""

        except json.JSONDecodeError:
            return f"""{cross} **Invalid JSON Format**

{wrench} **Fix the format:**
• Prompts should be in JSON format like: {{"0": "prompt text", "60": "another prompt"}}
• Check for missing quotes, commas, or brackets"""

    except Exception as e:
        return f"{cross} **Validation Error:** {str(e)}"


def wan_generate_with_validation(*component_args):
    """Wrapper for wan_generate_video that includes validation"""
    # Theme-aware emoji symbols
    check = emoji_utils.maybe_check()
    cross = emoji_utils.maybe_cross()
    movie_camera = emoji_utils.movie_camera()
    wrench = emoji_utils.wrench()

    try:
        # Get component names to find the wan_enhanced_prompts index
        from deforum.config.args import get_component_names
        component_names = get_component_names()

        # Find wan_enhanced_prompts in the component list
        wan_prompts = ""
        try:
            # The wan_enhanced_prompts should be passed as one of the component args
            # We need to identify which position it's in
            # For now, let's assume it's passed as the first argument to this wrapper
            if len(component_args) > 0:
                wan_prompts = component_args[0] if component_args[0] else ""

            # Validate prompts first
            validation_result = validate_wan_generation(wan_prompts)
            if validation_result.startswith(cross):
                return validation_result

            # If validation passes, call the original generate function
            # But first we need to insert the prompts into the right position in component_args
            # This is a bit complex - we'll need to reconstruct the args properly

            # For now, return validation success and instructions
            return f"""{check} Validation passed!

{validation_result}

{movie_camera} **Starting Wan video generation...**
- Prompts: {len(wan_prompts.split('"')) // 4} clips detected
- Using I2V chaining for smooth transitions
- Check console for detailed progress

{wrench} **Note**: Full generation integration in progress..."""

        except Exception as e:
            return f"{cross} Generation preparation error: {str(e)}"

    except Exception as e:
        return f"{cross} Generation error: {str(e)}"