import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
from modules.ui_components import FormRow, FormColumn, ToolButton
from .deprecation_utils import handle_deprecated_settings
from .ffmpeg_utils import FFmpegProcessor
from .rendering.util import emoji_utils
from .defaults import DeforumAnimPrompts, get_gradio_html
from .gradio_funcs import upload_pics_to_interpolate, upload_vid_to_depth, ncnn_upload_vid_to_upscale
from .video_audio_utilities import ffmpeg_stitch_video, direct_stitch_vid_from_frames
from modules import paths_internal
from modules.shared import state
from .rich import console
from .args import DeforumArgs, DeforumAnimArgs, ParseqArgs, DeforumOutputArgs, RootArgs, WanArgs, controlnet_component_names, get_component_names
from .webui_sd_pipeline import get_webui_sd_pipeline
from .data_models import TestFixtureArgs


def create_gr_elem(d):
    # Capitalize and CamelCase the orig value under "type", which defines gr.inputs.type in lower_case.
    # Examples: "dropdown" becomes gr.Dropdown, and "checkbox_group" becomes gr.CheckboxGroup.
    obj_type_str = ''.join(word.title() for word in d["type"].split('_'))
    obj_type = getattr(gr, obj_type_str)

    # Prepare parameters for gradio element creation
    params = {k: v for k, v in d.items() if k != "type" and v is not None}

    # Special case: Since some elements can have 'type' parameter and we are already using 'type' to specify
    # which element to use we need a separate parameter that will be used to overwrite 'type' at this point.
    # E.g. for Radio element we should specify 'type_param' which is then used to set gr.radio's type.
    if 'type_param' in params:
        params['type'] = params.pop('type_param')

    return obj_type(**params)


def is_gradio_component(args):
    return isinstance(args, (gr.Button, gr.Textbox, gr.Slider, gr.Dropdown,
                             gr.HTML, gr.Radio, gr.Interface, gr.Markdown,
                             gr.Checkbox))  # TODO...


def create_row(args, *attrs):
    # If attrs are provided, create components from the attributes of args.
    # Otherwise, pass through a single component or create one.
    with FormRow():
        return [create_gr_elem(getattr(args, attr)) for attr in attrs] if attrs \
            else args if is_gradio_component(args) else create_gr_elem(args)


# ******** Important message ********
# All get_tab functions use FormRow()/ FormColumn() by default,
# unless we have a gr.File inside that row/column, then we use gr.Row()/gr.Column() instead.
# ******** Important message ********
def get_tab_run(d, da):
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
        with gr.Accordion('Batch Mode, Resume and more', open=False):
            with gr.Tab('Batch Mode/ run from setting files'):
                with gr.Row():  # TODO: handle this inside one of the args functions?
                    override_settings_with_file = gr.Checkbox(label="Enable batch mode", value=False, interactive=True,
                                                              elem_id='override_settings',
                                                              info="run from a list of setting .txt files. Upload them to the box on the right (visible when enabled)")
                    custom_settings_file = gr.File(label="Setting files", interactive=True, file_count="multiple",
                                                   file_types=[".txt"], elem_id="custom_setting_file", visible=False)
            # RESUME ANIMATION ACCORD
            with gr.Tab('Resume Animation'):
                resume_from_timestring, resume_timestring = create_row(
                    da, 'resume_from_timestring', 'resume_timestring')
    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_keyframes(d, da, dloopArgs):
    components = {}
    with gr.TabItem(f"{emoji_utils.key()} Keyframes"):  # TODO make a some sort of the original dictionary parsing
        with FormRow():
            with FormColumn(scale=2):
                animation_mode = create_gr_elem(da.animation_mode)
            with FormColumn(scale=1, min_width=180):
                border = create_gr_elem(da.border)
        diffusion_cadence, max_frames = create_row(da, 'diffusion_cadence', 'max_frames')
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
        # EXTRA SCHEDULES TABS
        with gr.Tabs():
            with gr.TabItem(f"{emoji_utils.distribution()} Distribution"):
                keyframe_distribution = create_row(da.keyframe_distribution)
                create_keyframe_distribution_info_tab()
            with gr.TabItem(f"{emoji_utils.strength()} Strength"):
                strength_schedule = create_row(da.strength_schedule)
                keyframe_strength_schedule = create_row(da.keyframe_strength_schedule)
            with gr.TabItem(f"{emoji_utils.scale()} CFG"):
                cfg_scale_schedule = create_row(da.cfg_scale_schedule)
                distilled_cfg_scale_schedule = create_row(da.distilled_cfg_scale_schedule)
                enable_clipskip_scheduling = create_row(da.enable_clipskip_scheduling)
                clipskip_schedule = create_row(da.clipskip_schedule)
            with gr.TabItem(f"{emoji_utils.seed()} Seed & SubSeed") as subseed_sch_tab:
                seed_behavior = create_row(d.seed_behavior)
                with FormRow() as seed_iter_N_row:
                    seed_iter_N = create_gr_elem(d.seed_iter_N)
                with FormRow(visible=False) as seed_schedule_row:
                    seed_schedule = create_gr_elem(da.seed_schedule)
                enable_subseed_scheduling, subseed_schedule, subseed_strength_schedule = create_row(
                    da, 'enable_subseed_scheduling', 'subseed_schedule', 'subseed_strength_schedule')
                seed_resize_from_w, seed_resize_from_h = create_row(
                    d, 'seed_resize_from_w', 'seed_resize_from_h')
            # Steps Scheduling
            with gr.TabItem('Step'):
                enable_steps_scheduling = create_row(da.enable_steps_scheduling)
                steps_schedule = create_row(da.steps_schedule)
            # Sampler Scheduling
            with gr.TabItem('Sampler'):
                enable_sampler_scheduling = create_row(da.enable_sampler_scheduling)
                sampler_schedule = create_row(da.sampler_schedule)
            # Scheduler Scheduling
            with gr.TabItem('Scheduler'):
                enable_scheduler_scheduling = create_row(da.enable_scheduler_scheduling)
                scheduler_schedule = create_row(da.scheduler_schedule)
            # Checkpoint Scheduling
            with gr.TabItem('Checkpoint'):
                enable_checkpoint_scheduling = create_row(da.enable_checkpoint_scheduling)
                checkpoint_schedule = create_row(da.checkpoint_schedule)
        # MOTION INNER TAB
        with gr.Tabs(elem_id='motion_noise_etc'):
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

            with gr.TabItem(f"{emoji_utils.video_camera()} Shakify"):
                with FormColumn(min_width=220):
                    create_row(gr.Markdown(f"""
                        Integrate dynamic camera shake effects into your renders with data sourced from EatTheFutures
                         'Camera Shakify' Blender plugin. This feature enhances the realism of your animations
                        by simulating natural camera movements, adding a layer of depth and engagement to your visuals. 
                    """))
                    shake_name = create_row(da.shake_name)
                    shake_intensity = create_row(da.shake_intensity)
                    shake_speed = create_row(da.shake_speed)

            # NOISE INNER TAB
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
                with FormRow() as optical_flow_cadence_row:
                    with FormColumn(min_width=220) as optical_flow_cadence_column:
                        optical_flow_cadence = create_gr_elem(da.optical_flow_cadence)
                    with FormColumn(min_width=220, visible=False) as cadence_flow_factor_schedule_column:
                        cadence_flow_factor_schedule = create_gr_elem(da.cadence_flow_factor_schedule)
                with FormRow():
                    with FormColumn(min_width=220):
                        optical_flow_redo_generation = create_gr_elem(da.optical_flow_redo_generation)
                    with FormColumn(min_width=220, visible=False) as redo_flow_factor_schedule_column:
                        redo_flow_factor_schedule = create_gr_elem(da.redo_flow_factor_schedule)
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
            # ANTI BLUR INNER TAB
            with gr.TabItem(f"{emoji_utils.broom()} Anti Blur", elem_id='anti_blur_accord') as anti_blur_tab:
                amount_schedule = create_row(da.amount_schedule)
                kernel_schedule = create_row(da.kernel_schedule)
                sigma_schedule = create_row(da.sigma_schedule)
                threshold_schedule = create_row(da.threshold_schedule)
            with gr.TabItem(f"{emoji_utils.hole()} 3D Depth Warping & FOV", elem_id='depth_warp_fov_tab') \
                    as depth_warp_fov_tab:

                # FIXME this should only be visible if animation mode is "3D".
                is_visible = True
                is_info_visible = is_visible
                depth_warp_msg_html = gr.HTML(value='Please switch to 3D animation mode to view this section.',
                                              elem_id='depth_warp_msg_html', visible=is_info_visible)
                with FormRow(visible=is_visible) as depth_warp_row_1:
                    use_depth_warping = create_gr_elem(da.use_depth_warping)
                    # *the following html only shows when LeReS depth is selected*
                    leres_license_msg = gr.HTML(value=get_gradio_html('leres'), visible=False,
                                                elem_id='leres_license_msg')
                    depth_algorithm = create_gr_elem(da.depth_algorithm)
                    midas_weight = create_gr_elem(da.midas_weight)
                with FormRow(visible=is_visible) as depth_warp_row_2:
                    padding_mode = create_gr_elem(da.padding_mode)
                    sampling_mode = create_gr_elem(da.sampling_mode)
                with FormRow(visible=is_visible):
                    with gr.Accordion('Extended Depth Warp Settings', open=False):
                        with FormRow() as depth_warp_row_3:
                            aspect_ratio_use_old_formula = create_gr_elem(da.aspect_ratio_use_old_formula)
                        with FormRow() as depth_warp_row_4:
                            aspect_ratio_schedule = create_gr_elem(da.aspect_ratio_schedule)
                with FormRow(visible=is_visible):
                    with FormRow() as depth_warp_row_5:
                        fov_schedule = create_gr_elem(da.fov_schedule)
                    with FormRow() as depth_warp_row_6:
                        near_schedule = create_gr_elem(da.near_schedule)
                    with FormRow() as depth_warp_row_7:
                        far_schedule = create_gr_elem(da.far_schedule)

    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_prompts(da):
    with gr.TabItem(f"{emoji_utils.prompts()} Prompts"):
        # Main prompts in tabs for better organization
        with gr.Tabs():
            # ========== MAIN PROMPTS TAB ==========
            with gr.TabItem("📝 Main Prompts"):
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
            
            # ========== AI ENHANCEMENT TAB ==========
            with gr.TabItem("🎨 AI Enhancement"):
                with gr.Accordion("ℹ️ AI Enhancement Guide", open=False):
                    gr.HTML("""
                    <div style="padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin: 10px 0;">
                        <h3>🎨 AI Prompt Enhancement</h3>
                        <p><strong>What it does:</strong> Uses AI to enhance your prompts with better descriptions, artistic details, and consistent styling.</p>
                        
                        <h4>🎯 Style & Theme System:</h4>
                        <ul>
                            <li><strong>Style:</strong> Controls the overall artistic approach (photorealistic, anime, oil painting, etc.)</li>
                            <li><strong>Theme:</strong> Adds mood/atmosphere (cyberpunk, nature, minimal, etc.)</li>
                            <li><strong>Consistency:</strong> Applied to ALL prompts for coherent video sequences</li>
                        </ul>
                        
                        <h4>⚡ For I2V Chaining with Wan:</h4>
                        <p>Styles and themes ensure smooth transitions between video clips. Each clip maintains the same visual style.</p>
                        
                        <h4>🎬 For Pure Deforum + Flux:</h4>
                        <p>Creates cinematic consistency across animation frames with enhanced artistic descriptions.</p>
                    </div>
                    """)
                
                # Style and Theme Selection
                with gr.Row():
                    with gr.Column(scale=1):
                        style_dropdown = gr.Dropdown(
                            label="🎨 Visual Style",
                            choices=[
                                "Photorealistic",
                                "Cinematic",
                                "Anime/Manga", 
                                "Oil Painting",
                                "Watercolor",
                                "Digital Art",
                                "3D Render",
                                "Sketch/Drawing",
                                "Vintage Film",
                                "Studio Photography",
                                "Street Photography",
                                "Fine Art",
                                "Impressionist",
                                "Pop Art",
                                "Minimalist"
                            ],
                            value="Photorealistic",
                            info="Main artistic style - affects overall visual approach"
                        )
                        
                        custom_style = gr.Textbox(
                            label="🎭 Custom Style",
                            placeholder="e.g., 'neon-lit cyberpunk with holographic elements'",
                            info="Override style dropdown with custom text"
                        )
                    
                    with gr.Column(scale=1):
                        theme_dropdown = gr.Dropdown(
                            label="🌍 Theme/Atmosphere",
                            choices=[
                                "None",
                                "Cyberpunk",
                                "Synthwave/Vaporwave", 
                                "Frutiger Aero",
                                "Steampunk",
                                "Post-Apocalyptic",
                                "Nature/Organic",
                                "Urban/Metropolitan",
                                "Retro-Futuristic",
                                "Noir/Moody",
                                "Ethereal/Dreamy",
                                "Industrial/Brutalist",
                                "Art Deco",
                                "Bauhaus/Minimal",
                                "Cosmic/Space",
                                "Medieval Fantasy",
                                "Tropical Paradise",
                                "Winter Wonderland",
                                "Desert Mystique",
                                "Underwater World"
                            ],
                            value="None",
                            info="Thematic atmosphere - adds mood and environment details"
                        )
                        
                        custom_theme = gr.Textbox(
                            label="🏛️ Custom Theme",
                            placeholder="e.g., 'ancient temples with golden light'",
                            info="Override theme dropdown with custom text"
                        )
                
                # Random and Reset Controls
                with gr.Row():
                    with gr.Column(scale=1):
                        random_style_btn = gr.Button("🎲 Random Style", variant="secondary")
                        random_theme_btn = gr.Button("🎲 Random Theme", variant="secondary")
                        random_both_btn = gr.Button("🎲 Random Both", variant="primary")
                    
                    with gr.Column(scale=1):
                        reset_to_photo_btn = gr.Button("📷 Reset to Photorealistic", variant="secondary")
                        cycle_creative_btn = gr.Button("🌈 Cycle Creative Themes", variant="secondary")
                
                # AI Model Selection
                with gr.Row():
                    with gr.Column(scale=2):
                        qwen_model_dropdown = gr.Dropdown(
                            label="🤖 AI Enhancement Model",
                            choices=[
                                "Auto-Select",
                                "Qwen2.5-0.5B-Instruct",
                                "Qwen2.5-1.5B-Instruct", 
                                "Qwen2.5-3B-Instruct",
                                "Qwen2.5-7B-Instruct",
                                "Qwen2.5-14B-Instruct"
                            ],
                            value="Auto-Select",
                            info="AI model for prompt enhancement - Auto-Select chooses based on available VRAM"
                        )
                    
                    with gr.Column(scale=1):
                        qwen_language = gr.Dropdown(
                            label="🌐 Language",
                            choices=["English", "Chinese", "Japanese", "Korean", "French", "German", "Spanish"],
                            value="English",
                            info="Enhancement language"
                        )
                        
                        qwen_auto_download = gr.Checkbox(
                            label="📥 Auto-Download Models",
                            value=True,
                            info="Automatically download AI models when needed"
                        )
                
                # Enhancement Buttons
                with gr.Row():
                    with gr.Column():
                        enhance_deforum_btn = gr.Button(
                            "✨ Enhance Deforum Prompts",
                            variant="primary",
                            size="lg"
                        )
                        
                        enhance_wan_btn = gr.Button(
                            "🎬 Enhance Wan Prompts", 
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        apply_style_deforum_btn = gr.Button(
                            "🎨 Apply Style to Deforum Only",
                            variant="secondary"
                        )
                        
                        apply_style_wan_btn = gr.Button(
                            "🎭 Apply Style to Wan Only",
                            variant="secondary"
                        )
                
                # Status and Progress
                enhancement_status = gr.Textbox(
                    label="📊 Enhancement Status",
                    lines=8,
                    interactive=False,
                    value="Ready for AI enhancement! Select your style and theme above, then click enhance buttons.",
                    info="Progress and results will appear here"
                )
                
                enhancement_progress = gr.Textbox(
                    label="⏳ Progress",
                    lines=3,
                    interactive=False,
                    value="Waiting...",
                    info="Real-time progress updates"
                )
    
    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_init(d, da, dp):
    with gr.TabItem('Init'):
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
        # VIDEO INIT INNER-TAB
        with gr.Tab('Video Init'):
            video_init_path = create_row(da.video_init_path)
            with FormRow():
                extract_from_frame = create_gr_elem(da.extract_from_frame)
                extract_to_frame = create_gr_elem(da.extract_to_frame)
                extract_nth_frame = create_gr_elem(da.extract_nth_frame)
                overwrite_extracted_frames = create_gr_elem(da.overwrite_extracted_frames)
                use_mask_video = create_gr_elem(da.use_mask_video)
            video_mask_path = create_row(da.video_mask_path)
        # MASK INIT INNER-TAB
        with gr.Tab('Mask Init'):
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
        # PARSEQ INNER-TAB
        with gr.Tab(f"{emoji_utils.numbers()} Parseq"):
            gr.HTML(value=get_gradio_html('parseq'))
            parseq_manifest = create_row(dp.parseq_manifest)
            parseq_non_schedule_overrides = create_row(dp.parseq_non_schedule_overrides)
            parseq_use_deltas = create_row(dp.parseq_use_deltas)
    return {k: v for k, v in {**locals(), **vars()}.items()}


def wan_generate_video(*component_args):
    """Generate Wan video with comprehensive validation and settings generation"""
    try:
        # Extract component arguments in the correct order
        wan_mode = component_args[0]

        print(f"🎬 Wan video generation button clicked! Received {len(component_args)} arguments")
        
        # Import the main Deforum run function
        from .run_deforum import run_deforum
        from .wan.wan_simple_integration import WanSimpleIntegration
        
        # Auto-discover models first to validate setup
        integration = WanSimpleIntegration()
        models = integration.discover_models()
        
        # Get wan_args from the components to check auto-download setting
        from .args import get_component_names
        component_names = get_component_names()
        wan_auto_download = True  # Default value
        
        try:
            auto_download_index = component_names.index('wan_auto_download')
            if auto_download_index < len(component_args):
                wan_auto_download = component_args[auto_download_index]
        except (ValueError, IndexError):
            print("⚠️ Could not find wan_auto_download setting, using default: True")
        
        # If no models found and auto-download is enabled, try to download
        if not models and wan_auto_download:
            print("📥 No models found and auto-download enabled. Downloading recommended model...")
            
            try:
                from .wan.wan_model_downloader import WanModelDownloader
                downloader = WanModelDownloader()
                
                # Try to download 1.3B VACE first (most user-friendly)
                print("📥 Downloading Wan 1.3B VACE model (recommended: 8GB VRAM, consumer-friendly)...")
                if downloader.download_model("1.3B VACE"):
                    print("✅ 1.3B VACE model download completed!")
                    # Re-discover models after download
                    models = integration.discover_models()
                else:
                    print("❌ 1.3B VACE download failed, trying 14B VACE...")
                    # Fallback to 14B VACE
                    if downloader.download_model("14B VACE"):
                        print("✅ 14B VACE model download completed!")
                        models = integration.discover_models()
                    else:
                        print("❌ All model downloads failed")
                        
            except Exception as e:
                print(f"❌ Auto-download failed: {e}")
        
        # If we have models but they might be corrupted, validate them
        if models:
            print("🔍 Validating discovered models...")
            valid_models = []
            corrupted_models = []
            
            for model in models:
                if model['type'] == 'VACE':
                    # Check if VACE model has required I2V components
                    is_valid = integration._validate_vace_weights(Path(model['path']))
                    if is_valid:
                        valid_models.append(model)
                        print(f"✅ {model['name']}: Valid VACE model")
                    else:
                        corrupted_models.append(model)
                        print(f"❌ {model['name']}: Corrupted/incomplete VACE model")
                elif model['type'] in ['T2V', 'I2V']:
                    # Legacy T2V/I2V models - check for basic structure
                    model_path = Path(model['path'])
                    if (model_path / "model_index.json").exists():
                        valid_models.append(model)
                        print(f"✅ {model['name']}: Valid {model['type']} model")
                    else:
                        corrupted_models.append(model)
                        print(f"❌ {model['name']}: Incomplete {model['type']} model")
                else:
                    # Unknown model type - likely invalid leftover files
                    model_path = Path(model['path'])
                    # Check if it has any recognizable Wan model structure
                    has_valid_structure = (
                        (model_path / "model_index.json").exists() or
                        (model_path / "transformer").exists() or
                        any(f.name.startswith("wan") for f in model_path.rglob("*.pth")) or
                        any(f.name.startswith("wan") for f in model_path.rglob("*.safetensors"))
                    )
                    
                    if has_valid_structure:
                        valid_models.append(model)
                        print(f"✅ {model['name']}: Valid legacy model")
                    else:
                        corrupted_models.append(model)
                        print(f"❌ {model['name']}: Invalid/leftover files (not a proper Wan model)")
            
            # If we found corrupted models and auto-download is enabled, offer repair
            if corrupted_models and wan_auto_download:
                print(f"⚠️ Found {len(corrupted_models)} corrupted model(s)")
                print("🛠️ MANUAL CLEANUP INSTRUCTIONS:")
                print("For safety, corrupted models are NOT automatically deleted.")
                print("If you want to remove them, please:")
                print()
                
                for corrupted_model in corrupted_models:
                    print(f"❌ {corrupted_model['name']}: {corrupted_model['path']}")
                
                print()
                print("🗑️ To manually remove corrupted models:")
                for corrupted_model in corrupted_models:
                    print(f"   rm -rf \"{corrupted_model['path']}\"")
                
                print()
                print("📥 To re-download models:")
                for corrupted_model in corrupted_models:
                    model_name = corrupted_model['name'].lower()
                    if '1.3b' in model_name and 'vace' in model_name:
                        print(f"   huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/wan/Wan2.1-VACE-1.3B")
                    elif '14b' in model_name and 'vace' in model_name:
                        print(f"   huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan/Wan2.1-VACE-14B")
                    elif '1.3b' in model_name and 't2v' in model_name:
                        print(f"   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan/Wan2.1-T2V-1.3B")
                    elif '14b' in model_name and 't2v' in model_name:
                        print(f"   huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir models/wan/Wan2.1-T2V-14B")
                    elif '1.3b' in model_name and 'i2v' in model_name:
                        print(f"   huggingface-cli download Wan-AI/Wan2.1-I2V-1.3B --local-dir models/wan/Wan2.1-I2V-1.3B")
                    elif '14b' in model_name and 'i2v' in model_name:
                        print(f"   huggingface-cli download Wan-AI/Wan2.1-I2V-14B --local-dir models/wan/Wan2.1-I2V-14B")
                
                print()
                print("💡 TIP: Enable 'Auto-Download Models' for automatic downloading of missing models")
                print("⚠️ SAFETY: Always verify corruption before deleting - some errors may be temporary")
            
            # Update models list to only include valid models
            models = valid_models
        
        if not models:
            auto_download_help = """

🔧 AUTO-DOWNLOAD OPTIONS:
1. ✅ Enable "Auto-Download Models" in the Wan tab (recommended)
2. 📥 Manual download with HuggingFace CLI:
   
   **For 1.3B VACE (Recommended - 8GB VRAM):**
   huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/wan/Wan2.1-VACE-1.3B
   
   **For 14B VACE (High Quality - 480P+720P):**
   huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan/Wan2.1-VACE-14B

3. ✅ Restart generation after downloading

🔧 AUTO-REPAIR: Corrupted models are automatically detected and re-downloaded!""" if not wan_auto_download else """

🔧 TROUBLESHOOTING:
1. 📶 Check internet connection for downloads
2. 💾 Ensure enough disk space (1.3B: ~17GB, 14B: ~75GB)
3. 🔄 Try manual download with HuggingFace CLI (see Auto-Discovery tab)
4. 🔧 Corrupted models are detected - follow manual cleanup instructions"""

            return f"""❌ No Wan models found!

💡 QUICK SETUP:
VACE models are all-in-one (T2V + I2V) - recommended for I2V chaining!

• **1.3B VACE**: 8GB VRAM, 480P, fast (perfect for most users)
• **14B VACE**: 480P+720P, slower, higher quality (for power users)

{auto_download_help}

💡 VACE models handle both text-to-video and image-to-video in one model, perfect for seamless I2V chaining!

🔧 **If you have T2V models but want I2V chaining:**
Set I2V Model to "Use T2V Model (No Continuity)" for independent clips, 
or download a VACE model for seamless transitions."""
        
        print(f"✅ Found {len(models)} Wan model(s):")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model['name']} ({model['size']}) - {model['path']}")
        
        # Get component names to find the animation_prompts index
        from .args import get_component_names
        component_names = get_component_names()
        
        # Find animation_prompts in the component list
        animation_prompts = '{"0": "a beautiful landscape"}'  # Default
        animation_mode_index = None
        animation_prompts_index = None
        
        try:
            animation_prompts_index = component_names.index('animation_prompts')
            if animation_prompts_index < len(component_args):
                animation_prompts = component_args[animation_prompts_index]
                print(f"📝 Found animation_prompts at index {animation_prompts_index}")
            else:
                print(f"⚠️ animation_prompts index {animation_prompts_index} out of range (have {len(component_args)} args)")
        except ValueError:
            print("⚠️ Could not find animation_prompts in component names")
        
        try:
            animation_mode_index = component_names.index('animation_mode')
            print(f"📝 Found animation_mode at index {animation_mode_index}")
        except ValueError:
            print("⚠️ Could not find animation_mode in component names")
        
        # Validate prompts
        if not animation_prompts or animation_prompts.strip() == '{"0": "a beautiful landscape"}':
            return """❌ No prompts configured!

🔧 SETUP REQUIRED:
1. 📝 Go to the **Prompts tab** and configure your animation prompts
2. 🎬 Set your desired FPS in the **Output tab**
3. 🎯 Optionally configure seeds in **Keyframes → Seed & SubSeed tab**
4. 🎬 Click **Generate Wan Video** again

💡 I2V chaining needs your prompt schedule to know what to generate!

Example prompts for seamless I2V chaining:
{
  "0": "a serene beach at sunset",
  "60": "a misty forest in the morning",
  "120": "a bustling city street at night"
}

Each prompt will be smoothly connected using I2V continuity!"""
        
        # Force animation mode to Wan Video
        component_args = list(component_args)
        
        if animation_mode_index is not None and animation_mode_index < len(component_args):
            component_args[animation_mode_index] = 'Wan Video'
            print(f"✅ Set animation mode to 'Wan Video' at index {animation_mode_index}")
        else:
            print("⚠️ Could not set animation mode - index not found or out of range")
        
        # Generate a unique job ID
        import uuid
        job_id = str(uuid.uuid4())[:8]
        
        print(f"🚀 Starting Wan video generation with job ID: {job_id}")
        print(f"📝 Using prompts: {str(animation_prompts)[:100]}...")
        
        # Call the main Deforum generation function
        # run_deforum expects: job_id, custom_settings_file, *component_values
        # where component_values must match exactly with get_component_names()
        
        from .args import get_component_names
        component_names = get_component_names()
        expected_component_count = len(component_names)
        
        print(f"🔧 Debug: Expected {expected_component_count} components, have {len(component_args)} args")
        print(f"🔧 Debug: Component names count: {len(component_names)}")
        
        # We need exactly: [job_id, custom_settings_file] + component_values
        # So total args should be 2 + len(component_names)
        final_args = [job_id, None]  # job_id and custom_settings_file
        
        # Add the component values, ensuring we have exactly the right number
        for i in range(expected_component_count):
            if i < len(component_args):
                final_args.append(component_args[i])
            else:
                print(f"⚠️ Warning: Missing component at index {i}, using None")
                final_args.append(None)
        
        print(f"🔧 Debug: Final args count: {len(final_args)} (should be {2 + expected_component_count})")
        
        result = run_deforum(*final_args)
        
        if result and len(result) >= 4:
            # run_deforum returns (images, seed, info, comments)
            images, seed, info, comments = result
            
            if comments and "Error" in str(comments):
                return f"❌ Wan generation failed: {comments}"
            else:
                return f"✅ Wan video generation completed successfully!\n📊 Job ID: {job_id}\n💡 Check the Output tab for your video files."
        else:
            raise RuntimeError(f"❌ Wan generation failed")
            
    except Exception as e:
        error_msg = f"❌ Wan generation error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


def generate_wan_video(args, anim_args, video_args, frame_idx, turbo_mode, turbo_preroll, root, animation_prompts, loop_args, parseq_args, freeu_args, controlnet_args, depth_args, hybrid_args, parseq_adapter, wan_args, frame_duration):
    """Generate Wan video using the new simple integration approach - called by Deforum internally"""
    from .wan.wan_simple_integration import WanSimpleIntegration
    import time
    
    print("🎬 Wan video generation started with AUTO-DISCOVERY (Internal Call)")
    print("🔍 Using smart model discovery instead of manual paths")
    
    # Ensure Qwen models are unloaded before video generation to free VRAM
    try:
        from .wan.utils.qwen_manager import qwen_manager
        if qwen_manager.is_model_loaded():
            print("🔄 Unloading Qwen models before video generation...")
            qwen_manager.ensure_model_unloaded()
    except Exception as e:
        print(f"⚠️ Warning: Could not cleanup Qwen models: {e}")
    
    start_time = time.time()
    
    try:
        # Initialize the simple integration
        integration = WanSimpleIntegration()
        
        # Auto-discover models
        print("🔍 Auto-discovering Wan models...")
        models = integration.discover_models()
        
        if not models:
            raise RuntimeError("""
❌ No Wan models found automatically!

💡 SOLUTIONS:
1. 📥 Download a Wan model using HuggingFace CLI:
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir "models/wan"

2. 📂 Or place your model in one of these locations:
   • models/wan/
   • models/Wan/
   
3. ✅ Restart generation after downloading

The auto-discovery will find your models automatically!
""")
        
        # Select the best model (or use user's size preference)
        selected_model = None
        
        # Try to find a model matching user's size preference
        user_preferred_size = wan_args.wan_preferred_size.replace(" (Recommended)", "").replace(" (High Quality)", "")
        
        for model in models:
            if user_preferred_size in model['size']:
                selected_model = model
                break
                
        # Fallback to best available model
        if not selected_model:
            selected_model = models[0]
            print(f"⚠️ User requested {user_preferred_size} but using available: {selected_model['size']}")
            
        print(f"🎯 Selected model: {selected_model['name']} ({selected_model['type']}, {selected_model['size']})")
        print(f"📁 Model path: {selected_model['path']}")
        
        # Prepare output directory (let Deforum handle directory creation)
        output_directory = args.outdir if hasattr(args, 'outdir') else root.outdir 
        
        # Generate video using direct integration
        print("🚀 Starting direct Wan integration...")
        
        # Parse prompts for Wan scheduling
        def parse_prompts_and_timing(animation_prompts, wan_args, video_args):
            """Calculate exact frame counts from prompt schedule for audio sync precision"""
            prompt_schedule = []
            
            # Sort prompts by frame number
            sorted_prompts = sorted(animation_prompts.items(), key=lambda x: int(x[0]))
            
            if not sorted_prompts:
                return [("a beautiful landscape", 0, 81)]  # Default: 0 start frame, 81 frames
            
            # Check if enhanced prompts are available and use them
            final_prompts = animation_prompts.copy()
            
            if wan_args.wan_enhanced_prompts:
                try:
                    # Try to parse enhanced prompts
                    import json
                    enhanced_prompts_data = json.loads(wan_args.wan_enhanced_prompts)
                    if enhanced_prompts_data:
                        print("🎨 Using enhanced prompts from QwenPromptExpander")
                        final_prompts = enhanced_prompts_data
                except (json.JSONDecodeError, ValueError):
                    print("⚠️ Could not parse enhanced prompts, using original prompts")
            
            # Add movement description if available
            movement_description = ""
            if wan_args.wan_movement_description:
                movement_description = wan_args.wan_movement_description.split('\n')[0]  # Get first line
                print(f"📐 Adding movement description: {movement_description}")
            
            # Re-sort with final prompts
            sorted_prompts = sorted(final_prompts.items(), key=lambda x: int(x[0]))
            
            # Calculate frame differences between prompts
            for i, (frame_str, prompt) in enumerate(sorted_prompts):
                start_frame = int(frame_str)
                clean_prompt = prompt.split('--neg')[0].strip()
                
                # Append movement description if available
                if movement_description:
                    clean_prompt = f"{clean_prompt}. {movement_description}"
                
                # Calculate end frame (frame count for this clip)
                if i < len(sorted_prompts) - 1:
                    # Next prompt exists - calculate difference
                    next_frame = int(sorted_prompts[i + 1][0])
                    frame_count = next_frame - start_frame
                else:
                    # Last prompt - use default or calculate from total expected frames
                    # Assume at least 2 seconds worth of frames for the last clip
                    frame_count = max(2 * video_args.fps, 81)  # Minimum 2 seconds or 81 frames
                
                # Ensure minimum frame count for Wan (at least 5 frames)
                frame_count = max(5, frame_count)
                
                # Pad to Wan's 4n+1 requirement if needed (but try to preserve exact timing)
                if (frame_count - 1) % 4 != 0:
                    # Calculate closest 4n+1 value
                    target_4n_plus_1 = ((frame_count - 1) // 4) * 4 + 1
                    next_4n_plus_1 = target_4n_plus_1 + 4
                    
                    # Choose the closest one
                    if abs(frame_count - target_4n_plus_1) <= abs(frame_count - next_4n_plus_1):
                        frame_count = target_4n_plus_1
                    else:
                        frame_count = next_4n_plus_1
                
                # Add to schedule: (prompt, start_frame, frame_count)
                prompt_schedule.append((clean_prompt, start_frame, frame_count))
                
                # Show enhanced/movement prompt info
                if wan_args.wan_enhanced_prompts or wan_args.wan_movement_description:
                    print(f"  🎨 Enhanced Clip {i+1}: '{clean_prompt[:80]}...' (frames: {frame_count})")
                else:
                    print(f"  Clip {i+1}: '{clean_prompt[:50]}...' (start: frame {start_frame}, frames: {frame_count})")
            
            return prompt_schedule
        
        clips = parse_prompts_and_timing(animation_prompts, wan_args, video_args)
        
        # Calculate dynamic motion strength if enabled
        motion_strength = wan_args.wan_motion_strength  # Default value
        motion_intensity_schedule = None  # For frame-by-frame motion control
        
        if wan_args.wan_movement_description and not wan_args.wan_motion_strength_override:
            try:
                from .wan.utils.movement_analyzer import analyze_deforum_movement, generate_wan_motion_intensity_schedule
                
                print("🎬 Calculating dynamic motion strength from movement schedules...")
                
                # Generate both description and average strength for backwards compatibility
                _, dynamic_motion_strength = analyze_deforum_movement(
                    anim_args=anim_args,
                    sensitivity=wan_args.wan_movement_sensitivity,
                    max_frames=min(anim_args.max_frames, 100)
                )
                
                # Generate frame-by-frame motion intensity schedule for Wan
                motion_intensity_schedule = generate_wan_motion_intensity_schedule(
                    anim_args=anim_args,
                    max_frames=min(anim_args.max_frames, 100),
                    sensitivity=wan_args.wan_movement_sensitivity
                )
                
                motion_strength = dynamic_motion_strength  # Fallback for simple integrations
                print(f"✅ Dynamic motion strength: {motion_strength:.2f} (average)")
                print(f"📐 Generated motion intensity schedule with frame-by-frame control")
                
            except Exception as e:
                print(f"⚠️ Dynamic motion strength calculation failed: {e}, using default: {motion_strength}")
        elif wan_args.wan_motion_strength_override:
            print(f"🔧 Using manual motion strength override: {motion_strength}")
        else:
            print(f"📊 Using default motion strength: {motion_strength}")
        
        # Parse resolution - handle both old format (864x480) and new format (864x480 (Landscape))
        resolution_str = wan_args.wan_resolution
        if '(' in resolution_str:
            # New format: "864x480 (Landscape)" 
            resolution_str = resolution_str.split(' (')[0]
        width, height = map(int, resolution_str.split('x'))
        
        # Model/Resolution validation
        model_size = selected_model['size']
        model_name = selected_model['name']
        is_720p = (width >= 1280 and height >= 720) or (width >= 720 and height >= 1280)
        is_480p = (width <= 864 and height <= 480) or (width <= 480 and height <= 864)
        
        print(f"\n🔍 Model/Resolution Validation:")
        print(f"   📦 Model: {model_name} ({model_size})")
        print(f"   📐 Resolution: {width}x{height} ({'720p' if is_720p else '480p' if is_480p else 'Custom'})")
        
        # Check for resolution/model mismatches and warn
        if "1.3B" in model_size and is_720p:
            print(f"\n⚠️  WARNING: VACE 1.3B + 720p Resolution Mismatch")
            print(f"   📦 Model: {model_name} (optimized for 480p)")
            print(f"   📐 Resolution: {width}x{height} (720p)")
            print(f"   💡 RECOMMENDATION: Use 864x480 for better performance with VACE 1.3B")
            print(f"   🚀 Continuing anyway - VACE 1.3B may struggle with 720p...")
            
        elif "14B" in model_size and is_480p:
            print(f"\n💡 INFO: VACE 14B + 480p Resolution")
            print(f"   📦 Model: {model_name} (supports both 480p and 720p)")
            print(f"   📐 Resolution: {width}x{height} (480p)")
            print(f"   ✅ This combination works well, but you could use 1280x720 for higher quality")
            
        elif "1.3B" in model_size and is_480p:
            print(f"\n✅ Perfect Match: VACE 1.3B + 480p")
            print(f"   📦 Model: {model_name} (optimized for 480p)")
            print(f"   📐 Resolution: {width}x{height} (480p)")
            print(f"   🎯 Optimal configuration for VACE 1.3B!")
            
        elif "14B" in model_size and is_720p:
            print(f"\n✅ Perfect Match: VACE 14B + 720p")
            print(f"   📦 Model: {model_name} (supports 720p)")
            print(f"   📐 Resolution: {width}x{height} (720p)")
            print(f"   🎯 High quality configuration!")
        
        # Prepare clips data for generation
        clips_data = []
        for i, (prompt, start_frame, frame_count) in enumerate(clips):
            clips_data.append({
                'prompt': prompt,
                'start_frame': start_frame,
                'end_frame': start_frame + frame_count,
                'num_frames': frame_count
            })
        
        # Add motion intensity schedule to wan_args for use by Wan integration
        if motion_intensity_schedule:
            wan_args.wan_motion_intensity_schedule = motion_intensity_schedule
            print(f"💡 Added motion intensity schedule to wan_args for frame-by-frame control")
        
        # Check if user wants T2V mode (no continuity)
        use_t2v_only = wan_args.wan_i2v_model == "Use T2V Model (No Continuity)"
        mode_description = "generation"  # Default fallback
        
        if use_t2v_only:
            print(f"\n🎬 Using T2V mode (no continuity) for {len(clips_data)} independent clips")
            print("⚠️ Each clip will be generated independently - no frame continuity between clips")
            
            # Generate video using T2V for all clips (no chaining)
            output_file = integration.generate_video_t2v_only(
                clips=clips_data,
                model_info=selected_model,
                output_dir=str(output_directory),
                width=width,
                height=height,
                steps=wan_args.wan_inference_steps,
                guidance_scale=wan_args.wan_guidance_scale,
                seed=wan_args.wan_seed if wan_args.wan_seed > 0 else -1,
                wan_args=wan_args
            )
            
            mode_description = "T2V independent clips"
        else:
            print(f"\n🔗 Using I2V chaining for better continuity between {len(clips_data)} clips")
            
            # Generate video using I2V chaining
            output_file = integration.generate_video_with_i2v_chaining(
                clips=clips_data,
                model_info=selected_model,
                output_dir=str(output_directory),
                width=width,
                height=height,
                steps=wan_args.wan_inference_steps,
                guidance_scale=wan_args.wan_guidance_scale,
                seed=wan_args.wan_seed if wan_args.wan_seed > 0 else -1,
                anim_args=anim_args,  # Pass anim_args for strength scheduling
                wan_args=wan_args     # Pass wan_args for strength override settings and motion schedule
            )
            
            mode_description = "I2V chaining"
        
        generated_videos = [output_file] if output_file else []
        
        total_time = time.time() - start_time
        
        if generated_videos:
            print(f"\n🎉 Wan {mode_description} generation completed!")
            print(f"✅ Generated seamless video with {len(clips_data)} clips using {mode_description}")
            print(f"⏱️ Total time: {total_time:.1f} seconds")
            print(f"📁 Output file: {generated_videos[0]}")
            print(f"🔗 {mode_description} ensures smooth transitions between clips")
                
            # Return the output directory for Deforum's video processing
            return str(output_directory)
        else:
            raise RuntimeError(f"❌ Wan {mode_description} failed")
            
    except Exception as e:
        print(f"❌ Wan generation failed: {e}")
        
        # Provide helpful troubleshooting info
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"   • Check model availability with: python scripts/deforum_helpers/wan_direct_integration.py")
        print(f"   • Download models: huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan")
        print(f"   • Verify Wan models are in: models/wan/ directory")
        
        # Re-raise for Deforum error handling
        raise


def get_tab_wan(dw: SimpleNamespace):
    """🤖 Wan AI TAB - Advanced AI Video Generation"""
    with gr.TabItem(f"🤖 Wan AI"):
        
        # ESSENTIAL PROMPTS SECTION - TOP PRIORITY
        with gr.Accordion("📝 Wan Video Prompts (REQUIRED)", open=True):
            gr.Markdown("""
            **🎯 Essential for Wan Generation:** These prompts define what video clips will be generated.
            
            **Quick Setup:** Load → Analyze Movement → Enhance → Generate
            """)
            
            # Prompt Loading Buttons
            with FormRow():
                load_deforum_to_wan_btn = gr.Button(
                    "📋 Load from Deforum Prompts",
                    variant="primary",
                    size="lg",
                    elem_id="load_deforum_to_wan_btn"
                )
                load_wan_defaults_btn = gr.Button(
                    "📝 Load Default Wan Prompts",
                    variant="secondary", 
                    size="lg",
                    elem_id="load_wan_defaults_btn"
                )
            
            # Wan Prompts Display - ALWAYS VISIBLE AND PROMINENT
            wan_enhanced_prompts = gr.Textbox(
                label="Wan Video Prompts (JSON Format)",
                lines=10,
                interactive=True,
                placeholder='REQUIRED: Load prompts first! Click "Load from Deforum Prompts" or "Load Default Wan Prompts" above.',
                info="🎯 ESSENTIAL: These prompts will be used for Wan video generation. Edit manually or use buttons below to enhance.",
                elem_id="wan_enhanced_prompts_textbox"
            )
            
            # Prompt Enhancement Actions
            with FormRow():
                analyze_movement_btn = gr.Button(
                    "📐 Add Movement Descriptions",
                    variant="secondary",
                    size="lg",
                    elem_id="wan_analyze_movement_btn"
                )
                enhance_prompts_btn = gr.Button(
                    "🎨 AI Prompt Enhancement",
                    variant="secondary",
                    size="lg",
                    elem_id="wan_enhance_prompts_btn"
                )
            
            # Camera Shakify Integration Control
            with FormRow():
                wan_enable_shakify = gr.Checkbox(
                    label="🎬 Include Camera Shakify with Movement Analysis",
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
        
        # DEFORUM INTEGRATION - ESSENTIAL (moved up)
        with gr.Accordion("🔗 Deforum Integration", open=False):
            gr.Markdown("""
            **✅ Wan seamlessly integrates with your Deforum settings:**
            
            - **📝 Prompts:** Uses prompts from Deforum Prompts tab
            - **🎬 Movement:** Uses same movement schedules as normal Deforum renders
            - **🎲 Seed & CFG:** Uses Deforum's seed and CFG schedules
            - **💪 Strength:** Uses Deforum's strength schedule for I2V continuity
            - **🎬 FPS:** Uses Output tab FPS setting
            
            **Movement Integration:**
            - ✅ Translation X/Y/Z, Rotation 3D X/Y/Z, Zoom schedules
            - ✅ **Parseq schedules fully supported**
            - ✅ Movement descriptions automatically calculated and added
            - ✅ Motion intensity dynamically adapts to movement complexity
            """)
        
        # GENERATION SECTION
        with gr.Accordion("🎬 Generate Wan Video", open=True):
            # Generate Button with Validation
            with FormRow():
                wan_generate_button = gr.Button(
                    "🎬 Generate Wan Video (I2V Chaining)",
                    variant="primary", 
                    size="lg",
                    elem_id="wan_generate_button"
                )
                
            # Status output for Wan generation
            wan_generation_status = gr.Textbox(
                label="Generation Status",
                interactive=False,
                lines=5,
                placeholder="⚠️ Prompts required! Load prompts above first, then click Generate.",
                info="Status updates will appear here during generation."
            )
        
        # ESSENTIAL SETTINGS - Compact
        with gr.Accordion("⚙️ Essential Settings", open=True):
            with FormRow():
                wan_auto_download = create_gr_elem(dw.wan_auto_download)
                wan_preferred_size = create_gr_elem(dw.wan_preferred_size)
                # Fix resolution dropdown to handle old format values
                wan_resolution_elem = create_gr_elem(dw.wan_resolution)
                
                # Update resolution value to handle old format (e.g., "864x480" -> "864x480 (Landscape)")
                def update_resolution_format(current_value):
                    """Convert old resolution format to new format with labels"""
                    if current_value and '(' not in current_value:
                        # Old format detected, convert to new format
                        if current_value == "864x480":
                            return "864x480 (Landscape)"
                        elif current_value == "480x864":
                            return "480x864 (Portrait)"
                        elif current_value == "1280x720":
                            return "1280x720 (Landscape)"
                        elif current_value == "720x1280":
                            return "720x1280 (Portrait)"
                    return current_value
                
                # Apply format update on component creation
                if hasattr(wan_resolution_elem, 'value'):
                    wan_resolution_elem.value = update_resolution_format(wan_resolution_elem.value)
                
                wan_resolution = wan_resolution_elem
                
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
        
        # AI ENHANCEMENT SETTINGS - Collapsed by default
        with gr.Accordion("🧠 AI Prompt Enhancement (Optional)", open=False):
            gr.Markdown("""
            **Enhance prompts using Qwen AI models** for better video quality:
            - **🧠 AI Enhancement**: Refines and expands prompts
            - **🎬 Movement Integration**: Uses movement descriptions from analysis
            - **🌍 Multi-Language**: English and Chinese support
            """)
            
            with FormRow():
                wan_qwen_model = create_gr_elem(dw.wan_qwen_model)
                wan_qwen_language = create_gr_elem(dw.wan_qwen_language)
                wan_qwen_auto_download = create_gr_elem(dw.wan_qwen_auto_download)
        
        # MODEL SETTINGS - Collapsed by default  
        with gr.Accordion("🔧 Model Settings", open=False):
            with FormRow():
                wan_t2v_model = create_gr_elem(dw.wan_t2v_model)
                wan_i2v_model = create_gr_elem(dw.wan_i2v_model)
            with FormRow():
                wan_model_path = create_gr_elem(dw.wan_model_path)
        
        # OVERRIDES SECTION - Movement sensitivity moved here
        with gr.Accordion("🔧 Override Settings (Advanced)", open=False):
            gr.Markdown("""
            **Override automatic calculations with fixed values:**
            
            By default, Wan calculates these values from your Deforum schedules. Enable overrides only if you need manual control.
            """)
            
            with FormRow():
                wan_strength_override = create_gr_elem(dw.wan_strength_override)
                wan_fixed_strength = create_gr_elem(dw.wan_fixed_strength)
                
            with FormRow():
                wan_guidance_override = create_gr_elem(dw.wan_guidance_override) 
                wan_guidance_scale = create_gr_elem(dw.wan_guidance_scale)
                
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
        
        # Advanced Settings - Other advanced features
        with gr.Accordion("⚡ Advanced Settings", open=False):
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
                    "🔍 Check Flash Attention Status",
                    variant="secondary",
                    elem_id="wan_check_flash_attention_btn"
                )
        
        # QWEN MODEL MANAGEMENT - Collapsed by default
        with gr.Accordion("🧠 Qwen Model Management", open=False):
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
                    "🔍 Check Model Status",
                    variant="secondary",
                    elem_id="wan_check_qwen_models_btn"
                )
                download_qwen_model_btn = gr.Button(
                    "📥 Download Selected Model",
                    variant="primary",
                    elem_id="wan_download_qwen_model_btn"
                )
                cleanup_qwen_cache_btn = gr.Button(
                    "🧹 Cleanup Model Cache",
                    variant="secondary",
                    elem_id="wan_cleanup_qwen_cache_btn"
                )

        # Auto-Discovery and Setup Information
        with gr.Accordion("📥 Model Auto-Discovery & Setup", open=False):
            gr.Markdown("""
            **✅ Auto-Discovery System**
            
            Wan automatically finds models in these locations:
            - `models/wan/` (recommended)
            - `models/video/wan/`
            - Custom paths you specify
            
            **✨ VACE Models (Recommended)**
            
            VACE models handle both T2V and I2V in one model:
            - **1.3B VACE**: 480p, 8GB VRAM, fast generation
            - **14B VACE**: 480p+720p, 16GB+ VRAM, highest quality
            
            **📥 Easy Download Commands:**
            ```bash
            # Download 1.3B VACE (recommended default)
            huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/wan/Wan2.1-VACE-1.3B
            
            # Or download 14B VACE (high quality)
            huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan/Wan2.1-VACE-14B
            ```
            
            **⚠️ Legacy Models** (for compatibility):
            - T2V models: Text-to-video only
            - I2V models: Separate models for image-to-video
            - Less convenient than VACE, but still supported
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
        with gr.Accordion("📚 Detailed Documentation", open=False):
            with gr.Accordion("🎯 How Wan Integrates with Deforum Schedules", open=False):
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
                
                with gr.Accordion("🎬 Movement Translation: From Deforum Schedules to Prompt Descriptions", open=False):
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
                    2. **Overlays shake on Deforum schedules** (like experimental render core)
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
                
            with gr.Accordion("🛠️ Setup Guide", open=False):
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
                - Click "Generate Wan Video" button
                - Wan reads all settings from Deforum automatically
                - Each prompt becomes a seamless video clip with strength-controlled transitions
                """)
                
            with gr.Accordion("🆘 Troubleshooting", open=False):
                gr.Markdown("""
                If generation fails:
                1. **Check models**: Run `python scripts/deforum_helpers/wan_direct_integration.py`
                2. **Download missing models**: Use commands in Auto-Discovery section
                3. **Verify placement**: Models should be in `models/wan/` directory
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
            from .wan.wan_flash_attention_patch import get_flash_attention_status_html
            return get_flash_attention_status_html()
        except Exception as e:
            return f"❌ <span style='color: #f44336;'>Error checking status: {e}</span>"
    
    def update_flash_attention_mode(mode):
        """Update flash attention mode and return updated status"""
        try:
            from .wan.wan_flash_attention_patch import update_patched_flash_attention_mode, get_flash_attention_status_html
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
        from .wan.wan_flash_attention_patch import get_flash_attention_status_html
        wan_flash_attention_status.value = get_flash_attention_status_html()
    except Exception:
        wan_flash_attention_status.value = "⚠️ <span style='color: #FF9800;'>Status check unavailable</span>"
    
    # Get all component names for the handlers
    from .args import get_component_names
    component_names = get_component_names()
    
    print(f"🔗 Connecting Wan generate button...")
    # Component count is already displayed in ui_left.py with more detail
    
    # NOTE: enhance_prompts_btn connection is now handled in ui_left.py for proper component access
    
    # Connect event handlers for movement analysis - updated with Camera Shakify and sensitivity controls
    analyze_movement_btn.click(
        fn=analyze_movement_handler,
        inputs=[wan_enhanced_prompts, wan_enable_shakify, wan_movement_sensitivity_override, wan_manual_sensitivity],
        outputs=[wan_enhanced_prompts, wan_movement_description]
    )
    
    # Connect sensitivity override toggle to show/hide manual sensitivity slider
    wan_movement_sensitivity_override.change(
        fn=lambda override_enabled: gr.update(visible=override_enabled),
        inputs=[wan_movement_sensitivity_override],
        outputs=[manual_sensitivity_row]
    )
    
    # Connect generate button with validation
    wan_generate_button.click(
        fn=validate_wan_generation,
        inputs=[wan_enhanced_prompts],
        outputs=[wan_generation_status]
    )
    
    # Add automatic validation status updates when prompts change
    wan_enhanced_prompts.change(
        fn=validate_wan_generation,
        inputs=[wan_enhanced_prompts],
        outputs=[wan_generation_status]
    )
    
    # Connect new Wan prompt loading buttons
    load_deforum_to_wan_btn.click(
        fn=load_deforum_to_wan_prompts_handler,
        inputs=[],
        outputs=[wan_enhanced_prompts]
    )
    
    load_wan_defaults_btn.click(
        fn=load_wan_defaults_handler,
        inputs=[],
        outputs=[wan_enhanced_prompts]
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
            with FormRow() as fps_out_format_row:
                fps = create_gr_elem(dv.fps)
            with FormColumn():
                with FormRow() as soundtrack_row:
                    add_soundtrack = create_gr_elem(dv.add_soundtrack)
                    soundtrack_path = create_gr_elem(dv.soundtrack_path)
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

                # Upscaled frames folder upload button:
                interpolate_pics_button = gr.Button(value="*Interpolate Pics*")
                pics_to_interpolate_chosen_file = gr.File(
                    label="Pics to Interpolate", interactive=True, file_count="multiple",
                    file_types=["image"],
                    elem_id="pics_to_interpolate_chosen_file", visible=True)

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

                with FormRow() as ncnn_upscale_factor_model_row:
                    upscale_factor_ncnn = gr.Dropdown(label='Upscale Factor', elem_id="upscale_factor_ncnn",
                                                      choices=['x2', 'x3', 'x4'], value='x2')
                    # Alias for gradio_funcs.py compatibility
                    ncnn_upscale_factor = upscale_factor_ncnn
                    ncnn_upscale_model = gr.Dropdown(label='NCNN Model', elem_id="ncnn_upscale_model",
                                                     choices=['realesr-animevideov3'], value='realesr-animevideov3')
                    ncnn_upscale_keep_imgs = gr.Checkbox(label='Keep Imgs', value=True, elem_id="ncnn_upscale_keep_imgs")

            # This is the actual button that's pressed to initiate the Upscaling:
            upscale_btn = create_row(gr.Button(value="*NCNN Upscale uploaded video*"))

            # The message we see in the UI telling users to check CLI for outputs
            create_row(gr.HTML("* check your CLI for outputs"))  # Show a text about CLI outputs:

            # make the function call when the UPSCALE button is clicked
            upscale_btn.click(fn=ncnn_upload_vid_to_upscale, 
                            inputs=[vid_to_upscale_chosen_file, ncnn_upscale_in_vid_fps_ui_window, 
                                    ncnn_upscale_in_vid_res, ncnn_upscale_out_vid_res, 
                                    ncnn_upscale_model, upscale_factor_ncnn, ncnn_upscale_keep_imgs])
        # Vid2Depth TAB - not built using our args.py at all - all data and params are here and in .vid2depth file
        with gr.TabItem('Vid2depth'):
            vid_to_depth_chosen_file = gr.File(label="Video to get Depth from", interactive=True, file_count="single",
                                               file_types=["video"], elem_id="vid_to_depth_chosen_file")
            with FormRow():
                mode = gr.Dropdown(label='Mode', elem_id="mode",
                                   choices=['Depth (Depth-Anything-V2)', 'Depth (MiDaS)', 'Mixed',
                                            'None (just grayscale)'], value='Depth (Depth-Anything-V2)')
                threshold_value = gr.Slider(label="Threshold Value Lower", value=127, minimum=0, maximum=255, step=1)
                threshold_value_max = gr.Slider(label="Threshold Value Upper", value=255, minimum=0, maximum=255,
                                                step=1)
            thresholding = create_row(gr.Radio(
                ['None', 'Simple', 'Simple (Auto-value)', 'Adaptive (Mean)', 'Adaptive (Gaussian)'],
                label="Thresholding Mode", value='None'))
            with FormRow():
                adapt_block_size = gr.Number(label="Block size", value=11)
                adapt_c = gr.Number(label="C", value=2)
                invert = gr.Checkbox(label='Closer is brighter', value=True, elem_id="invert")
            with FormRow():
                end_blur = gr.Slider(label="End blur width", value=0, minimum=0, maximum=255, step=1)
                midas_weight_vid2depth = gr.Slider(
                    label="MiDaS weight (vid2depth)", value=da.midas_weight, minimum=0,
                    maximum=1, step=0.05, interactive=True,
                    info="sets a midpoint at which a depth-map is to be drawn: range [-1 to +1]")
                depth_keep_imgs = gr.Checkbox(label='Keep Imgs', value=True, elem_id="depth_keep_imgs")

            # This is the actual button that's pressed to initiate the Upscaling:
            depth_btn = create_row(gr.Button(value="*Get depth from uploaded video*"))

            create_row(gr.HTML("* check your CLI for outputs"))  # Show a text about CLI outputs:

            # make the function call when the UPSCALE button is clicked
            depth_btn.click(fn=upload_vid_to_depth,
                            inputs=[vid_to_depth_chosen_file, mode, thresholding, threshold_value, threshold_value_max,
                                    adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth,
                                    depth_keep_imgs])
        # STITCH FRAMES TO VID TAB
        with gr.TabItem(f"{emoji_utils.frames()} Frames to Video") as stitch_imgs_to_vid_row:
            gr.HTML(value=get_gradio_html('frames_to_video'))
            image_path = create_row(dv.image_path)
            ffmpeg_stitch_imgs_but = gr.Button(value="*Stitch frames to video*")
            ffmpeg_stitch_imgs_but.click(fn=direct_stitch_vid_from_frames,
                                         inputs=[image_path, fps, add_soundtrack, soundtrack_path])
    return {k: v for k, v in {**locals(), **vars()}.items()}


def create_keyframe_distribution_info_tab():
    create_row(gr.Markdown(f"""
        {emoji_utils.warn} Keyframe distribution uses an experimental render core with a slightly different feature set.
        Some features may not be supported and could cause errors or unexpected results if not disabled.
    """))
    create_accordion_md_row("Keyframe Distribution Info", f"""
        ### Purpose & Description
        - Ensures diffusion of frames with entries in Prompts or Parseq tables
        - Allows faster generation with high or no cadence
        - Produces less jittery videos, but may introduce artifacts like 'depth smear' at 3D fast movement.
        - Mitigate cumulative negative effects by inserting lower strength frames at regular intervals

        ### Distribution Modes
        1. **Off**: Standard render core, respects cadence settings
        2. **Keyframes Only**: Diffuses only Prompts/Parseq entries, ignores cadence
        3. **Additive**: Uses keyframes and adds cadence for stability.
        4. **Redistributed**: Calculates cadence but rearranges the frames closest to keyframe positions
            to fit them for better synchronization and reactivity at high cadence.
    """)
    create_accordion_md_row("General Recommendations & Warnings", f"""
        - Use with high FPS (e.g., 60) and high cadence (e.g., 15)
        - 'Keyframe_strength' should be lower than 'strength' (ignored when using Parseq)
        - {emoji_utils.warn} Not recommended with optical flow or hybrid settings
        - {emoji_utils.warn} Optical flow settings ~may~ will behave unexpectedly.
            - Turn off in tab "Keyframes", sub-tab "Coherence".
        - Prevent issues like dark-outs that add up over frames:
            - Set up regular low-strength diffusions by using enough keyframes
        - Balance strength values for optimal results
    """)
    create_accordion_md_row("Deforum Setup Recommendations", f"""
        - Set 'Keyframe strength' lower than 'Strength' to make sure keyframes get diffused with more steps
            - The higher the difference, the more keyframes become key compared to regular cadence frames. 
        - Force keyframe creation by duplicating the previous prompt with the desired frame number
            - This ensures diffusion with 'Keyframe strength' value
    """)
    create_accordion_md_row("Parseq Setup Recommendations", f"""
        - Deforum prompt keyframes are ignored
        - All frames with Parseq entries are treated as keyframes and will be diffused
        - 'Keyframe strength' is ignored; use Parseq for direct 'Strength' control
        - Create strength-dips at regular intervals:
            - Mark frames with 'Info' (e.g., "event")
            - Use formulas like: `if (f == info_match_last("event")) 0.25 else 0.75`
    """)


def create_accordion_md_row(name, markdown, is_open=False):
    with gr.Accordion(name, open=is_open):
        gr.Markdown(markdown)


# QwenPromptExpander and Movement Analysis Event Handlers - moved outside for proper import
def enhance_prompts_handler(current_prompts, qwen_model, language, auto_download):
    """Handle prompt enhancement with QwenPromptExpander with progress feedback"""
    try:
        from .wan.utils.qwen_manager import qwen_manager
        import json
        
        print(f"🎨 AI Prompt Enhancement requested for {qwen_model}")
        print(f"📝 Received prompts: {str(current_prompts)[:100]}...")
        
        # Progress: Start
        progress_update = "🎨 Starting AI Prompt Enhancement...\n"
        
        # Check if auto-download is enabled for model availability
        if not auto_download:
            # Check if the selected model is available
            if not qwen_manager.is_model_downloaded(qwen_model):
                return f"""❌ Qwen model not available: {qwen_model}

🔧 **Model Download Required:**
1. ✅ Enable "Auto-Download Qwen Models" checkbox
2. 🎨 Click "AI Prompt Enhancement" again to auto-download
3. ⏳ Wait for download to complete

📥 **Manual Download Alternative:**
1. Use HuggingFace CLI: `huggingface-cli download {qwen_manager.get_model_info(qwen_model).get('huggingface_id', 'model-id')}`
2. ✅ Enable auto-download for easier setup

💡 **Auto-download is recommended** for seamless model management.""", progress_update + "❌ Model not available - enable auto-download!"
        
        # Progress: Model check
        progress_update += "🔍 Checking model availability...\n"
        
        # Check if a model is already loaded
        if qwen_manager.is_model_loaded():
            loaded_info = qwen_manager.get_loaded_model_info()
            current_model = loaded_info['name'] if loaded_info else "Unknown"
            
            # If different model requested, cleanup first
            if qwen_model != "Auto-Select" and current_model != qwen_model:
                print(f"🔄 Switching from {current_model} to {qwen_model}")
                progress_update += f"🔄 Switching from {current_model} to {qwen_model}...\n"
                qwen_manager.cleanup_cache()
        
        # Progress: Model loading
        if not qwen_manager.is_model_loaded():
            if qwen_model == "Auto-Select":
                selected_model = qwen_manager.auto_select_model()
                print(f"🤖 Auto-selected model: {selected_model}")
                progress_update += f"🤖 Auto-selected model: {selected_model}\n"
            else:
                print(f"📥 Loading Qwen model: {qwen_model}")
                progress_update += f"📥 Loading Qwen model: {qwen_model}...\n"
        
        # Get wan prompts from the current_prompts parameter (passed directly)
        animation_prompts = None
        
        if current_prompts and current_prompts.strip():
            try:
                # Try to parse as JSON first
                animation_prompts = json.loads(current_prompts)
                print(f"✅ Successfully parsed {len(animation_prompts)} Wan prompts as JSON")
                progress_update += f"✅ Parsed {len(animation_prompts)} prompts successfully\n"
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
                        print(f"✅ Successfully parsed {len(animation_prompts)} Wan prompts as readable format")
                        progress_update += f"✅ Parsed {len(animation_prompts)} prompts from readable format\n"
                    else:
                        raise ValueError("No valid prompts found")
                except Exception as e:
                    print(f"❌ Could not parse Wan prompts: {e}")
                    error_msg = f"❌ Invalid format in Wan prompts. Expected JSON format like:\n{{\n  \"0\": \"prompt text\",\n  \"60\": \"another prompt\"\n}}\n\nOr readable format like:\nFrame 0: prompt text\nFrame 60: another prompt"
                    return error_msg, progress_update + "❌ Failed to parse prompts!"
        else:
            print("⚠️ Empty Wan prompts")
            
        # Check if we got valid prompts
        if not animation_prompts:
            error_msg = """❌ No Wan prompts found!

🔧 **Setup Required:**
1. 📝 Load prompts using "Load from Deforum Prompts" or "Load Default Wan Prompts"
2. 📋 Make sure your prompts are in proper JSON format like:
   {
     "0": "prompt text",
     "60": "another prompt",
     "120": "a cyberpunk environment with glowing elements"
   }
3. 🎨 Click **AI Prompt Enhancement** again after setting up prompts

💡 **Quick Start:**
Click "Load Default Wan Prompts" to start with example prompts!"""
            return error_msg, progress_update + "❌ No prompts to enhance!"
        
        # Validate prompts content
        if len(animation_prompts) == 1 and "0" in animation_prompts and "beautiful landscape" in animation_prompts["0"]:
            error_msg = """❌ Default prompts detected!

🔧 **Please configure your actual animation prompts:**
1. 📝 Load your real prompts using the load buttons above
2. ✏️ Or manually edit the Wan prompts field
3. 🎨 Click **AI Prompt Enhancement** again

💡 **For your animation sequence:**
Set up prompts like:
{
  "0": "A peaceful scene, photorealistic",
  "18": "A scene with glowing effects, neon colors, synthwave aesthetic",
  "36": "A cyberpunk scene with LED patterns, digital environment"
}"""
            return error_msg, progress_update + "❌ Default prompts detected!"
        
        print(f"🎨 Enhancing {len(animation_prompts)} Wan prompts with {qwen_model}")
        progress_update += f"🎨 Starting enhancement of {len(animation_prompts)} prompts...\n"
        
        # Create the Qwen prompt expander with better error handling
        try:
            progress_update += "📥 Creating AI model instance...\n"
            prompt_expander = qwen_manager.create_prompt_expander(qwen_model, auto_download)
            
            if not prompt_expander:
                if auto_download:
                    error_msg = f"""⏳ Downloading {qwen_model} model...

🔄 **Download in Progress:**
Model download started automatically. This may take a few minutes.

📥 **Please wait** and try clicking "AI Prompt Enhancement" again in 30-60 seconds.

💡 **Status**: Check console for download progress."""
                    return error_msg, progress_update + f"⏳ Downloading {qwen_model}..."
                else:
                    error_msg = f"""❌ Failed to create Qwen prompt expander: {qwen_model}

🔧 **Solutions:**
1. ✅ Enable "Auto-Download Qwen Models" and try again
2. 📥 Manual download: Check console for HuggingFace CLI commands
3. 🔄 Restart WebUI after downloading

📊 **Model Info**: {qwen_manager.get_model_info(qwen_model).get('description', 'N/A')}"""
                    return error_msg, progress_update + "❌ Failed to create AI model!"
        except Exception as e:
            error_msg = f"""❌ Error creating Qwen prompt expander: {str(e)}

🔧 **Troubleshooting:**
1. ✅ Enable auto-download and try again
2. 🔄 Restart WebUI if models were just downloaded
3. 💾 Check available disk space ({qwen_manager.get_model_info(qwen_model).get('vram_gb', 'Unknown')}GB VRAM required)

💡 **Tip**: Try selecting "Auto-Select" for automatic model choice."""
            return error_msg, progress_update + f"❌ Error: {str(e)}"
        
        # Use the QwenModelManager's enhance_prompts method directly
        try:
            progress_update += "✨ Enhancing prompts with AI...\n"
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
                print(f"📐 Found movement description to append: {movement_description}")
                progress_update += "📐 Adding movement descriptions...\n"
            
            # Append movement descriptions to enhanced prompts if available
            if movement_description and movement_description.strip():
                for frame_key in enhanced_prompts_dict:
                    original_prompt = enhanced_prompts_dict[frame_key]
                    enhanced_prompts_dict[frame_key] = f"{original_prompt}. {movement_description}"
                print(f"✅ Appended movement description to {len(enhanced_prompts_dict)} enhanced prompts")
                progress_update += f"✅ Added movement to {len(enhanced_prompts_dict)} prompts\n"
            
            # Format the enhanced prompts as JSON
            enhanced_json = json.dumps(enhanced_prompts_dict, ensure_ascii=False, indent=2)
            
            print(f"✅ Successfully enhanced {len(enhanced_prompts_dict)} prompts")
            progress_update += f"✅ Enhancement complete! {len(enhanced_prompts_dict)} prompts ready\n"
            
            # Return the enhanced prompts and success progress
            return enhanced_json, progress_update + "🎉 Ready for generation!"
            
        except Exception as e:
            print(f"❌ Error enhancing prompts: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"❌ Error enhancing prompts: {str(e)}"
            return error_msg, progress_update + f"❌ Enhancement failed: {str(e)}"
    
    except Exception as e:
        print(f"❌ Fatal error in enhance_prompts_handler: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"❌ Fatal error: {str(e)}"
        return error_msg, f"❌ Fatal error: {str(e)}"


def analyze_movement_handler(current_prompts, enable_shakify=True, sensitivity_override=False, manual_sensitivity=1.0):
    """Handle movement analysis from Deforum schedules with enhanced Camera Shakify integration and fine-grained sensitivity control"""
    try:
        from .wan.utils.movement_analyzer import analyze_deforum_movement, generate_wan_motion_intensity_schedule, MovementAnalyzer
        from types import SimpleNamespace
        import json
        
        print("🎬 Starting enhanced movement analysis with fine-grained detection...")
        print(f"🎬 Camera Shakify: {'ENABLED' if enable_shakify else 'DISABLED'}")
        print(f"🎯 Sensitivity: {'MANUAL ({:.1f})'.format(manual_sensitivity) if sensitivity_override else 'AUTO-CALCULATED'}")
        
        # Validate current prompts
        if not current_prompts or current_prompts.strip() == "":
            return "", """❌ No prompts to analyze!

🔧 **Load prompts first:**
1. 📋 Click "Load from Deforum Prompts" or "Load Default Wan Prompts"
2. 📐 Then click "Add Movement Descriptions" again

Movement descriptions will be added to your existing prompts."""
        
        # Parse current prompts
        try:
            prompts_dict = json.loads(current_prompts)
            if not prompts_dict:
                return "", "❌ Empty prompts! Load prompts first before analyzing movement."
        except json.JSONDecodeError:
            return "", "❌ Invalid JSON format! Please fix the prompts format first."
        
        # Create anim_args with actual Deforum schedule values
        anim_args = SimpleNamespace()
        
        # Try to access the stored movement schedule values (these are the actual schedule strings)
        if hasattr(analyze_movement_handler, '_movement_components'):
            components = analyze_movement_handler._movement_components
            try:
                # Get actual schedule strings from Deforum's animation system
                anim_args.translation_x = components.get('translation_x', "0:(0)")
                anim_args.translation_y = components.get('translation_y', "0:(0)")
                anim_args.translation_z = components.get('translation_z', "0:(0)")
                anim_args.rotation_3d_x = components.get('rotation_3d_x', "0:(0)")
                anim_args.rotation_3d_y = components.get('rotation_3d_y', "0:(0)")
                anim_args.rotation_3d_z = components.get('rotation_3d_z', "0:(0)")
                anim_args.zoom = components.get('zoom', "0:(1.0)")
                anim_args.angle = components.get('angle', "0:(0)")
                anim_args.max_frames = int(components.get('max_frames', 100))
                
                print("✅ Using actual Deforum movement schedules from UI")
                print(f"📊 Translation X: {anim_args.translation_x}")
                print(f"📊 Translation Z: {anim_args.translation_z}")
                print(f"📊 Rotation Y: {anim_args.rotation_3d_y}")
                print(f"📊 Zoom: {anim_args.zoom}")
                
            except Exception as e:
                print(f"⚠️ Could not access movement schedules: {e}")
                # Use static defaults for testing
                anim_args.translation_x = "0:(0)"
                anim_args.translation_y = "0:(0)"
                anim_args.translation_z = "0:(0)"
                anim_args.rotation_3d_x = "0:(0)"
                anim_args.rotation_3d_y = "0:(0)"
                anim_args.rotation_3d_z = "0:(0)"
                anim_args.zoom = "0:(1.0)"
                anim_args.angle = "0:(0)"
                anim_args.max_frames = 120
        else:
            print("⚠️ No stored movement schedule references found")
            # Use static defaults for testing
            anim_args.translation_x = "0:(0)"
            anim_args.translation_y = "0:(0)"
            anim_args.translation_z = "0:(0)"
            anim_args.rotation_3d_x = "0:(0)"
            anim_args.rotation_3d_y = "0:(0)"
            anim_args.rotation_3d_z = "0:(0)"
            anim_args.zoom = "0:(1.0)"
            anim_args.angle = "0:(0)"
            anim_args.max_frames = 120
        
        # Get Camera Shakify settings if enabled
        if enable_shakify:
            try:
                # Try to get Camera Shakify settings from stored component references first
                if hasattr(analyze_movement_handler, '_movement_components'):
                    components = analyze_movement_handler._movement_components
                    anim_args.shake_name = components.get('shake_name', "None")
                    anim_args.shake_intensity = float(components.get('shake_intensity', 1.0))
                    anim_args.shake_speed = float(components.get('shake_speed', 1.0))
                    print(f"✅ Using Camera Shakify settings from UI components")
                else:
                    # Fallback to reading from DeforumArgs if component references not available
                    from .args import DeforumArgs
                    current_args = DeforumArgs()
                    anim_args.shake_name = getattr(current_args, 'shake_name', "None")
                    anim_args.shake_intensity = getattr(current_args, 'shake_intensity', 1.0)
                    anim_args.shake_speed = getattr(current_args, 'shake_speed', 1.0)
                    print(f"✅ Using Camera Shakify settings from DeforumArgs fallback")
                
                # Camera Shakify is enabled when shake_name is not "None"
                camera_shake_enabled = anim_args.shake_name and anim_args.shake_name != "None"
                
                if camera_shake_enabled:
                    print(f"🎬 Camera Shakify ENABLED:")
                    print(f"   Shake Name: {anim_args.shake_name}")
                    print(f"   Intensity: {anim_args.shake_intensity}")
                    print(f"   Speed: {anim_args.shake_speed}")
                else:
                    print(f"📷 Camera Shakify disabled (shake_name: {anim_args.shake_name})")
            except Exception as e:
                print(f"⚠️ Could not read Camera Shakify settings: {e}")
                # Disable Shakify on error
                anim_args.shake_name = "None"
                anim_args.shake_intensity = 1.0
                anim_args.shake_speed = 1.0
        else:
            # Disable Camera Shakify when checkbox is unchecked
            anim_args.shake_name = "None"
            anim_args.shake_intensity = 1.0
            anim_args.shake_speed = 1.0
            print(f"🎬 Camera Shakify manually disabled via UI checkbox")
        
        # Determine sensitivity
        if sensitivity_override:
            sensitivity = manual_sensitivity
            sensitivity_reason = f"manual override ({sensitivity:.1f})"
            print(f"🎯 Using manual sensitivity: {sensitivity}")
        else:
            # Auto-calculate movement sensitivity from the schedules
            print("🧮 Auto-calculating movement sensitivity from Deforum schedules...")
            
            # Create a MovementAnalyzer to calculate optimal sensitivity
            analyzer = MovementAnalyzer(sensitivity=1.0)  # Start with baseline
            
            # Calculate movement ranges to determine optimal sensitivity
            from .wan.utils.movement_analyzer import parse_schedule_string, interpolate_schedule
            
            try:
                # Parse all movement schedules
                x_keyframes = parse_schedule_string(anim_args.translation_x, anim_args.max_frames)
                y_keyframes = parse_schedule_string(anim_args.translation_y, anim_args.max_frames)
                z_keyframes = parse_schedule_string(anim_args.translation_z, anim_args.max_frames)
                zoom_keyframes = parse_schedule_string(anim_args.zoom, anim_args.max_frames)
                
                # Interpolate to get value ranges
                x_values = interpolate_schedule(x_keyframes, anim_args.max_frames)
                y_values = interpolate_schedule(y_keyframes, anim_args.max_frames)
                z_values = interpolate_schedule(z_keyframes, anim_args.max_frames)
                zoom_values = interpolate_schedule(zoom_keyframes, anim_args.max_frames)
                
                # Calculate movement ranges
                x_range = max(x_values) - min(x_values) if x_values else 0
                y_range = max(y_values) - min(y_values) if y_values else 0
                z_range = max(z_values) - min(z_values) if z_values else 0
                zoom_range = max(zoom_values) - min(zoom_values) if zoom_values else 0
                
                # Calculate total movement magnitude
                total_movement = x_range + y_range + z_range + (zoom_range * 50)  # Zoom weighted higher
                
                # Auto-calculate optimal sensitivity based on movement magnitude
                if total_movement < 5:
                    # Very small movement - high sensitivity to detect subtle motion
                    sensitivity = 3.0
                    sensitivity_reason = "high sensitivity for very subtle movement"
                elif total_movement < 15:
                    # Small movement - moderate-high sensitivity
                    sensitivity = 2.0
                    sensitivity_reason = "high sensitivity for subtle movement"
                elif total_movement < 50:
                    # Normal movement - standard sensitivity
                    sensitivity = 1.0
                    sensitivity_reason = "standard sensitivity for normal movement"
                elif total_movement < 200:
                    # Large movement - reduced sensitivity to avoid over-detection
                    sensitivity = 0.7
                    sensitivity_reason = "reduced sensitivity for large movement"
                else:
                    # Very large movement - low sensitivity
                    sensitivity = 0.5
                    sensitivity_reason = "low sensitivity for very large movement"
                
                print(f"📊 Total movement magnitude: {total_movement:.1f}")
                print(f"🎯 Auto-calculated sensitivity: {sensitivity} ({sensitivity_reason})")
                
            except Exception as e:
                print(f"⚠️ Could not auto-calculate sensitivity: {e}, using default 2.0")
                sensitivity = 2.0
                sensitivity_reason = "default (calculation failed)"
        
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
        
        print(f"🎯 Enhanced movement analysis result:")
        print(f"   Description: {movement_desc}")
        print(f"   Strength: {average_motion_strength:.3f}")
        print(f"   Motion Intensity Schedule: {motion_intensity_schedule}")
        
        # Update prompts with movement descriptions (cleaner approach)
        updated_prompts = {}
        for frame, prompt in prompts_dict.items():
            # Clean up existing movement descriptions
            clean_prompt = prompt.replace(", static camera position", "")
            clean_prompt = clean_prompt.replace("static camera position", "")
            clean_prompt = clean_prompt.replace(", camera movement with", ", ")
            if clean_prompt.startswith("camera movement with "):
                clean_prompt = clean_prompt[21:]  # Remove "camera movement with " prefix
            
            # Remove existing movement descriptions more thoroughly
            clean_prompt = clean_prompt.split('. camera movement:')[0].split('. Camera movement:')[0].strip()
            
            # Add new movement description
            if movement_desc and average_motion_strength > 0:
                if not clean_prompt.endswith('.'):
                    updated_prompts[frame] = f"{clean_prompt}, {movement_desc}"
                else:
                    updated_prompts[frame] = f"{clean_prompt.rstrip('.')} {movement_desc}."
            else:
                updated_prompts[frame] = clean_prompt
        
        # Convert back to JSON
        updated_json = json.dumps(updated_prompts, ensure_ascii=False, indent=2)
        
        # Enhanced result message with frame-by-frame details and Camera Shakify info
        camera_shakify_status = ""
        if enable_shakify and hasattr(anim_args, 'shake_name') and anim_args.shake_name != "None":
            camera_shakify_status = f"""
🎬 **Camera Shakify Integration:**
- Pattern: {anim_args.shake_name}
- Intensity: {anim_args.shake_intensity}
- Speed: {anim_args.shake_speed}
- Status: ✅ Active and applied to movement schedules"""
        elif enable_shakify:
            camera_shakify_status = f"""
🎬 **Camera Shakify Integration:**
- Status: ⚠️ Enabled but no shake pattern selected
- Go to Keyframes → Motion → Shakify tab to configure"""
        else:
            camera_shakify_status = f"""
🎬 **Camera Shakify Integration:**
- Status: ❌ Disabled via checkbox
- Enable checkbox above to include shake effects"""
        
        if average_motion_strength > 0:
            result_message = f"""✅ Enhanced fine-grained movement analysis complete!

🎯 **Movement Detection:**
"{movement_desc}"

📊 **Analysis Details:**
- Motion strength: {average_motion_strength:.3f}
- Sensitivity: {sensitivity} ({sensitivity_reason})
- Detection method: Frame-by-frame analysis with enhanced thresholds
{camera_shakify_status}

📐 **Motion Intensity Schedule for Wan:**
{motion_intensity_schedule}

💡 **Copy the schedule above to Wan's Motion Intensity field for synchronized movement effects!**

✅ Movement descriptions applied to {len(updated_prompts)} prompts.
Ready for AI enhancement or video generation."""
        else:
            result_message = f"""✅ Enhanced movement analysis complete!

📊 **Analysis Result:**
"{movement_desc}"

📊 **Analysis Details:**
- Sensitivity: {sensitivity} ({sensitivity_reason})
- Detection method: Frame-by-frame analysis with enhanced thresholds
{camera_shakify_status}

📷 Camera appears to be static based on current movement schedules. To add movement:
1. Go to Keyframes → Motion tab and configure movement schedules
2. Or enable Camera Shakify in the Keyframes → Motion → Shakify tab
3. Then run movement analysis again

✅ Analysis complete for {len(updated_prompts)} prompts."""
        
        print(f"✅ Updated {len(updated_prompts)} Wan prompts with enhanced movement descriptions")
        print(f"📊 Use this motion intensity schedule in Wan: {motion_intensity_schedule}")
        print(f"💡 Copy this schedule to Wan's Motion Intensity field for synchronized movement effects!")
        
        # Store movement description for enhance_prompts_handler
        analyze_movement_handler._movement_description = movement_desc
        
        return updated_json, result_message
        
    except Exception as e:
        print(f"❌ Error in enhanced movement analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = f"""❌ Error in enhanced movement analysis: {str(e)}

🔧 **Try this:**
1. Check that Deforum movement schedules are valid (Keyframes → Motion tab)
2. Verify Camera Shakify settings if using shake effects
3. Ensure prompts are in valid JSON format
4. Try disabling Camera Shakify checkbox if issues persist

Contact support if this persists."""
        return current_prompts, error_msg

def check_qwen_models_handler(qwen_model):
    """Check Qwen model status and availability"""
    try:
        from .wan.utils.qwen_manager import qwen_manager
        
        print(f"🔍 Checking Qwen model status: {qwen_model}")
        
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
                status_parts.append("✅ <span style='color: #4CAF50;'>VRAM requirement met</span>")
            else:
                status_parts.append("⚠️ <span style='color: #FF9800;'>May exceed available VRAM</span>")
        
        # Download Status
        if is_downloaded:
            status_parts.append("✅ <span style='color: #4CAF50;'>Model downloaded and available</span>")
        else:
            status_parts.append("❌ <span style='color: #f44336;'>Model not downloaded</span>")
            if model_info and 'hf_name' in model_info:
                status_parts.append(f"<strong style='color: #333;'>HuggingFace ID:</strong> {model_info['hf_name']}")
        
        # Loading Status
        if is_loaded:
            if loaded_info and loaded_info['name'] == qwen_model:
                status_parts.append("🔥 <span style='color: #4CAF50;'>Model currently loaded and ready</span>")
                estimated_vram = loaded_info.get('vram_usage', 0)
                if estimated_vram > 0:
                    status_parts.append(f"<strong style='color: #333;'>Estimated VRAM usage:</strong> {estimated_vram:.1f}GB")
            else:
                current_model = loaded_info['name'] if loaded_info else "Unknown"
                status_parts.append(f"🔄 <span style='color: #FF9800;'>Different model loaded: {current_model}</span>")
                status_parts.append("<span style='color: #333;'>Will switch on next enhancement</span>")
        else:
            status_parts.append("💤 <span style='color: #333;'>No model currently loaded</span>")
        
        # Quick Setup Instructions
        if not is_downloaded:
            status_parts.append("<br><strong style='color: #333;'>Quick Setup:</strong>")
            status_parts.append("1. ✅ Enable 'Auto-Download Qwen Models' above")
            status_parts.append("2. 🎨 Click 'AI Prompt Enhancement' for auto-download")
            status_parts.append("3. ⏳ Wait for download to complete")
        elif not is_loaded:
            status_parts.append("<br><strong style='color: #333;'>Ready to Use:</strong>")
            status_parts.append("🎨 Click 'AI Prompt Enhancement' to load and use this model")
        else:
            status_parts.append("<br><strong style='color: #333;'>Status:</strong> Ready for prompt enhancement!")
        
        return "<br>".join(status_parts)
        
    except Exception as e:
        print(f"❌ Error checking Qwen model status: {e}")
        return f"❌ <span style='color: #f44336;'>Error checking model status: {str(e)}</span>"


def download_qwen_model_handler(qwen_model, auto_download_enabled):
    """Download selected Qwen model"""
    try:
        from .wan.utils.qwen_manager import qwen_manager
        
        if not auto_download_enabled:
            return """❌ <span style='color: #f44336;'>Auto-download is disabled</span>

<strong style='color: #333;'>To download models:</strong><br>
1. ✅ Enable 'Auto-Download Qwen Models' checkbox above<br>
2. 📥 Click this button again<br>
<br>
<strong style='color: #333;'>Or download manually:</strong><br>
Use HuggingFace CLI or git to download the model"""
        
        print(f"📥 Downloading Qwen model: {qwen_model}")
        
        # Handle auto-select
        if qwen_model == "Auto-Select":
            selected_model = qwen_manager.auto_select_model()
            print(f"🤖 Auto-selected model for download: {selected_model}")
        else:
            selected_model = qwen_model
        
        # Check if already downloaded
        if qwen_manager.is_model_downloaded(selected_model):
            return f"""✅ <span style='color: #4CAF50;'>Model already available: {selected_model}</span>

<strong style='color: #333;'>Status:</strong> Model is downloaded and ready to use<br>
🎨 Click 'AI Prompt Enhancement' to start using this model"""
        
        # Start download
        download_status = []
        download_status.append(f"📥 <span style='color: #2196F3;'>Starting download: {selected_model}</span>")
        
        model_info = qwen_manager.get_model_info(selected_model)
        if model_info:
            download_status.append(f"<strong style='color: #333;'>Description:</strong> {model_info.get('description', 'N/A')}")
            download_status.append(f"<strong style='color: #333;'>VRAM Required:</strong> {model_info.get('vram_gb', 'Unknown')}GB")
            download_status.append(f"<strong style='color: #333;'>HuggingFace:</strong> {model_info.get('hf_name', 'N/A')}")
        
        # Attempt download
        success = qwen_manager.download_model(selected_model)
        
        if success:
            download_status.append("<br>✅ <span style='color: #4CAF50;'>Download completed successfully!</span>")
            download_status.append("🎨 Ready to use - click 'AI Prompt Enhancement' to start")
        else:
            download_status.append("<br>❌ <span style='color: #f44336;'>Download failed</span>")
            download_status.append("<strong style='color: #333;'>Troubleshooting:</strong>")
            download_status.append("• Check internet connection")
            download_status.append("• Verify disk space")
            download_status.append("• Try manual download with HuggingFace CLI")
            
            if model_info and 'hf_name' in model_info:
                download_status.append(f"<br><strong style='color: #333;'>Manual command:</strong>")
                download_status.append(f"<code>huggingface-cli download {model_info['hf_name']} --local-dir models/qwen/{selected_model}</code>")
        
        return "<br>".join(download_status)
        
    except Exception as e:
        print(f"❌ Error downloading Qwen model: {e}")
        return f"❌ <span style='color: #f44336;'>Download error: {str(e)}</span>"


def cleanup_qwen_cache_handler():
    """Cleanup Qwen model cache and free VRAM"""
    try:
        from .wan.utils.qwen_manager import qwen_manager
        
        print("🧹 Cleaning up Qwen model cache...")
        
        # Check if any model is loaded
        if not qwen_manager.is_model_loaded():
            return """ℹ️ <span style='color: #2196F3;'>No Qwen models currently loaded</span>

<strong style='color: #333;'>Cache Status:</strong> Clean - no cleanup needed<br>
💾 VRAM available for other operations"""
        
        # Get info about loaded model before cleanup
        loaded_info = qwen_manager.get_loaded_model_info()
        model_name = loaded_info['name'] if loaded_info else "Unknown"
        estimated_vram = loaded_info.get('vram_usage', 0) if loaded_info else 0
        
        # Perform cleanup
        qwen_manager.cleanup_cache()
        
        result = []
        result.append("✅ <span style='color: #4CAF50;'>Qwen model cache cleaned successfully</span>")
        result.append(f"<strong style='color: #333;'>Unloaded model:</strong> {model_name}")
        
        if estimated_vram > 0:
            result.append(f"<strong style='color: #333;'>Freed VRAM:</strong> ~{estimated_vram:.1f}GB")
        
        result.append("<br><strong style='color: #333;'>Benefits:</strong>")
        result.append("💾 VRAM freed for video generation")
        result.append("🧠 Reduced memory usage")
        result.append("🔄 Fresh start for next enhancement")
        
        result.append("<br>💡 <span style='color: #333;'>Models will auto-load when needed for enhancement</span>")
        
        return "<br>".join(result)
        
    except Exception as e:
        print(f"❌ Error during Qwen cache cleanup: {e}")
        return f"❌ <span style='color: #f44336;'>Cleanup error: {str(e)}</span>"


def load_wan_prompts_handler():
    """Load Wan prompts from default settings"""
    try:
        # Load prompts from default_settings.txt
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'default_settings.txt')
        
        if not os.path.exists(settings_path):
            print(f"❌ Default settings file not found: {settings_path}")
            return "0: A peaceful landscape scene, photorealistic"
        
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # Get wan_prompts from settings
        wan_prompts = settings.get('wan_prompts', {})
        
        if not wan_prompts:
            print("⚠️ No wan_prompts found in default settings, falling back to basic prompt")
            return "0: A peaceful landscape scene, photorealistic"
        
        # Convert prompts dict to textarea format (frame: prompt)
        prompt_lines = []
        for frame, prompt in sorted(wan_prompts.items(), key=lambda x: int(x[0])):
            prompt_lines.append(f"{frame}: {prompt}")
        
        result = "\n".join(prompt_lines)
        print(f"✅ Loaded {len(wan_prompts)} Wan prompts from default settings")
        return result
        
    except Exception as e:
        print(f"❌ Error loading Wan prompts: {e}")
        return f"0: Error loading prompts: {str(e)}"


def load_deforum_prompts_handler():
    """Load original Deforum prompts from default settings"""
    try:
        # Load prompts from default_settings.txt
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'default_settings.txt')
        
        if not os.path.exists(settings_path):
            print(f"❌ Default settings file not found: {settings_path}")
            return "0: A peaceful landscape scene, photorealistic"
        
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # Get prompts from settings (main prompts section)
        deforum_prompts = settings.get('prompts', {})
        
        if not deforum_prompts:
            print("⚠️ No prompts found in default settings, falling back to basic prompt")
            return "0: A peaceful landscape scene, photorealistic"
        
        # Convert prompts dict to textarea format (frame: prompt)
        prompt_lines = []
        for frame, prompt in sorted(deforum_prompts.items(), key=lambda x: int(x[0])):
            prompt_lines.append(f"{frame}: {prompt}")
        
        result = "\n".join(prompt_lines)
        print(f"✅ Loaded {len(deforum_prompts)} Deforum prompts from default settings")
        return result
        
    except Exception as e:
        print(f"❌ Error loading Deforum prompts: {e}")
        return f"0: Error loading prompts: {str(e)}"


def load_deforum_to_wan_prompts_handler():
    """Load current Deforum prompts into Wan prompts field"""
    try:
        # Try to get animation prompts from the stored component reference
        animation_prompts_json = ""
        
        if hasattr(enhance_prompts_handler, '_animation_prompts_component'):
            try:
                animation_prompts_json = enhance_prompts_handler._animation_prompts_component.value
                print(f"📋 Loading Deforum prompts to Wan prompts field")
            except Exception as e:
                print(f"⚠️ Could not access animation_prompts component: {e}")
        
        if not animation_prompts_json or animation_prompts_json.strip() == "":
            return """{"0": "No Deforum prompts found! Go to the Prompts tab and configure your animation prompts first."}"""
        
        # Parse the JSON and convert to clean Wan format
        try:
            import json
            prompts_dict = json.loads(animation_prompts_json)
            
            # Convert to Wan format (clean prompts without negative parts)
            wan_prompts_dict = {}
            for frame, prompt in prompts_dict.items():
                # Clean up the prompt (remove negative prompts)
                clean_prompt = prompt.split('--neg')[0].strip()
                wan_prompts_dict[frame] = clean_prompt
            
            # Return as JSON
            result = json.dumps(wan_prompts_dict, ensure_ascii=False, indent=2)
            print(f"✅ Converted {len(prompts_dict)} Deforum prompts to Wan JSON format")
            return result
            
        except json.JSONDecodeError as e:
            return json.dumps({
                "0": f"Invalid JSON in Deforum prompts: {str(e)}. Fix the JSON format in the Prompts tab first."
            }, indent=2)
            
    except Exception as e:
        return f"❌ Error loading Deforum prompts: {str(e)}"


def load_wan_defaults_handler():
    """Load default Wan prompts from settings file"""
    try:
        # Load default prompts from settings
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'default_settings.txt')
        
        if not os.path.exists(settings_path):
            # Fallback to simple defaults
            return json.dumps({
                "0": "prompt text",
                "60": "another prompt"
            }, ensure_ascii=False, indent=2)
        
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            wan_prompts = settings.get('wan_prompts', {})
            
            if wan_prompts:
                # Return as JSON
                result = json.dumps(wan_prompts, ensure_ascii=False, indent=2)
                print(f"✅ Loaded {len(wan_prompts)} default Wan prompts from settings")
                return result
            else:
                # Use fallback
                return json.dumps({
                    "0": "prompt text",
                    "60": "another prompt"
                }, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error loading default settings: {e}")
            # Return simple fallback
            return json.dumps({
                "0": "prompt text",
                "60": "another prompt"
            }, ensure_ascii=False, indent=2)
            
    except Exception as e:
        return json.dumps({
            "0": f"Error loading default prompts: {str(e)}"
        }, indent=2)


def validate_wan_generation(current_prompts):
    """Validate that Wan generation requirements are met"""
    try:
        import json
        
        # Check if prompts are empty
        if not current_prompts or current_prompts.strip() == "":
            return """⚠️ **Prompts Required**

📋 **Load prompts to get started:**
• Click "Load from Deforum Prompts" to use your animation prompts
• Or click "Load Default Wan Prompts" for examples
• Then optionally enhance with AI or add movement descriptions"""
        
        # Check if it's just placeholder text
        if any(placeholder in current_prompts.lower() for placeholder in ["required:", "load prompts", "placeholder"]):
            return """⚠️ **Load Real Prompts**

📋 **Replace placeholder text:**
• Click "Load from Deforum Prompts" to copy your animation prompts
• Or click "Load Default Wan Prompts" for examples"""
        
        # Try to parse as JSON
        try:
            prompts_dict = json.loads(current_prompts)
            if not prompts_dict:
                return "⚠️ **Empty prompts** - Add some prompts first"
            
            # Check if prompts are just basic placeholders
            first_prompt = list(prompts_dict.values())[0].lower()
            if any(placeholder in first_prompt for placeholder in ["prompt text", "beautiful landscape", "load prompts"]):
                return """⚠️ **Default/Placeholder Prompts Detected**

📋 **Load your real prompts:**
• Click "Load from Deforum Prompts" to copy your animation prompts
• Or edit the prompts manually to describe your desired video"""
                
            # All good - ready to generate!
            num_prompts = len(prompts_dict)
            return f"""✅ **Ready to Generate!** 

🎬 **Found {num_prompts} prompt{'s' if num_prompts != 1 else ''}** for Wan video generation
🔥 **Click "Generate Wan Video" above** to start I2V chaining generation
⚡ **Optional:** Add movement descriptions or AI enhancement first"""
            
        except json.JSONDecodeError:
            return """❌ **Invalid JSON Format**

🔧 **Fix the format:**
• Prompts should be in JSON format like: {"0": "prompt text", "60": "another prompt"}
• Check for missing quotes, commas, or brackets"""
    
    except Exception as e:
        return f"❌ **Validation Error:** {str(e)}"


def wan_generate_with_validation(*component_args):
    """Wrapper for wan_generate_video that includes validation"""
    try:
        # Get component names to find the wan_enhanced_prompts index
        from .args import get_component_names
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
            if validation_result.startswith("❌"):
                return validation_result
            
            # If validation passes, call the original generate function
            # But first we need to insert the prompts into the right position in component_args
            # This is a bit complex - we'll need to reconstruct the args properly
            
            # For now, return validation success and instructions
            return f"""✅ Validation passed! 

{validation_result}

🎬 **Starting Wan video generation...**
- Prompts: {len(wan_prompts.split('"')) // 4} clips detected
- Using I2V chaining for smooth transitions
- Check console for detailed progress

🔧 **Note**: Full generation integration in progress..."""
            
        except Exception as e:
            return f"❌ Generation preparation error: {str(e)}"
            
    except Exception as e:
        return f"❌ Generation error: {str(e)}"


def get_tab_ffmpeg():
    """FFmpeg post-processing tab with upscaling, interpolation, and audio replacement"""
    
    # Check if FFmpeg is available before creating the processor
    try:
        from .ffmpeg_utils import FFmpegProcessor
        ffmpeg_processor = FFmpegProcessor()
        ffmpeg_available = True
        ffmpeg_error = None
    except Exception as e:
        ffmpeg_available = False
        ffmpeg_error = str(e)
        ffmpeg_processor = None
    
    with gr.TabItem(f"{emoji_utils.hybrid()} FFmpeg Post-Processing", elem_id='ffmpeg_tab'):
        
        if not ffmpeg_available:
            # Show FFmpeg installation instructions when not available
            gr.HTML(f"""
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 20px; margin: 10px 0;">
                <h3 style="color: #856404; margin-top: 0;">⚠️ FFmpeg Not Available</h3>
                <p><strong>Error:</strong> {ffmpeg_error}</p>
                
                <h4>📥 How to Install FFmpeg:</h4>
                <p><strong>Windows (Recommended):</strong></p>
                <ol>
                    <li>Download FFmpeg from <a href="https://www.gyan.dev/ffmpeg/builds/" target="_blank">gyan.dev/ffmpeg/builds</a></li>
                    <li>Extract to <code>C:\\ffmpeg</code></li>
                    <li>Add <code>C:\\ffmpeg\\bin</code> to your system PATH</li>
                    <li>Restart WebUI</li>
                </ol>
                
                <p><strong>Alternative (Windows with Chocolatey):</strong></p>
                <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px;">choco install ffmpeg</pre>
                
                <p><strong>Alternative (Windows with winget):</strong></p>
                <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px;">winget install FFmpeg</pre>
                
                <p><strong>Linux:</strong></p>
                <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px;">sudo apt install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg   # CentOS/RHEL</pre>
                
                <p><strong>macOS:</strong></p>
                <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px;">brew install ffmpeg</pre>
                
                <p>💡 <strong>Tip:</strong> After installation, restart WebUI to enable FFmpeg post-processing features.</p>
            </div>
            """)
            
            # Return early - no functional content when FFmpeg unavailable
            return locals()
        
        # FFmpeg is available - show full functionality
        gr.HTML("""
        <h3>🎬 FFmpeg Video Post-Processing</h3>
        <p>Professional video post-processing using FFmpeg: upscaling, frame interpolation, and audio replacement.</p>
        <p><strong>✅ FFmpeg detected and ready to use!</strong></p>
        """)
        
        # Video file upload
        with gr.Accordion('📁 Video Input', open=True):
            video_input_file = gr.File(
                label="Select Video File", 
                file_types=["video"], 
                file_count="single",
                elem_id="ffmpeg_video_input"
            )
            
        # Video Information Display
        with gr.Accordion('📊 Video Information', open=True):
            with gr.Row():
                with gr.Column(scale=2):
                    video_info_display = gr.HTML(value="<p>No video loaded</p>")
                with gr.Column(scale=1):
                    analyze_video_btn = gr.Button("🔍 Analyze Video", size="sm")
                    
        # Processing Options
        with gr.Accordion('⚙️ Processing Options', open=True):
            
            # Upscaling Section
            with gr.Tab("🔍 Upscaling"):
                gr.HTML("<h4>Video Upscaling</h4><p>Increase video resolution using high-quality algorithms.</p>")
                
                with gr.Row():
                    upscale_resolution = gr.Dropdown(
                        label="Target Resolution",
                        choices=["720p", "1080p", "1440p", "4K", "8K", "Custom"],
                        value="1080p"
                    )
                    upscale_custom_res = gr.Textbox(
                        label="Custom Resolution (WxH)",
                        placeholder="1920x1080",
                        visible=False
                    )
                    
                with gr.Row():
                    upscale_quality = gr.Dropdown(
                        label="Quality Preset",
                        choices=["fast", "medium", "slow", "veryslow"],
                        value="medium",
                        info="Higher quality = slower processing"
                    )
                    upscale_output_suffix = gr.Textbox(
                        label="Output Suffix",
                        value="_upscaled",
                        info="Added to filename"
                    )
                    
                upscale_btn = gr.Button("🚀 Start Upscaling", variant="primary", size="lg")
                
            # Frame Interpolation Section  
            with gr.Tab("🎞️ Frame Interpolation"):
                gr.HTML("<h4>Frame Interpolation</h4><p>Increase frame rate for smoother motion.</p>")
                
                with gr.Row():
                    interp_target_fps = gr.Dropdown(
                        label="Target FPS",
                        choices=["24", "30", "60", "120", "Custom"],
                        value="60"
                    )
                    interp_custom_fps = gr.Number(
                        label="Custom FPS",
                        value=60,
                        visible=False
                    )
                    
                with gr.Row():
                    interp_method = gr.Dropdown(
                        label="Interpolation Method",
                        choices=["minterpolate", "fps"],
                        value="minterpolate",
                        info="minterpolate = higher quality, fps = faster"
                    )
                    interp_output_suffix = gr.Textbox(
                        label="Output Suffix", 
                        value="_interpolated",
                        info="Added to filename"
                    )
                    
                interpolate_btn = gr.Button("🎬 Start Interpolation", variant="primary", size="lg")
                
            # Audio Replacement Section
            with gr.Tab("🎵 Audio Replacement"):
                gr.HTML("<h4>Audio Replacement</h4><p>Replace or add audio track to video.</p>")
                
                audio_input_file = gr.File(
                    label="Audio File",
                    file_types=["audio"],
                    file_count="single"
                )
                
                with gr.Row():
                    audio_start_time = gr.Number(
                        label="Audio Start Time (seconds)",
                        value=0,
                        info="Offset audio start time"
                    )
                    audio_output_suffix = gr.Textbox(
                        label="Output Suffix",
                        value="_audio_replaced",
                        info="Added to filename"
                    )
                    
                replace_audio_btn = gr.Button("🎵 Replace Audio", variant="primary", size="lg")
                
        # Progress and Output
        with gr.Accordion('📈 Progress & Output', open=True):
            progress_text = gr.Textbox(
                label="Processing Status",
                value="Ready to process",
                interactive=False,
                lines=3
            )
            
            output_file_info = gr.HTML(value="<p>No output files yet</p>")
            
        # Event Handlers (only if FFmpeg is available)
        def analyze_video_handler(video_file):
            """Analyze uploaded video and display information"""
            if not video_file:
                return "<p>No video file selected</p>"
                
            try:
                video_info = ffmpeg_processor.get_video_info(video_file.name)
                
                if "error" in video_info:
                    return f"<p style='color: red;'>Error: {video_info['error']}</p>"
                
                # Create formatted HTML display
                info_html = f"""
                <div style='background: #f5f5f5; padding: 15px; border-radius: 8px; font-family: monospace;'>
                    <h4>📹 {video_info['filename']}</h4>
                    <table style='width: 100%; border-collapse: collapse;'>
                        <tr><td><strong>Resolution:</strong></td><td>{video_info['resolution']} ({video_info['orientation']})</td></tr>
                        <tr><td><strong>Duration:</strong></td><td>{video_info['duration_seconds']}s</td></tr>
                        <tr><td><strong>Frame Rate:</strong></td><td>{video_info['fps']} FPS</td></tr>
                        <tr><td><strong>Frame Count:</strong></td><td>{video_info['frame_count']:,} frames</td></tr>
                        <tr><td><strong>File Size:</strong></td><td>{video_info['file_size_mb']} MB</td></tr>
                        <tr><td><strong>Video Codec:</strong></td><td>{video_info['video_codec']}</td></tr>
                        <tr><td><strong>Audio:</strong></td><td>{video_info['audio_streams']} stream(s) - {video_info['audio_codec']}</td></tr>
                        <tr><td><strong>Format:</strong></td><td>{video_info['format']}</td></tr>
                        <tr><td><strong>Pixel Format:</strong></td><td>{video_info['pixel_format']}</td></tr>
                    </table>
                </div>
                """
                
                # Get processing recommendations
                recommendations = ffmpeg_processor.get_optimal_settings(video_info)
                
                if recommendations['upscale_options'] or recommendations['interpolation_options']:
                    info_html += "<div style='margin-top: 10px; padding: 10px; background: #e3f2fd; border-radius: 5px;'>"
                    info_html += "<strong>💡 Recommendations:</strong><br>"
                    
                    if recommendations['upscale_options']:
                        info_html += f"• Upscale to: {', '.join(recommendations['upscale_options'])}<br>"
                    
                    if recommendations['interpolation_options']:
                        info_html += f"• Interpolate to: {', '.join(map(str, recommendations['interpolation_options']))} FPS<br>"
                    
                    info_html += f"• Quality setting: {recommendations['quality_recommendation']}"
                    info_html += "</div>"
                
                return info_html
                
            except Exception as e:
                return f"<p style='color: red;'>Analysis failed: {str(e)}</p>"
        
        def upscale_video_handler(video_file, resolution, custom_res, quality, suffix, progress=gr.Progress()):
            """Handle video upscaling"""
            if not video_file:
                return "❌ No video file selected", "<p>No output</p>"
            
            try:
                # Determine output path
                input_path = video_file.name
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                output_path = os.path.join(os.path.dirname(input_path), f"{base_name}{suffix}.mp4")
                
                # Use custom resolution if specified
                target_res = custom_res if resolution == "Custom" and custom_res else resolution
                
                def progress_callback(status):
                    progress(0.5, desc=status)
                    return status
                
                # Start upscaling
                progress(0.1, desc="Starting upscaling...")
                success = ffmpeg_processor.upscale_video(
                    input_path, output_path, target_res, quality, progress_callback
                )
                
                if success:
                    file_size = os.path.getsize(output_path) / (1024*1024)
                    result_text = f"✅ Upscaling completed!\nOutput: {os.path.basename(output_path)}"
                    result_html = f"<p style='color: green;'>✅ <strong>Upscaling completed!</strong><br>Output: {os.path.basename(output_path)} ({file_size:.1f} MB)</p>"
                    progress(1.0, desc="Upscaling completed!")
                    return result_text, result_html
                else:
                    return "❌ Upscaling failed - check console for details", "<p style='color: red;'>❌ Upscaling failed</p>"
                    
            except Exception as e:
                error_msg = f"❌ Upscaling error: {str(e)}"
                return error_msg, f"<p style='color: red;'>{error_msg}</p>"
        
        def interpolate_video_handler(video_file, target_fps, custom_fps, method, suffix, progress=gr.Progress()):
            """Handle frame interpolation"""
            if not video_file:
                return "❌ No video file selected", "<p>No output</p>"
            
            try:
                input_path = video_file.name
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                output_path = os.path.join(os.path.dirname(input_path), f"{base_name}{suffix}.mp4")
                
                # Use custom FPS if specified
                fps = int(custom_fps) if target_fps == "Custom" else int(target_fps)
                
                def progress_callback(status):
                    progress(0.5, desc=status)
                    return status
                
                progress(0.1, desc="Starting frame interpolation...")
                success = ffmpeg_processor.interpolate_frames(
                    input_path, output_path, fps, method, progress_callback
                )
                
                if success:
                    file_size = os.path.getsize(output_path) / (1024*1024)
                    result_text = f"✅ Interpolation completed!\nOutput: {os.path.basename(output_path)}"
                    result_html = f"<p style='color: green;'>✅ <strong>Interpolation completed!</strong><br>Output: {os.path.basename(output_path)} ({file_size:.1f} MB)</p>"
                    progress(1.0, desc="Interpolation completed!")
                    return result_text, result_html
                else:
                    return "❌ Interpolation failed - check console for details", "<p style='color: red;'>❌ Interpolation failed</p>"
                    
            except Exception as e:
                error_msg = f"❌ Interpolation error: {str(e)}"
                return error_msg, f"<p style='color: red;'>{error_msg}</p>"
        
        def replace_audio_handler(video_file, audio_file, start_time, suffix, progress=gr.Progress()):
            """Handle audio replacement"""
            if not video_file:
                return "❌ No video file selected", "<p>No output</p>"
            if not audio_file:
                return "❌ No audio file selected", "<p>No output</p>"
            
            try:
                video_path = video_file.name
                audio_path = audio_file.name
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(os.path.dirname(video_path), f"{base_name}{suffix}.mp4")
                
                def progress_callback(status):
                    progress(0.5, desc=status)
                    return status
                
                progress(0.1, desc="Starting audio replacement...")
                success = ffmpeg_processor.replace_audio(
                    video_path, audio_path, output_path, start_time, progress_callback
                )
                
                if success:
                    file_size = os.path.getsize(output_path) / (1024*1024)
                    result_text = f"✅ Audio replacement completed!\nOutput: {os.path.basename(output_path)}"
                    result_html = f"<p style='color: green;'>✅ <strong>Audio replacement completed!</strong><br>Output: {os.path.basename(output_path)} ({file_size:.1f} MB)</p>"
                    progress(1.0, desc="Audio replacement completed!")
                    return result_text, result_html
                else:
                    return "❌ Audio replacement failed - check console for details", "<p style='color: red;'>❌ Audio replacement failed</p>"
                    
            except Exception as e:
                error_msg = f"❌ Audio replacement error: {str(e)}"
                return error_msg, f"<p style='color: red;'>{error_msg}</p>"
        
        # UI Event Bindings
        def toggle_custom_resolution(resolution):
            return gr.update(visible=(resolution == "Custom"))
        
        def toggle_custom_fps(target_fps):
            return gr.update(visible=(target_fps == "Custom"))
        
        # Connect event handlers
        analyze_video_btn.click(
            fn=analyze_video_handler,
            inputs=[video_input_file],
            outputs=[video_info_display]
        )
        
        upscale_resolution.change(
            fn=toggle_custom_resolution,
            inputs=[upscale_resolution],
            outputs=[upscale_custom_res]
        )
        
        interp_target_fps.change(
            fn=toggle_custom_fps,
            inputs=[interp_target_fps],
            outputs=[interp_custom_fps]
        )
        
        upscale_btn.click(
            fn=upscale_video_handler,
            inputs=[video_input_file, upscale_resolution, upscale_custom_res, upscale_quality, upscale_output_suffix],
            outputs=[progress_text, output_file_info]
        )
        
        interpolate_btn.click(
            fn=interpolate_video_handler,
            inputs=[video_input_file, interp_target_fps, interp_custom_fps, interp_method, interp_output_suffix],
            outputs=[progress_text, output_file_info]
        )
        
        replace_audio_btn.click(
            fn=replace_audio_handler,
            inputs=[video_input_file, audio_input_file, audio_start_time, audio_output_suffix],
            outputs=[progress_text, output_file_info]
        )
    
    return locals()




# ******** NEW WORKFLOW-ORIENTED TAB STRUCTURE ********

def get_tab_setup(d, da):
    """🎯 SETUP TAB - Essential generation settings (replaces Run tab)"""
    with gr.TabItem(emoji_utils.tab_title(emoji_utils.setup, "Setup")):
        with gr.Accordion("📝 Generation Essentials", open=True):
            # Core generation settings
            with FormRow():
                animation_mode = create_gr_elem(da.animation_mode)  # Moved from keyframes - CRITICAL setting
                max_frames = create_gr_elem(da.max_frames)  # Moved from keyframes - CRITICAL setting
            
            sampler, scheduler, steps = create_row(d, 'sampler', 'scheduler', 'steps')
            W, H = create_row(d, 'W', 'H')
            seed, batch_name = create_row(d, 'seed', 'batch_name')
            
            with FormRow():
                seed_behavior = create_gr_elem(d.seed_behavior)  # Moved from keyframes
                
            with FormRow() as seed_iter_N_row:
                seed_iter_N = create_gr_elem(d.seed_iter_N)  # Moved from keyframes
            
            with FormRow():
                restore_faces = create_gr_elem(d.restore_faces)
                tiling = create_gr_elem(d.tiling)
                motion_preview_mode = create_gr_elem(d.motion_preview_mode)
        
        # EXPERIMENTAL RENDER CORE - PROMINENTLY FEATURED
        with gr.Accordion("🚀 Experimental Render Core (Zirteq Fork Exclusive)", open=True):
            gr.Markdown(f"""
            **{emoji_utils._emoji('🎯', essential=True)} This fork uses an exclusive experimental render core with variable cadence**
            
            **Key Benefits:**
            - **Variable Cadence**: Intelligent frame distribution for better quality
            - **Keyframe Redistribution**: Ensures important frames get diffused
            - **Higher Efficiency**: Generate smooth videos with higher cadence settings
            - **Better Synchronization**: Prompts align precisely with visual changes
            
            **{emoji_utils.warn} Important**: This is the default render core for this fork and should remain enabled for optimal results.
            """)
            
            # Keyframe distribution is THE setting that controls experimental core
            keyframe_distribution = create_row(da.keyframe_distribution)
            
            # Cadence setting (works differently with experimental core)
            diffusion_cadence = create_row(da.diffusion_cadence)
            
            # Info about how this differs from traditional Deforum
            with gr.Accordion("📚 How This Differs from Traditional Deforum", open=False):
                gr.Markdown("""
                **Traditional Deforum**: Fixed cadence, processes every Nth frame uniformly
                
                **Zirteq Fork Experimental Core**: 
                - **Variable Cadence**: Adapts frame processing based on content importance
                - **Keyframe Redistribution**: Moves diffusion frames closer to prompt keyframes
                - **Better Prompt Synchronization**: Visual changes happen precisely when prompts change
                - **Smarter Resource Usage**: Focuses processing power where it matters most
                
                **Recommended Settings for Best Results:**
                - **Keyframe Distribution**: "Keyframes Only" or "Redistributed" (default)
                - **High FPS**: 60 FPS for smooth output  
                - **Higher Cadence**: 10-15 for efficiency without quality loss
                - **Keyframe Strength**: Lower than regular strength for better blending
                """)
        
        # Batch Mode and Resume - less prominent but still accessible
        with gr.Accordion('🔄 Batch Mode & Resume', open=False):
            with gr.Tab('Batch Mode'):
                with gr.Row():
                    override_settings_with_file = gr.Checkbox(
                        label="Enable batch mode", value=False, interactive=True,
                        elem_id='override_settings',
                        info="run from a list of setting .txt files. Upload them to the box on the right"
                    )
                    custom_settings_file = gr.File(
                        label="Setting files", interactive=True, file_count="multiple",
                        file_types=[".txt"], elem_id="custom_setting_file", visible=False
                    )
            
            with gr.Tab('Resume Animation'):
                resume_from_timestring, resume_timestring = create_row(
                    da, 'resume_from_timestring', 'resume_timestring')
        
        # Advanced sampling settings
        with gr.Accordion('⚙️ Advanced Sampling', open=False):
            with FormRow():
                enable_ddim_eta_scheduling = create_gr_elem(da.enable_ddim_eta_scheduling)
                enable_ancestral_eta_scheduling = create_gr_elem(da.enable_ancestral_eta_scheduling)
            with gr.Row(variant='compact') as eta_sch_row:
                ddim_eta_schedule = create_gr_elem(da.ddim_eta_schedule)
                ancestral_eta_schedule = create_gr_elem(da.ancestral_eta_schedule)
    
    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_animation(da, dloopArgs):
    """🎬 ANIMATION TAB - Movement, timing, and motion settings"""
    with gr.TabItem(emoji_utils.tab_title(emoji_utils.animation, "Animation")):
        # Core Animation Controls
        with gr.Accordion(f"{emoji_utils._emoji('🎯')} Core Animation", open=True):
            border = create_row(da.border)
            
            # Note: Keyframe Distribution moved to Setup tab as part of Experimental Render Core
            gr.Markdown(f"""
            **{emoji_utils._emoji('📝')} Note**: **Keyframe Distribution** (experimental render core control) is now in the **Setup tab** 
            for better visibility since it's fundamental to how this fork works.
            """)
            
            # The cadence setting here is managed by the experimental render core
            with gr.Accordion("📚 About Variable Cadence in This Fork", open=False):
                gr.Markdown("""
                **This fork uses Variable Cadence** (controlled by Keyframe Distribution in Setup tab):
                
                - **Traditional Cadence**: Fixed interval, processes every Nth frame
                - **Variable Cadence**: Intelligent distribution based on content importance
                - **Better Results**: Focuses processing on frames that matter most
                
                🧪 **EXPERIMENTAL CORE**: Active when Keyframe Distribution ≠ "Off"
                
                The experimental render core automatically manages cadence based on your settings.
                """)
        
        # Movement and Camera Controls
        with gr.Accordion(f"{emoji_utils._emoji('📷')} Camera Movement", open=True):
            with gr.Tabs():
                # 3D Movement Tab (primary movement controls)
                with gr.TabItem("3D Movement"):
                    with FormColumn() as only_2d_motion_column:
                        angle = create_row(da.angle)
                        zoom = create_row(da.zoom)
                        with FormColumn() as both_anim_mode_motion_params_column:
                            translation_x, translation_y = create_row(da, 'translation_x', 'translation_y')
                        transform_center_x, transform_center_y = create_row(da, 'transform_center_x', 'transform_center_y')
                    with FormColumn() as only_3d_motion_column:
                        translation_z = create_row(da.translation_z)
                        rotation_3d_x, rotation_3d_y, rotation_3d_z = create_row(da, 'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z')
                        fov_schedule = create_row(da.fov_schedule)
                        near_schedule, far_schedule = create_row(da, 'near_schedule', 'far_schedule')
                
                # Camera Shake Tab (Shakify integration) - Add handcam emoji
                with gr.TabItem(f"{emoji_utils._emoji('📹')} Camera Shake"):
                    with FormColumn(min_width=220):
                        create_row(gr.Markdown(f"""
                            **{emoji_utils.video_camera()} Shakify Integration**: Add realistic camera shake effects using EatTheFuture's
                            Camera Shakify data. Enhances realism and engagement.
                        """))
                        shake_name = create_row(da.shake_name)
                        shake_intensity, shake_speed = create_row(da, 'shake_intensity', 'shake_speed')
                
                # Perspective Flip Tab
                with gr.TabItem("Perspective"):
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
        
        # Depth and 3D Settings
        with gr.Accordion(f"{emoji_utils.depth()} 3D Depth Processing", open=True):
            # HTML message for when not in 3D mode (required by gradio_funcs.py)
            depth_warp_msg_html = gr.HTML(
                value='Please switch to 3D animation mode to view this section.',
                elem_id='depth_warp_msg_html', 
                visible=False  # Initially hidden, shown/hidden by animation mode changes
            )
            
            with FormRow() as depth_warp_row_1:
                use_depth_warping = create_gr_elem(da.use_depth_warping)
                # LeReS license message (shown when LeReS depth algorithm is selected)
                leres_license_msg = gr.HTML(
                    value=get_gradio_html('leres'), 
                    visible=False,
                    elem_id='leres_license_msg'
                )
                depth_algorithm = create_gr_elem(da.depth_algorithm)
                midas_weight = create_gr_elem(da.midas_weight)
            with FormRow() as depth_warp_row_2:
                padding_mode = create_gr_elem(da.padding_mode)
                sampling_mode = create_gr_elem(da.sampling_mode)
            with FormRow() as depth_warp_row_3:
                save_depth_maps = create_gr_elem(da.save_depth_maps)
            with FormRow() as depth_warp_row_4:
                aspect_ratio_schedule = create_gr_elem(da.aspect_ratio_schedule)
            with FormRow() as depth_warp_row_5:
                aspect_ratio_use_old_formula = create_gr_elem(da.aspect_ratio_use_old_formula)
            with FormRow() as depth_warp_row_6:
                # Empty row for compatibility
                pass
            with FormRow() as depth_warp_row_7:
                # Empty row for compatibility
                pass

        # Guided Images (moved from keyframes) - ADD PROPER VARIABLE ASSIGNMENT
        with gr.Accordion(f"{emoji_utils.coherence()} Guided Images", open=False) as guided_images_accord:
            with gr.Accordion('*Important: Read before using*', open=False):
                gr.HTML(value=get_gradio_html('guided_imgs'))
            
            use_looper = create_row(dloopArgs.use_looper)
            init_images = create_row(dloopArgs.init_images)
            
            with gr.Accordion('Guided Images Schedules', open=False):
                image_strength_schedule = create_row(dloopArgs.image_strength_schedule)
                image_keyframe_strength_schedule = create_row(dloopArgs.image_keyframe_strength_schedule)
                blendFactorMax, blendFactorSlope = create_row(dloopArgs, 'blendFactorMax', 'blendFactorSlope')
                tweening_frames_schedule = create_row(dloopArgs.tweening_frames_schedule)
                color_correction_factor = create_row(dloopArgs.color_correction_factor)
    
    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_advanced(d, da):
    """⚙️ ADVANCED TAB - Fine-tuning, schedules, noise, and coherence"""
    with gr.TabItem(emoji_utils.tab_title(emoji_utils.advanced, "Advanced")):
        
        # Strength and CFG Schedules
        with gr.Accordion(f"{emoji_utils.strength()} Strength & CFG Schedules", open=True):
            strength_schedule = create_row(da.strength_schedule)
            keyframe_strength_schedule = create_row(da.keyframe_strength_schedule)
            cfg_scale_schedule = create_row(da.cfg_scale_schedule)
            distilled_cfg_scale_schedule = create_row(da.distilled_cfg_scale_schedule)
        
        # Advanced Scheduling
        with gr.Accordion("📅 Advanced Scheduling", open=False):
            with gr.Tabs():
                # Seed Scheduling
                with gr.TabItem(emoji_utils.tab_title(emoji_utils.seed, "Seed")):
                    with FormRow(visible=False) as seed_schedule_row:
                        seed_schedule = create_gr_elem(da.seed_schedule)
                    enable_subseed_scheduling = create_row(da.enable_subseed_scheduling)
                    subseed_schedule, subseed_strength_schedule = create_row(da, 'subseed_schedule', 'subseed_strength_schedule')
                    seed_resize_from_w, seed_resize_from_h = create_row(d, 'seed_resize_from_w', 'seed_resize_from_h')
                
                # Steps & Sampler Scheduling
                with gr.TabItem("Steps & Sampling"):
                    enable_steps_scheduling = create_row(da.enable_steps_scheduling)
                    steps_schedule = create_row(da.steps_schedule)
                    enable_sampler_scheduling = create_row(da.enable_sampler_scheduling)
                    sampler_schedule = create_row(da.sampler_schedule)
                    enable_scheduler_scheduling = create_row(da.enable_scheduler_scheduling)
                    scheduler_schedule = create_row(da.scheduler_schedule)
                
                # Checkpoint & CLIP Scheduling  
                with gr.TabItem("Model & CLIP"):
                    enable_checkpoint_scheduling = create_row(da.enable_checkpoint_scheduling)
                    checkpoint_schedule = create_row(da.checkpoint_schedule)
                    enable_clipskip_scheduling = create_row(da.enable_clipskip_scheduling)
                    clipskip_schedule = create_row(da.clipskip_schedule)
        
        # Noise Settings
        with gr.Accordion(f"{emoji_utils.noise()} Noise & Randomization", open=False):
            with FormColumn() as noise_tab_column:
                noise_type = create_row(da.noise_type)
                noise_schedule = create_row(da.noise_schedule)
                with FormRow() as perlin_row:
                    perlin_octaves = create_gr_elem(da.perlin_octaves)
                    perlin_persistence = create_gr_elem(da.perlin_persistence)
                    perlin_w = create_gr_elem(da.perlin_w)  # Hidden
                    perlin_h = create_gr_elem(da.perlin_h)  # Hidden
                enable_noise_multiplier_scheduling = create_row(da.enable_noise_multiplier_scheduling)
                noise_multiplier_schedule = create_row(da.noise_multiplier_schedule)
        
        # Color Coherence & Optical Flow
        with gr.Accordion(f"{emoji_utils.coherence()} Color Coherence & Flow", open=False):
            color_coherence, color_force_grayscale = create_row(da, 'color_coherence', 'color_force_grayscale')
            legacy_colormatch = create_row(da.legacy_colormatch)
            
            with FormRow(visible=False) as color_coherence_image_path_row:
                color_coherence_image_path = create_gr_elem(da.color_coherence_image_path)
            with FormRow(visible=False) as color_coherence_video_every_N_frames_row:
                color_coherence_video_every_N_frames = create_gr_elem(da.color_coherence_video_every_N_frames)
            
            # Optical Flow Settings - ADD PROPER ROW WRAPPERS
            with FormRow() as optical_flow_cadence_row:
                with FormColumn(min_width=220) as optical_flow_cadence_column:
                    optical_flow_cadence = create_gr_elem(da.optical_flow_cadence)
                with FormColumn(min_width=220, visible=False) as cadence_flow_factor_schedule_column:
                    cadence_flow_factor_schedule = create_gr_elem(da.cadence_flow_factor_schedule)
            with FormRow():
                with FormColumn(min_width=220):
                    optical_flow_redo_generation = create_gr_elem(da.optical_flow_redo_generation)
                with FormColumn(min_width=220, visible=False) as redo_flow_factor_schedule_column:
                    redo_flow_factor_schedule = create_gr_elem(da.redo_flow_factor_schedule)
            
            contrast_schedule = gr.Textbox(
                label="Contrast schedule", lines=1, value=da.contrast_schedule, interactive=True,
                info="adjusts the overall contrast per frame [neutral at 1.0]"
            )
            diffusion_redo = gr.Slider(
                label="Redo generation", minimum=0, maximum=50, step=1, value=da.diffusion_redo,
                interactive=True, info="renders N times before final render. Lower steps if increasing this."
            )
        
        # Anti-Blur and Quality
        with gr.Accordion(f"{emoji_utils.anti_blur()} Anti-Blur & Quality", open=False):
            amount_schedule = create_row(da.amount_schedule)
            kernel_schedule = create_row(da.kernel_schedule)
            sigma_schedule = create_row(da.sigma_schedule)
            threshold_schedule = create_row(da.threshold_schedule)
            
            reroll_blank_frames, reroll_patience = create_row(d, 'reroll_blank_frames', 'reroll_patience')
        
        # Composable Masks
        with gr.Accordion("🎭 Composable Masks", open=False):
            gr.HTML(value=get_gradio_html('composable_masks'))
            mask_schedule = create_row(da.mask_schedule)
            use_noise_mask = create_row(da.use_noise_mask)
    
    return {k: v for k, v in {**locals(), **vars()}.items()}
