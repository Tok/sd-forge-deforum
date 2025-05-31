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
from .rendering.util import emoji_utils
from .defaults import get_gradio_html, DeforumAnimPrompts
from .gradio_funcs import (upload_vid_to_interpolate, upload_pics_to_interpolate,
                           ncnn_upload_vid_to_upscale, upload_vid_to_depth)
from .video_audio_utilities import direct_stitch_vid_from_frames


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
                    seed_iter_N = create_row(d.seed_iter_N)
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
        # COMPOSABLE MASK SCHEDULING ACCORD
        with gr.Accordion('Composable Mask scheduling', open=False):
            gr.HTML(value=get_gradio_html('composable_masks'))
            mask_schedule = create_row(da.mask_schedule)
            use_noise_mask = create_row(da.use_noise_mask)
            noise_mask_schedule = create_row(da.noise_mask_schedule)

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


def get_tab_freeu(dfu: SimpleNamespace):
    with gr.TabItem('FreeU'):
        freeu_enabled = create_row(dfu.freeu_enabled)
        freeu_b1 = create_row(dfu.freeu_b1)
        freeu_b2 = create_row(dfu.freeu_b2)
        freeu_s1 = create_row(dfu.freeu_s1)
        freeu_s2 = create_row(dfu.freeu_s2)
    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_kohya_hrfix(dku: SimpleNamespace):
    with gr.TabItem('Kohya HR Fix'):
        kohya_hrfix_enabled = create_row(dku.kohya_hrfix_enabled)
        kohya_hrfix_block_number = create_row(dku.kohya_hrfix_block_number)
        kohya_hrfix_downscale_factor = create_row(dku.kohya_hrfix_downscale_factor)
        kohya_hrfix_start_percent = create_row(dku.kohya_hrfix_start_percent)
        kohya_hrfix_end_percent = create_row(dku.kohya_hrfix_end_percent)
        kohya_hrfix_downscale_after_skip = create_row(dku.kohya_hrfix_downscale_after_skip)
        kohya_hrfix_downscale_method = create_row(dku.kohya_hrfix_downscale_method)
        kohya_hrfix_upscale_method = create_row(dku.kohya_hrfix_upscale_method)
    return {k: v for k, v in {**locals(), **vars()}.items()}


def wan_generate_video(*component_args):
    """
    Function to handle Wan video generation from the Wan tab
    This function calls the main Deforum generation pipeline with Wan mode
    """
    try:
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
        
        # Parse resolution
        width, height = map(int, wan_args.wan_resolution.split('x'))
        
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
    """Wan 2.1 Video Generation Tab - Integrated with Deforum Schedules"""
    with gr.TabItem(f"{emoji_utils.wan_video()} Wan Video"):
        # Quick Start Section - Most Important
        with gr.Accordion("🚀 Quick Start", open=True):
            gr.Markdown("""
            **Ready to Generate Wan Videos with I2V Chaining:**
            
            1. **Configure prompts** in the **Prompts tab** (REQUIRED)
            2. **Set FPS** in the **Output tab** (REQUIRED)  
            3. **Choose VACE model** below (1.3B recommended for most users)
            4. **Click Generate Wan Video** for seamless I2V chaining
            
            **💡 I2V Chaining**: Each clip uses the last frame from the previous clip as input, creating smooth transitions!
            """)
            
            # Generate Button - Prominently placed at top
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
                lines=2,
                placeholder="Ready to generate Wan video with I2V chaining using Deforum schedules..."
            )
        
        # Essential Settings - Open by default
        with gr.Accordion("Essential Settings", open=True):
            gr.Markdown("""
            **🔥 VACE Models Recommended**: All-in-one models that handle both T2V and I2V in one package!
            - **1.3B VACE**: 8GB VRAM, 480P, fast (perfect for most users)
            - **14B VACE**: 480P+720P, slower, higher quality (for power users)
            """)
            
            with FormRow():
                wan_t2v_model = create_gr_elem(dw.wan_t2v_model)
                wan_i2v_model = create_gr_elem(dw.wan_i2v_model)
                
            with FormRow():
                wan_auto_download = create_gr_elem(dw.wan_auto_download)
                wan_preferred_size = create_gr_elem(dw.wan_preferred_size)
                
            with FormRow():
                wan_model_path = create_gr_elem(dw.wan_model_path)
                
            with FormRow():
                wan_resolution = create_gr_elem(dw.wan_resolution)
                
            with FormRow():
                # Explicitly create steps slider with unique ID to force refresh
                wan_inference_steps = gr.Slider(
                    label="Inference Steps (Min: 5)",
                    minimum=5,
                    maximum=100,
                    step=1,
                    value=20,
                    elem_id="wan_inference_steps_fixed_min_5",
                    info="Number of inference steps for Wan generation. Lower values (5-15) for quick testing, higher values (20-50) for quality"
                )
                # wan_guidance_scale = create_gr_elem(dw.wan_guidance_scale)  # Moved to override section
        
        # Prompt Enhancement Section - MOVED UP and REORGANIZED for better workflow
        with gr.Accordion("🎨 Wan Prompt Management & AI Enhancement", open=True):
            gr.Markdown("""
            **Wan Video Prompt Workflow:**
            
            1. **📋 Load Initial Prompts** from Deforum or defaults
            2. **📐 Analyze Movement** (optional) - adds camera movement descriptions  
            3. **🎨 Enhance with AI** (optional) - improves prompt quality with Qwen models
            4. **✏️ Manual Edit** (optional) - fine-tune the final prompts
            5. **🎬 Generate Video** - create your Wan video with enhanced prompts
            """)
            
            # Step 1: Load Initial Prompts
            gr.Markdown("### **Step 1: Load Initial Prompts**")
            with FormRow():
                load_deforum_to_wan_btn = gr.Button(
                    "📋 Load Wan Prompts from Deforum Prompts",
                    variant="primary",
                    size="lg",
                    elem_id="load_deforum_to_wan_btn"
                )
                load_wan_defaults_btn = gr.Button(
                    "📝 Load Default Wan Prompt",
                    variant="secondary", 
                    size="lg",
                    elem_id="load_wan_defaults_btn"
                )
            
            # Wan Prompts Display - ALWAYS VISIBLE
            wan_enhanced_prompts = gr.Textbox(
                label="Wan Video Prompts (Editable JSON)",
                lines=12,
                interactive=True,
                placeholder='Click "Load Wan Prompts from Deforum Prompts" to copy your current prompts, or "Load Default Wan Prompt" to start with defaults.',
                info="These are the prompts that will be used for Wan video generation in JSON format. You can manually edit these or enhance them with AI.",
                elem_id="wan_enhanced_prompts_textbox"
            )
            
            # Step 2: Movement Analysis (Optional)
            gr.Markdown("### **Step 2: Analyze Movement (Optional)**")
            gr.Markdown("""
            **🎬 Uses the Same Movement System as Normal Deforum Renders:**
            
            Analyzes your **exact same movement schedules** from **Keyframes → Motion tab**:
            - **Translation X/Y/Z**: Camera position → "left pan", "forward dolly", "upward crane"  
            - **Rotation 3D X/Y/Z**: Camera rotation → "upward pitch", "right yaw", "clockwise roll"
            - **Zoom**: Zoom changes → "slow zoom in", "fast zoom out"
            - **Parseq Compatible**: Also works with Parseq movement schedules
            
            **Movement descriptions are added to prompts** to help Wan understand camera motion.
            """)
            
            with FormRow():
                wan_movement_sensitivity = create_gr_elem(dw.wan_movement_sensitivity)
                analyze_movement_btn = gr.Button(
                    "📐 Analyze Deforum Movement & Add Descriptions",
                    variant="primary",
                    size="lg",
                    elem_id="wan_analyze_movement_btn"
                )
                
            # Movement Analysis Results - ALWAYS VISIBLE
            wan_movement_description = gr.Textbox(
                label="Deforum Movement Analysis Results",
                lines=8,
                interactive=False,
                placeholder="Click 'Analyze Deforum Movement' to see detailed movement analysis from your Keyframes → Motion schedules...",
                info="Movement analysis results from your Deforum schedules. Movement descriptions are automatically added to wan prompts.",
                elem_id="wan_movement_description_textbox"
            )
            
            # Step 3: AI Enhancement (Optional)  
            gr.Markdown("### **Step 3: AI Enhancement (Optional)**")
            gr.Markdown("""
            **Enhance prompts using Qwen AI models** for better video quality:
            - **🧠 AI Enhancement**: Refines and expands prompts
            - **🎬 Movement Integration**: Uses movement descriptions from Step 2
            - **🌍 Multi-Language**: English and Chinese support
            """)
            
            with FormRow():
                wan_qwen_model = create_gr_elem(dw.wan_qwen_model)
                wan_qwen_language = create_gr_elem(dw.wan_qwen_language)
                wan_qwen_auto_download = create_gr_elem(dw.wan_qwen_auto_download)
                
            with FormRow():
                enhance_prompts_btn = gr.Button(
                    "🎨 Enhance Prompts with AI",
                    variant="primary",
                    size="lg",
                    elem_id="wan_enhance_prompts_btn"
                )
                
        # Qwen Model Management - Moved to separate accordion
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

        # Advanced Settings - Moved down but still accessible
        with gr.Accordion("Advanced Settings", open=False):
            with FormRow():
                wan_frame_overlap = create_gr_elem(dw.wan_frame_overlap)
                # Remove wan_motion_strength from here - moved to overrides
                
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
                
                with FormRow():
                    wan_flash_attention_mode = create_gr_elem(dw.wan_flash_attention_mode)
                    
                with FormRow():
                    wan_flash_attention_status = gr.HTML(
                        label="Flash Attention Status",
                        value="⏳ Checking availability...",
                        elem_id="wan_flash_attention_status"
                    )
                    check_flash_attention_btn = gr.Button(
                        "🔍 Check Flash Attention",
                        variant="secondary",
                        size="sm",
                        elem_id="wan_check_flash_attention_btn"
                    )
                
            # Strength Override Section
            with gr.Accordion("🔗 Schedule Override Controls", open=True):
                gr.Markdown("""
                **Deforum Schedule Integration (Recommended)**
                
                By default, Wan respects Deforum's scheduling system:
                - **📈 Strength Schedule**: From Keyframes → Strength tab (controls I2V continuity)
                - **📊 CFG Scale Schedule**: From Keyframes → CFG tab (controls prompt adherence)
                
                **Override Options (Advanced)**
                
                Enable overrides below to use fixed values instead of Deforum schedules:
                
                **Strength Override**: 
                - Controls how much the previous frame influences the next clip
                - High (0.8-0.9): Strong continuity, smooth transitions
                - Low (0.3-0.5): More creative freedom, less continuity
                
                **Guidance Scale Override**:
                - Controls how closely generation follows the prompt
                - Higher values (7.5-12): Strong prompt adherence
                - Lower values (3-6): More creative interpretation
                
                **💡 When to Use Overrides:**
                - For consistent I2V chaining (strength override)
                - For stable prompt adherence across clips (guidance override)
                - When you want fixed values instead of scheduled changes
                """)
                
                gr.Markdown("**🔧 Strength Control (I2V Continuity)**")
                with FormRow():
                    wan_strength_override = create_gr_elem(dw.wan_strength_override)
                    wan_fixed_strength = create_gr_elem(dw.wan_fixed_strength)
                
                gr.Markdown("**🎯 Guidance Scale Control (Prompt Adherence)**")
                with FormRow():
                    wan_guidance_override = create_gr_elem(dw.wan_guidance_override)
                    wan_guidance_scale = create_gr_elem(dw.wan_guidance_scale)
                    
                gr.Markdown("**🎬 Motion Control (Advanced)**")
                with FormRow():
                    wan_motion_strength_override = create_gr_elem(dw.wan_motion_strength_override)
                    wan_motion_strength = create_gr_elem(dw.wan_motion_strength)
        
        # Auto-Discovery Info - Collapsed by default
        with gr.Accordion("🔍 Auto-Discovery & Setup", open=False):
            gr.Markdown("""
            **Smart Model Discovery & Auto-Download**
            
            Wan models are automatically discovered from common locations:
            - `models/wan/` (recommended location)
            - HuggingFace cache
            
            **🔥 VACE Models (Recommended for I2V Chaining):**
            - **1.3B VACE (Default)**: ~17GB download, 8GB VRAM, 480P, consumer-friendly
            - **14B VACE**: ~75GB download, 480P+720P, slower, higher quality
            
            **💡 Why VACE Models?**
            - All-in-one: Handle both T2V and I2V in single model
            - Perfect for I2V chaining with seamless transitions
            - No need for separate I2V models
            """)
            
            # Model Validation Section
            with gr.Accordion("🔍 Model Validation & Integrity Check", open=False):
                gr.Markdown("""
                **Advanced Model Validation with Checksum Verification**
                
                Verify your downloaded Wan models are complete and not corrupted:
                - ✅ **File size validation**: Detect suspiciously small files
                - ✅ **JSON config validation**: Verify configuration files
                - ✅ **Safetensors integrity**: Check model weights are readable
                - ✅ **Git LFS detection**: Find incomplete downloads (pointer files)
                - ✅ **Checksum verification**: Calculate and verify file hashes
                - ✅ **Structure validation**: Ensure all required files are present
                """)
                
                with FormRow():
                    validate_models_btn = gr.Button(
                        "🔍 Validate All Models",
                        variant="secondary",
                        elem_id="wan_validate_models_btn"
                    )
                    cleanup_invalid_btn = gr.Button(
                        "🗑️ Clean Up Invalid Models",
                        variant="secondary",
                        elem_id="wan_cleanup_invalid_btn"
                    )
                    
                with FormRow():
                    compute_checksums_btn = gr.Button(
                        "🔐 Compute Model Checksums",
                        variant="secondary",
                        elem_id="wan_compute_checksums_btn"
                    )
                    verify_integrity_btn = gr.Button(
                        "✅ Full Integrity Check",
                        variant="primary",
                        elem_id="wan_verify_integrity_btn"
                    )
                
                # Validation output
                validation_output = gr.Textbox(
                    label="Validation Results",
                    interactive=False,
                    lines=10,
                    placeholder="Click a validation button to check your Wan models...",
                    elem_id="wan_validation_output"
                )
                
                # Model details output
                model_details_output = gr.JSON(
                    label="Model Details & Checksums",
                    visible=False,
                    elem_id="wan_model_details"
                )
            
            gr.Markdown("""
            **Quick Download Commands:**
            ```bash
            # Install HuggingFace CLI (if not already installed)
            pip install huggingface_hub
            
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
        
        # Integration Status - Collapsed by default
        with gr.Accordion("🔗 Deforum Integration", open=False):
            gr.Markdown("""
            **Wan uses these settings from other Deforum tabs:**
            
            - **📝 Prompts:** From Prompts tab
            - **🎬 FPS:** From Output tab (no separate Wan FPS needed)
            - **🎲 Seed:** From Keyframes → Seed & SubSeed tab
            - **💪 Strength:** From Keyframes → Strength tab (for I2V chaining)
            - **📊 CFG Scale:** From Keyframes → CFG tab (for prompt adherence)
            - **🎬 Movement:** From Keyframes → Motion tab (same schedules as normal Deforum renders)
            - **⏱️ Duration:** Auto-calculated from prompt timing
            
            **Movement Integration:**
            - ✅ Uses **exact same movement schedules** as normal Deforum renders
            - ✅ Translation X/Y/Z, Rotation 3D X/Y/Z, Zoom, Angle schedules
            - ✅ **Parseq schedules fully supported** for advanced movement
            - ✅ Movement descriptions automatically added to prompts
            - ✅ Motion intensity dynamically adapts to movement complexity
            
            **Features:**
            - ✅ Uses Deforum's prompt scheduling system
            - ✅ Uses Deforum's FPS and seed settings  
            - ✅ Uses Deforum's movement scheduling system
            - ✅ Auto-discovery finds your models automatically
            - ✅ Smart model selection based on your preference
            - ✅ Calls the full Deforum generation pipeline
            - ✅ I2V chaining for seamless transitions between clips
            - ✅ PNG frame output with 4n+1 calculation
            - ✅ Strength scheduling for continuity control
            - ✅ CFG scale scheduling for prompt adherence control
            """)
        
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
    print(f"📊 Found {len(component_names)} UI components for Wan generation")
    
    # NOTE: enhance_prompts_btn connection is now handled in ui_left.py for proper component access
    
    # Connect event handlers for movement analysis - simplified without enable checkbox
    analyze_movement_btn.click(
        fn=analyze_movement_handler,
        inputs=[wan_movement_sensitivity],
        outputs=[wan_movement_description]
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


def get_tab_hybrid(da):
    with gr.TabItem('Hybrid Video'):
        # this html only shows when not in 2d/3d mode
        hybrid_msg_html = gr.HTML(value='Change animation mode to 2D or 3D to enable Hybrid Mode', visible=False,
                                  elem_id='hybrid_msg_html')
        # HYBRID INFO ACCORD
        with gr.Accordion("Info & Help", open=False):
            gr.HTML(value=get_gradio_html('hybrid_video'))
        # HYBRID SETTINGS ACCORD
        with gr.Accordion("Hybrid Settings", open=True) as hybrid_settings_accord:
            hybrid_composite = create_row(gr.Radio(['None', 'Normal', 'Before Motion', 'After Generation'],
                                                   label="Hybrid composite", value=da.hybrid_composite,
                                                   elem_id="hybrid_composite"))
            with FormRow():
                with FormColumn(min_width=340):
                    with FormRow():
                        hybrid_generate_inputframes = create_gr_elem(da.hybrid_generate_inputframes)
                        hybrid_use_first_frame_as_init_image = create_gr_elem(da.hybrid_use_first_frame_as_init_image)
                        hybrid_use_init_image = create_gr_elem(da.hybrid_use_init_image)
            with FormRow():
                with FormColumn():
                    hybrid_motion = create_row(da.hybrid_motion)
                with FormColumn():
                    with FormRow():
                        with FormColumn(scale=1):
                            hybrid_flow_method = create_gr_elem(da.hybrid_flow_method)
                    with FormRow():
                        with FormColumn():
                            hybrid_flow_consistency = create_gr_elem(da.hybrid_flow_consistency)
                            hybrid_consistency_blur = create_gr_elem(da.hybrid_consistency_blur)
                        with FormColumn():
                            hybrid_motion_use_prev_img = create_gr_elem(da.hybrid_motion_use_prev_img)
            hybrid_comp_mask_type = create_row(da.hybrid_comp_mask_type)
            with gr.Row(visible=False, variant='compact') as hybrid_comp_mask_row:
                hybrid_comp_mask_equalize = create_gr_elem(da.hybrid_comp_mask_equalize)
                with FormColumn():
                    hybrid_comp_mask_auto_contrast = gr.Checkbox(
                        label="Comp mask auto contrast", value=False, interactive=True)
                    hybrid_comp_mask_inverse = gr.Checkbox(
                        label="Comp mask inverse", value=da.hybrid_comp_mask_inverse, interactive=True)
            hybrid_comp_save_extra_frames = create_row(gr.Checkbox(
                label="Comp save extra frames", value=False, interactive=True))
        # HYBRID SCHEDULES ACCORD
        with gr.Accordion("Hybrid Schedules", open=False, visible=False) as hybrid_sch_accord:
            with FormRow() as hybrid_comp_alpha_schedule_row:
                hybrid_comp_alpha_schedule = create_gr_elem(da.hybrid_comp_alpha_schedule)
            with FormRow() as hybrid_flow_factor_schedule_row:
                hybrid_flow_factor_schedule = create_gr_elem(da.hybrid_flow_factor_schedule)
            with FormRow(visible=False) as hybrid_comp_mask_blend_alpha_schedule_row:
                hybrid_comp_mask_blend_alpha_schedule = create_gr_elem(da.hybrid_comp_mask_blend_alpha_schedule)
            with FormRow(visible=False) as hybrid_comp_mask_contrast_schedule_row:
                hybrid_comp_mask_contrast_schedule = create_gr_elem(da.hybrid_comp_mask_contrast_schedule)
            with FormRow(visible=False) as hybrid_comp_mask_auto_contrast_cutoff_high_schedule_row:
                hybrid_comp_mask_auto_contrast_cutoff_high_schedule = create_gr_elem(
                    da.hybrid_comp_mask_auto_contrast_cutoff_high_schedule)
            with FormRow(visible=False) as hybrid_comp_mask_auto_contrast_cutoff_low_schedule_row:
                hybrid_comp_mask_auto_contrast_cutoff_low_schedule = create_gr_elem(
                    da.hybrid_comp_mask_auto_contrast_cutoff_low_schedule)
        # HUMANS MASKING ACCORD
        with gr.Accordion("Humans Masking", open=False, visible=False) as humans_masking_accord:
            hybrid_generate_human_masks = create_row(da.hybrid_generate_human_masks)

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
        # Vid2Depth TAB - not built using our args.py at all - all data and params are here and in .vid2depth file
        with gr.TabItem('Vid2depth'):
            vid_to_depth_chosen_file = gr.File(label="Video to get Depth from", interactive=True, file_count="single",
                                               file_types=["video"], elem_id="vid_to_depth_chosen_file")
            with FormRow():
                mode = gr.Dropdown(label='Mode', elem_id="mode",
                                   choices=['Depth (Midas/Adabins)', 'Anime Remove Background', 'Mixed',
                                            'None (just grayscale)'], value='Depth (Midas/Adabins)')
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
    with FormRow():
        with gr.Accordion(name, open=is_open):
            gr.Markdown(markdown)


# QwenPromptExpander and Movement Analysis Event Handlers - moved outside for proper import
def enhance_prompts_handler(qwen_model, language, auto_download, movement_sensitivity):
    """Handle prompt enhancement with QwenPromptExpander"""
    try:
        from .wan.utils.qwen_manager import qwen_manager
        import json
        
        # Check if auto-download is enabled for model availability
        if not auto_download:
            # Check if the selected model is available
            if not qwen_manager.is_model_downloaded(qwen_model):
                return f"""❌ Qwen model not available: {qwen_model}

🔧 **Model Download Required:**
1. ✅ Enable "Auto-Download Qwen Models" checkbox
2. 🎨 Click "Enhance Prompts" again to auto-download
3. ⏳ Wait for download to complete

📥 **Manual Download Alternative:**
1. Use HuggingFace CLI: `huggingface-cli download {qwen_manager.get_model_info(qwen_model).get('huggingface_id', 'model-id')}`
2. ✅ Enable auto-download for easier setup

💡 **Auto-download is recommended** for seamless model management."""
        
        # Check if a model is already loaded
        if qwen_manager.is_model_loaded():
            loaded_info = qwen_manager.get_loaded_model_info()
            current_model = loaded_info['name'] if loaded_info else "Unknown"
            
            # If different model requested, cleanup first
            if qwen_model != "Auto-Select" and current_model != qwen_model:
                print(f"🔄 Switching from {current_model} to {qwen_model}")
                qwen_manager.cleanup_cache()
        
        # Provide loading feedback
        if not qwen_manager.is_model_loaded():
            if qwen_model == "Auto-Select":
                selected_model = qwen_manager.auto_select_model()
                print(f"🤖 Auto-selected model: {selected_model}")
            else:
                print(f"📥 Loading Qwen model: {qwen_model}")
        
        # Try to get wan prompts from the stored wan_enhanced_prompts component
        animation_prompts = None
        
        # Use a global reference approach - store the wan_enhanced_prompts component
        # This will be set when the UI is created
        if hasattr(enhance_prompts_handler, '_wan_enhanced_prompts_component'):
            try:
                wan_prompts_value = enhance_prompts_handler._wan_enhanced_prompts_component.value
                print(f"📝 Got wan prompts from stored component reference")
                print(f"🔍 Raw wan prompts value: {str(wan_prompts_value)[:200]}...")
                
                # Parse the wan prompts - could be JSON or readable format
                if wan_prompts_value and wan_prompts_value.strip():
                    try:
                        # Try to parse as JSON first
                        animation_prompts = json.loads(wan_prompts_value)
                        print(f"✅ Successfully parsed {len(animation_prompts)} wan prompts as JSON")
                    except json.JSONDecodeError:
                        # Try to parse as readable format (Frame X: prompt)
                        try:
                            animation_prompts = {}
                            for line in wan_prompts_value.strip().split('\n'):
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
                                print(f"✅ Successfully parsed {len(animation_prompts)} wan prompts as readable format")
                            else:
                                raise ValueError("No valid prompts found")
                        except Exception as e:
                            print(f"❌ Could not parse wan prompts: {e}")
                            return f"❌ Invalid format in wan prompts. Expected JSON format like:\n{{\n  \"0\": \"prompt text\",\n  \"60\": \"another prompt\"\n}}\n\nOr readable format like:\nFrame 0: prompt text\nFrame 60: another prompt"
                else:
                    print("⚠️ Empty wan prompts")
                    
            except Exception as e:
                print(f"❌ Error accessing stored wan prompts component: {e}")
        else:
            print("⚠️ No stored wan_enhanced_prompts component reference found")
            
        # Check if we got valid prompts
        if not animation_prompts:
            return """❌ No wan prompts found!

🔧 **Setup Required:**
1. 📝 Load prompts using "Load Wan Prompts from Deforum Prompts" or "Load Default Wan Prompt"
2. 📋 Make sure your prompts are in proper JSON format like:
   {
     "0": "prompt text",
     "60": "another prompt",
     "120": "a cyberpunk environment with glowing elements"
   }
3. 🎨 Click **Enhance Prompts** again after setting up prompts

💡 **Quick Start:**
Click "Load Default Wan Prompt" to start with example prompts!"""
        
        # Validate prompts content
        if len(animation_prompts) == 1 and "0" in animation_prompts and "beautiful landscape" in animation_prompts["0"]:
            return """❌ Default prompts detected!

🔧 **Please configure your actual animation prompts:**
1. 📝 Load your real prompts using the load buttons above
2. ✏️ Or manually edit the wan prompts field
3. 🎨 Click **Enhance Prompts** again

💡 **For your animation sequence:**
Set up prompts like:
{
  "0": "A peaceful scene, photorealistic",
  "18": "A scene with glowing effects, neon colors, synthwave aesthetic",
  "36": "A cyberpunk scene with LED patterns, digital environment"
}"""
        
        print(f"🎨 Enhancing {len(animation_prompts)} wan prompts with {qwen_model}")
        
        # Create the Qwen prompt expander with better error handling
        try:
            prompt_expander = qwen_manager.create_prompt_expander(qwen_model, auto_download)
            
            if not prompt_expander:
                if auto_download:
                    return f"""⏳ Downloading {qwen_model} model...

🔄 **Download in Progress:**
Model download started automatically. This may take a few minutes.

📥 **Please wait** and try clicking "Enhance Prompts" again in 30-60 seconds.

💡 **Status**: Check console for download progress."""
                else:
                    return f"""❌ Failed to create Qwen prompt expander: {qwen_model}

🔧 **Solutions:**
1. ✅ Enable "Auto-Download Qwen Models" and try again
2. 📥 Manual download: Check console for HuggingFace CLI commands
3. 🔄 Restart WebUI after downloading

📊 **Model Info**: {qwen_manager.get_model_info(qwen_model).get('description', 'N/A')}"""
        except Exception as e:
            return f"""❌ Error creating Qwen prompt expander: {str(e)}

🔧 **Troubleshooting:**
1. ✅ Enable auto-download and try again
2. 🔄 Restart WebUI if models were just downloaded
3. 💾 Check available disk space ({qwen_manager.get_model_info(qwen_model).get('vram_gb', 'Unknown')}GB VRAM required)

💡 **Tip**: Try selecting "Auto-Select" for automatic model choice."""
        
        # Use the QwenModelManager's enhance_prompts method directly
        try:
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
            
            # Append movement descriptions to enhanced prompts if available
            if movement_description and movement_description.strip():
                for frame_key in enhanced_prompts_dict:
                    original_prompt = enhanced_prompts_dict[frame_key]
                    enhanced_prompts_dict[frame_key] = f"{original_prompt}. {movement_description}"
                print(f"✅ Appended movement description to {len(enhanced_prompts_dict)} enhanced prompts")
            
            # Format the enhanced prompts as JSON
            enhanced_json = json.dumps(enhanced_prompts_dict, ensure_ascii=False, indent=2)
            
            print(f"✅ Successfully enhanced {len(enhanced_prompts_dict)} prompts")
            
            # Return the enhanced prompts
            return enhanced_json
            
        except Exception as e:
            print(f"❌ Error enhancing prompts: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error enhancing prompts: {str(e)}"
    
    except Exception as e:
        print(f"❌ Error in enhance_prompts_handler: {e}")
        import traceback
        traceback.print_exc()
        return f"""❌ Error in prompt enhancement: {str(e)}

🔧 **Troubleshooting:**
1. 🔄 Restart WebUI and try again
2. ✅ Check that Qwen models are properly installed
3. 📝 Verify your wan prompts are in valid JSON format

💡 **Need Help?** Check the console for detailed error messages."""

def analyze_movement_handler(movement_sensitivity):
    """Handle movement analysis from Deforum schedules and add movement descriptions"""
    try:
        from .wan.utils.movement_analyzer import analyze_deforum_movement, generate_wan_motion_intensity_schedule
        from types import SimpleNamespace
        
        print("📐 Movement analysis requested - analyzing Deforum movement schedules...")
        
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
                
                # Check if we have any actual movement (not all default values)
                has_movement = (
                    anim_args.translation_x != "0:(0)" or
                    anim_args.translation_y != "0:(0)" or
                    anim_args.translation_z != "0:(0)" or
                    anim_args.rotation_3d_x != "0:(0)" or
                    anim_args.rotation_3d_y != "0:(0)" or
                    anim_args.rotation_3d_z != "0:(0)" or
                    anim_args.zoom != "0:(1.0)" or
                    anim_args.angle != "0:(0)"
                )
                
                if not has_movement:
                    print("📊 No movement detected in schedules, using demo values for illustration")
                    # Provide demo values to show how it works
                    anim_args.translation_x = "0:(0), 100:(50)"  # Right pan demo
                    anim_args.translation_z = "0:(0), 100:(30)"  # Forward dolly demo
                    anim_args.rotation_3d_y = "0:(0), 100:(15)" # Right yaw demo
                    anim_args.zoom = "0:(1.0), 100:(1.3)"       # Zoom in demo
                
            except Exception as e:
                print(f"⚠️ Could not access movement schedules: {e}")
                print("📊 Using demo movement data to show functionality")
                # Fallback to demo data
                anim_args.translation_x = "0:(0), 100:(50)"  # Right pan
                anim_args.translation_y = "0:(0)"
                anim_args.translation_z = "0:(0), 100:(30)"  # Forward dolly
                anim_args.rotation_3d_x = "0:(0)"
                anim_args.rotation_3d_y = "0:(0), 100:(15)"  # Right yaw
                anim_args.rotation_3d_z = "0:(0)"
                anim_args.zoom = "0:(1.0), 100:(1.3)"  # Zoom in
                anim_args.angle = "0:(0)"
                anim_args.max_frames = 100
        else:
            print("⚠️ No stored movement schedule references found")
            print("📊 Using demo movement data to show functionality")
            # Use demo data to show how it works
            anim_args.translation_x = "0:(0), 100:(50)"  # Right pan
            anim_args.translation_y = "0:(0)"
            anim_args.translation_z = "0:(0), 100:(30)"  # Forward dolly
            anim_args.rotation_3d_x = "0:(0)"
            anim_args.rotation_3d_y = "0:(0), 100:(15)"  # Right yaw
            anim_args.rotation_3d_z = "0:(0)"
            anim_args.zoom = "0:(1.0), 100:(1.3)"  # Zoom in
            anim_args.angle = "0:(0)"
            anim_args.max_frames = 100
        
        # Generate movement description for appending to prompts
        movement_desc, average_motion_strength = analyze_deforum_movement(
            anim_args=anim_args,
            sensitivity=movement_sensitivity,
            max_frames=anim_args.max_frames
        )
        
        # Generate Wan motion intensity schedule
        motion_intensity_schedule = generate_wan_motion_intensity_schedule(
            anim_args=anim_args,
            max_frames=anim_args.max_frames,
            sensitivity=movement_sensitivity
        )
        
        # Store movement description for enhance_prompts_handler to use
        enhance_prompts_handler._movement_description = movement_desc
        print(f"💾 Stored movement description for prompt enhancement: {movement_desc}")
        
        # Update wan_enhanced_prompts with movement descriptions if prompts exist
        if hasattr(enhance_prompts_handler, '_wan_enhanced_prompts_component'):
            try:
                import json
                current_prompts_value = enhance_prompts_handler._wan_enhanced_prompts_component.value
                
                if current_prompts_value and current_prompts_value.strip():
                    # Parse current prompts
                    try:
                        # Try JSON first
                        current_prompts = json.loads(current_prompts_value)
                    except json.JSONDecodeError:
                        # Try readable format
                        current_prompts = {}
                        for line in current_prompts_value.strip().split('\n'):
                            if ':' in line:
                                parts = line.split(':', 1)
                                frame_part = parts[0].strip()
                                prompt_part = parts[1].strip()
                                
                                if frame_part.lower().startswith('frame '):
                                    frame_num = frame_part[6:].strip()
                                else:
                                    frame_num = frame_part
                                
                                current_prompts[frame_num] = prompt_part
                    
                    # Add movement description to each prompt
                    updated_prompts = {}
                    for frame, prompt in current_prompts.items():
                        # Remove existing movement description if present
                        clean_prompt = prompt.split('. camera movement:')[0].split('. Camera movement:')[0]
                        # Add new movement description
                        updated_prompts[frame] = f"{clean_prompt}. {movement_desc}"
                    
                    # Convert back to JSON and update component
                    updated_json = json.dumps(updated_prompts, ensure_ascii=False, indent=2)
                    enhance_prompts_handler._wan_enhanced_prompts_component.value = updated_json
                    
                    print(f"✅ Updated {len(updated_prompts)} wan prompts with movement descriptions")
                    
            except Exception as e:
                print(f"⚠️ Could not update wan prompts: {e}")
        
        # Simple result text for UI display
        return f"""✅ Movement analysis complete!

Movement: "{movement_desc}"
Motion strength: {average_motion_strength:.2f}

✅ Wan prompts updated with movement descriptions.
Ready for enhancement or generation."""
        
    except Exception as e:
        error_msg = f"❌ Error in movement analysis: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg

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
            status_parts.append("2. 🎨 Click 'Enhance Prompts' for auto-download")
            status_parts.append("3. ⏳ Wait for download to complete")
        elif not is_loaded:
            status_parts.append("<br><strong style='color: #333;'>Ready to Use:</strong>")
            status_parts.append("🎨 Click 'Enhance Prompts' to load and use this model")
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
🎨 Click 'Enhance Prompts' to start using this model"""
        
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
            download_status.append("🎨 Ready to use - click 'Enhance Prompts' to start")
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
        import json
        import os
        
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
        import json
        import os
        
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
        import json
        import os
        
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