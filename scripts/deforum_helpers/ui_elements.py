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
            with gr.Row(variant='compact') as pix2pix_img_cfg_scale_row:
                pix2pix_img_cfg_scale_schedule = create_gr_elem(da.pix2pix_img_cfg_scale_schedule)
                pix2pix_img_distilled_cfg_scale_schedule = create_gr_elem(da.pix2pix_img_distilled_cfg_scale_schedule)
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
            with gr.TabItem(f"{emoji_utils.seed()} Seed"):
                seed_behavior = create_row(d.seed_behavior)
                with FormRow() as seed_iter_N_row:
                    seed_iter_N = create_row(d.seed_iter_N)
                with FormRow(visible=False) as seed_schedule_row:
                    seed_schedule = create_gr_elem(da.seed_schedule)
            with gr.TabItem('SubSeed', open=False) as subseed_sch_tab:
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
                with FormColumn(visible=False) as only_3d_motion_column:
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
            with gr.TabItem(f"{emoji_utils.hole()} Depth Warping & FOV", elem_id='depth_warp_fov_tab') \
                    as depth_warp_fov_tab:
                # this html only shows when not in 2d/3d mode
                depth_warp_msg_html = gr.HTML(value='Please switch to 3D animation mode to view this section.',
                                              elem_id='depth_warp_msg_html')
                with FormRow(visible=False) as depth_warp_row_1:
                    use_depth_warping = create_gr_elem(da.use_depth_warping)
                    # *the following html only shows when LeReS depth is selected*
                    leres_license_msg = gr.HTML(value=get_gradio_html('leres'), visible=False,
                                                elem_id='leres_license_msg')
                    depth_algorithm = create_gr_elem(da.depth_algorithm)
                    midas_weight = create_gr_elem(da.midas_weight)
                with FormRow(visible=False) as depth_warp_row_2:
                    padding_mode = create_gr_elem(da.padding_mode)
                    sampling_mode = create_gr_elem(da.sampling_mode)
                with FormRow(visible=False) as depth_warp_row_3:
                    aspect_ratio_use_old_formula = create_gr_elem(da.aspect_ratio_use_old_formula)
                with FormRow(visible=False) as depth_warp_row_4:
                    aspect_ratio_schedule = create_gr_elem(da.aspect_ratio_schedule)
                with FormRow(visible=False) as depth_warp_row_5:
                    fov_schedule = create_gr_elem(da.fov_schedule)
                with FormRow(visible=False) as depth_warp_row_6:
                    near_schedule = create_gr_elem(da.near_schedule)
                with FormRow(visible=False) as depth_warp_row_7:
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
                       info="full prompts list in a JSON format.  value on left side is the frame number"))
        animation_prompts_positive = create_row(
            gr.Textbox(label="Prompts positive", lines=1, interactive=True,
                       placeholder="words in here will be added to the start of all positive prompts"))
        animation_prompts_negative = create_row(
            gr.Textbox(label="Prompts negative", value="nsfw, nude", lines=1, interactive=True,
                       placeholder="words in here will be added to the end of all negative prompts"))
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
        {emoji_utils.warn} If Keyframe distribution is activated, the rendering will be processed in a
            different, more experimental render core. **Some features are currently not- or not fully supported in the
            new render core** and could lead to an error or to undesired results if not turned off."""))
    create_accordion_md_row("Keyframe Distribution Info", f"""
        ### Purpose & Description
        * Keyframe distribution ensures that every frame with an entry in the Prompts-
            or in the Parseq-table is diffused, ideally at a lower Strength.
        * Since the experimental render core allows for high or no cadence,
            the generation can be performed **much faster** compared to a traditional low cadence setup.
        * Resulting videos tend to be **less jittery** at high or no cadence, but may introduce other visual artifacts
            like 'depth smear' when combined with fast movement.
            * {emoji_utils.bulb()} Most negative effects can still be mitigated,
                by inserting lower strength frames in at regular intervals.

        ### Distribution Modes
        * **Off**: Key frames are not redistributed. Cadence settings are fully respected.
            Process runs on stable render core with better support for complicated setups.
        * **Keyframes Only**: Only frames with an entry in Prompts or in the Parseq table are diffused.
            Actual cadence settings are ignored and all non-keyframes are handled as cadence frames.
        * **Additive**: Is using cadence but adds keyframes. Takes more time to generate, 
            but may help stabilizing the frames by doing diffusions in regular intervals.
            Parseq Recommendation: Make high cadence setup and mark your frames with an 'Info' like "event", then
            in 'i_strength' dip deeper on key frames. E.g.: 'if (f == info_match_last("event")) 0.25 else 0.50'.
        * **Redistributed**: Calculates uniform cadence distribution from cadence, 
            but rearranges some keyframes to preserve proper keyframe synchronization at high cadence (e.g. '30').
            Helps to prevent diffusion frames from being too close and should be slightly faster than additive,
            but is also less predictable because some cadence frames are moved to match keyframes.
        """)
    create_accordion_md_row("General Recommendations & Warnings", f"""
        * Keyframe Distribution modes may most effectively be used at **high FPS** and with **high cadence**.
            * Try FPS "60" and cadence "20".
        * The diffusion process is either using 'strength' or 'keyframe_strength', depending on frame type.
            * 'keyframe_strength' should be lower than 'strength', but is ignored when using Parseq.
        * {emoji_utils.warn} It's currently not recommended to try and use keyframe distribution together
            with optical flow or with hybrid. Better keep this turned off.
        * {emoji_utils.warn} Optical Flow related settings may not behave as expected and are recommended
            to be turned off when keyframe distribution is used (see tab "Keyframes", sub-tab "Coherence").
        * Avoid Dark Out: High cadence generations may have a tendency to dark out over time.
            Make sure to still setup some diffusions with low strength at regular intervals.
            Setting "Sampling mode" to "nearest" in "Depth Warping & FOW" can help a great deal against dark-outs.
        * Avoid Depth Smear: If you get 'depth smear', try to calculate and set the correct 
            Aspect Ratio Schedule. Eg. "0: (1.777)" for 16:9 landscape or "0: (0.5625)" for 9:16 portrait.
            It doesn't solve the problem, but may help mitigating it.
            * Help clearing the canvas from time to time by diffusing some frames at low strength.
            * A render at 0 keyframe strength could in theory solve practically all visual problems by resetting
                the context, however that would is generally undesirable because it results in a cut in 
                the resulting clip. Finding an ideal balance for strength values is key to a good setup.
                * The optimal values highly depend on other variables such as cadence or calculated pseudo cadence.
    """)
    create_accordion_md_row("Deforum Setup Recommendations", f"""
        * Set 'Keyframe strength' somewhat lower than 'Strength' so the keyframes can really be key.
        * When using keyframe distributions, you can force keyframe creation at a specific frame by duplicating 
            the previous prompt and give it the desired frame number. It will make sure that the frame is diffused with
            the value defined in 'Keyframe strength' (ideally at higher denoise than cadence frames).
    """)
    create_accordion_md_row("Parseq Setup Recommendations", f"""
        * Deforum prompt keyframes are ignored, instead all frames with an entry in Parseq are handled as keyframes.
        * If Parseq is active, 'Keyframe strength' is also ignored, because you can provide the value for regular
            'Strength' directly from Parseq, where you have more options to control the value directly, 
            so there's no need for a secondary value.
            * However, it's still recommended to create a similar setup with strength-dips at regular intervals.
            * E.g: mark your all frames with an 'Info' like "event", then in 'i_strength' dip deeper on key frames 
                with a formula like "if (f == info_match_last("event")) 0.25 else 0.50" (tune values as needed).
    """)


def create_accordion_md_row(name, markdown, is_open=False):
    with FormRow():
        with gr.Accordion(name, open=is_open):
            gr.Markdown(markdown)
