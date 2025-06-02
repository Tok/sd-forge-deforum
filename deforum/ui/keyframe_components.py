"""
Keyframe Components
Contains keyframe and scheduling UI components
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn, ToolButton
from .component_builders import create_gr_elem, create_row
from ..utils import emoji_utils
from ..config.defaults import get_gradio_html


def get_tab_keyframes(d, da, dloopArgs):
    """Create the Keyframes tab with motion and scheduling controls.
    
    Args:
        d: DeforumArgs instance
        da: DeforumAnimArgs instance 
        dloopArgs: LoopArgs instance
        
    Returns:
        Dict of component references
    """
    components = {}
    with gr.TabItem(f"{emoji_utils.key()} Keyframes"):
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
            components.update(_create_motion_tab(da, components))
            _create_shakify_tab(da)
            _create_noise_tab(da)
            _create_coherence_tab(da, d)
            _create_anti_blur_tab(da)
            _create_depth_warp_tab(da)
            
    return {k: v for k, v in {**locals(), **vars()}.items()}


def _create_motion_tab(da, components):
    """Create the Motion sub-tab."""
    motion_components = {}
    
    with gr.TabItem(f"{emoji_utils.bicycle()} Motion") as motion_tab:
        with FormColumn() as only_2d_motion_column:
            with FormRow(variant="compact"):
                zoom = create_gr_elem(da.zoom)
                reset_zoom_button = ToolButton(
                    elem_id='reset_zoom_btn', 
                    value=emoji_utils.refresh,
                    tooltip="Reset zoom to static."
                )
                components['zoom'] = zoom
                motion_components['zoom'] = zoom

                def reset_zoom_field():
                    return {zoom: gr.update(value='0:(1)', visible=True)}

                # Only set up click handler if zoom component is valid
                if zoom is not None and hasattr(zoom, '_id'):
                    reset_zoom_button.click(fn=reset_zoom_field, inputs=[], outputs=[zoom])
                else:
                    print("⚠️ Warning: Cannot set up zoom reset button - zoom component is invalid")
                
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
                reset_tr_z_button = ToolButton(
                    elem_id='reset_tr_z_btn', 
                    value=emoji_utils.refresh,
                    tooltip="Reset translation Z to static."
                )
                components['tr_z'] = translation_z
                motion_components['tr_z'] = translation_z

                def reset_tr_z_field():
                    return {translation_z: gr.update(value='0:(0)', visible=True)}

                # Only set up click handler if translation_z component is valid
                if translation_z is not None and hasattr(translation_z, '_id'):
                    reset_tr_z_button.click(fn=reset_tr_z_field, inputs=[], outputs=[translation_z])
                else:
                    print("⚠️ Warning: Cannot set up translation_z reset button - component is invalid")
                
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
            
    return motion_components


def _create_shakify_tab(da):
    """Create the Shakify sub-tab."""
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


def _create_noise_tab(da):
    """Create the Noise sub-tab."""
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


def _create_coherence_tab(da, d):
    """Create the Coherence sub-tab."""
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
            d, 'reroll_blank_frames', 'reroll_patience')


def _create_anti_blur_tab(da):
    """Create the Anti Blur sub-tab."""
    with gr.TabItem(f"{emoji_utils.broom()} Anti Blur", elem_id='anti_blur_accord') as anti_blur_tab:
        amount_schedule = create_row(da.amount_schedule)
        kernel_schedule = create_row(da.kernel_schedule)
        sigma_schedule = create_row(da.sigma_schedule)
        threshold_schedule = create_row(da.threshold_schedule)


def _create_depth_warp_tab(da):
    """Create the 3D Depth Warping & FOV sub-tab."""
    with gr.TabItem(f"{emoji_utils.hole()} 3D Depth Warping & FOV", elem_id='depth_warp_fov_tab') as depth_warp_fov_tab:
        # FIXME this should only be visible if animation mode is "3D".
        is_visible = True
        is_info_visible = is_visible
        
        depth_warp_msg_html = gr.HTML(
            value='Please switch to 3D animation mode to view this section.',
            elem_id='depth_warp_msg_html', 
            visible=is_info_visible
        )
        
        with FormRow(visible=is_visible) as depth_warp_row_1:
            use_depth_warping = create_gr_elem(da.use_depth_warping)
            # *the following html only shows when LeReS depth is selected*
            leres_license_msg = gr.HTML(
                value=get_gradio_html('leres'), 
                visible=False,
                elem_id='leres_license_msg'
            )
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


def create_keyframe_distribution_info_tab():
    """Create the keyframe distribution information tab."""
    create_row(gr.Markdown(f"""
        {emoji_utils.warn} Keyframe distribution uses an experimental render core with a slightly different feature set.
        Some features may not be supported and could cause errors or unexpected results if not disabled.
    """))
    
    _create_keyframe_info_accordion()
    _create_recommendations_accordion()
    _create_deforum_setup_accordion()
    _create_parseq_setup_accordion()


def _create_keyframe_info_accordion():
    """Create the keyframe distribution info accordion."""
    with gr.Accordion("Keyframe Distribution Info", open=False):
        gr.Markdown(f"""
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


def _create_recommendations_accordion():
    """Create the general recommendations accordion."""
    with gr.Accordion("General Recommendations & Warnings", open=False):
        gr.Markdown(f"""
            - Use with high FPS (e.g., 60) and high cadence (e.g., 15)
            - 'Keyframe_strength' should be lower than 'strength' (ignored when using Parseq)
            - {emoji_utils.warn} Not recommended with optical flow or hybrid settings
            - {emoji_utils.warn} Optical flow settings ~may~ will behave unexpectedly.
                - Turn off in tab "Keyframes", sub-tab "Coherence".
            - Prevent issues like dark-outs that add up over frames:
                - Set up regular low-strength diffusions by using enough keyframes
            - Balance strength values for optimal results
        """)


def _create_deforum_setup_accordion():
    """Create the Deforum setup recommendations accordion."""
    with gr.Accordion("Deforum Setup Recommendations", open=False):
        gr.Markdown(f"""
            - Set 'Keyframe strength' lower than 'Strength' to make sure keyframes get diffused with more steps
                - The higher the difference, the more keyframes become key compared to regular cadence frames. 
            - Force keyframe creation by duplicating the previous prompt with the desired frame number
                - This ensures diffusion with 'Keyframe strength' value
        """)


def _create_parseq_setup_accordion():
    """Create the Parseq setup recommendations accordion."""
    with gr.Accordion("Parseq Setup Recommendations", open=False):
        gr.Markdown(f"""
            - Deforum prompt keyframes are ignored
            - All frames with Parseq entries are treated as keyframes and will be diffused
            - 'Keyframe strength' is ignored; use Parseq for direct 'Strength' control
            - Create strength-dips at regular intervals:
                - Mark frames with 'Info' (e.g., "event")
                - Use formulas like: `if (f == info_match_last("event")) 0.25 else 0.75`
        """) 