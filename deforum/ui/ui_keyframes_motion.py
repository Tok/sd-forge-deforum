"""
UI Keyframes Motion Module

Contains the Keyframes tab with motion controls, scheduling parameters,
distribution settings, and guided images functionality for Deforum.

Functions:
    - get_tab_keyframes: Creates the main Keyframes tab interface
    - Motion and animation parameter controls
    - Keyframe distribution and scheduling utilities
    - Guided images configuration
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn, ToolButton
from .ui_core_components import create_row, create_gr_elem
from .rendering.util import emoji_utils
from .defaults import get_gradio_html


def get_tab_keyframes(d, da, dloopArgs):
    """
    Create the Keyframes tab with motion controls and scheduling options.
    
    Args:
        d: DeforumArgs instance with basic parameters
        da: DeforumAnimArgs instance with animation parameters
        dloopArgs: Loop args for guided images
        
    Returns:
        dict: Dictionary containing all created components
    """
    components = {}
    
    with gr.TabItem(f"{emoji_utils.key()} Keyframes"):
        
        # Core Animation Settings
        with FormRow():
            with FormColumn(scale=2):
                components['animation_mode'] = create_gr_elem(da.animation_mode)
            with FormColumn(scale=1, min_width=180):
                components['border'] = create_gr_elem(da.border)
        
        components['diffusion_cadence'], components['max_frames'] = create_row(
            da, 'diffusion_cadence', 'max_frames'
        )
        
        # Guided Images Section
        guided_components = create_guided_images_section(dloopArgs)
        components.update(guided_components)
        
        # Scheduling Tabs
        schedule_components = create_scheduling_tabs(d, da)
        components.update(schedule_components)
        
        # Motion Controls
        motion_components = create_motion_controls(da)
        components.update(motion_components)
    
    return components


def create_guided_images_section(dloopArgs):
    """
    Create the Guided Images section with settings and schedules.
    
    Args:
        dloopArgs: Loop arguments for guided images
        
    Returns:
        dict: Dictionary of guided images components
    """
    components = {}
    
    with gr.Accordion('Guided Images', open=False, elem_id='guided_images_accord') as guided_images_accord:
        components['guided_images_accord'] = guided_images_accord
        
        # Information Section
        with gr.Accordion('*READ ME before you use this mode!*', open=False):
            gr.HTML(value=get_gradio_html('guided_imgs'))
        
        # Core Settings
        components['use_looper'] = create_row(dloopArgs.use_looper)
        components['init_images'] = create_row(dloopArgs.init_images)
        
        # Schedules Section
        with gr.Accordion('Guided images schedules', open=False):
            components['image_strength_schedule'] = create_row(dloopArgs.image_strength_schedule)
            components['image_keyframe_strength_schedule'] = create_row(dloopArgs.image_keyframe_strength_schedule)
            components['blendFactorMax'] = create_row(dloopArgs.blendFactorMax)
            components['blendFactorSlope'] = create_row(dloopArgs.blendFactorSlope)
            components['tweening_frames_schedule'] = create_row(dloopArgs.tweening_frames_schedule)
            components['color_correction_factor'] = create_row(dloopArgs.color_correction_factor)
    
    return components


def create_scheduling_tabs(d, da):
    """
    Create the scheduling tabs with various parameter schedules.
    
    Args:
        d: DeforumArgs instance
        da: DeforumAnimArgs instance
        
    Returns:
        dict: Dictionary of scheduling components
    """
    components = {}
    
    with gr.Tabs():
        
        # Distribution Tab
        with gr.TabItem(f"{emoji_utils.distribution()} Distribution"):
            components['keyframe_distribution'] = create_row(da.keyframe_distribution)
            create_keyframe_distribution_info_tab()
        
        # Strength Tab
        with gr.TabItem(f"{emoji_utils.strength()} Strength"):
            components['strength_schedule'] = create_row(da.strength_schedule)
            components['keyframe_strength_schedule'] = create_row(da.keyframe_strength_schedule)
        
        # CFG Tab
        with gr.TabItem(f"{emoji_utils.scale()} CFG"):
            components['cfg_scale_schedule'] = create_row(da.cfg_scale_schedule)
            components['distilled_cfg_scale_schedule'] = create_row(da.distilled_cfg_scale_schedule)
            components['enable_clipskip_scheduling'] = create_row(da.enable_clipskip_scheduling)
            components['clipskip_schedule'] = create_row(da.clipskip_schedule)
        
        # Seed & SubSeed Tab
        with gr.TabItem(f"{emoji_utils.seed()} Seed & SubSeed") as subseed_sch_tab:
            components['subseed_sch_tab'] = subseed_sch_tab
            
            # Seed behavior and iteration controls
            components['seed_behavior'] = create_row(d.seed_behavior)
            
            with FormRow() as seed_iter_N_row:
                components['seed_iter_N_row'] = seed_iter_N_row
                components['seed_iter_N'] = create_gr_elem(d.seed_iter_N)
            
            with FormRow(visible=False) as seed_schedule_row:
                components['seed_schedule_row'] = seed_schedule_row
                components['seed_schedule'] = create_gr_elem(da.seed_schedule)
            
            # SubSeed scheduling
            enable_subseed_scheduling, subseed_schedule, subseed_strength_schedule = create_row(
                da, 'enable_subseed_scheduling', 'subseed_schedule', 'subseed_strength_schedule'
            )
            components['enable_subseed_scheduling'] = enable_subseed_scheduling
            components['subseed_schedule'] = subseed_schedule
            components['subseed_strength_schedule'] = subseed_strength_schedule
            
            # Seed resize settings
            seed_resize_from_w, seed_resize_from_h = create_row(
                d, 'seed_resize_from_w', 'seed_resize_from_h'
            )
            components['seed_resize_from_w'] = seed_resize_from_w
            components['seed_resize_from_h'] = seed_resize_from_h
        
        # Additional Scheduling Tabs
        steps_components = create_steps_scheduling_tab(da)
        components.update(steps_components)
        
        sampler_components = create_sampler_scheduling_tab(da)
        components.update(sampler_components)
        
        scheduler_components = create_scheduler_scheduling_tab(da)
        components.update(scheduler_components)
        
        checkpoint_components = create_checkpoint_scheduling_tab(da)
        components.update(checkpoint_components)
    
    return components


def create_steps_scheduling_tab(da):
    """Create the steps scheduling tab."""
    components = {}
    
    with gr.TabItem('Step'):
        components['enable_steps_scheduling'] = create_row(da.enable_steps_scheduling)
        components['steps_schedule'] = create_row(da.steps_schedule)
    
    return components


def create_sampler_scheduling_tab(da):
    """Create the sampler scheduling tab."""
    components = {}
    
    with gr.TabItem('Sampler'):
        components['enable_sampler_scheduling'] = create_row(da.enable_sampler_scheduling)
        components['sampler_schedule'] = create_row(da.sampler_schedule)
    
    return components


def create_scheduler_scheduling_tab(da):
    """Create the scheduler scheduling tab."""
    components = {}
    
    with gr.TabItem('Scheduler'):
        components['enable_scheduler_scheduling'] = create_row(da.enable_scheduler_scheduling)
        components['scheduler_schedule'] = create_row(da.scheduler_schedule)
    
    return components


def create_checkpoint_scheduling_tab(da):
    """Create the checkpoint scheduling tab."""
    components = {}
    
    with gr.TabItem('Checkpoint'):
        components['enable_checkpoint_scheduling'] = create_row(da.enable_checkpoint_scheduling)
        components['checkpoint_schedule'] = create_row(da.checkpoint_schedule)
    
    return components


def create_motion_controls(da):
    """
    Create the motion controls section with movement parameters.
    
    Args:
        da: DeforumAnimArgs instance
        
    Returns:
        dict: Dictionary of motion components
    """
    components = {}
    
    with gr.Tabs(elem_id='motion_noise_etc'):
        
        # Main Motion Tab
        with gr.TabItem(f"{emoji_utils.bicycle()} Motion") as motion_tab:
            components['motion_tab'] = motion_tab
            
            # 2D Motion Parameters
            motion_2d_components = create_2d_motion_controls(da)
            components.update(motion_2d_components)
            
            # Both Animation Mode Parameters
            both_anim_components = create_both_anim_motion_controls(da)
            components.update(both_anim_components)
            
            # 3D Motion Parameters
            motion_3d_components = create_3d_motion_controls(da)
            components.update(motion_3d_components)
            
            # Perspective Flip Controls
            perspective_components = create_perspective_flip_controls(da)
            components.update(perspective_components)
        
        # Shakify Tab
        shakify_components = create_shakify_tab(da)
        components.update(shakify_components)
        
        # Noise Tab
        noise_components = create_noise_tab(da)
        components.update(noise_components)
        
        # Coherence Tab
        coherence_components = create_coherence_tab(da)
        components.update(coherence_components)
        
        # Anti-Blur Tab
        anti_blur_components = create_anti_blur_tab(da)
        components.update(anti_blur_components)
        
        # 3D Depth Warping & FOV Tab
        depth_warp_components = create_depth_warp_tab(da)
        components.update(depth_warp_components)
    
    return components


def create_2d_motion_controls(da):
    """Create 2D motion control components."""
    components = {}
    
    with FormColumn() as only_2d_motion_column:
        components['only_2d_motion_column'] = only_2d_motion_column
        
        # Zoom with reset button
        with FormRow(variant="compact"):
            zoom = create_gr_elem(da.zoom)
            reset_zoom_button = ToolButton(
                elem_id='reset_zoom_btn', 
                value=emoji_utils.refresh,
                tooltip="Reset zoom to static."
            )
            components['zoom'] = zoom
            components['reset_zoom_button'] = reset_zoom_button
            
            def reset_zoom_field():
                return {zoom: gr.update(value='0:(1)', visible=True)}
            
            reset_zoom_button.click(fn=reset_zoom_field, inputs=[], outputs=[zoom])
        
        # Other 2D motion parameters
        components['angle'] = create_row(da.angle)
        components['transform_center_x'] = create_row(da.transform_center_x)
        components['transform_center_y'] = create_row(da.transform_center_y)
    
    return components


def create_both_anim_motion_controls(da):
    """Create motion controls used in both 2D and 3D modes."""
    components = {}
    
    with FormColumn() as both_anim_mode_motion_params_column:
        components['both_anim_mode_motion_params_column'] = both_anim_mode_motion_params_column
        components['translation_x'] = create_row(da.translation_x)
        components['translation_y'] = create_row(da.translation_y)
    
    return components


def create_3d_motion_controls(da):
    """Create 3D motion control components."""
    components = {}
    
    is_3d_motion_column_visible = True  # FIXME: init, overridden because default is 3D
    
    with FormColumn(visible=is_3d_motion_column_visible) as only_3d_motion_column:
        components['only_3d_motion_column'] = only_3d_motion_column
        
        # Translation Z with reset button
        with FormRow():
            translation_z = create_gr_elem(da.translation_z)
            reset_tr_z_button = ToolButton(
                elem_id='reset_tr_z_btn', 
                value=emoji_utils.refresh,
                tooltip="Reset translation Z to static."
            )
            components['translation_z'] = translation_z
            components['reset_tr_z_button'] = reset_tr_z_button
            
            def reset_tr_z_field():
                return {translation_z: gr.update(value='0:(0)', visible=True)}
            
            reset_tr_z_button.click(fn=reset_tr_z_field, inputs=[], outputs=[translation_z])
        
        # 3D Rotation parameters
        components['rotation_3d_x'] = create_row(da.rotation_3d_x)
        components['rotation_3d_y'] = create_row(da.rotation_3d_y)
        components['rotation_3d_z'] = create_row(da.rotation_3d_z)
    
    return components


def create_perspective_flip_controls(da):
    """Create perspective flip control components."""
    components = {}
    
    # Enable perspective flip
    with FormRow() as enable_per_f_row:
        components['enable_per_f_row'] = enable_per_f_row
        components['enable_perspective_flip'] = create_gr_elem(da.enable_perspective_flip)
    
    # Perspective flip parameters (initially hidden)
    with FormRow(visible=False) as per_f_th_row:
        components['per_f_th_row'] = per_f_th_row
        components['perspective_flip_theta'] = create_gr_elem(da.perspective_flip_theta)
    
    with FormRow(visible=False) as per_f_ph_row:
        components['per_f_ph_row'] = per_f_ph_row
        components['perspective_flip_phi'] = create_gr_elem(da.perspective_flip_phi)
    
    with FormRow(visible=False) as per_f_ga_row:
        components['per_f_ga_row'] = per_f_ga_row
        components['perspective_flip_gamma'] = create_gr_elem(da.perspective_flip_gamma)
    
    with FormRow(visible=False) as per_f_f_row:
        components['per_f_f_row'] = per_f_f_row
        components['perspective_flip_fv'] = create_gr_elem(da.perspective_flip_fv)
    
    return components


def create_shakify_tab(da):
    """Create the Shakify camera shake tab."""
    components = {}
    
    with gr.TabItem(f"{emoji_utils.video_camera()} Shakify"):
        with FormColumn(min_width=220):
            create_row(gr.Markdown("""
                Integrate dynamic camera shake effects into your renders with data sourced from EatTheFutures
                'Camera Shakify' Blender plugin. This feature enhances the realism of your animations
                by simulating natural camera movements, adding a layer of depth and engagement to your visuals.
            """))
            
            components['shake_name'] = create_row(da.shake_name)
            components['shake_intensity'] = create_row(da.shake_intensity)
            components['shake_speed'] = create_row(da.shake_speed)
    
    return components


def create_noise_tab(da):
    """Create the noise settings tab."""
    components = {}
    
    with gr.TabItem(f"{emoji_utils.wave()} Noise"):
        with FormColumn() as noise_tab_column:
            components['noise_tab_column'] = noise_tab_column
            
            components['noise_type'] = create_row(da.noise_type)
            components['noise_schedule'] = create_row(da.noise_schedule)
            
            with FormRow() as perlin_row:
                components['perlin_row'] = perlin_row
                with FormColumn(min_width=220):
                    components['perlin_octaves'] = create_gr_elem(da.perlin_octaves)
                with FormColumn(min_width=220):
                    components['perlin_persistence'] = create_gr_elem(da.perlin_persistence)
                    # Hidden parameters
                    components['perlin_w'] = create_gr_elem(da.perlin_w)
                    components['perlin_h'] = create_gr_elem(da.perlin_h)
            
            components['enable_noise_multiplier_scheduling'] = create_row(da.enable_noise_multiplier_scheduling)
            components['noise_multiplier_schedule'] = create_row(da.noise_multiplier_schedule)
    
    return components


def create_coherence_tab(da):
    """Create the color coherence tab."""
    components = {}
    
    with gr.TabItem(f"{emoji_utils.palette()} Coherence", open=False) as coherence_accord:
        components['coherence_accord'] = coherence_accord
        
        # Color coherence settings
        color_coherence, color_force_grayscale = create_row(
            da, 'color_coherence', 'color_force_grayscale'
        )
        components['color_coherence'] = color_coherence
        components['color_force_grayscale'] = color_force_grayscale
        
        components['legacy_colormatch'] = create_row(da.legacy_colormatch)
        
        # Hidden color coherence parameters
        with FormRow(visible=False) as color_coherence_image_path_row:
            components['color_coherence_image_path_row'] = color_coherence_image_path_row
            components['color_coherence_image_path'] = create_gr_elem(da.color_coherence_image_path)
        
        with FormRow(visible=False) as color_coherence_video_every_N_frames_row:
            components['color_coherence_video_every_N_frames_row'] = color_coherence_video_every_N_frames_row
            components['color_coherence_video_every_N_frames'] = create_gr_elem(da.color_coherence_video_every_N_frames)
        
        # Optical flow settings
        optical_flow_components = create_optical_flow_controls(da)
        components.update(optical_flow_components)
        
        # Additional coherence controls
        contrast_components = create_contrast_controls(da)
        components.update(contrast_components)
        
        # Blank frame handling
        reroll_components = create_reroll_controls(da)
        components.update(reroll_components)
    
    return components


def create_optical_flow_controls(da):
    """Create optical flow control components."""
    components = {}
    
    with FormRow() as optical_flow_cadence_row:
        components['optical_flow_cadence_row'] = optical_flow_cadence_row
        with FormColumn(min_width=220) as optical_flow_cadence_column:
            components['optical_flow_cadence_column'] = optical_flow_cadence_column
            components['optical_flow_cadence'] = create_gr_elem(da.optical_flow_cadence)
        with FormColumn(min_width=220, visible=False) as cadence_flow_factor_schedule_column:
            components['cadence_flow_factor_schedule_column'] = cadence_flow_factor_schedule_column
            components['cadence_flow_factor_schedule'] = create_gr_elem(da.cadence_flow_factor_schedule)
    
    with FormRow():
        with FormColumn(min_width=220):
            components['optical_flow_redo_generation'] = create_gr_elem(da.optical_flow_redo_generation)
        with FormColumn(min_width=220, visible=False) as redo_flow_factor_schedule_column:
            components['redo_flow_factor_schedule_column'] = redo_flow_factor_schedule_column
            components['redo_flow_factor_schedule'] = create_gr_elem(da.redo_flow_factor_schedule)
    
    return components


def create_contrast_controls(da):
    """Create contrast and diffusion redo controls."""
    components = {}
    
    with FormRow():
        components['contrast_schedule'] = gr.Textbox(
            label="Contrast schedule", 
            lines=1, 
            value=da.contrast_schedule, 
            interactive=True,
            info="adjusts the overall contrast per frame [neutral at 1.0, recommended to *not* play with this param]"
        )
        
        components['diffusion_redo'] = gr.Slider(
            label="Redo generation", 
            minimum=0, 
            maximum=50,
            step=1, 
            value=da.diffusion_redo, 
            interactive=True,
            info="this option renders N times before the final render. it is suggested to lower your steps if you up your redo. seed is randomized during redo generations and restored afterwards"
        )
    
    return components


def create_reroll_controls(da):
    """Create reroll controls for blank frames."""
    components = {}
    
    # Import from the parent module where these are defined
    from .args import DeforumArgs
    d = DeforumArgs()  # Create instance to access these parameters
    
    reroll_blank_frames, reroll_patience = create_row(
        d, 'reroll_blank_frames', 'reroll_patience'
    )
    components['reroll_blank_frames'] = reroll_blank_frames
    components['reroll_patience'] = reroll_patience
    
    return components


def create_anti_blur_tab(da):
    """Create the anti-blur settings tab."""
    components = {}
    
    with gr.TabItem(f"{emoji_utils.broom()} Anti Blur", elem_id='anti_blur_accord') as anti_blur_tab:
        components['anti_blur_tab'] = anti_blur_tab
        
        components['amount_schedule'] = create_row(da.amount_schedule)
        components['kernel_schedule'] = create_row(da.kernel_schedule)
        components['sigma_schedule'] = create_row(da.sigma_schedule)
        components['threshold_schedule'] = create_row(da.threshold_schedule)
    
    return components


def create_depth_warp_tab(da):
    """Create the 3D depth warping and FOV tab."""
    components = {}
    
    with gr.TabItem(f"{emoji_utils.hole()} 3D Depth Warping & FOV", elem_id='depth_warp_fov_tab') as depth_warp_fov_tab:
        components['depth_warp_fov_tab'] = depth_warp_fov_tab
        
        # FIXME: this should only be visible if animation mode is "3D"
        is_visible = True
        is_info_visible = is_visible
        
        components['depth_warp_msg_html'] = gr.HTML(
            value='Please switch to 3D animation mode to view this section.',
            elem_id='depth_warp_msg_html', 
            visible=is_info_visible
        )
        
        # Depth warping controls
        depth_controls = create_depth_warping_controls(da, is_visible)
        components.update(depth_controls)
        
        # Extended depth settings
        extended_controls = create_extended_depth_controls(da, is_visible)
        components.update(extended_controls)
        
        # FOV and near/far controls
        fov_controls = create_fov_controls(da, is_visible)
        components.update(fov_controls)
    
    return components


def create_depth_warping_controls(da, is_visible):
    """Create basic depth warping controls."""
    components = {}
    
    with FormRow(visible=is_visible) as depth_warp_row_1:
        components['depth_warp_row_1'] = depth_warp_row_1
        components['use_depth_warping'] = create_gr_elem(da.use_depth_warping)
        
        # LeReS license message (shown when LeReS depth algorithm is selected)
        components['leres_license_msg'] = gr.HTML(
            value=get_gradio_html('leres'), 
            visible=False,
            elem_id='leres_license_msg'
        )
        
        components['depth_algorithm'] = create_gr_elem(da.depth_algorithm)
        components['midas_weight'] = create_gr_elem(da.midas_weight)
    
    with FormRow(visible=is_visible) as depth_warp_row_2:
        components['depth_warp_row_2'] = depth_warp_row_2
        components['padding_mode'] = create_gr_elem(da.padding_mode)
        components['sampling_mode'] = create_gr_elem(da.sampling_mode)
    
    return components


def create_extended_depth_controls(da, is_visible):
    """Create extended depth warping settings."""
    components = {}
    
    with FormRow(visible=is_visible):
        with gr.Accordion('Extended Depth Warp Settings', open=False):
            with FormRow() as depth_warp_row_3:
                components['depth_warp_row_3'] = depth_warp_row_3
                components['aspect_ratio_use_old_formula'] = create_gr_elem(da.aspect_ratio_use_old_formula)
            
            with FormRow() as depth_warp_row_4:
                components['depth_warp_row_4'] = depth_warp_row_4
                components['aspect_ratio_schedule'] = create_gr_elem(da.aspect_ratio_schedule)
    
    return components


def create_fov_controls(da, is_visible):
    """Create FOV and near/far controls."""
    components = {}
    
    with FormRow(visible=is_visible):
        with FormRow() as depth_warp_row_5:
            components['depth_warp_row_5'] = depth_warp_row_5
            components['fov_schedule'] = create_gr_elem(da.fov_schedule)
        
        with FormRow() as depth_warp_row_6:
            components['depth_warp_row_6'] = depth_warp_row_6
            components['near_schedule'] = create_gr_elem(da.near_schedule)
        
        with FormRow() as depth_warp_row_7:
            components['depth_warp_row_7'] = depth_warp_row_7
            components['far_schedule'] = create_gr_elem(da.far_schedule)
    
    return components


def create_keyframe_distribution_info_tab():
    """Create the keyframe distribution information tab."""
    # This function is referenced in the original code but not defined
    # Adding a placeholder implementation
    with gr.Accordion("ℹ️ Keyframe Distribution Info", open=False):
        gr.Markdown("""
        **Keyframe Distribution Modes:**
        
        - **Off**: Traditional Deforum cadence-based generation
        - **Keyframes Only**: Generate only at prompt keyframes
        - **Redistributed**: Intelligent redistribution for better quality
        
        This setting controls the experimental render core behavior.
        """)


def setup_keyframes_motion_events(components):
    """
    Set up event handlers for keyframes and motion components.
    
    Args:
        components (dict): Dictionary of UI components
    """
    # Seed behavior visibility toggle
    if 'seed_behavior' in components and 'seed_schedule_row' in components:
        def toggle_seed_schedule(seed_behavior):
            return gr.update(visible=(seed_behavior == 'schedule'))
        
        components['seed_behavior'].change(
            fn=toggle_seed_schedule,
            inputs=[components['seed_behavior']],
            outputs=[components['seed_schedule_row']]
        )
    
    # Perspective flip visibility toggles
    if 'enable_perspective_flip' in components:
        def toggle_perspective_flip(enabled):
            return (
                gr.update(visible=enabled),
                gr.update(visible=enabled),
                gr.update(visible=enabled),
                gr.update(visible=enabled)
            )
        
        perspective_outputs = [
            components.get('per_f_th_row'),
            components.get('per_f_ph_row'),
            components.get('per_f_ga_row'),
            components.get('per_f_f_row')
        ]
        
        if all(output is not None for output in perspective_outputs):
            components['enable_perspective_flip'].change(
                fn=toggle_perspective_flip,
                inputs=[components['enable_perspective_flip']],
                outputs=perspective_outputs
            )


def get_keyframes_motion_info():
    """Get informational HTML for keyframes and motion."""
    return """
    <div style="padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px 0;">
        <h4>🎯 Keyframes & Motion Guide</h4>
        <ul>
            <li><strong>Animation Mode:</strong> Choose between 2D, 3D, or Video modes</li>
            <li><strong>Keyframe Distribution:</strong> Controls the experimental render core</li>
            <li><strong>Motion Parameters:</strong> Set camera movement and animation paths</li>
            <li><strong>Scheduling:</strong> Create dynamic parameter changes over time</li>
            <li><strong>Shakify:</strong> Add realistic camera shake effects</li>
        </ul>
    </div>
    """ 