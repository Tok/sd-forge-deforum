"""
Missing Components Patch
Contains placeholder components to prevent UI warnings about missing elements
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn
from .component_builders import create_gr_elem


def create_missing_component_placeholders():
    """Create placeholder components for missing UI elements to prevent warnings.
    
    Returns:
        Dict of placeholder component references
    """
    components = {}
    
    # Create hidden placeholders for missing components from the error log
    
    # Optical Flow components
    with gr.Row(visible=False) as optical_flow_cadence_row:
        components['optical_flow_cadence'] = gr.Dropdown(
            label="Optical Flow Cadence",
            choices=["None", "1", "2", "3", "4", "5"],
            value="None",
            visible=False
        )
    components['optical_flow_cadence_row'] = optical_flow_cadence_row
    
    with gr.Column(visible=False) as cadence_flow_factor_schedule_column:
        components['cadence_flow_factor_schedule'] = gr.Textbox(
            label="Cadence Flow Factor Schedule",
            value="0: (1)",
            visible=False
        )
    components['cadence_flow_factor_schedule_column'] = cadence_flow_factor_schedule_column
    
    components['optical_flow_redo_generation'] = gr.Dropdown(
        label="Optical Flow Redo Generation",
        choices=["None", "1", "2", "3"],
        value="None",
        visible=False
    )
    
    with gr.Column(visible=False) as redo_flow_factor_schedule_column:
        components['redo_flow_factor_schedule'] = gr.Textbox(
            label="Redo Flow Factor Schedule",
            value="0: (1)",
            visible=False
        )
    components['redo_flow_factor_schedule_column'] = redo_flow_factor_schedule_column
    
    components['diffusion_redo'] = gr.Number(
        label="Diffusion Redo",
        value=0,
        visible=False
    )
    
    # Motion columns
    with gr.Column(visible=False) as only_3d_motion_column:
        components['only_3d_motion_column'] = only_3d_motion_column
    
    with gr.Column(visible=False) as only_2d_motion_column:
        components['only_2d_motion_column'] = only_2d_motion_column
    
    with gr.Column(visible=False) as both_anim_mode_motion_params_column:
        components['both_anim_mode_motion_params_column'] = both_anim_mode_motion_params_column
    
    # Depth warp rows
    for i in range(1, 8):
        with gr.Row(visible=False) as depth_warp_row:
            components[f'depth_warp_row_{i}'] = depth_warp_row
    
    components['enable_perspective_flip'] = gr.Checkbox(
        label="Enable Perspective Flip",
        value=False,
        visible=False
    )
    
    components['depth_warp_msg_html'] = gr.HTML(
        value="Depth warp message placeholder",
        visible=False
    )
    
    # Color components
    components['color_force_grayscale'] = gr.Checkbox(
        label="Force Grayscale",
        value=False,
        visible=False
    )
    
    components['color_coherence'] = gr.Dropdown(
        label="Color Coherence",
        choices=["None", "Match Frame 0 HSV", "Match Frame 0 LAB", "Match Frame 0 RGB"],
        value="None",
        visible=False
    )
    
    with gr.Row(visible=False) as color_coherence_video_every_N_frames_row:
        components['color_coherence_video_every_N_frames'] = gr.Number(
            label="Color Coherence Video Every N Frames",
            value=1,
            visible=False
        )
    components['color_coherence_video_every_N_frames_row'] = color_coherence_video_every_N_frames_row
    
    with gr.Row(visible=False) as color_coherence_image_path_row:
        components['color_coherence_image_path'] = gr.Textbox(
            label="Color Coherence Image Path",
            value="",
            visible=False
        )
    components['color_coherence_image_path_row'] = color_coherence_image_path_row
    
    # Noise components
    with gr.Column(visible=False) as noise_tab_column:
        components['noise_type'] = gr.Dropdown(
            label="Noise Type",
            choices=["perlin", "uniform"],
            value="perlin",
            visible=False
        )
        
        with gr.Row(visible=False) as perlin_row:
            components['perlin_octaves'] = gr.Number(
                label="Perlin Octaves",
                value=4,
                visible=False
            )
            components['perlin_persistence'] = gr.Number(
                label="Perlin Persistence", 
                value=0.5,
                visible=False
            )
    components['noise_tab_column'] = noise_tab_column
    components['perlin_row'] = perlin_row
    
    with gr.Row(visible=False) as enable_per_f_row:
        components['enable_per_f'] = gr.Checkbox(
            label="Enable Per Frame",
            value=False,
            visible=False
        )
    components['enable_per_f_row'] = enable_per_f_row
    
    # Aspect ratio components
    components['aspect_ratio_use_old_formula'] = gr.Checkbox(
        label="Aspect Ratio Use Old Formula",
        value=False,
        visible=False
    )
    
    components['aspect_ratio_schedule'] = gr.Textbox(
        label="Aspect Ratio Schedule",
        value="0: (1.0)",
        visible=False
    )
    
    # Depth algorithm components
    components['depth_algorithm'] = gr.Dropdown(
        label="Depth Algorithm",
        choices=["Zoe", "MiDaS", "LeReS"],
        value="Zoe",
        visible=False
    )
    
    components['midas_weight'] = gr.Number(
        label="MiDaS Weight",
        value=0.3,
        visible=False
    )
    
    components['leres_license_msg'] = gr.HTML(
        value="LeReS license message placeholder",
        visible=False
    )
    
    # Video output components
    components['make_gif'] = gr.Checkbox(
        label="Make GIF",
        value=False,
        visible=False
    )
    
    components['skip_video_creation'] = gr.Checkbox(
        label="Skip Video Creation",
        value=False,
        visible=False
    )
    
    # NCNN upscaling components
    components['ncnn_upscale_model'] = gr.Dropdown(
        label="NCNN Upscale Model",
        choices=["None"],
        value="None",
        visible=False
    )
    
    components['ncnn_upscale_factor'] = gr.Dropdown(
        label="NCNN Upscale Factor",
        choices=["2x", "4x"],
        value="2x",
        visible=False
    )
    
    # Frame interpolation components
    with gr.Column(visible=False) as frame_interp_slow_mo_amount_column:
        components['frame_interp_slow_mo_amount'] = gr.Number(
            label="Frame Interpolation Slow Mo Amount",
            value=2,
            visible=False
        )
    components['frame_interp_slow_mo_amount_column'] = frame_interp_slow_mo_amount_column
    
    with gr.Row(visible=False) as frame_interp_amounts_row:
        components['frame_interp_x_amount'] = gr.Number(
            label="Frame Interpolation X Amount",
            value=2,
            visible=False
        )
    components['frame_interp_amounts_row'] = frame_interp_amounts_row
    
    return components
