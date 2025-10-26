"""3D Depth Warping & FOV tab for Deforum UI.

Configure depth estimation models, optical flow (RAFT), Flux ControlNet,
and field-of-view settings for 3D animation mode.
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.ui.builders import create_gr_elem, create_row


def get_tab_depth_warping(da, skip_tabitem=False):
    """Create the 3D Depth Warping & FOV tab.

    Args:
        da: DeforumAnimArgs namespace
        skip_tabitem: If True, don't create TabItem wrapper (default: False)

    Returns:
        dict: Component dictionary for event binding
    """
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

    with gr.Accordion("🌊 Optical Flow / Cadence", open=False):
        gr.Markdown("""
        **Optical flow** estimates motion between frames for smooth in-between (cadence) frames.
        Enable RAFT to generate only keyframes and use motion estimation for tweens (10x speedup).

        ⚠️ **WARNING:** Can produce "smear-core" artifacts with many cadence frames.
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

    with gr.Accordion("🌐 Flux ControlNet", open=False):
        gr.Markdown("""
        **Flux ControlNet** adds structural control to keyframe generation using:
        - **Canny edges** from previous frame (preserves shapes and lines)
        - **Depth maps** from Depth-Anything V2 (preserves 3D structure)

        ⚠️ **Only applies to keyframes** (not tween frames). Requires Flux model.
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
