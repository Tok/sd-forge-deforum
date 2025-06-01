"""
Input Components
Contains the main Run tab and basic input controls
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn, ToolButton
from .component_builders import create_gr_elem, create_row
from ..utils import emoji_utils


def get_tab_run(d, da):
    """Create the Run tab with basic generation controls.
    
    Args:
        d: DeforumArgs instance
        da: DeforumAnimArgs instance
        
    Returns:
        Dict of component references
    """
    with gr.TabItem(f"{emoji_utils.run()} Run"):  # RUN TAB
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
                with gr.Row():
                    override_settings_with_file = gr.Checkbox(
                        label="Enable batch mode", 
                        value=False, 
                        interactive=True,
                        elem_id='override_settings',
                        info="run from a list of setting .txt files. Upload them to the box on the right (visible when enabled)"
                    )
                    custom_settings_file = gr.File(
                        label="Setting files", 
                        interactive=True, 
                        file_count="multiple",
                        file_types=[".txt"], 
                        elem_id="custom_setting_file", 
                        visible=False
                    )
                    
            # RESUME ANIMATION ACCORD
            with gr.Tab('Resume Animation'):
                resume_from_timestring, resume_timestring = create_row(
                    da, 'resume_from_timestring', 'resume_timestring')
                    
    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_init(d, da, dp):
    """Create the Init tab with initialization controls.
    
    Args:
        d: DeforumArgs instance
        da: DeforumAnimArgs instance
        dp: DeforumParseqArgs instance
        
    Returns:
        Dict of component references
    """
    with gr.TabItem(f"{emoji_utils.microscope()} Init"):
        with FormRow():
            use_init = create_gr_elem(d.use_init)
            init_image = create_gr_elem(d.init_image)
        
        with FormRow():
            strength = create_gr_elem(d.strength)
            strength_0_no_init = create_gr_elem(d.strength_0_no_init)
            init_scale = create_gr_elem(d.init_scale)
            
        with FormRow():
            use_mask = create_gr_elem(d.use_mask)
            use_alpha_as_mask = create_gr_elem(d.use_alpha_as_mask)
            invert_mask = create_gr_elem(d.invert_mask)
            overlay_mask = create_gr_elem(d.overlay_mask)
            
        with FormRow():
            mask_file = create_gr_elem(d.mask_file)
            mask_brightness_adjust = create_gr_elem(d.mask_brightness_adjust)
            mask_contrast_adjust = create_gr_elem(d.mask_contrast_adjust)
            
        with FormRow():
            mask_overlay_blur = create_gr_elem(d.mask_overlay_blur)
            fill = create_gr_elem(d.fill)
            full_res_mask = create_gr_elem(d.full_res_mask)
            full_res_mask_padding = create_gr_elem(d.full_res_mask_padding)
            
        with FormRow():
            reroll_blank_frames = create_gr_elem(d.reroll_blank_frames)
            reroll_patience = create_gr_elem(d.reroll_patience)
        
        # Parseq section
        with gr.Accordion("Parseq", open=False):
            with FormRow():
                parseq_manifest = create_gr_elem(dp.parseq_manifest)
                parseq_use_deltas = create_gr_elem(dp.parseq_use_deltas)
                
            with FormRow():
                parseq_non_schedule_overrides = create_gr_elem(dp.parseq_non_schedule_overrides)
                
    return {k: v for k, v in {**locals(), **vars()}.items()} 