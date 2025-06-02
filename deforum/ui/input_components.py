"""
Input Components - FIXED VERSION
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
    components = {}
    
    with gr.TabItem(f"{emoji_utils.run()} Run"):  # RUN TAB
        components['motion_preview_mode'] = create_gr_elem(d.motion_preview_mode)
        
        with FormRow():
            components['sampler'] = create_gr_elem(d.sampler)
            components['scheduler'] = create_gr_elem(d.scheduler)
            components['steps'] = create_gr_elem(d.steps)
            
        with FormRow():
            components['W'] = create_gr_elem(d.W)
            components['H'] = create_gr_elem(d.H)
            
        with FormRow():
            components['seed'] = create_gr_elem(d.seed)
            components['batch_name'] = create_gr_elem(d.batch_name)
        
        with FormRow():
            components['restore_faces'] = create_gr_elem(d.restore_faces)
            components['tiling'] = create_gr_elem(d.tiling)
            components['enable_ddim_eta_scheduling'] = create_gr_elem(da.enable_ddim_eta_scheduling)
            components['enable_ancestral_eta_scheduling'] = create_gr_elem(da.enable_ancestral_eta_scheduling)
            
        with gr.Row(variant='compact') as eta_sch_row:
            components['ddim_eta_schedule'] = create_gr_elem(da.ddim_eta_schedule)
            components['ancestral_eta_schedule'] = create_gr_elem(da.ancestral_eta_schedule)
        components['eta_sch_row'] = eta_sch_row

        # BATCH MODE AND RESUME SECTION - CRITICAL FIX
        with gr.Accordion('Batch Mode, Resume and more', open=False):
            with gr.Tab('Batch Mode/ run from setting files'):
                with gr.Row():
                    components['override_settings_with_file'] = gr.Checkbox(
                        label="Enable batch mode", 
                        value=False, 
                        interactive=True,
                        elem_id='override_settings',
                        info="run from a list of setting .txt files. Upload them to the box on the right (visible when enabled)"
                    )
                
                with gr.Row(visible=False) as custom_settings_file_row:
                    components['custom_settings_file'] = gr.File(
                        label="Setting files", 
                        interactive=True, 
                        file_count="multiple",
                        file_types=[".txt"], 
                        elem_id="custom_setting_file"
                    )
                components['custom_settings_file_row'] = custom_settings_file_row
                
                # Connect visibility toggle - safely
                def toggle_file_upload_visibility(enabled):
                    return gr.update(visible=enabled)
                
                if (components['override_settings_with_file'] is not None and 
                    hasattr(components['override_settings_with_file'], '_id') and
                    components['custom_settings_file_row'] is not None and 
                    hasattr(components['custom_settings_file_row'], '_id')):
                    try:
                        components['override_settings_with_file'].change(
                            fn=toggle_file_upload_visibility,
                            inputs=[components['override_settings_with_file']],
                            outputs=[components['custom_settings_file_row']]
                        )
                    except Exception as e:
                        print(f"⚠️ Failed to connect batch mode visibility toggle: {e}")
                    
            # RESUME ANIMATION TAB
            with gr.Tab('Resume Animation'):
                with FormRow():
                    components['resume_from_timestring'] = create_gr_elem(da.resume_from_timestring)
                    components['resume_timestring'] = create_gr_elem(da.resume_timestring)
                    
    return components


def get_tab_init(d, da, dp):
    """Create the Init tab with initialization controls.
    
    Args:
        d: DeforumArgs instance
        da: DeforumAnimArgs instance
        dp: DeforumParseqArgs instance
        
    Returns:
        Dict of component references
    """
    components = {}
    
    with gr.TabItem(f"{emoji_utils.microscope()} Init"):
        
        # INITIALIZATION IMAGE SETTINGS
        with gr.Accordion("🖼️ Initialization Image", open=True):
            with FormRow():
                components['use_init'] = create_gr_elem(d.use_init)
                components['init_image'] = create_gr_elem(d.init_image)
            
            with FormRow():
                components['strength'] = create_gr_elem(d.strength)
                components['strength_0_no_init'] = create_gr_elem(d.strength_0_no_init)
                components['init_scale'] = create_gr_elem(d.init_scale)
        
        # MASK SETTINGS
        with gr.Accordion("🎭 Mask Settings", open=False):
            with FormRow():
                components['use_mask'] = create_gr_elem(d.use_mask)
                components['use_alpha_as_mask'] = create_gr_elem(d.use_alpha_as_mask)
                components['invert_mask'] = create_gr_elem(d.invert_mask)
                components['overlay_mask'] = create_gr_elem(d.overlay_mask)
                
            with FormRow():
                components['mask_file'] = create_gr_elem(d.mask_file)
                components['mask_brightness_adjust'] = create_gr_elem(d.mask_brightness_adjust)
                components['mask_contrast_adjust'] = create_gr_elem(d.mask_contrast_adjust)
                
            with FormRow():
                components['mask_overlay_blur'] = create_gr_elem(d.mask_overlay_blur)
                components['fill'] = create_gr_elem(d.fill)
                components['full_res_mask'] = create_gr_elem(d.full_res_mask)
                components['full_res_mask_padding'] = create_gr_elem(d.full_res_mask_padding)
        
        # ERROR HANDLING
        with gr.Accordion("⚠️ Error Handling", open=False):
            with FormRow():
                components['reroll_blank_frames'] = create_gr_elem(d.reroll_blank_frames)
                components['reroll_patience'] = create_gr_elem(d.reroll_patience)
        
        # PARSEQ INTEGRATION
        with gr.Accordion("📊 Parseq Integration", open=False):
            gr.Markdown("""
            **Parseq Integration:** Use Parseq for advanced keyframe management.
            Paste your Parseq manifest JSON here to use external keyframe data.
            """)
            
            with FormRow():
                components['parseq_manifest'] = create_gr_elem(dp.parseq_manifest)
                
            with FormRow():
                components['parseq_use_deltas'] = create_gr_elem(dp.parseq_use_deltas)
                components['parseq_non_schedule_overrides'] = create_gr_elem(dp.parseq_non_schedule_overrides)
                
    return components
