"""
UI Run Controls Module

Contains the Run tab interface with sampling controls, scheduling options,
and batch mode functionality for Deforum.

Functions:
    - get_tab_run: Creates the main Run tab interface
    - Sampling and scheduling parameter controls
    - Batch mode and resume animation functionality
"""

import gradio as gr
from modules.ui_components import FormRow, ToolButton
from .ui_core_components import create_row, create_gr_elem
from ..utils import emoji_utils


def get_tab_run(d, da):
    """
    Create the Run tab with sampling controls and execution options.
    
    Args:
        d: DeforumArgs instance with basic parameters
        da: DeforumAnimArgs instance with animation parameters
        
    Returns:
        dict: Dictionary containing all created components
    """
    with (gr.TabItem(f"{emoji_utils.run()} Run")):  # RUN TAB
        
        # Motion Preview and Core Sampling Controls
        motion_preview_mode = create_row(d.motion_preview_mode)
        sampler, scheduler, steps = create_row(d, 'sampler', 'scheduler', 'steps')
        W, H = create_row(d, 'W', 'H')
        seed, batch_name = create_row(d, 'seed', 'batch_name')
        
        # Advanced Sampling Options
        with FormRow():
            restore_faces = create_gr_elem(d.restore_faces)
            tiling = create_gr_elem(d.tiling)
            enable_ddim_eta_scheduling = create_gr_elem(da.enable_ddim_eta_scheduling)
            enable_ancestral_eta_scheduling = create_gr_elem(da.enable_ancestral_eta_scheduling)
        
        # ETA Scheduling Controls
        with gr.Row(variant='compact') as eta_sch_row:
            ddim_eta_schedule = create_gr_elem(da.ddim_eta_schedule)
            ancestral_eta_schedule = create_gr_elem(da.ancestral_eta_schedule)

        # Batch Mode and Resume Controls
        with gr.Accordion('Batch Mode, Resume and more', open=False):
            
            # Batch Mode Tab
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
            
            # Resume Animation Tab
            with gr.Tab('Resume Animation'):
                resume_from_timestring, resume_timestring = create_row(
                    da, 'resume_from_timestring', 'resume_timestring'
                )
    
    return {k: v for k, v in {**locals(), **vars()}.items()}


def create_sampling_controls(d):
    """
    Create the core sampling parameter controls.
    
    Args:
        d: DeforumArgs instance
        
    Returns:
        dict: Dictionary of sampling control components
    """
    components = {}
    
    # Main sampling parameters
    components['sampler'] = create_gr_elem(d.sampler)
    components['scheduler'] = create_gr_elem(d.scheduler) 
    components['steps'] = create_gr_elem(d.steps)
    components['cfg_scale'] = create_gr_elem(d.cfg_scale)
    
    # Image dimensions
    components['W'] = create_gr_elem(d.W)
    components['H'] = create_gr_elem(d.H)
    
    # Seed and batch controls
    components['seed'] = create_gr_elem(d.seed)
    components['batch_name'] = create_gr_elem(d.batch_name)
    
    return components


def create_quality_controls(d):
    """
    Create quality and enhancement controls.
    
    Args:
        d: DeforumArgs instance
        
    Returns:
        dict: Dictionary of quality control components
    """
    components = {}
    
    with FormRow():
        components['restore_faces'] = create_gr_elem(d.restore_faces)
        components['tiling'] = create_gr_elem(d.tiling)
    
    return components


def create_eta_scheduling_controls(da):
    """
    Create ETA scheduling controls for advanced sampling.
    
    Args:
        da: DeforumAnimArgs instance
        
    Returns:
        dict: Dictionary of ETA scheduling components
    """
    components = {}
    
    # Enable/disable controls
    with FormRow():
        components['enable_ddim_eta_scheduling'] = create_gr_elem(da.enable_ddim_eta_scheduling)
        components['enable_ancestral_eta_scheduling'] = create_gr_elem(da.enable_ancestral_eta_scheduling)
    
    # Schedule inputs
    with gr.Row(variant='compact'):
        components['ddim_eta_schedule'] = create_gr_elem(da.ddim_eta_schedule)
        components['ancestral_eta_schedule'] = create_gr_elem(da.ancestral_eta_schedule)
    
    return components


def create_batch_mode_controls():
    """
    Create batch mode execution controls.
    
    Returns:
        dict: Dictionary of batch mode components
    """
    components = {}
    
    with gr.Accordion('Batch Mode, Resume and more', open=False):
        
        # Batch processing from settings files
        with gr.Tab('Batch Mode/ run from setting files'):
            with gr.Row():
                components['override_settings_with_file'] = gr.Checkbox(
                    label="Enable batch mode", 
                    value=False, 
                    interactive=True,
                    elem_id='override_settings',
                    info="run from a list of setting .txt files. Upload them to the box on the right (visible when enabled)"
                )
                components['custom_settings_file'] = gr.File(
                    label="Setting files", 
                    interactive=True, 
                    file_count="multiple",
                    file_types=[".txt"], 
                    elem_id="custom_setting_file", 
                    visible=False
                )
    
    return components


def create_resume_controls(da):
    """
    Create resume animation controls.
    
    Args:
        da: DeforumAnimArgs instance
        
    Returns:
        dict: Dictionary of resume control components
    """
    components = {}
    
    with gr.Tab('Resume Animation'):
        components['resume_from_timestring'] = create_gr_elem(da.resume_from_timestring)
        components['resume_timestring'] = create_gr_elem(da.resume_timestring)
    
    return components


def setup_run_tab_events(components):
    """
    Set up event handlers for run tab components.
    
    Args:
        components (dict): Dictionary of UI components
    """
    # Enable/disable batch file upload based on checkbox
    if 'override_settings_with_file' in components and 'custom_settings_file' in components:
        components['override_settings_with_file'].change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=[components['override_settings_with_file']],
            outputs=[components['custom_settings_file']]
        )
    
    # Validate resume timestring format
    if 'resume_timestring' in components:
        def validate_timestring(timestring):
            """Validate timestring format (YYYYMMDDHHMMSS)"""
            import re
            if not timestring:
                return gr.update()
            
            pattern = r'^\d{14}$'  # 14 digits for YYYYMMDDHHMMSS
            if not re.match(pattern, timestring):
                return gr.update(
                    value=timestring,
                    elem_classes=["error-input"]
                )
            return gr.update(elem_classes=[])
        
        components['resume_timestring'].change(
            fn=validate_timestring,
            inputs=[components['resume_timestring']],
            outputs=[components['resume_timestring']]
        )


def get_run_tab_info_html():
    """
    Get informational HTML content for the Run tab.
    
    Returns:
        str: HTML content with usage information
    """
    return """
    <div style="padding: 10px; background: #f0f0f0; border-radius: 5px; margin: 5px 0;">
        <h4>🚀 Run Tab Guide</h4>
        <ul>
            <li><strong>Sampling:</strong> Choose sampler, scheduler, and step count for generation quality</li>
            <li><strong>Dimensions:</strong> Set width/height - higher resolutions require more VRAM</li>
            <li><strong>Seed:</strong> Use -1 for random, or set specific values for reproducible results</li>
            <li><strong>Batch Mode:</strong> Run multiple generations from uploaded settings files</li>
            <li><strong>Resume:</strong> Continue interrupted animations from specific timestring</li>
        </ul>
    </div>
    """


def validate_run_parameters(components_dict):
    """
    Validate run parameters before execution.
    
    Args:
        components_dict (dict): Dictionary of component values
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check dimensions
    if 'W' in components_dict and 'H' in components_dict:
        w, h = components_dict['W'], components_dict['H']
        if w * h > 2048 * 2048:
            errors.append("Resolution too high - may cause VRAM issues")
        if w % 8 != 0 or h % 8 != 0:
            errors.append("Width and height must be multiples of 8")
    
    # Check steps
    if 'steps' in components_dict:
        steps = components_dict['steps']
        if steps < 1:
            errors.append("Steps must be at least 1")
        if steps > 150:
            errors.append("Steps over 150 may be unnecessarily slow")
    
    # Check timestring format for resume
    if 'resume_timestring' in components_dict and components_dict['resume_timestring']:
        import re
        timestring = components_dict['resume_timestring']
        if not re.match(r'^\d{14}$', timestring):
            errors.append("Resume timestring must be 14 digits (YYYYMMDDHHMMSS)")
    
    return len(errors) == 0, errors 