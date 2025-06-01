"""
UI Initialization Module

Contains the Init tab with image/video initialization, mask settings,
and Parseq integration functionality for Deforum.

Functions:
    - get_tab_init: Creates the main Init tab interface
    - Image and video initialization controls
    - Mask configuration utilities
    - Parseq integration settings
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn
from .ui_core_components import create_row, create_gr_elem
from .defaults import get_gradio_html


def get_tab_init(d, da, dp):
    """
    Create the Init tab with initialization and mask settings.
    
    Args:
        d: DeforumArgs instance with basic parameters
        da: DeforumAnimArgs instance with animation parameters
        dp: ParseqArgs instance with Parseq parameters
        
    Returns:
        dict: Dictionary containing all created components
    """
    components = {}
    
    with gr.TabItem('Init'):
        
        # Image Init Sub-tab
        image_init_components = create_image_init_tab(d)
        components.update(image_init_components)
        
        # Video Init Sub-tab
        video_init_components = create_video_init_tab(da)
        components.update(video_init_components)
        
        # Mask Init Sub-tab
        mask_init_components = create_mask_init_tab(d)
        components.update(mask_init_components)
        
        # Parseq Sub-tab
        parseq_components = create_parseq_tab(dp)
        components.update(parseq_components)
    
    return components


def create_image_init_tab(d):
    """
    Create the image initialization tab.
    
    Args:
        d: DeforumArgs instance
        
    Returns:
        dict: Dictionary of image init components
    """
    components = {}
    
    with gr.Tab('Image Init'):
        
        # Init settings row
        with FormRow():
            with gr.Column(min_width=150):
                components['use_init'] = create_gr_elem(d.use_init)
            with gr.Column(min_width=150):
                components['strength_0_no_init'] = create_gr_elem(d.strength_0_no_init)
            with gr.Column(min_width=170):
                components['strength'] = create_gr_elem(d.strength)  # TODO: rename to init_strength
        
        # Image upload controls
        components['init_image'] = create_row(d.init_image)
        components['init_image_box'] = create_row(d.init_image_box)
    
    return components


def create_video_init_tab(da):
    """
    Create the video initialization tab.
    
    Args:
        da: DeforumAnimArgs instance
        
    Returns:
        dict: Dictionary of video init components
    """
    components = {}
    
    with gr.Tab('Video Init'):
        
        # Video path input
        components['video_init_path'] = create_row(da.video_init_path)
        
        # Video extraction settings
        with FormRow():
            components['extract_from_frame'] = create_gr_elem(da.extract_from_frame)
            components['extract_to_frame'] = create_gr_elem(da.extract_to_frame)
            components['extract_nth_frame'] = create_gr_elem(da.extract_nth_frame)
            components['overwrite_extracted_frames'] = create_gr_elem(da.overwrite_extracted_frames)
            components['use_mask_video'] = create_gr_elem(da.use_mask_video)
        
        # Video mask path
        components['video_mask_path'] = create_row(da.video_mask_path)
    
    return components


def create_mask_init_tab(d):
    """
    Create the mask initialization tab.
    
    Args:
        d: DeforumArgs instance
        
    Returns:
        dict: Dictionary of mask init components
    """
    components = {}
    
    with gr.Tab('Mask Init'):
        
        # Mask enable settings
        with FormRow():
            components['use_mask'] = create_gr_elem(d.use_mask)
            components['use_alpha_as_mask'] = create_gr_elem(d.use_alpha_as_mask)
            components['invert_mask'] = create_gr_elem(d.invert_mask)
            components['overlay_mask'] = create_gr_elem(d.overlay_mask)
        
        # Mask file upload
        components['mask_file'] = create_row(d.mask_file)
        
        # Mask processing settings
        components['mask_overlay_blur'] = create_row(d.mask_overlay_blur)
        components['fill'] = create_row(d.fill)
        
        # Full resolution mask settings
        full_res_mask, full_res_mask_padding = create_row(
            d, 'full_res_mask', 'full_res_mask_padding'
        )
        components['full_res_mask'] = full_res_mask
        components['full_res_mask_padding'] = full_res_mask_padding
        
        # Mask adjustment controls
        with FormRow():
            with FormColumn(min_width=240):
                components['mask_contrast_adjust'] = create_gr_elem(d.mask_contrast_adjust)
            with FormColumn(min_width=250):
                components['mask_brightness_adjust'] = create_gr_elem(d.mask_brightness_adjust)
    
    return components


def create_parseq_tab(dp):
    """
    Create the Parseq integration tab.
    
    Args:
        dp: ParseqArgs instance
        
    Returns:
        dict: Dictionary of Parseq components
    """
    components = {}
    
    with gr.Tab(f"📊 Parseq"):
        
        # Parseq information
        gr.HTML(value=get_gradio_html('parseq'))
        
        # Parseq settings
        components['parseq_manifest'] = create_row(dp.parseq_manifest)
        components['parseq_non_schedule_overrides'] = create_row(dp.parseq_non_schedule_overrides)
        components['parseq_use_deltas'] = create_row(dp.parseq_use_deltas)
    
    return components


def create_advanced_init_settings(d, da):
    """
    Create advanced initialization settings.
    
    Args:
        d: DeforumArgs instance
        da: DeforumAnimArgs instance
        
    Returns:
        dict: Dictionary of advanced init components
    """
    components = {}
    
    with gr.Accordion("⚙️ Advanced Init Settings", open=False):
        
        # Color coherence for initialization
        with FormRow():
            components['init_color_coherence'] = create_gr_elem(da.color_coherence)
            components['init_color_force_grayscale'] = create_gr_elem(da.color_force_grayscale)
        
        # Init strength scheduling
        with FormRow():
            components['init_strength_schedule'] = gr.Textbox(
                label="Init Strength Schedule",
                lines=1,
                interactive=True,
                placeholder="0:(0.8), 30:(0.6), 60:(0.4)",
                info="Schedule for varying initialization strength over time"
            )
    
    return components


def create_video_processing_settings():
    """
    Create video processing and extraction settings.
    
    Returns:
        dict: Dictionary of video processing components
    """
    components = {}
    
    with gr.Accordion("🎬 Video Processing Settings", open=False):
        
        # Frame extraction quality
        with FormRow():
            components['video_extract_quality'] = gr.Slider(
                label="Extraction Quality",
                minimum=1,
                maximum=100,
                step=1,
                value=95,
                info="JPEG quality for extracted frames (higher = better quality, larger files)"
            )
        
        # Video format settings
        with FormRow():
            components['video_extract_format'] = gr.Dropdown(
                label="Extract Format",
                choices=["jpg", "png", "webp"],
                value="jpg",
                info="Format for extracted video frames"
            )
            
            components['video_fps_detection'] = gr.Checkbox(
                label="Auto-detect FPS",
                value=True,
                info="Automatically detect video FPS for frame extraction"
            )
    
    return components


def create_mask_processing_settings():
    """
    Create advanced mask processing settings.
    
    Returns:
        dict: Dictionary of mask processing components
    """
    components = {}
    
    with gr.Accordion("🎭 Advanced Mask Settings", open=False):
        
        # Mask preprocessing
        with FormRow():
            components['mask_gaussian_blur'] = gr.Slider(
                label="Gaussian Blur",
                minimum=0,
                maximum=50,
                step=1,
                value=0,
                info="Apply Gaussian blur to mask for softer edges"
            )
            
            components['mask_threshold'] = gr.Slider(
                label="Threshold",
                minimum=0,
                maximum=255,
                step=1,
                value=128,
                info="Threshold value for binary mask conversion"
            )
        
        # Mask dilation/erosion
        with FormRow():
            components['mask_dilate'] = gr.Slider(
                label="Dilate",
                minimum=0,
                maximum=20,
                step=1,
                value=0,
                info="Expand mask areas (morphological dilation)"
            )
            
            components['mask_erode'] = gr.Slider(
                label="Erode",
                minimum=0,
                maximum=20,
                step=1,
                value=0,
                info="Shrink mask areas (morphological erosion)"
            )
    
    return components


def setup_init_tab_events(components):
    """
    Set up event handlers for initialization tab components.
    
    Args:
        components (dict): Dictionary of UI components
    """
    # Toggle mask settings visibility based on use_mask
    if 'use_mask' in components:
        def toggle_mask_settings(use_mask):
            return gr.update(visible=use_mask)
        
        mask_dependent_components = [
            'mask_file',
            'mask_overlay_blur',
            'mask_contrast_adjust',
            'mask_brightness_adjust'
        ]
        
        for comp_name in mask_dependent_components:
            if comp_name in components:
                components['use_mask'].change(
                    fn=toggle_mask_settings,
                    inputs=[components['use_mask']],
                    outputs=[components[comp_name]]
                )
    
    # Toggle video mask path visibility
    if 'use_mask_video' in components and 'video_mask_path' in components:
        def toggle_video_mask_path(use_mask_video):
            return gr.update(visible=use_mask_video)
        
        components['use_mask_video'].change(
            fn=toggle_video_mask_path,
            inputs=[components['use_mask_video']],
            outputs=[components['video_mask_path']]
        )
    
    # Validate frame extraction ranges
    if all(key in components for key in ['extract_from_frame', 'extract_to_frame']):
        def validate_frame_range(from_frame, to_frame):
            if from_frame >= to_frame:
                return (
                    gr.update(value=0),
                    gr.update(value=from_frame + 1)
                )
            return gr.update(), gr.update()
        
        components['extract_from_frame'].change(
            fn=validate_frame_range,
            inputs=[components['extract_from_frame'], components['extract_to_frame']],
            outputs=[components['extract_from_frame'], components['extract_to_frame']]
        )


def validate_init_settings(components_dict):
    """
    Validate initialization settings.
    
    Args:
        components_dict (dict): Dictionary of component values
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check strength values
    if 'strength' in components_dict:
        strength = components_dict['strength']
        if not (0.0 <= strength <= 1.0):
            errors.append("Strength must be between 0.0 and 1.0")
    
    # Check frame extraction range
    if 'extract_from_frame' in components_dict and 'extract_to_frame' in components_dict:
        from_frame = components_dict['extract_from_frame']
        to_frame = components_dict['extract_to_frame']
        if from_frame >= to_frame:
            errors.append("Extract from frame must be less than extract to frame")
    
    # Check nth frame value
    if 'extract_nth_frame' in components_dict:
        nth_frame = components_dict['extract_nth_frame']
        if nth_frame < 1:
            errors.append("Extract nth frame must be at least 1")
    
    return len(errors) == 0, errors


def get_init_tab_info():
    """
    Get informational HTML for the Init tab.
    
    Returns:
        str: HTML content with usage information
    """
    return """
    <div style="padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px 0;">
        <h4>🎯 Initialization Guide</h4>
        <ul>
            <li><strong>Image Init:</strong> Start generation from an initial image</li>
            <li><strong>Video Init:</strong> Extract frames from video for initialization</li>
            <li><strong>Mask Init:</strong> Use masks to control generation areas</li>
            <li><strong>Parseq:</strong> Import external parameter schedules</li>
            <li><strong>Strength:</strong> Controls how much the init image influences generation</li>
        </ul>
    </div>
    """


def create_init_preview_controls():
    """
    Create preview controls for initialization.
    
    Returns:
        dict: Dictionary of preview components
    """
    components = {}
    
    with gr.Accordion("👁️ Init Preview", open=False):
        
        # Preview buttons
        with FormRow():
            components['preview_init_btn'] = gr.Button(
                "Preview Init Image",
                variant="secondary"
            )
            
            components['preview_mask_btn'] = gr.Button(
                "Preview Mask",
                variant="secondary"
            )
        
        # Preview display
        components['init_preview'] = gr.Image(
            label="Init Preview",
            interactive=False,
            visible=False
        )
        
        components['mask_preview'] = gr.Image(
            label="Mask Preview", 
            interactive=False,
            visible=False
        )
    
    return components


def get_parseq_integration_info():
    """
    Get information about Parseq integration.
    
    Returns:
        str: HTML content with Parseq information
    """
    return """
    <div style="padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 10px; margin: 10px 0;">
        <h3>📊 Parseq Integration</h3>
        <p><strong>What is Parseq?</strong> External parameter scheduling system for advanced animation control.</p>
        
        <h4>🎯 Key Features:</h4>
        <ul>
            <li><strong>External Schedules:</strong> Import parameter schedules from external tools</li>
            <li><strong>Delta Mode:</strong> Apply relative changes instead of absolute values</li>
            <li><strong>Override Control:</strong> Selectively override specific parameters</li>
            <li><strong>Advanced Timing:</strong> Precise frame-level parameter control</li>
        </ul>
        
        <h4>📝 Usage:</h4>
        <ol>
            <li>Create parameter schedules in Parseq or compatible tool</li>
            <li>Export as JSON manifest file</li>
            <li>Upload manifest to Parseq Manifest field</li>
            <li>Configure override and delta settings as needed</li>
        </ol>
    </div>
    """


def create_initialization_presets():
    """
    Create initialization preset controls.
    
    Returns:
        dict: Dictionary of preset components
    """
    components = {}
    
    with gr.Accordion("🎨 Init Presets", open=False):
        
        # Preset selection
        components['init_preset'] = gr.Dropdown(
            label="Initialization Preset",
            choices=[
                "Custom",
                "Photo Enhancement",
                "Artistic Style Transfer", 
                "Animation Starting Frame",
                "Video Continuation",
                "Masked Region Editing"
            ],
            value="Custom",
            info="Pre-configured settings for common use cases"
        )
        
        # Preset description
        components['preset_description'] = gr.Markdown(
            "Select a preset above to automatically configure initialization settings for specific use cases."
        )
    
    return components


def apply_init_preset(preset_name, components):
    """
    Apply initialization preset settings.
    
    Args:
        preset_name (str): Name of the preset to apply
        components (dict): Dictionary of UI components
        
    Returns:
        dict: Updated component values
    """
    presets = {
        "Photo Enhancement": {
            "strength": 0.3,
            "use_init": True,
            "strength_0_no_init": True
        },
        "Artistic Style Transfer": {
            "strength": 0.6,
            "use_init": True,
            "strength_0_no_init": False
        },
        "Animation Starting Frame": {
            "strength": 0.8,
            "use_init": True,
            "strength_0_no_init": False
        },
        "Video Continuation": {
            "strength": 0.9,
            "use_init": True,
            "extract_nth_frame": 1
        },
        "Masked Region Editing": {
            "strength": 0.5,
            "use_init": True,
            "use_mask": True,
            "invert_mask": False
        }
    }
    
    if preset_name in presets:
        return presets[preset_name]
    
    return {} 