"""
Animation Components
Contains animation controls and settings
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn, ToolButton
from .component_builders import create_gr_elem, create_row
from ..utils import emoji_utils


def get_tab_animation(da, dloopArgs):
    """Create the Animation tab with animation-specific controls.
    
    Args:
        da: DeforumAnimArgs instance
        dloopArgs: LoopArgs instance
        
    Returns:
        Dict of component references
    """
    with gr.TabItem(f"{emoji_utils.movie_camera()} Animation"):
        
        # PROMPTS SECTION
        with gr.Accordion("📝 Animation Prompts", open=True):
            gr.Markdown("""
            **Animation Prompts:** Define what happens at each keyframe of your animation.
            Use frame numbers to specify when prompts should change.
            """)
            
            with FormRow():
                animation_prompts = gr.Textbox(
                    label="Animation prompts",
                    lines=12,
                    value="",
                    interactive=True,
                    elem_id="animation_prompts",
                    info="JSON format: {\"0\": \"prompt for frame 0\", \"60\": \"prompt for frame 60\"}"
                )
                
            with FormRow():
                animation_prompts_positive = gr.Textbox(
                    label="Positive prompts",
                    lines=4,
                    value="",
                    interactive=True,
                    elem_id="animation_prompts_positive",
                    visible=False
                )
                
            with FormRow():
                animation_prompts_negative = gr.Textbox(
                    label="Negative prompts", 
                    lines=4,
                    value="",
                    interactive=True,
                    elem_id="animation_prompts_negative",
                    visible=False
                )
        
        # ANIMATION SETTINGS
        with gr.Accordion("⚙️ Animation Settings", open=True):
            with FormRow():
                animation_mode = create_gr_elem(da.animation_mode)
                border = create_gr_elem(da.border)
                
            with FormRow():
                max_frames = create_gr_elem(da.max_frames)
                diffusion_cadence = create_gr_elem(da.diffusion_cadence)
                
        # VIDEO INPUT SETTINGS
        with gr.Accordion("📹 Video Input", open=False):
            with FormRow():
                video_init_path = create_gr_elem(da.video_init_path)
                
            with FormRow():
                extract_from_frame = create_gr_elem(da.extract_from_frame)
                extract_to_frame = create_gr_elem(da.extract_to_frame)
                
            with FormRow():
                extract_nth_frame = create_gr_elem(da.extract_nth_frame)
                overwrite_extracted_frames = create_gr_elem(da.overwrite_extracted_frames)
                
            with FormRow():
                use_mask_video = create_gr_elem(da.use_mask_video)
                video_mask_path = create_gr_elem(da.video_mask_path)
        
        # GUIDED IMAGES SECTION (moved from keyframes)
        with gr.Accordion("🖼️ Guided Images", open=False):
            gr.Markdown("""
            **Guided Images Mode:** Use a sequence of images to guide the animation.
            Each image will influence the generation at its corresponding frame.
            """)
            
            with FormRow():
                use_looper = create_gr_elem(dloopArgs.use_looper)
                
            with FormRow():
                init_images = create_gr_elem(dloopArgs.init_images)
                
            # Guided Images Schedules
            with gr.Accordion("📊 Guided Images Schedules", open=False):
                with FormRow():
                    image_strength_schedule = create_gr_elem(dloopArgs.image_strength_schedule)
                    
                with FormRow():
                    image_keyframe_strength_schedule = create_gr_elem(dloopArgs.image_keyframe_strength_schedule)
                    
                with FormRow():
                    blendFactorMax = create_gr_elem(dloopArgs.blendFactorMax)
                    blendFactorSlope = create_gr_elem(dloopArgs.blendFactorSlope)
                    
                with FormRow():
                    tweening_frames_schedule = create_gr_elem(dloopArgs.tweening_frames_schedule)
                    color_correction_factor = create_gr_elem(dloopArgs.color_correction_factor)
        
        # RESUME ANIMATION
        with gr.Accordion("🔄 Resume Animation", open=False):
            with FormRow():
                resume_from_timestring = create_gr_elem(da.resume_from_timestring)
                resume_timestring = create_gr_elem(da.resume_timestring)
        
    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_prompts(da):
    """Create the Prompts tab with detailed prompt editing.
    
    Args:
        da: DeforumAnimArgs instance
        
    Returns:
        Dict of component references
    """
    with gr.TabItem(f"{emoji_utils.pencil()} Prompts"):
        gr.Markdown("""
        ## Animation Prompts
        
        Define what happens at each keyframe of your animation. Use JSON format to specify 
        which prompts should be active at which frames.
        
        **Format:** `{"0": "first prompt", "60": "second prompt", "120": "third prompt"}`
        """)
        
        # Main prompts editor
        with FormRow():
            animation_prompts = gr.Textbox(
                label="Animation prompts",
                lines=15,
                value='{\n    "0": "a beautiful landscape, detailed"\n}',
                interactive=True,
                elem_id="animation_prompts_main",
                info="JSON format animation prompts. Each key is a frame number, each value is the prompt for that frame."
            )
        
        # Prompt utilities
        with gr.Accordion("🛠️ Prompt Utilities", open=False):
            with FormRow():
                load_prompts_btn = gr.Button(
                    "📁 Load Prompts from File",
                    variant="secondary"
                )
                save_prompts_btn = gr.Button(
                    "💾 Save Prompts to File", 
                    variant="secondary"
                )
                validate_prompts_btn = gr.Button(
                    "✅ Validate JSON",
                    variant="primary"
                )
            
            # Validation results
            validation_output = gr.Textbox(
                label="Validation Results",
                lines=3,
                interactive=False,
                placeholder="Click 'Validate JSON' to check your prompts format..."
            )
        
        # Positive/Negative prompts (advanced)
        with gr.Accordion("➕➖ Advanced Positive/Negative Prompts", open=False):
            gr.Markdown("""
            **Advanced Mode:** Separate positive and negative prompts for fine control.
            Leave empty to use the main prompts above.
            """)
            
            with FormRow():
                animation_prompts_positive = gr.Textbox(
                    label="Positive prompts only",
                    lines=8,
                    value="",
                    interactive=True,
                    placeholder='{\n    "0": "beautiful, detailed, high quality"\n}',
                    info="Optional: Positive prompts only (overrides main prompts if provided)"
                )
            
            with FormRow():
                animation_prompts_negative = gr.Textbox(
                    label="Negative prompts only",
                    lines=8,
                    value="",
                    interactive=True,
                    placeholder='{\n    "0": "blurry, low quality, distorted"\n}',
                    info="Optional: Negative prompts for each frame"
                )
        
        # Event handlers
        def validate_prompts_handler(prompts_text):
            """Validate the JSON format of prompts."""
            try:
                import json
                
                if not prompts_text.strip():
                    return "❌ Empty prompts. Please add some prompts."
                
                # Try to parse as JSON
                prompts = json.loads(prompts_text)
                
                if not isinstance(prompts, dict):
                    return "❌ Prompts must be a JSON object (dictionary)"
                
                # Validate frame numbers
                frame_numbers = []
                for key in prompts.keys():
                    try:
                        frame_num = int(key)
                        frame_numbers.append(frame_num)
                    except ValueError:
                        return f"❌ Invalid frame number: '{key}'. Frame numbers must be integers."
                
                # Check for negative frame numbers
                if any(f < 0 for f in frame_numbers):
                    return "❌ Frame numbers cannot be negative."
                
                frame_numbers.sort()
                num_prompts = len(prompts)
                max_frame = max(frame_numbers) if frame_numbers else 0
                
                return f"""✅ **Valid JSON!** 
                
📊 **Statistics:**
• {num_prompts} prompt(s) defined
• Frame range: {min(frame_numbers)} to {max_frame}
• Total animation length: {max_frame + 1} frames

🎬 **Ready for animation generation!**"""
                
            except json.JSONDecodeError as e:
                return f"❌ **JSON Error:** {str(e)}\n\nPlease check your JSON syntax (quotes, commas, brackets)."
            except Exception as e:
                return f"❌ **Validation Error:** {str(e)}"
        
        # Connect validation handler
        if validate_prompts_btn is not None and hasattr(validate_prompts_btn, '_id'):
            valid_inputs = [comp for comp in [animation_prompts] if comp is not None and hasattr(comp, '_id')]
            valid_outputs = [comp for comp in [validation_output] if comp is not None and hasattr(comp, '_id')]
            if valid_inputs and valid_outputs:
                try:
                    validate_prompts_btn.click(
                        fn=validate_prompts_handler,
                        inputs=valid_inputs,
                        outputs=valid_outputs
                    )
                except Exception as e:
                    print(f"⚠️ Failed to connect validation button: {e}")
        
        # Auto-validate on change (debounced) - safely
        if animation_prompts is not None and hasattr(animation_prompts, '_id'):
            valid_outputs = [comp for comp in [validation_output] if comp is not None and hasattr(comp, '_id')]
            if valid_outputs:
                try:
                    animation_prompts.change(
                        fn=validate_prompts_handler,
                        inputs=[animation_prompts], 
                        outputs=valid_outputs
                    )
                except Exception as e:
                    print(f"⚠️ Failed to connect animation prompts change handler: {e}")
    
    return {k: v for k, v in {**locals(), **vars()}.items()} 