"""
UI Prompts and Enhancement Module

Contains the Prompts tab with main prompt editing interface and comprehensive
AI enhancement functionality for Deforum.

Functions:
    - get_tab_prompts: Creates the main Prompts tab interface
    - AI enhancement controls with style and theme systems
    - Prompt validation and management utilities
"""

import gradio as gr
from .ui_core_components import create_row
from ..utils import emoji_utils
from .defaults import DeforumAnimPrompts, get_gradio_html


def get_tab_prompts(da):
    """
    Create the Prompts tab with main prompt editing and AI enhancement.
    
    Args:
        da: DeforumAnimArgs instance
        
    Returns:
        dict: Dictionary containing all created components
    """
    with gr.TabItem(f"{emoji_utils.prompts()} Prompts"):
        # Main prompts in tabs for better organization
        with gr.Tabs():
            
            # ========== MAIN PROMPTS TAB ==========
            with gr.TabItem("📝 Main Prompts"):
                # PROMPTS INFO ACCORDION
                with gr.Accordion(label='*Important* notes on Prompts', elem_id='prompts_info_accord',
                                  open=False) as prompts_info_accord:
                    gr.HTML(value=get_gradio_html('prompts'))
                
                # Main animation prompts editor
                animation_prompts = create_row(
                    gr.Textbox(
                        label="Prompts", 
                        lines=8, 
                        interactive=True, 
                        value=DeforumAnimPrompts(),
                        info="""Full prompts list in a JSON format. The value on left side is the frame number and
                             its presence also defines the frame as a keyframe if a 'keyframe distribution' mode
                             is active. Duplicating the same prompt multiple times to define keyframes
                             is therefore expected and fine."""
                    )
                )
                
                # Positive and negative prompt modifiers
                animation_prompts_positive = create_row(
                    gr.Textbox(
                        label="Prompts positive", 
                        lines=1, 
                        interactive=True,
                        placeholder="words in here will be added to the start of all positive prompts"
                    )
                )
                
                animation_prompts_negative = create_row(
                    gr.Textbox(
                        label="Prompts negative", 
                        value="nsfw, nude", 
                        lines=1, 
                        interactive=True,
                        placeholder="words here will be added to the end of all negative prompts. ignored with Flux."
                    )
                )
            
            # ========== AI ENHANCEMENT TAB ==========
            enhancement_components = create_ai_enhancement_tab()
    
    # Combine all components for return
    return {
        'animation_prompts': animation_prompts,
        'animation_prompts_positive': animation_prompts_positive,
        'animation_prompts_negative': animation_prompts_negative,
        **enhancement_components
    }


def create_ai_enhancement_tab():
    """
    Create the AI Enhancement tab with comprehensive styling and enhancement controls.
    
    Returns:
        dict: Dictionary of enhancement components
    """
    components = {}
    
    with gr.TabItem("🎨 AI Enhancement"):
        
        # Enhancement Guide
        create_enhancement_guide()
        
        # Style and Theme Selection
        style_components = create_style_theme_controls()
        components.update(style_components)
        
        # Random and Reset Controls
        random_components = create_random_reset_controls()
        components.update(random_components)
        
        # AI Model Selection
        model_components = create_model_selection_controls()
        components.update(model_components)
        
        # Enhancement Action Buttons
        action_components = create_enhancement_action_buttons()
        components.update(action_components)
        
        # Status and Progress Displays
        status_components = create_status_progress_displays()
        components.update(status_components)
    
    return components


def create_enhancement_guide():
    """Create the AI Enhancement information guide."""
    with gr.Accordion("ℹ️ AI Enhancement Guide", open=False):
        gr.HTML("""
        <div style="padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin: 10px 0;">
            <h3>🎨 AI Prompt Enhancement</h3>
            <p><strong>What it does:</strong> Uses AI to enhance your prompts with better descriptions, artistic details, and consistent styling.</p>
            
            <h4>🎯 Style & Theme System:</h4>
            <ul>
                <li><strong>Style:</strong> Controls the overall artistic approach (photorealistic, anime, oil painting, etc.)</li>
                <li><strong>Theme:</strong> Adds mood/atmosphere (cyberpunk, nature, minimal, etc.)</li>
                <li><strong>Consistency:</strong> Applied to ALL prompts for coherent video sequences</li>
            </ul>
            
            <h4>⚡ For I2V Chaining with Wan:</h4>
            <p>Styles and themes ensure smooth transitions between video clips. Each clip maintains the same visual style.</p>
            
            <h4>🎬 For Pure Deforum + Flux:</h4>
            <p>Creates cinematic consistency across animation frames with enhanced artistic descriptions.</p>
        </div>
        """)


def create_style_theme_controls():
    """Create style and theme selection controls."""
    components = {}
    
    with gr.Row():
        with gr.Column(scale=1):
            components['style_dropdown'] = gr.Dropdown(
                label="🎨 Visual Style",
                choices=[
                    "Photorealistic",
                    "Cinematic",
                    "Anime/Manga", 
                    "Oil Painting",
                    "Watercolor",
                    "Digital Art",
                    "3D Render",
                    "Sketch/Drawing",
                    "Vintage Film",
                    "Studio Photography",
                    "Street Photography",
                    "Fine Art",
                    "Impressionist",
                    "Pop Art",
                    "Minimalist"
                ],
                value="Photorealistic",
                info="Main artistic style - affects overall visual approach"
            )
            
            components['custom_style'] = gr.Textbox(
                label="🎭 Custom Style",
                placeholder="e.g., 'neon-lit cyberpunk with holographic elements'",
                info="Override style dropdown with custom text"
            )
        
        with gr.Column(scale=1):
            components['theme_dropdown'] = gr.Dropdown(
                label="🌍 Theme/Atmosphere",
                choices=[
                    "None",
                    "Cyberpunk",
                    "Synthwave/Vaporwave", 
                    "Frutiger Aero",
                    "Steampunk",
                    "Post-Apocalyptic",
                    "Nature/Organic",
                    "Urban/Metropolitan",
                    "Retro-Futuristic",
                    "Noir/Moody",
                    "Ethereal/Dreamy",
                    "Industrial/Brutalist",
                    "Art Deco",
                    "Bauhaus/Minimal",
                    "Cosmic/Space",
                    "Medieval Fantasy",
                    "Tropical Paradise",
                    "Winter Wonderland",
                    "Desert Mystique",
                    "Underwater World"
                ],
                value="None",
                info="Thematic atmosphere - adds mood and environment details"
            )
            
            components['custom_theme'] = gr.Textbox(
                label="🏛️ Custom Theme",
                placeholder="e.g., 'ancient temples with golden light'",
                info="Override theme dropdown with custom text"
            )
    
    return components


def create_random_reset_controls():
    """Create random and reset control buttons."""
    components = {}
    
    with gr.Row():
        with gr.Column(scale=1):
            components['random_style_btn'] = gr.Button("🎲 Random Style", variant="secondary")
            components['random_theme_btn'] = gr.Button("🎲 Random Theme", variant="secondary")
            components['random_both_btn'] = gr.Button("🎲 Random Both", variant="primary")
        
        with gr.Column(scale=1):
            components['reset_to_photo_btn'] = gr.Button("📷 Reset to Photorealistic", variant="secondary")
            components['cycle_creative_btn'] = gr.Button("🌈 Cycle Creative Themes", variant="secondary")
    
    return components


def create_model_selection_controls():
    """Create AI model selection controls."""
    components = {}
    
    with gr.Row():
        with gr.Column(scale=2):
            components['qwen_model_dropdown'] = gr.Dropdown(
                label="🤖 AI Enhancement Model",
                choices=[
                    "Auto-Select",
                    "Qwen2.5-0.5B-Instruct",
                    "Qwen2.5-1.5B-Instruct", 
                    "Qwen2.5-3B-Instruct",
                    "Qwen2.5-7B-Instruct",
                    "Qwen2.5-14B-Instruct"
                ],
                value="Auto-Select",
                info="AI model for prompt enhancement - Auto-Select chooses based on available VRAM"
            )
        
        with gr.Column(scale=1):
            components['qwen_language'] = gr.Dropdown(
                label="🌐 Language",
                choices=["English", "Chinese", "Japanese", "Korean", "French", "German", "Spanish"],
                value="English",
                info="Enhancement language"
            )
            
            components['qwen_auto_download'] = gr.Checkbox(
                label="📥 Auto-Download Models",
                value=True,
                info="Automatically download AI models when needed"
            )
    
    return components


def create_enhancement_action_buttons():
    """Create enhancement action buttons."""
    components = {}
    
    with gr.Row():
        with gr.Column():
            components['enhance_deforum_btn'] = gr.Button(
                "✨ Enhance Deforum Prompts",
                variant="primary",
                size="lg"
            )
            
            components['enhance_wan_btn'] = gr.Button(
                "🎬 Enhance Wan Prompts", 
                variant="primary",
                size="lg"
            )
        
        with gr.Column():
            components['apply_style_deforum_btn'] = gr.Button(
                "🎨 Apply Style to Deforum Only",
                variant="secondary"
            )
            
            components['apply_style_wan_btn'] = gr.Button(
                "🎭 Apply Style to Wan Only",
                variant="secondary"
            )
    
    return components


def create_status_progress_displays():
    """Create status and progress display components."""
    components = {}
    
    components['enhancement_status'] = gr.Textbox(
        label="📊 Enhancement Status",
        lines=8,
        interactive=False,
        value="Ready for AI enhancement! Select your style and theme above, then click enhance buttons.",
        info="Progress and results will appear here"
    )
    
    components['enhancement_progress'] = gr.Textbox(
        label="⏳ Progress",
        lines=3,
        interactive=False,
        value="Waiting...",
        info="Real-time progress updates"
    )
    
    return components


def get_available_styles():
    """Get list of available visual styles."""
    return [
        "Photorealistic",
        "Cinematic",
        "Anime/Manga", 
        "Oil Painting",
        "Watercolor",
        "Digital Art",
        "3D Render",
        "Sketch/Drawing",
        "Vintage Film",
        "Studio Photography",
        "Street Photography",
        "Fine Art",
        "Impressionist",
        "Pop Art",
        "Minimalist"
    ]


def get_available_themes():
    """Get list of available themes/atmospheres."""
    return [
        "None",
        "Cyberpunk",
        "Synthwave/Vaporwave", 
        "Frutiger Aero",
        "Steampunk",
        "Post-Apocalyptic",
        "Nature/Organic",
        "Urban/Metropolitan",
        "Retro-Futuristic",
        "Noir/Moody",
        "Ethereal/Dreamy",
        "Industrial/Brutalist",
        "Art Deco",
        "Bauhaus/Minimal",
        "Cosmic/Space",
        "Medieval Fantasy",
        "Tropical Paradise",
        "Winter Wonderland",
        "Desert Mystique",
        "Underwater World"
    ]


def get_creative_themes():
    """Get list of creative/artistic themes."""
    return [
        "Cyberpunk",
        "Synthwave/Vaporwave",
        "Steampunk",
        "Post-Apocalyptic",
        "Ethereal/Dreamy",
        "Cosmic/Space",
        "Medieval Fantasy",
        "Tropical Paradise",
        "Winter Wonderland"
    ]


def validate_prompt_json(prompt_text):
    """
    Validate JSON format of animation prompts.
    
    Args:
        prompt_text (str): JSON prompt text to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    import json
    
    try:
        prompt_dict = json.loads(prompt_text)
        
        # Check if all keys are numeric (frame numbers)
        for key in prompt_dict.keys():
            try:
                int(key)
            except ValueError:
                return False, f"Invalid frame number: {key}. Frame numbers must be integers."
        
        # Check if all values are strings (prompts)
        for frame, prompt in prompt_dict.items():
            if not isinstance(prompt, str):
                return False, f"Invalid prompt at frame {frame}. Prompts must be strings."
            
            if not prompt.strip():
                return False, f"Empty prompt at frame {frame}. All prompts must have content."
        
        return True, "Valid JSON format"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"


def setup_enhancement_events(components):
    """
    Set up event handlers for enhancement components.
    
    Args:
        components (dict): Dictionary of UI components
    """
    import random
    
    # Random style selection
    if 'random_style_btn' in components and 'style_dropdown' in components:
        def random_style():
            styles = get_available_styles()
            return gr.update(value=random.choice(styles))
        
        components['random_style_btn'].click(
            fn=random_style,
            outputs=[components['style_dropdown']]
        )
    
    # Random theme selection
    if 'random_theme_btn' in components and 'theme_dropdown' in components:
        def random_theme():
            themes = [t for t in get_available_themes() if t != "None"]
            return gr.update(value=random.choice(themes))
        
        components['random_theme_btn'].click(
            fn=random_theme,
            outputs=[components['theme_dropdown']]
        )
    
    # Random both style and theme
    if 'random_both_btn' in components:
        def random_both():
            styles = get_available_styles()
            themes = [t for t in get_available_themes() if t != "None"]
            return (
                gr.update(value=random.choice(styles)),
                gr.update(value=random.choice(themes))
            )
        
        components['random_both_btn'].click(
            fn=random_both,
            outputs=[components['style_dropdown'], components['theme_dropdown']]
        )
    
    # Reset to photorealistic
    if 'reset_to_photo_btn' in components:
        def reset_to_photo():
            return (
                gr.update(value="Photorealistic"),
                gr.update(value="None"),
                gr.update(value=""),
                gr.update(value="")
            )
        
        components['reset_to_photo_btn'].click(
            fn=reset_to_photo,
            outputs=[
                components['style_dropdown'], 
                components['theme_dropdown'],
                components['custom_style'],
                components['custom_theme']
            ]
        )
    
    # Cycle creative themes
    if 'cycle_creative_btn' in components and 'theme_dropdown' in components:
        def cycle_creative_themes():
            creative_themes = get_creative_themes()
            return gr.update(value=random.choice(creative_themes))
        
        components['cycle_creative_btn'].click(
            fn=cycle_creative_themes,
            outputs=[components['theme_dropdown']]
        )


def get_prompt_enhancement_info():
    """Get informational HTML for prompt enhancement."""
    return """
    <div style="padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 5px 0;">
        <h4>📝 Prompt Enhancement Tips</h4>
        <ul>
            <li><strong>JSON Format:</strong> Use proper JSON with frame numbers as keys</li>
            <li><strong>Frame Numbers:</strong> Must be integers (0, 30, 60, etc.)</li>
            <li><strong>Prompts:</strong> Keep prompts descriptive but not overly complex</li>
            <li><strong>Consistency:</strong> Use similar styles across frames for smooth transitions</li>
            <li><strong>AI Enhancement:</strong> Works best with clear, specific base prompts</li>
        </ul>
    </div>
    """ 