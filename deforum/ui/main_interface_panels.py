"""
Main Interface Panels
Contains the primary user interface setup and tab organization
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn

# Import from our modular components instead of large elements.py
from .input_components import get_tab_run, get_tab_init
from .keyframe_components import get_tab_keyframes
from .animation_components import get_tab_animation, get_tab_prompts
from .wan_components import get_tab_wan
from .output_components import get_tab_output, get_tab_ffmpeg
from .settings_components import get_tab_setup, get_tab_advanced
from .component_builders import create_gr_elem, create_row

# Legacy imports for compatibility (to be phased out)
from .args import set_arg_lists, get_component_names
from .defaults import get_gradio_html
from .gradio_funcs import change_css, handle_change_functions

# ControlNet import (conditional)
try:
    from .controlnet import setup_controlnet_ui
except ImportError:
    def setup_controlnet_ui():
        return {}

# WAN integration imports
try:
    from .wan_event_handlers import (
        analyze_movement_handler, enhance_prompts_handler,
        load_deforum_to_wan_prompts_handler, load_wan_defaults_handler,
        validate_wan_generation, check_qwen_models_handler,
        download_qwen_model_handler, cleanup_qwen_cache_handler
    )
    WAN_AVAILABLE = True
except ImportError:
    WAN_AVAILABLE = False
    
try:
    from .prompt_enhancement_handlers import (
        random_style_handler, random_theme_handler, random_both_handler,
        reset_to_photorealistic_handler, cycle_creative_themes_handler,
        enhance_deforum_prompts_handler, enhance_wan_prompts_handler_with_style,
        apply_style_only_handler
    )
    ENHANCEMENT_AVAILABLE = True
except ImportError:
    ENHANCEMENT_AVAILABLE = False


def setup_deforum_left_side_ui():
    """Set up the main left-side UI with all tabs using modular components."""
    
    # Initialize argument sets
    d, da, dp, dv, dr, dw, dloopArgs = set_arg_lists()
    
    # Main info accordion
    with gr.Accordion("Info, Links and Help", open=False, elem_id='main_top_info_accord'):
        gr.HTML(value=get_gradio_html('main'))
    
    # Show info toggle
    with gr.Row(variant='compact'):
        show_info_on_ui = gr.Checkbox(label="Show more info", value=d.show_info_on_ui, interactive=True)
    
    # Main tab interface
    with gr.Blocks():
        with gr.Tabs():
            # ========== WORKFLOW-ORIENTED TAB STRUCTURE ==========
            
            # 1. RUN TAB - Quick generation controls
            tab_run_params = get_tab_run(d, da)
            
            # 2. SETUP TAB - Basic configuration  
            tab_setup_params = get_tab_setup(d, da)
            
            # 3. PROMPTS TAB - Content creation
            tab_prompts_params = get_tab_prompts(da)
            
            # 4. KEYFRAMES TAB - Motion and scheduling
            tab_keyframes_params = get_tab_keyframes(d, da, dloopArgs)
            
            # 5. ANIMATION TAB - Animation settings
            tab_animation_params = get_tab_animation(da, dloopArgs)
            
            # 6. INIT TAB - Input sources
            tab_init_params = get_tab_init(d, da, dp)
            
            # 7. WAN AI TAB - Advanced AI generation
            if WAN_AVAILABLE:
                tab_wan_params = get_tab_wan(dw)
            else:
                tab_wan_params = {}
                print("⚠️ WAN components not available")
            
            # 8. OUTPUT TAB - Video and output settings
            tab_output_params = get_tab_output(da, dv)
            
            # 9. FFMPEG TAB - Video processing
            tab_ffmpeg_params = get_tab_ffmpeg()
            
            # 10. ADVANCED TAB - Expert controls
            tab_advanced_params = get_tab_advanced(d, da)
            
            # ControlNet tab (conditional)
            controlnet_dict = {}
            if d.show_controlnet_tab:
                controlnet_dict = setup_controlnet_ui()
            
            # Merge all component dictionaries
            all_components = {
                **tab_run_params,
                **tab_setup_params, 
                **tab_prompts_params,
                **tab_keyframes_params,
                **tab_animation_params,
                **tab_init_params,
                **tab_wan_params,
                **tab_output_params,
                **tab_ffmpeg_params,
                **tab_advanced_params,
                **controlnet_dict
            }
            
            # Add components to locals for backward compatibility
            for key, value in all_components.items():
                locals()[key] = value
    
    # Set up UI interactions
    show_info_on_ui.change(fn=change_css, inputs=show_info_on_ui, outputs=[gr.HTML()])
    handle_change_functions(locals())
    
    # ========== WAN AI INTEGRATION ==========
    if WAN_AVAILABLE:
        _setup_wan_integration(locals())
    
    # ========== PROMPT ENHANCEMENT INTEGRATION ==========
    if ENHANCEMENT_AVAILABLE:
        _setup_prompt_enhancement(locals())
    
    # ========== MODEL VALIDATION ==========
    _setup_model_validation(locals())
    
    return all_components


def _setup_wan_integration(components):
    """Set up WAN AI integration with event handlers."""
    print("🔗 Setting up WAN AI integration...")
    
    # Wan Generate button
    if 'wan_generate_button' in components and 'wan_generation_status' in components:
        try:
            # Import the main generation function
            from .elements import wan_generate_video as wan_generate_video_main
            
            component_names = get_component_names()
            component_inputs = []
            
            for name in component_names:
                if name in components:
                    component_inputs.append(components[name])
            
            print(f"📊 Found {len(component_inputs)} UI components for WAN generation")
            
            components['wan_generate_button'].click(
                fn=wan_generate_video_main,
                inputs=component_inputs,
                outputs=[components['wan_generation_status']]
            )
            print("✅ WAN generate button connected")
            
        except Exception as e:
            print(f"⚠️ Failed to connect WAN generate button: {e}")
    
    # Movement Analysis
    if 'analyze_movement_btn' in components:
        try:
            components['analyze_movement_btn'].click(
                fn=analyze_movement_handler,
                inputs=[
                    components.get('wan_enhanced_prompts', gr.Textbox()),
                    components.get('wan_enable_shakify', gr.Checkbox(value=True)),
                    components.get('wan_movement_sensitivity_override', gr.Checkbox()),
                    components.get('wan_manual_sensitivity', gr.Slider(value=1.0))
                ],
                outputs=[
                    components.get('wan_enhanced_prompts', gr.Textbox()),
                    components.get('wan_movement_description', gr.Textbox())
                ]
            )
            print("✅ Movement analysis connected")
        except Exception as e:
            print(f"⚠️ Failed to connect movement analysis: {e}")
    
    # Prompt Enhancement  
    if 'enhance_prompts_btn' in components:
        try:
            components['enhance_prompts_btn'].click(
                fn=enhance_prompts_handler,
                inputs=[
                    components.get('wan_enhanced_prompts', gr.Textbox()),
                    components.get('wan_qwen_model', gr.Dropdown()),
                    components.get('wan_qwen_language', gr.Dropdown()),
                    components.get('wan_qwen_auto_download', gr.Checkbox())
                ],
                outputs=[
                    components.get('wan_enhanced_prompts', gr.Textbox()),
                    components.get('enhancement_progress', gr.Textbox())
                ]
            )
            print("✅ Prompt enhancement connected")
        except Exception as e:
            print(f"⚠️ Failed to connect prompt enhancement: {e}")
    
    # Prompt Loading Buttons
    _setup_prompt_loading_buttons(components)


def _setup_prompt_loading_buttons(components):
    """Set up prompt loading buttons for WAN integration."""
    
    # Load Deforum to WAN prompts
    if 'load_deforum_to_wan_btn' in components:
        try:
            if 'animation_prompts' in components:
                enhance_prompts_handler._animation_prompts_component = components['animation_prompts']
            
            components['load_deforum_to_wan_btn'].click(
                fn=load_deforum_to_wan_prompts_handler,
                inputs=[],
                outputs=[components.get('wan_enhanced_prompts', gr.Textbox())]
            )
            print("✅ Load Deforum to WAN connected")
        except Exception as e:
            print(f"⚠️ Failed to connect Deforum to WAN: {e}")
    
    # Load WAN defaults
    if 'load_wan_defaults_btn' in components:
        try:
            components['load_wan_defaults_btn'].click(
                fn=load_wan_defaults_handler,
                inputs=[],
                outputs=[components.get('wan_enhanced_prompts', gr.Textbox())]
            )
            print("✅ Load WAN defaults connected")
        except Exception as e:
            print(f"⚠️ Failed to connect WAN defaults: {e}")
    
    # Model management buttons
    _setup_model_management_buttons(components)


def _setup_model_management_buttons(components):
    """Set up model management buttons for WAN."""
    
    if 'check_qwen_models_btn' in components:
        try:
            components['check_qwen_models_btn'].click(
                fn=check_qwen_models_handler,
                inputs=[components.get('wan_qwen_model', gr.Dropdown())],
                outputs=[components.get('qwen_model_status', gr.HTML())]
            )
            print("✅ Check Qwen models connected")
        except Exception as e:
            print(f"⚠️ Failed to connect Qwen model check: {e}")
    
    if 'download_qwen_model_btn' in components:
        try:
            components['download_qwen_model_btn'].click(
                fn=download_qwen_model_handler,
                inputs=[
                    components.get('wan_qwen_model', gr.Dropdown()),
                    components.get('wan_qwen_auto_download', gr.Checkbox())
                ],
                outputs=[components.get('qwen_model_status', gr.HTML())]
            )
            print("✅ Download Qwen model connected")
        except Exception as e:
            print(f"⚠️ Failed to connect Qwen model download: {e}")
    
    if 'cleanup_qwen_cache_btn' in components:
        try:
            components['cleanup_qwen_cache_btn'].click(
                fn=cleanup_qwen_cache_handler,
                inputs=[],
                outputs=[components.get('qwen_model_status', gr.HTML())]
            )
            print("✅ Cleanup Qwen cache connected")
        except Exception as e:
            print(f"⚠️ Failed to connect Qwen cache cleanup: {e}")


def _setup_prompt_enhancement(components):
    """Set up prompt enhancement system."""
    print("🔗 Setting up prompt enhancement...")
    
    try:
        # Random style/theme buttons
        if 'random_style_btn' in components:
            components['random_style_btn'].click(
                fn=random_style_handler,
                inputs=[],
                outputs=[components.get('style_dropdown', gr.Dropdown())]
            )
        
        if 'random_theme_btn' in components:
            components['random_theme_btn'].click(
                fn=random_theme_handler,
                inputs=[],
                outputs=[components.get('theme_dropdown', gr.Dropdown())]
            )
        
        if 'random_both_btn' in components:
            components['random_both_btn'].click(
                fn=random_both_handler,
                inputs=[],
                outputs=[
                    components.get('style_dropdown', gr.Dropdown()),
                    components.get('theme_dropdown', gr.Dropdown())
                ]
            )
        
        # Enhancement buttons
        if 'enhance_deforum_btn' in components:
            components['enhance_deforum_btn'].click(
                fn=enhance_deforum_prompts_handler,
                inputs=[
                    components.get('animation_prompts', gr.Textbox()),
                    components.get('style_dropdown', gr.Dropdown()),
                    components.get('theme_dropdown', gr.Dropdown()),
                    components.get('custom_style', gr.Textbox()),
                    components.get('custom_theme', gr.Textbox()),
                    components.get('qwen_model_dropdown', gr.Dropdown()),
                    components.get('qwen_language', gr.Dropdown()),
                    components.get('qwen_auto_download', gr.Checkbox())
                ],
                outputs=[
                    components.get('animation_prompts', gr.Textbox()),
                    components.get('enhancement_status', gr.Textbox()),
                    components.get('enhancement_progress', gr.Textbox())
                ]
            )
        
        print("✅ Prompt enhancement connected")
        
    except Exception as e:
        print(f"⚠️ Failed to setup prompt enhancement: {e}")


def _setup_model_validation(components):
    """Set up model validation system."""
    print("🔗 Setting up model validation...")
    
    try:
        from .wan.wan_model_validator import WanModelValidator
        
        def wan_validate_models():
            """Validate WAN models with checksums."""
            try:
                validator = WanModelValidator()
                models = validator.discover_models()
                
                if not models:
                    return "❌ No WAN models found for validation."
                
                results = []
                results.append("🔐 WAN MODEL VALIDATION")
                results.append("=" * 40)
                
                valid_models = 0
                for model in models:
                    model_path = Path(model['path'])
                    hf_validation = validator.validate_against_huggingface_checksums(model_path)
                    
                    if hf_validation['valid']:
                        results.append(f"✅ {model['name']} - VALID")
                        valid_models += 1
                    else:
                        results.append(f"❌ {model['name']} - INVALID")
                
                results.append(f"\n📊 {valid_models}/{len(models)} models valid")
                return "\n".join(results)
                
            except Exception as e:
                return f"❌ Validation error: {str(e)}"
        
        # Connect validation buttons if they exist
        validation_buttons = [
            ('wan_validate_models_btn', wan_validate_models),
        ]
        
        for button_name, callback in validation_buttons:
            if button_name in components:
                components[button_name].click(
                    fn=callback,
                    inputs=[],
                    outputs=[components.get('validation_output', gr.Textbox())]
                )
        
        print("✅ Model validation connected")
        
    except ImportError:
        print("⚠️ WAN model validator not available")
    except Exception as e:
        print(f"⚠️ Failed to setup model validation: {e}")
