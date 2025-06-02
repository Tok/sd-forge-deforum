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
from .missing_components import create_missing_component_placeholders
from .component_builders import create_gr_elem, create_row

# Legacy imports for compatibility (to be phased out)
from ..config.args import set_arg_lists, get_component_names
from ..config.defaults import get_gradio_html
from .gradio_functions import change_css, handle_change_functions
from ..utils.core_utilities import debug_print

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
        # Create a proper HTML component for CSS output instead of creating it inline
        # css_output = gr.HTML(visible=False)
    
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
        
        # CRITICAL FIX: Add missing component placeholders to prevent warnings
        missing_components = create_missing_component_placeholders()
        
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
            **controlnet_dict,
            **missing_components  # Add missing components to prevent warnings
        }
        
        # Add components to locals for backward compatibility
        for key, value in all_components.items():
            locals()[key] = value
    
    # Set up UI interactions
    # show_info_on_ui.change(fn=change_css, inputs=show_info_on_ui, outputs=[css_output])
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
    debug_print("Setting up WAN AI integration...")
    
    # Wan Generate button - with safe import
    if 'wan_generate_button' in components and 'wan_generation_status' in components:
        try:
            # Try to import the main generation function safely
            try:
                from .elements import wan_generate_video as wan_generate_video_main
            except ImportError as import_error:
                debug_print(f"Could not import wan_generate_video: {import_error}")
                # Skip WAN generate button setup if import fails
                print("⚠️ Skipping WAN generate button - import failed")
                return
            
            component_names = get_component_names()
            component_inputs = []
            
            # Filter out None components and validate _id
            for name in component_names:
                if name in components:
                    comp = components[name]
                    if comp is not None and hasattr(comp, '_id'):
                        component_inputs.append(comp)
            
            debug_print(f"Found {len(component_inputs)} valid UI components for WAN generation")
            
            # Only set up the click handler if we have valid components
            if component_inputs and components.get('wan_generation_status') is not None and hasattr(components['wan_generation_status'], '_id'):
                components['wan_generate_button'].click(
                    fn=wan_generate_video_main,
                    inputs=component_inputs,
                    outputs=[components['wan_generation_status']]
                )
                debug_print("WAN generate button connected")
            else:
                debug_print("Skipping WAN generate button - missing valid components")
            
        except Exception as e:
            print(f"⚠️ Failed to connect WAN generate button: {e}")
    
    # Movement Analysis
    if 'analyze_movement_btn' in components:
        try:
            # Filter out None components and validate _id
            inputs = [comp for comp in [
                components.get('wan_enhanced_prompts'),
                components.get('wan_enable_shakify'),
                components.get('wan_movement_sensitivity_override'),
                components.get('wan_manual_sensitivity')
            ] if comp is not None and hasattr(comp, '_id')]
            
            outputs = [comp for comp in [
                components.get('wan_enhanced_prompts'),
                components.get('wan_movement_description')
            ] if comp is not None and hasattr(comp, '_id')]
            
            if inputs and outputs and len(inputs) >= 1 and len(outputs) >= 1:  # Ensure minimum required components
                components['analyze_movement_btn'].click(
                    fn=analyze_movement_handler,
                    inputs=inputs,
                    outputs=outputs
                )
                debug_print("Movement analysis connected")
            else:
                debug_print("Skipping movement analysis - missing components")
        except Exception as e:
            print(f"⚠️ Failed to connect movement analysis: {e}")
    
    # Prompt Enhancement  
    if 'enhance_prompts_btn' in components:
        try:
            # Filter out None components and validate _id
            inputs = [comp for comp in [
                components.get('wan_enhanced_prompts'),
                components.get('wan_qwen_model'),
                components.get('wan_qwen_language'),
                components.get('wan_qwen_auto_download')
            ] if comp is not None and hasattr(comp, '_id')]
            
            outputs = [comp for comp in [
                components.get('wan_enhanced_prompts'),
                components.get('enhancement_progress')
            ] if comp is not None and hasattr(comp, '_id')]
            
            if inputs and outputs and len(inputs) >= 1 and len(outputs) >= 1:  # Ensure minimum required components
                components['enhance_prompts_btn'].click(
                    fn=enhance_prompts_handler,
                    inputs=inputs,
                    outputs=outputs
                )
                debug_print("Prompt enhancement connected")
            else:
                debug_print("Skipping prompt enhancement - missing components")
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
            
            # Filter outputs for None components and validate _id
            outputs = [comp for comp in [components.get('wan_enhanced_prompts')] if comp is not None and hasattr(comp, '_id')]
            
            if outputs:
                components['load_deforum_to_wan_btn'].click(
                    fn=load_deforum_to_wan_prompts_handler,
                    inputs=[],
                    outputs=outputs
                )
                debug_print("Load Deforum to WAN connected")
            else:
                debug_print("Skipping Load Deforum to WAN - missing output component")
        except Exception as e:
            print(f"⚠️ Failed to connect Deforum to WAN: {e}")
    
    # Load WAN defaults
    if 'load_wan_defaults_btn' in components:
        try:
            # Filter outputs for None components and validate _id
            outputs = [comp for comp in [components.get('wan_enhanced_prompts')] if comp is not None and hasattr(comp, '_id')]
            
            if outputs:
                components['load_wan_defaults_btn'].click(
                    fn=load_wan_defaults_handler,
                    inputs=[],
                    outputs=outputs
                )
                debug_print("Load WAN defaults connected")
            else:
                debug_print("Skipping Load WAN defaults - missing output component")
        except Exception as e:
            print(f"⚠️ Failed to connect WAN defaults: {e}")
    
    # Model management buttons
    _setup_model_management_buttons(components)


def _setup_model_management_buttons(components):
    """Set up model management buttons for WAN."""
    
    if 'check_qwen_models_btn' in components:
        try:
            # Filter inputs and outputs for None components
            inputs = [comp for comp in [components.get('wan_qwen_model')] if comp is not None and hasattr(comp, '_id')]
            outputs = [comp for comp in [components.get('qwen_model_status')] if comp is not None and hasattr(comp, '_id')]
            
            if inputs and outputs:
                components['check_qwen_models_btn'].click(
                    fn=check_qwen_models_handler,
                    inputs=inputs,
                    outputs=outputs
                )
                debug_print("Check Qwen models connected")
            else:
                debug_print("Skipping Check Qwen models - missing components")
        except Exception as e:
            print(f"⚠️ Failed to connect Qwen model check: {e}")
    
    if 'download_qwen_model_btn' in components:
        try:
            # Filter inputs and outputs for None components
            inputs = [comp for comp in [
                components.get('wan_qwen_model'),
                components.get('wan_qwen_auto_download')
            ] if comp is not None and hasattr(comp, '_id')]
            outputs = [comp for comp in [components.get('qwen_model_status')] if comp is not None and hasattr(comp, '_id')]
            
            if inputs and outputs and len(inputs) >= 1:
                components['download_qwen_model_btn'].click(
                    fn=download_qwen_model_handler,
                    inputs=inputs,
                    outputs=outputs
                )
                debug_print("Download Qwen model connected")
            else:
                debug_print("Skipping Download Qwen model - missing components")
        except Exception as e:
            print(f"⚠️ Failed to connect Qwen model download: {e}")
    
    if 'cleanup_qwen_cache_btn' in components:
        try:
            # Filter outputs for None components
            outputs = [comp for comp in [components.get('qwen_model_status')] if comp is not None and hasattr(comp, '_id')]
            
            if outputs:
                components['cleanup_qwen_cache_btn'].click(
                    fn=cleanup_qwen_cache_handler,
                    inputs=[],
                    outputs=outputs
                )
                debug_print("Cleanup Qwen cache connected")
            else:
                debug_print("Skipping Cleanup Qwen cache - missing output component")
        except Exception as e:
            print(f"⚠️ Failed to connect Qwen cache cleanup: {e}")


def _setup_prompt_enhancement(components):
    """Set up prompt enhancement system."""
    print("🔗 Setting up prompt enhancement...")
    
    try:
        # Random style/theme buttons
        if 'random_style_btn' in components:
            outputs = [comp for comp in [components.get('style_dropdown')] if comp is not None and hasattr(comp, '_id')]
            if outputs:
                components['random_style_btn'].click(
                    fn=random_style_handler,
                    inputs=[],
                    outputs=outputs
                )
        
        if 'random_theme_btn' in components:
            outputs = [comp for comp in [components.get('theme_dropdown')] if comp is not None and hasattr(comp, '_id')]
            if outputs:
                components['random_theme_btn'].click(
                    fn=random_theme_handler,
                    inputs=[],
                    outputs=outputs
                )
        
        if 'random_both_btn' in components:
            outputs = [comp for comp in [
                components.get('style_dropdown'),
                components.get('theme_dropdown')
            ] if comp is not None and hasattr(comp, '_id')]
            if outputs and len(outputs) >= 2:
                components['random_both_btn'].click(
                    fn=random_both_handler,
                    inputs=[],
                    outputs=outputs
                )
        
        # Enhancement buttons
        if 'enhance_deforum_btn' in components:
            inputs = [comp for comp in [
                components.get('animation_prompts'),
                components.get('style_dropdown'),
                components.get('theme_dropdown'),
                components.get('custom_style'),
                components.get('custom_theme'),
                components.get('qwen_model_dropdown'),
                components.get('qwen_language'),
                components.get('qwen_auto_download')
            ] if comp is not None and hasattr(comp, '_id')]
            
            outputs = [comp for comp in [
                components.get('animation_prompts'),
                components.get('enhancement_status'),
                components.get('enhancement_progress')
            ] if comp is not None and hasattr(comp, '_id')]
            
            if inputs and outputs and len(inputs) >= 1 and len(outputs) >= 1:
                components['enhance_deforum_btn'].click(
                    fn=enhance_deforum_prompts_handler,
                    inputs=inputs,
                    outputs=outputs
                )
            else:
                print("⚠️ Skipping enhance_deforum_btn - missing components")
        
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
                outputs = [comp for comp in [components.get('validation_output')] if comp is not None and hasattr(comp, '_id')]
                if outputs:
                    components[button_name].click(
                        fn=callback,
                        inputs=[],
                        outputs=outputs
                    )
        
        print("✅ Model validation connected")
        
    except ImportError:
        print("⚠️ WAN model validator not available")
    except Exception as e:
        print(f"⚠️ Failed to setup model validation: {e}")
