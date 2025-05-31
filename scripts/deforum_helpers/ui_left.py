# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

from types import SimpleNamespace
import gradio as gr
from .defaults import get_gradio_html
from .gradio_funcs import change_css, handle_change_functions
from .args import DeforumArgs, DeforumAnimArgs, ParseqArgs, DeforumOutputArgs, RootArgs, LoopArgs, FreeUArgs, KohyaHRFixArgs, WanArgs
from .deforum_controlnet import setup_controlnet_ui
from .ui_elements import (get_tab_run, get_tab_keyframes, get_tab_prompts, get_tab_init,
                          get_tab_hybrid, get_tab_output, get_tab_freeu, get_tab_kohya_hrfix)

def set_arg_lists():
    # convert dicts to NameSpaces for easy working (args.param instead of args['param']
    d = SimpleNamespace(**DeforumArgs())  # default args
    da = SimpleNamespace(**DeforumAnimArgs())  # default anim args
    dp = SimpleNamespace(**ParseqArgs())  # default parseq ars
    dv = SimpleNamespace(**DeforumOutputArgs())  # default video args
    dr = SimpleNamespace(**RootArgs())  # ROOT args
    dfu = SimpleNamespace(**FreeUArgs()) 
    dku = SimpleNamespace(**KohyaHRFixArgs()) 
    dw = SimpleNamespace(**WanArgs())  # Wan args
    dloopArgs = SimpleNamespace(**LoopArgs())  # Guided imgs args
    return d, da, dp, dv, dr, dfu, dku, dw, dloopArgs

def wan_generate_video():
    """
    Simple placeholder function for Wan video generation button
    Returns a status message indicating the feature is integrated but needs models
    """
    try:
        print("🎬 Wan video generation button clicked!")
        
        # Try to discover models to check if setup is complete
        try:
            from .wan.wan_simple_integration import WanSimpleIntegration
            integration = WanSimpleIntegration()
            models = integration.discover_models()
            
            if models:
                return f"""✅ Wan integration is working!

Found {len(models)} model(s):
{chr(10).join([f"• {model['name']} ({model['size']})" for model in models[:3]])}

🔧 TODO: Full generation requires connecting to Deforum's argument system.
For now, you can test model discovery is working.

💡 Next steps:
1. Ensure your prompts are configured in the Prompts tab
2. Set your desired FPS in the Output tab  
3. Choose animation mode 'Wan Video' in the Keyframes tab
4. Click the main Generate button in Deforum

📁 Models found in: {models[0]['path']}"""
            else:
                return """❌ No Wan models found!

💡 SETUP REQUIRED:
1. Download a Wan model:
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan

2. Or place your Wan models in:
   • models/wan/
   • models/Wan/
   • HuggingFace cache (automatic)

3. Restart the WebUI after downloading

The auto-discovery will find your models automatically!"""
                
        except ImportError as e:
            return f"""⚠️ Wan integration partially loaded

The Wan tab is integrated but some dependencies may be missing.

Error: {str(e)}

💡 To complete setup:
1. Download Wan models as instructed above
2. Ensure all Wan dependencies are installed
3. Check the console for any import errors"""
            
        except Exception as e:
            return f"""❌ Wan integration error: {str(e)}

💡 Troubleshooting:
1. Check that Wan models are downloaded and placed correctly
2. Verify all dependencies are installed
3. Check console output for detailed error messages
4. Try restarting the WebUI"""
            
    except Exception as e:
        print(f"❌ Wan button error: {e}")
        return f"❌ Error: {str(e)}"

def setup_deforum_left_side_ui():
    d, da, dp, dv, dr, dfu, dku, dw, dloopArgs = set_arg_lists()
    # set up main info accordion on top of the UI
    with gr.Accordion("Info, Links and Help", open=False, elem_id='main_top_info_accord'):
        gr.HTML(value=get_gradio_html('main'))
    # show button to hide/ show gradio's info texts for each element in the UI
    with gr.Row(variant='compact'):
        show_info_on_ui = gr.Checkbox(label="Show more info", value=d.show_info_on_ui, interactive=True)
    with gr.Blocks():
        with gr.Tabs():
            # Get main tab contents:
            tab_run_params = get_tab_run(d, da)  # Run tab
            tab_keyframes_params = get_tab_keyframes(d, da, dloopArgs)  # Keyframes tab
            tab_prompts_params = get_tab_prompts(da)  # Prompts tab
            tab_init_params = get_tab_init(d, da, dp)  # Init tab
            controlnet_dict = setup_controlnet_ui()  # ControlNet tab
            tab_freeu_params = get_tab_freeu(dfu)  # FreeU tab
            tab_kohya_hrfix_params = get_tab_kohya_hrfix(dku)  # Kohya tab
            # Re-enable Wan tab (UI only, imports still isolated)
            from .ui_elements import get_tab_wan
            tab_wan_params = get_tab_wan(dw)  # Re-enable Wan tab
            tab_hybrid_params = get_tab_hybrid(da)  # Hybrid tab
            tab_output_params = get_tab_output(da, dv)  # Output tab
            # add returned gradio elements from main tabs to locals()
            for key, value in {**tab_run_params, **tab_keyframes_params, **tab_prompts_params, **tab_init_params, **controlnet_dict, **tab_freeu_params, **tab_kohya_hrfix_params, **tab_wan_params, **tab_hybrid_params, **tab_output_params}.items():
                locals()[key] = value

    # Gradio's Change functions - hiding and renaming elements based on other elements
    show_info_on_ui.change(fn=change_css, inputs=show_info_on_ui, outputs=[gr.HTML()])
    handle_change_functions(locals())

    # Set up Wan Generate button if it exists - with better error handling
    if 'wan_generate_button' in locals() and 'wan_generation_status' in locals():
        try:
            print("🔗 Connecting Wan generate button...")
            
            # Import the real Wan generation function from ui_elements
            from .ui_elements import wan_generate_video as wan_generate_video_main
            
            # Get all component values to pass to the Wan generation function
            from .args import get_component_names
            component_names = get_component_names()
            
            # Create list of all UI components in the correct order
            component_inputs = []
            missing_components = []
            for name in component_names:
                if name in locals():
                    component_inputs.append(locals()[name])
                else:
                    missing_components.append(name)
                    print(f"⚠️ Warning: Component '{name}' not found in locals()")
            
            print(f"📊 Found {len(component_inputs)} UI components for Wan generation")
            if missing_components:
                print(f"⚠️ Missing {len(missing_components)} components: {missing_components[:5]}...")
            
            # Create a wrapper function with better error handling
            def wan_generate_wrapper(*args):
                try:
                    print(f"🎬 Wan generate button clicked! Received {len(args)} arguments")
                    print("🔄 Calling wan_generate_video_main...")
                    result = wan_generate_video_main(*args)
                    print(f"✅ Wan generation completed: {str(result)[:100]}...")
                    return result
                except Exception as e:
                    error_msg = f"❌ Wan generation error: {str(e)}"
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    return error_msg
            
            locals()['wan_generate_button'].click(
                fn=wan_generate_wrapper,
                inputs=component_inputs,  # Pass all UI component values
                outputs=[locals()['wan_generation_status']]
            )
            print("✅ Wan generate button connected successfully")
        except Exception as e:
            print(f"⚠️ Warning: Failed to connect Wan generate button: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to the simple placeholder function
            def simple_wan_test():
                return "🧪 Simple Wan test - button connection working but full integration failed"
            
            locals()['wan_generate_button'].click(
                fn=simple_wan_test,
                inputs=[],
                outputs=[locals()['wan_generation_status']]
            )

    # Set up Wan Prompt Enhancement button with proper wan_enhanced_prompts access
    if 'enhance_prompts_btn' in locals() and 'wan_enhanced_prompts' in locals():
        try:
            print("🔗 Connecting Wan prompt enhancement button...")
            
            from .ui_elements import enhance_prompts_handler
            
            # Check if enhancement_progress component exists for progress feedback
            enhancement_progress_available = 'enhancement_progress' in locals()
            
            # Connect the enhance button with current prompts as first parameter
            if enhancement_progress_available:
                # Connect with progress feedback
                locals()['enhance_prompts_btn'].click(
                    fn=enhance_prompts_handler,
                    inputs=[
                        locals()['wan_enhanced_prompts'],  # current_prompts - first parameter
                        locals()['wan_qwen_model'], 
                        locals()['wan_qwen_language'],
                        locals()['wan_qwen_auto_download']
                    ],
                    outputs=[locals()['wan_enhanced_prompts'], locals()['enhancement_progress']]
                )
                print("✅ Wan prompt enhancement button connected successfully with progress feedback")
            else:
                # Fallback connection without progress feedback
                def enhance_wrapper(*args):
                    result = enhance_prompts_handler(*args)
                    if isinstance(result, tuple):
                        return result[0]  # Return only the enhanced prompts
                    return result
                
                locals()['enhance_prompts_btn'].click(
                    fn=enhance_wrapper,
                    inputs=[
                        locals()['wan_enhanced_prompts'],  # current_prompts - first parameter
                        locals()['wan_qwen_model'], 
                        locals()['wan_qwen_language'],
                        locals()['wan_qwen_auto_download']
                    ],
                    outputs=[locals()['wan_enhanced_prompts']]
                )
                print("✅ Wan prompt enhancement button connected successfully (without progress feedback)")
        except Exception as e:
            print(f"⚠️ Warning: Failed to connect Wan prompt enhancement button: {e}")
            import traceback
            traceback.print_exc()

    # Set up movement component references for analyze_movement_handler
    try:
        from .ui_elements import analyze_movement_handler, enhance_prompts_handler
        
        # Store references to movement schedule components to get actual schedule strings
        movement_components = {}
        movement_component_names = [
            'translation_x', 'translation_y', 'translation_z',
            'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z',
            'zoom', 'angle', 'max_frames',
            # Add Camera Shakify components
            'shake_name', 'shake_intensity', 'shake_speed'
        ]
        
        # Get actual schedule values from the UI components (these are the schedule strings)
        for comp_name in movement_component_names:
            if comp_name in locals():
                component = locals()[comp_name]
                # Get the actual value from the component
                if comp_name == 'max_frames':
                    # max_frames is a number, not a schedule
                    movement_components[comp_name] = getattr(component, 'value', 100)
                elif comp_name in ['shake_name', 'shake_intensity', 'shake_speed']:
                    # Camera Shakify settings
                    if comp_name == 'shake_name':
                        movement_components[comp_name] = getattr(component, 'value', "None")
                    elif comp_name == 'shake_intensity':
                        movement_components[comp_name] = getattr(component, 'value', 1.0)
                    elif comp_name == 'shake_speed':
                        movement_components[comp_name] = getattr(component, 'value', 1.0)
                else:
                    # These are schedule strings used by Deforum's animation system
                    movement_components[comp_name] = getattr(component, 'value', f"0:(0)")
            else:
                # Fallback defaults (same as Deforum defaults)
                if comp_name == 'max_frames':
                    movement_components[comp_name] = 100
                elif comp_name == 'zoom':
                    movement_components[comp_name] = "0:(1.0)"  # Default zoom schedule
                elif comp_name == 'shake_name':
                    movement_components[comp_name] = "None"     # Default shake disabled
                elif comp_name == 'shake_intensity':
                    movement_components[comp_name] = 1.0       # Default shake intensity
                elif comp_name == 'shake_speed':
                    movement_components[comp_name] = 1.0       # Default shake speed
                else:
                    movement_components[comp_name] = "0:(0)"    # Default movement schedule
        
        # Store the movement components dictionary for the handler
        analyze_movement_handler._movement_components = movement_components
        
        # Store reference to wan_enhanced_prompts component for updating prompts with movement
        if 'wan_enhanced_prompts' in locals():
            enhance_prompts_handler._wan_enhanced_prompts_component = locals()['wan_enhanced_prompts']
        
        # Store reference to wan_movement_description component
        if 'wan_movement_description' in locals():
            analyze_movement_handler._wan_movement_description_component = locals()['wan_movement_description']
        
        print(f"✅ Movement schedule references set up for {len(movement_components)} Deforum schedules")
        print(f"📊 Sample schedules: translation_x='{movement_components.get('translation_x', 'N/A')[:30]}...', zoom='{movement_components.get('zoom', 'N/A')}'")
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to set up movement schedule references: {e}")
        import traceback
        traceback.print_exc()

    # Set up Wan prompt template loading buttons
    if 'load_wan_prompts_btn' in locals() and 'wan_enhanced_prompts' in locals():
        try:
            print("🔗 Connecting Wan prompt loading button...")
            
            from .ui_elements import load_wan_prompts_handler
            
            locals()['load_wan_prompts_btn'].click(
                fn=load_wan_prompts_handler,
                inputs=[],
                outputs=[locals()['wan_enhanced_prompts']]
            )
            print("✅ Wan prompt loading button connected")
        except Exception as e:
            print(f"⚠️ Warning: Failed to connect Wan prompt button: {e}")
    
    if 'load_deforum_prompts_btn' in locals() and 'wan_enhanced_prompts' in locals():
        try:
            print("🔗 Connecting Deforum prompts loading button...")
            
            from .ui_elements import load_deforum_prompts_handler
            
            locals()['load_deforum_prompts_btn'].click(
                fn=load_deforum_prompts_handler,
                inputs=[],
                outputs=[locals()['wan_enhanced_prompts']]
            )
            print("✅ Deforum prompts loading button connected")
        except Exception as e:
            print(f"⚠️ Warning: Failed to connect Deforum prompts button: {e}")
    
    # Set up load Deforum to Wan and load defaults buttons
    if 'load_deforum_to_wan_btn' in locals() and 'wan_enhanced_prompts' in locals():
        try:
            print("🔗 Connecting Load Deforum to Wan button...")
            
            from .ui_elements import load_deforum_to_wan_prompts_handler, enhance_prompts_handler
            
            # Store animation_prompts reference for deforum-to-wan loading
            if 'animation_prompts' in locals():
                enhance_prompts_handler._animation_prompts_component = locals()['animation_prompts']
            
            locals()['load_deforum_to_wan_btn'].click(
                fn=load_deforum_to_wan_prompts_handler,
                inputs=[],
                outputs=[locals()['wan_enhanced_prompts']]
            )
            print("✅ Load Deforum to Wan button connected")
        except Exception as e:
            print(f"⚠️ Warning: Failed to connect Load Deforum to Wan button: {e}")
    
    if 'load_wan_defaults_btn' in locals() and 'wan_enhanced_prompts' in locals():
        try:
            print("🔗 Connecting Load Wan Defaults button...")
            
            from .ui_elements import load_wan_defaults_handler
            
            locals()['load_wan_defaults_btn'].click(
                fn=load_wan_defaults_handler,
                inputs=[],
                outputs=[locals()['wan_enhanced_prompts']]
            )
            print("✅ Load Wan Defaults button connected")
        except Exception as e:
            print(f"⚠️ Warning: Failed to connect Load Wan Defaults button: {e}")

    # Set up Wan Model Validation buttons
    try:
        from .wan.wan_model_validator import WanModelValidator
        
        # Validation functions
        def wan_validate_models():
            """Validate Wan models with HuggingFace checksums when possible"""
            try:
                validator = WanModelValidator()
                models = validator.discover_models()
                
                if not models:
                    return "❌ No Wan models found for validation."
                
                results = []
                results.append("🔐 WAN MODEL VALIDATION WITH OFFICIAL CHECKSUMS")
                results.append("=" * 55)
                
                valid_models = 0
                total_models = len(models)
                
                for model in models:
                    results.append(f"\n📁 {model['name']} ({model['size_formatted']}):")
                    
                    from pathlib import Path
                    model_path = Path(model['path'])
                    
                    # Try HuggingFace checksum validation first
                    hf_validation = validator.validate_against_huggingface_checksums(model_path)
                    
                    if hf_validation['checked_files']:
                        # HuggingFace validation was possible
                        if hf_validation['valid']:
                            valid_count = len([f for f in hf_validation['checked_files'].values() if f['status'] == 'valid'])
                            total_count = len(hf_validation['checked_files'])
                            results.append(f"   ✅ VALID - {valid_count}/{total_count} files verified with official checksums")
                            valid_models += 1
                        else:
                            results.append(f"   ❌ INVALID - Checksum verification failed")
                            for error in hf_validation['errors']:
                                results.append(f"      🚨 {error}")
                    else:
                        # Fall back to basic validation if HuggingFace validation not possible
                        results.append(f"   ⚠️ Official checksums not available, using basic validation...")
                        validation_result = validator.validate_model_integrity(model_path)
                        
                        if validation_result['valid']:
                            results.append(f"   ✅ VALID (basic structure check)")
                            valid_models += 1
                        else:
                            results.append(f"   ❌ INVALID")
                            for error in validation_result['errors']:
                                results.append(f"      🚨 {error}")
                    
                    if hf_validation['warnings']:
                        for warning in hf_validation['warnings']:
                            results.append(f"   ⚠️ {warning}")
                
                summary = f"📊 SUMMARY: {valid_models}/{total_models} models valid"
                if valid_models == total_models:
                    summary = f"✅ {summary} - All models verified!"
                else:
                    summary = f"⚠️ {summary} - Some models have issues"
                
                results.append(f"\n{summary}")
                results.append(f"💡 Using official HuggingFace checksums for maximum reliability")
                
                return "\n".join(results)
                
            except Exception as e:
                return f"❌ Validation error: {str(e)}"
        
        def cleanup_invalid_models():
            """Clean up invalid models with confirmation"""
            try:
                validator = WanModelValidator()
                models = validator.discover_models()
                
                if not models:
                    return "❌ No models found to validate."
                
                # Find invalid models
                invalid_models = []
                results = []
                results.append("🔍 Checking all models for corruption...")
                results.append("=" * 50)
                
                for model in models:
                    from pathlib import Path
                    model_path = Path(model['path'])
                    
                    # Use HuggingFace validation if possible, otherwise basic validation
                    hf_validation = validator.validate_against_huggingface_checksums(model_path)
                    
                    if hf_validation['checked_files']:
                        # Use HuggingFace validation result
                        is_valid = hf_validation['valid']
                        errors = hf_validation['errors']
                    else:
                        # Fall back to basic validation
                        validation_result = validator.validate_model_integrity(model_path)
                        is_valid = validation_result['valid']
                        errors = validation_result['errors']
                    
                    if not is_valid:
                        invalid_models.append({
                            'name': model['name'],
                            'path': model['path'],
                            'size': model['size_formatted'],
                            'errors': errors
                        })
                
                if not invalid_models:
                    return "✅ All models passed validation! No cleanup needed."
                
                results.append(f"\n⚠️ Found {len(invalid_models)} invalid model(s):")
                for i, invalid in enumerate(invalid_models, 1):
                    results.append(f"\n{i}. {invalid['name']} ({invalid['size']})")
                    results.append(f"   Issues: {', '.join(invalid['errors'])}")
                
                results.append(f"\n🗑️ Use the 'Clean Up Invalid Models' button to remove these automatically.")
                results.append("⚠️ This action will permanently delete the invalid model directories!")
                
                return "\n".join(results)
                
            except Exception as e:
                return f"❌ Cleanup scan error: {str(e)}"
        
        def compute_model_checksums():
            """Compute checksums for all model files"""
            try:
                validator = WanModelValidator()
                models = validator.discover_models()
                
                if not models:
                    return "❌ No models found to checksum.", {}
                
                results = []
                results.append("🔐 Computing checksums for all models...")
                results.append("=" * 60)
                
                checksums = {}
                
                for model in models:
                    results.append(f"\n📁 {model['name']}:")
                    model_checksums = {}
                    
                    from pathlib import Path
                    model_path = Path(model['path'])
                    
                    # Compute checksums for important files
                    important_files = [
                        "diffusion_pytorch_model.safetensors",
                        "diffusion_pytorch_model-00001-of-00007.safetensors",
                        "models_t5_umt5-xxl-enc-bf16.pth",
                        "Wan2.1_VAE.pth",
                        "config.json"
                    ]
                    
                    for file_name in important_files:
                        file_path = model_path / file_name
                        if file_path.exists():
                            file_hash = validator.compute_file_hash(file_path)
                            if file_hash:
                                model_checksums[file_name] = file_hash
                                results.append(f"   ✅ {file_name}: {file_hash[:16]}...")
                            else:
                                results.append(f"   ❌ {file_name}: Failed to compute hash")
                    
                    checksums[model['name']] = model_checksums
                
                results.append(f"\n✅ Checksum computation complete!")
                results.append("💾 Full checksums available in the Model Details output below.")
                
                return "\n".join(results), checksums
                
            except Exception as e:
                return f"❌ Checksum error: {str(e)}", {}
        
        def full_integrity_check():
            """Comprehensive integrity check with HuggingFace checksum validation"""
            try:
                validator = WanModelValidator()
                models = validator.discover_models()
                
                if not models:
                    return "❌ No models found for integrity check.", {}
                
                results = []
                results.append("🔍 COMPREHENSIVE INTEGRITY CHECK WITH OFFICIAL CHECKSUMS")
                results.append("=" * 70)
                
                all_details = {}
                overall_status = "✅ ALL GOOD"
                
                for model in models:
                    results.append(f"\n📁 {model['name']} ({model['size_formatted']}):")
                    results.append("-" * 50)
                    
                    from pathlib import Path
                    model_path = Path(model['path'])
                    
                    # Run HuggingFace checksum validation first
                    hf_validation = validator.validate_against_huggingface_checksums(model_path)
                    
                    model_details = {
                        'path': model['path'],
                        'size': model['size_formatted'],
                        'type': model['type'],
                        'hf_checksum_validation': hf_validation,
                        'basic_validation': None
                    }
                    
                    # Report HuggingFace validation results
                    if hf_validation['valid']:
                        checked_count = len(hf_validation['checked_files'])
                        valid_count = sum(1 for f in hf_validation['checked_files'].values() if f['status'] == 'valid')
                        results.append(f"   ✅ HuggingFace Checksum Validation: {valid_count}/{checked_count} files verified")
                        
                        for file_name, file_info in hf_validation['checked_files'].items():
                            if file_info['status'] == 'valid':
                                results.append(f"      ✅ {file_name}: Official checksum verified")
                            else:
                                results.append(f"      ❌ {file_name}: Checksum mismatch")
                                overall_status = "⚠️ CHECKSUM ISSUES FOUND"
                    else:
                        results.append(f"   ❌ HuggingFace Checksum Validation: FAILED")
                        overall_status = "⚠️ CHECKSUM ISSUES FOUND"
                        for error in hf_validation['errors']:
                            results.append(f"      🚨 {error}")
                    
                    if hf_validation['warnings']:
                        for warning in hf_validation['warnings']:
                            results.append(f"      ⚠️ {warning}")
                    
                    # Only run basic validation if HuggingFace validation had issues
                    if not hf_validation['valid'] or hf_validation['warnings']:
                        basic_validation = validator.validate_model_integrity(model_path)
                        model_details['basic_validation'] = basic_validation
                        
                        if basic_validation['valid']:
                            results.append(f"   ✅ Basic Structure Validation: PASS")
                        else:
                            results.append(f"   ❌ Basic Structure Validation: FAIL")
                            for error in basic_validation['errors']:
                                results.append(f"      🚨 {error}")
                    
                    all_details[model['name']] = model_details
                
                results.insert(1, f"🎯 OVERALL STATUS: {overall_status}")
                results.append(f"\n💡 **Using Official HuggingFace Checksums for Maximum Reliability**")
                results.append(f"💾 Detailed results saved to Model Details output.")
                
                return "\n".join(results), all_details
                
            except Exception as e:
                return f"❌ Integrity check error: {str(e)}", {}
        
        # Connect validation buttons if they exist
        validation_buttons = [
            ('validate_models_btn', wan_validate_models),
            ('cleanup_invalid_btn', cleanup_invalid_models),
            ('compute_checksums_btn', compute_model_checksums),
            ('verify_integrity_btn', full_integrity_check)
        ]
        
        for button_name, callback_fn in validation_buttons:
            if button_name in locals():
                # Determine output based on function return signature
                if button_name in ['compute_checksums_btn', 'verify_integrity_btn']:
                    # These functions return tuple (text, dict)
                    locals()[button_name].click(
                        fn=callback_fn,
                        inputs=[],
                        outputs=[locals()['validation_output'], locals()['model_details_output']]
                    )
                else:
                    # These functions return just text
                    locals()[button_name].click(
                        fn=callback_fn,
                        inputs=[],
                        outputs=[locals()['validation_output']]
                    )
                print(f"✅ Connected {button_name}")
            
        print("✅ All Wan model validation buttons connected")
        
    except ImportError:
        print("⚠️ WanModelValidator not available - validation buttons will not work")
    except Exception as e:
        print(f"⚠️ Failed to set up validation buttons: {e}")

    return locals()
