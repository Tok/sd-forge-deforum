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
            from .wan_simple_integration import WanSimpleIntegration
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

    return locals()
