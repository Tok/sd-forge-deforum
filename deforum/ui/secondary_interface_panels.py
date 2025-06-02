from deforum.config.args import DeforumOutputArgs, get_component_names, get_settings_component_names
from modules.shared import opts, state
from modules.ui import create_output_panel, wrap_gradio_call
from modules.call_queue import wrap_gradio_gpu_call
from deforum.core.run_deforum import run_deforum
from deforum.config.settings import save_settings, load_all_settings, load_video_settings, get_default_settings_path, update_settings_path
from deforum.utils.core_utilities import get_deforum_version, get_commit_date, debug_print
from .main_interface_panels import setup_deforum_left_side_ui
from deforum_extend_paths import deforum_sys_extend
import gradio as gr

def on_ui_tabs():
    # extend paths using sys.path.extend so we can access all of our files and folders
    deforum_sys_extend()
    
    try:
        # set text above generate button
        style = '"text-align:center;font-weight:bold;margin-bottom:0em"'
        extension_url = "https://github.com/Tok/sd-forge-deforum"
        link = f"<a href='{extension_url}' target='_blank'>Zirteqs Fluxabled Fork</a>"
        extension_name = f"{link} of the Deforum Fork for WebUI Forge"

        commit_info = f"Git commit: {get_deforum_version()}"
        i1_store_backup = f"<p style={style}>{extension_name} - Version: {get_commit_date()} | {commit_info}</p>"
        i1_store = i1_store_backup

        debug_print("Creating Gradio interface...")
        
        try:
            debug_print("Creating Deforum interface with proper structure...")
            
            with gr.Blocks(analytics_enabled=False) as deforum_interface:
                debug_print("Inside Gradio Blocks context...")
                
                # Header
                style = '"text-align:center;font-weight:bold;margin-bottom:0em"'
                extension_url = "https://github.com/Tok/sd-forge-deforum"
                link = f"<a href='{extension_url}' target='_blank'>Zirteqs Fluxabled Fork</a>"
                extension_name = f"{link} of the Deforum Fork for WebUI Forge"
                commit_info = f"Git commit: {get_deforum_version()}"
                i1_store_backup = f"<p style={style}>{extension_name} - Version: {get_commit_date()} | {commit_info}</p>"
                
                with gr.Row(elem_id='deforum_progress_row', equal_height=False, variant='compact'):
                    with gr.Column(scale=1, variant='panel'):
                        # Try to set up the left side UI, but catch any errors
                        try:
                            debug_print("Setting up left side UI...")
                            components = setup_deforum_left_side_ui()
                            debug_print(f"Left side UI setup completed, got {len(components) if isinstance(components, dict) else 0} components")
                        except Exception as e:
                            print(f"⚠️ Warning: Could not set up full left side UI: {e}")
                            components = {}
                            # Create minimal fallback components
                            with gr.HTML("<h3>⚠️ Basic Deforum Interface</h3>"):
                                pass
                            with gr.HTML("<p>Some components could not load. Using simplified interface.</p>"):
                                pass
                        
                    with gr.Column(scale=1, variant='compact'):
                        # Right side - video display and controls
                        with gr.Row(variant='compact'):
                            btn = gr.Button("Click here after generation to show the video")
                            close_btn = gr.Button("Close the video", visible=False)
                        
                        with gr.Row(variant='compact'):
                            i1 = gr.HTML(i1_store_backup, elem_id='deforum_header')
                        
                        # Control buttons
                        id_part = 'deforum'
                        with gr.Row(elem_id=f"{id_part}_generate_box", variant='compact'):
                            skip = gr.Button('Pause/Resume', elem_id=f"{id_part}_skip", visible=False)
                            interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt", visible=True)
                            submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
                        
                        # Output panel
                        try:
                            res = create_output_panel("deforum", opts.outdir_img2img_samples)
                            deforum_gallery = res.gallery
                            generation_info = res.generation_info
                            html_info = res.html_log
                        except Exception as e:
                            print(f"⚠️ Warning: Could not create output panel: {e}")
                            deforum_gallery = gr.Gallery(label="Generated Images")
                            generation_info = gr.Textbox(label="Generation Info")
                            html_info = gr.HTML()
                        
                        # Settings controls
                        with gr.Row(variant='compact'):
                            settings_path = gr.Textbox(get_default_settings_path(), elem_id='deforum_settings_path', label="Settings File")
                        with gr.Row(variant='compact'):
                            save_settings_btn = gr.Button('Save Settings')
                            load_settings_btn = gr.Button('Load All Settings')
                            load_video_settings_btn = gr.Button('Load Video Settings')
                
                # Set up basic event handlers safely
                debug_print("Setting up essential event handlers...")
                
                # Simple interrupt handler - only set up if component exists and has _id
                try:
                    if hasattr(interrupt, '_id') and hasattr(state, 'interrupt'):
                        interrupt.click(fn=lambda: state.interrupt(), inputs=None, outputs=None)
                        debug_print("Interrupt handler connected")
                except Exception as e:
                    print(f"⚠️ Could not connect interrupt handler: {e}")
                
                # Simple skip handler - only set up if component exists and has _id
                try:
                    if hasattr(skip, '_id') and hasattr(state, 'skip'):
                        skip.click(fn=lambda: state.skip(), inputs=None, outputs=None)
                        debug_print("Skip handler connected")
                except Exception as e:
                    print(f"⚠️ Could not connect skip handler: {e}")
                
                # CRITICAL FIX: Connect the Generate button to run_deforum
                try:
                    debug_print("Setting up Generate button...")
                    
                    # Get all component names for input collection
                    component_names = get_component_names()
                    debug_print(f"Component names: {component_names[:10]}...")  # Show first 10
                    
                    # Create input list matching run_deforum expectations exactly
                    input_components = []
                    
                    # Add fixed inputs that run_deforum expects (args[0] and args[1])
                    input_components.append(gr.State(value="deforum_job"))  # job_id 
                    input_components.append(gr.State(value="placeholder"))  # second placeholder
                    
                    # Add ALL components from component_names in exact order (args[2] onwards)
                    for i, name in enumerate(component_names):
                        if name in components and components[name] is not None:
                            comp = components[name]
                            if hasattr(comp, '_id'):
                                input_components.append(comp)
                                debug_print(f"Added component {name} at position {i+2}")
                            else:
                                # Add a dummy component for invalid ones
                                input_components.append(gr.State(value=None))
                                debug_print(f"Added dummy for invalid component {name} at position {i+2}")
                        else:
                            # Add a dummy component for missing ones
                            input_components.append(gr.State(value=None))
                            debug_print(f"Added dummy for missing component {name} at position {i+2}")
                    
                    debug_print(f"Prepared {len(input_components)} input components for Generate button")
                    debug_print(f"Expected {len(component_names) + 2} total inputs")
                    
                    # Set up the generate button click handler
                    submit.click(
                        fn=wrap_gradio_gpu_call(run_deforum, extra_outputs=[None, '', '']),
                        inputs=input_components,
                        outputs=[
                            deforum_gallery,
                            generation_info,
                            html_info,
                            html_info
                        ],
                        show_progress=True
                    )
                    
                    debug_print("✅ Generate button connected successfully!")
                    
                except Exception as e:
                    print(f"⚠️ Failed to connect Generate button: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Connect settings buttons
                try:
                    if components:
                        # Create a simple list of all components for settings operations
                        all_components_list = []
                        for name in get_component_names():
                            if name in components and components[name] is not None:
                                all_components_list.append(components[name])
                        
                        if all_components_list:
                            # Save settings
                            save_settings_btn.click(
                                fn=lambda *args: save_settings(settings_path.value, *args),
                                inputs=[settings_path] + all_components_list,
                                outputs=[html_info]
                            )
                            
                            # Load all settings
                            load_settings_btn.click(
                                fn=lambda path: load_all_settings(path),
                                inputs=[settings_path],
                                outputs=all_components_list + [html_info]
                            )
                            
                            # Load video settings
                            load_video_settings_btn.click(
                                fn=lambda path: load_video_settings(path),
                                inputs=[settings_path],
                                outputs=all_components_list + [html_info]
                            )
                            
                            debug_print("Settings buttons connected")
                        else:
                            debug_print("No components available for settings buttons")
                except Exception as e:
                    print(f"⚠️ Could not connect settings buttons: {e}")
                
                debug_print("Interface setup completed successfully")
                
            debug_print("Deforum interface created successfully!")
            return [(deforum_interface, "Deforum", "deforum_interface")]
        except Exception as e:
            import traceback
            print(f"⚠️ CRITICAL: Failed to create Deforum extension UI: {e}")
            print(f"⚠️ Full traceback:")
            print(traceback.format_exc())
            print(f"⚠️ Creating minimal fallback interface...")
            
            # Create a minimal fallback interface that should always work
            try:
                with gr.Blocks(analytics_enabled=False) as minimal_interface:
                    gr.HTML("<h2>⚠️ Deforum Extension Partially Loaded</h2>")
                    gr.HTML("<p>The extension encountered issues during full UI creation but is attempting to load with basic functionality.</p>")
                    gr.HTML(f"<p>Error: {str(e)}</p>")
                    
                return [(minimal_interface, "Deforum", "deforum_interface")]
            except Exception as fallback_error:
                print(f"⚠️ Even fallback interface failed: {fallback_error}")
                return []
    except Exception as e:
        print(f"⚠️ CRITICAL: Failed to create Deforum extension UI: {e}")
        print(f"⚠️ Creating minimal fallback interface...")
        
        # Create a minimal fallback interface that should always work
        try:
            with gr.Blocks(analytics_enabled=False) as minimal_interface:
                gr.HTML("<h2>⚠️ Deforum Extension Partially Loaded</h2>")
                gr.HTML("<p>The extension encountered issues during full UI creation but is attempting to load with basic functionality.</p>")
                gr.HTML(f"<p>Error: {str(e)}</p>")
                
            return [(minimal_interface, "Deforum", "deforum_interface")]
        except Exception as fallback_error:
            print(f"⚠️ Even fallback interface failed: {fallback_error}")
            return []