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

from .args import DeforumOutputArgs, get_component_names, get_settings_component_names
from modules.shared import opts, state
from modules.ui import create_output_panel, wrap_gradio_call
from modules.call_queue import wrap_gradio_gpu_call
from .run_deforum import run_deforum
from .settings import save_settings, load_all_settings, load_video_settings, get_default_settings_path, update_settings_path
from .general_utils import get_deforum_version, get_commit_date
from .ui_left import setup_deforum_left_side_ui
from scripts.deforum_extend_paths import deforum_sys_extend
import gradio as gr

def on_ui_tabs():
    # extend paths using sys.path.extend so we can access all of our files and folders
    deforum_sys_extend()
    # set text above generate button
    style = '"text-align:center;font-weight:bold;margin-bottom:0em"'
    extension_url = "https://github.com/Tok/sd-forge-deforum"
    link = f"<a href='{extension_url}' target='_blank'>Zirteqs Fluxabled Fork</a>"
    extension_name = f"{link} of the Deforum Fork for WebUI Forge"

    commit_info = f"Git commit: {get_deforum_version()}"
    i1_store_backup = f"<p style={style}>{extension_name} - Version: {get_commit_date()} | {commit_info}</p>"
    i1_store = i1_store_backup

    with gr.Blocks(analytics_enabled=False) as deforum_interface:
        components = {}
        dummy_component = gr.Button(visible=False)
        with gr.Row(elem_id='deforum_progress_row', equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='panel'):
                # setting the left side of the ui:
                components = setup_deforum_left_side_ui()
            with gr.Column(scale=1, variant='compact'):
                with gr.Row(variant='compact'):
                    btn = gr.Button("Click here after the generation to show the video")
                    components['btn'] = btn
                    close_btn = gr.Button("Close the video", visible=False)
                with gr.Row(variant='compact'):
                    i1 = gr.HTML(i1_store, elem_id='deforum_header')
                    components['i1'] = i1
                    def show_vid(): # Show video button related func
                        from .run_deforum import last_vid_data # get latest vid preview data (this import needs to stay inside the function!)
                        return {
                            i1: gr.update(value=last_vid_data, visible=True),
                            close_btn: gr.update(visible=True),
                            btn: gr.update(value="Update the video", visible=True),
                        }
                    btn.click(
                        fn=show_vid,
                        inputs=[],
                        outputs=[i1, close_btn, btn],
                        )
                    def close_vid(): # Close video button related func
                        return {
                            i1: gr.update(value=i1_store_backup, visible=True),
                            close_btn: gr.update(visible=False),
                            btn: gr.update(value="Click here after the generation to show the video", visible=True),
                        }
                    
                    close_btn.click(
                        fn=close_vid,
                        inputs=[],
                        outputs=[i1, close_btn, btn],
                        )
                id_part = 'deforum'
                with gr.Row(elem_id=f"{id_part}_generate_box", variant='compact'):
                    skip = gr.Button('Pause/Resume', elem_id=f"{id_part}_skip", visible=False)
                    interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt", visible=True)
                    interrupting = gr.Button('Interrupting...', elem_id=f"{id_part}_interrupting", elem_classes="generate-box-interrupting", tooltip="Interrupting generation...")
                    submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                    skip.click(
                        fn=lambda: state.skip(),
                        inputs=[],
                        outputs=[],
                    )

                    interrupt.click(
                        fn=lambda: state.interrupt(),
                        inputs=[],
                        outputs=[],
                    )

                    interrupting.click(
                        fn=lambda: state.interrupt(),
                        inputs=[],
                        outputs=[],
                    )
                
                res = create_output_panel("deforum", opts.outdir_img2img_samples)
                
                #deforum_gallery, generation_info, html_info, _ 

                generation_info = res.generation_info
                html_info= res.html_log
                deforum_gallery = res.gallery

                with gr.Row(variant='compact'):
                    settings_path = gr.Textbox(get_default_settings_path(), elem_id='deforum_settings_path', label="Settings File", info="Settings are automatically loaded on startup. Path can be relative to webui folder OR full/absolute.")
                with gr.Row(variant='compact'):
                    save_settings_btn = gr.Button('Save Settings', elem_id='deforum_save_settings_btn')
                    load_settings_btn = gr.Button('Load All Settings', elem_id='deforum_load_settings_btn')
                    load_video_settings_btn = gr.Button('Load Video Settings', elem_id='deforum_load_video_settings_btn')

        # Handle missing components by creating dummy values for them
        # This prevents KeyErrors while maintaining backward compatibility
        missing_components = {}
        for name in get_component_names():
            if name not in components:
                print(f"⚠️ Creating dummy component for missing: {name}")
                # Create a dummy component - use a hidden textbox as a safe default
                missing_components[name] = gr.Textbox(value="", visible=False, elem_id=f"dummy_{name}")
        
        # Merge dummy components with actual components
        all_components = {**components, **missing_components}
        
        component_list = [all_components[name] for name in get_component_names()]

        submit.click(
                    fn=wrap_gradio_gpu_call(run_deforum),
                    _js="submit_deforum",
                    inputs=[dummy_component, dummy_component] + component_list,
                    outputs=[
                         deforum_gallery,
                         components["resume_timestring"],
                         generation_info,
                         html_info                 
                    ],
                )
        
        settings_component_names = get_settings_component_names()
        settings_missing_components = {}
        for name in settings_component_names:
            if name not in all_components:
                print(f"⚠️ Creating dummy settings component for missing: {name}")
                settings_missing_components[name] = gr.Textbox(value="", visible=False, elem_id=f"dummy_settings_{name}")
        
        # Merge settings dummy components
        all_settings_components = {**all_components, **settings_missing_components}
        
        settings_component_list = [all_settings_components[name] for name in settings_component_names]
        video_settings_component_list = [all_settings_components[name] for name in list(DeforumOutputArgs().keys())]

        save_settings_btn.click(
            fn=wrap_gradio_call(save_settings),
            inputs=[settings_path] + settings_component_list + video_settings_component_list,
            outputs=[],
        )
        
        # Create a path update function
        def path_updating_load_settings(*args):
            path = args[0]
            settings_path.value = path
            return load_all_settings(*args)
            
        load_settings_btn.click(
            fn=wrap_gradio_call(path_updating_load_settings),
            inputs=[settings_path] + settings_component_list,
            outputs=settings_component_list,
        )

        # Create a path update function for video settings
        def path_updating_load_video_settings(*args):
            path = args[0]
            settings_path.value = path
            return load_video_settings(*args)
            
        load_video_settings_btn.click(
            fn=wrap_gradio_call(path_updating_load_video_settings),
            inputs=[settings_path] + video_settings_component_list,
            outputs=video_settings_component_list,
        )
        
    # handle settings loading on UI launch
    def trigger_load_general_settings():
        print("Loading general settings...")
        
        # First check if deforum_settings.txt exists in webui root
        import os
        from modules import paths_internal
        webui_root_settings = os.path.join(paths_internal.script_path, "deforum_settings.txt")
        
        # Determine the settings file to load
        if os.path.isfile(webui_root_settings):
            # Use the settings file from webui root if it exists
            settings_file_path = webui_root_settings
            print(f"Loading existing settings from webui root: {settings_file_path}")
        else:
            # Fall back to default settings provided by the fork
            settings_file_path = get_default_settings_path()
            print(f"No settings found in webui root, using default settings from: {settings_file_path}")
        
        # Update the settings path field with the path
        settings_path.value = settings_file_path
        
        # Now call load_all_settings with ui_launch=True to update all components
        wrapped_fn = wrap_gradio_call(lambda *args, **kwargs: load_all_settings(*args, ui_launch=True, **kwargs))
        inputs = [settings_file_path] + [component.value for component in settings_component_list]
        outputs = settings_component_list
        updated_values = wrapped_fn(*inputs, *outputs)[0]
        
        # Update all the component values
        settings_component_name_to_obj = {name: component for name, component in zip(get_settings_component_names(), settings_component_list)}
        for key, value in updated_values.items():
            if key in settings_component_name_to_obj:
                settings_component_name_to_obj[key].value = value['value']
    # Always load settings on startup - either from persistent settings path (if enabled),
    # from webui root, or from the fork's default settings
    trigger_load_general_settings()
        
    return [(deforum_interface, "Deforum", "deforum_interface")]