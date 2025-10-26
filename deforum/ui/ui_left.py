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
from deforum.config.defaults import get_gradio_html
from deforum.ui.gradio_funcs import change_css, handle_change_functions
from deforum.config.args import DeforumArgs, DeforumAnimArgs, ParseqArgs, AudioSyncArgs, DeforumOutputArgs, RootArgs, LoopArgs, WanArgs
from deforum.utils.system.logging import emoji as emoji_utils
# TEMPORARILY DISABLED: ControlNet support disabled until Flux-specific reimplementation
# from .deforum_controlnet import setup_controlnet_ui
from deforum.ui.ui_elements import (get_tab_run, get_tab_keyframes, get_tab_prompts, get_tab_init,
                          get_tab_output, get_tab_masking)

def set_arg_lists():
    # convert dicts to NameSpaces for easy working (args.param instead of args['param']
    d = SimpleNamespace(**DeforumArgs())  # default args
    da = SimpleNamespace(**DeforumAnimArgs())  # default anim args
    dp = SimpleNamespace(**ParseqArgs())  # default parseq ars
    dau = SimpleNamespace(**AudioSyncArgs())  # default audio sync args
    dv = SimpleNamespace(**DeforumOutputArgs())  # default video args
    dr = SimpleNamespace(**RootArgs())  # ROOT args
    dw = SimpleNamespace(**WanArgs())  # Wan args
    dloopArgs = SimpleNamespace(**LoopArgs())  # Guided imgs args
    return d, da, dp, dau, dv, dr, dw, dloopArgs

def wan_generate_video():
    """
    Simple placeholder function for Wan video generation button
    Returns a status message indicating the feature is integrated but needs models
    """
    try:
        print("🎬 Wan video generation button clicked!")
        
        # Try to discover models to check if setup is complete
        try:
            from deforum.integrations.wan.wan_simple_integration import WanSimpleIntegration
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
3. Choose animation mode 'Flux + Interpolation' in the Keyframes tab
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
    d, da, dp, dau, dv, dr, dw, dloopArgs = set_arg_lists()

    # FLUX AVAILABILITY CHECK - All Deforum modes require Flux
    from deforum.utils.system.flux_check import should_show_flux_blocker, get_flux_setup_message

    if should_show_flux_blocker():
        # Show blocker message and minimal UI
        with gr.Row(variant='compact'):
            show_info_on_ui = gr.Checkbox(label="Show more info", value=d.show_info_on_ui, interactive=True, visible=False)

        gr.HTML(value=get_flux_setup_message())

        # Return minimal component set for compatibility
        return {
            'show_info_on_ui': show_info_on_ui,
        }

    # Normal UI setup continues if Flux is available
    # show button to hide/ show gradio's info texts for each element in the UI
    with gr.Row(variant='compact'):
        show_info_on_ui = gr.Checkbox(label="Show more info", value=d.show_info_on_ui, interactive=True)

    # Top-level essential settings before tabs - NEW RENDER MODE SYSTEM
    with gr.Row(variant='compact'):
        from .ui_elements import create_gr_elem
        render_mode = create_gr_elem(da.render_mode)

    with gr.Row(variant='compact'):
        fps = create_gr_elem(dv.fps)
        steps = create_gr_elem(d.steps)

    # Mode-dependent controls row - cadence OR pseudo-cadence (mutually exclusive)
    with gr.Row(variant='compact'):
        with gr.Column(scale=1, visible=True) as cadence_column:
            cadence = create_gr_elem(da.diffusion_cadence)
        with gr.Column(scale=1, visible=False) as pseudo_cadence_column:
            pseudo_cadence_display = gr.Textbox(
                label="Calculated Pseudo-Cadence",
                value="Will be calculated on render",
                interactive=False,
                info="Average frames between diffusions (calculated from keyframes)"
            )

    # Strength schedules - smart sliders + textboxes
    # Sliders for quick constant values, textboxes for complex schedules
    with gr.Row(variant='compact'):
        # Normal strength (Classic 3D, New 3D)
        with gr.Column(scale=1, visible=True) as normal_strength_column:
            normal_strength_slider = gr.Slider(
                label="Normal Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.05,  # Will be updated dynamically based on steps
                value=0.85,
                info="Resolution: 1/20 = 0.05. Slider sets constant '0:(value)'. Edit textbox for complex schedules."
            )
            normal_strength = create_gr_elem(da.strength_schedule)

        # Keyframe strength (New 3D, Keyframes Only, Flux + Interpolation)
        with gr.Column(scale=1, visible=True) as keyframe_strength_column:
            keyframe_strength_slider = gr.Slider(
                label="Keyframe Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.05,  # Will be updated dynamically based on steps
                value=0.50,
                info="Resolution: 1/20 = 0.05. Slider sets constant '0:(value)'. Edit textbox for complex schedules."
            )
            keyframe_strength = create_gr_elem(da.keyframe_strength_schedule)

    # Keep legacy animation_mode hidden for backwards compatibility
    animation_mode = gr.Radio(
        label="Animation mode (Legacy - Hidden)",
        choices=['3D', 'Interpolation', 'Flux + Interpolation'],
        value="3D",
        visible=False
    )

    with gr.Blocks():
        with gr.Tabs() as main_tabs:
            # Get main tab contents in new workflow order:
            # Tabs visible in all modes:
            tab_init_params = get_tab_init(d, da, dp, dau, dv)  # 1. Init - all modes
            from .ui_elements import get_tab_distribution, get_tab_shakify, get_tab_depth_warping
            tab_distribution_params = get_tab_distribution(da)  # 2. Distribution - all modes
            tab_prompts_params = get_tab_prompts(da, dw, dv)  # 3. Prompts - all modes (now includes audio/timing)
            tab_keyframes_params = get_tab_keyframes(d, da, dloopArgs)  # 4. Keyframes - all modes

            # Mode-specific tabs (with visibility control):
            # 3D mode only tabs:
            with gr.TabItem(f"{emoji_utils.hole()} 3D Depth", visible=True) as tab_depth:
                tab_depth_params = get_tab_depth_warping(da, skip_tabitem=True)  # 5. 3D Depth - 3D mode only
            with gr.TabItem(f"{emoji_utils.bicycle()} Shakify", visible=True) as tab_shakify:
                tab_shakify_params = get_tab_shakify(da, skip_tabitem=True)  # 6. Shakify - 3D mode only
            with gr.TabItem(f"{emoji_utils.masking()} Masking", visible=True) as tab_masking:
                tab_masking_params = get_tab_masking(d, da, skip_tabitem=True)  # 7. Masking - all modes

            # Flux + Interpolation mode tab:
            from .ui_elements import get_tab_wan
            with gr.TabItem(f"{emoji_utils.wan_video()} Interpolation", visible=True) as tab_wan:
                tab_wan_params = get_tab_wan(dw, skip_tabitem=True)  # 8. Interpolation - Flux + Interpolation mode

            # Always visible tabs:
            tab_run_params = get_tab_run(d, da)  # 8. Run - all modes
            tab_output_params = get_tab_output(da, dv)  # 9. Output - all modes

            # TEMPORARILY DISABLED: ControlNet support disabled until Flux-specific reimplementation
            controlnet_dict = {}  # Empty dict for backwards compatibility

            # add returned gradio elements from main tabs to locals()
            for key, value in {**tab_run_params, **tab_keyframes_params, **tab_distribution_params, **tab_prompts_params, **tab_shakify_params, **tab_masking_params, **tab_depth_params, **tab_init_params, **controlnet_dict, **tab_wan_params, **tab_output_params}.items():
                locals()[key] = value

            # Add top-level settings to locals()
            locals()['render_mode'] = render_mode
            locals()['animation_mode'] = animation_mode
            locals()['fps'] = fps
            locals()['steps'] = steps
            locals()['cadence'] = cadence
            locals()['diffusion_cadence'] = cadence  # Alias for backward compatibility with gradio_funcs
            locals()['cadence_column'] = cadence_column
            locals()['pseudo_cadence_display'] = pseudo_cadence_display
            locals()['pseudo_cadence_column'] = pseudo_cadence_column
            locals()['strength_schedule'] = normal_strength
            locals()['keyframe_strength_schedule'] = keyframe_strength
            locals()['normal_strength_slider'] = normal_strength_slider
            locals()['keyframe_strength_slider'] = keyframe_strength_slider
            locals()['normal_strength_column'] = normal_strength_column
            locals()['keyframe_strength_column'] = keyframe_strength_column

            # Store tab references for visibility control
            locals()['tab_depth'] = tab_depth
            locals()['tab_shakify'] = tab_shakify
            locals()['tab_wan'] = tab_wan

    # Mode change handler - updates all UI based on selected render mode
    def handle_render_mode_change(mode):
        """
        Update UI components when render mode changes.
        Returns updates for: tab_depth, tab_shakify, tab_wan, cadence, pseudo_cadence,
                             fps, steps, sliders, strength_columns, animation_mode
        """
        from deforum.rendering.data.render_mode import RenderMode

        render_mode_enum = RenderMode.from_string(mode)
        config = render_mode_enum.config

        # Determine tab visibility
        show_3d_tabs = render_mode_enum.should_show_3d_tabs()
        show_wan_tab = render_mode_enum.should_show_wan_tab()

        # Determine cadence/pseudo-cadence visibility
        show_real_cadence = render_mode_enum.should_show_cadence_slider()
        show_pseudo_cadence = config.shows_pseudo_cadence

        # Determine strength slider visibility
        show_normal_strength = render_mode_enum.should_show_normal_strength()
        show_keyframe_strength = render_mode_enum.should_show_keyframe_strength()

        # Calculate strength resolution (slider step size) based on steps
        strength_step = 1.0 / config.default_steps
        strength_info = f"Resolution: 1/{config.default_steps} = {strength_step:.4f}. Slider sets constant '0:(value)'. Edit textbox for complex schedules."

        # Mode-specific steps info text
        steps_info_map = {
            RenderMode.CLASSIC_3D: "Sampling steps for all diffusions (every cadence frames)",
            RenderMode.NEW_3D: "Sampling steps for all diffusions (keyframes + cadence frames)",
            RenderMode.KEYFRAMES_ONLY: "Sampling steps for keyframe diffusions only",
            RenderMode.FLUX_WAN: "Sampling steps for Flux keyframe generation (Wan FLF2V steps in Wan Models tab)",
        }
        steps_info = steps_info_map.get(render_mode_enum, "Sampling steps for diffusion")

        # Update legacy animation_mode for backward compatibility
        legacy_mode = render_mode_enum.to_legacy_animation_mode()

        return [
            gr.update(visible=show_3d_tabs),           # tab_depth
            gr.update(visible=show_3d_tabs),           # tab_shakify
            gr.update(visible=show_wan_tab),           # tab_wan
            gr.update(visible=show_real_cadence),      # cadence_column
            gr.update(visible=show_pseudo_cadence),    # pseudo_cadence_column
            gr.update(value=config.default_fps),       # fps
            gr.update(value=config.default_steps,      # steps
                     info=steps_info),
            gr.update(step=strength_step, info=strength_info),  # normal_strength_slider
            gr.update(step=strength_step, info=strength_info),  # keyframe_strength_slider
            gr.update(visible=show_normal_strength),   # normal_strength_column
            gr.update(visible=show_keyframe_strength), # keyframe_strength_column
            gr.update(value=legacy_mode)               # animation_mode (hidden)
        ]

    # Connect render mode change handler
    render_mode.change(
        fn=handle_render_mode_change,
        inputs=[render_mode],
        outputs=[
            tab_depth,
            tab_shakify,
            tab_wan,
            cadence_column,
            pseudo_cadence_column,
            fps,
            steps,
            normal_strength_slider,
            keyframe_strength_slider,
            normal_strength_column,
            keyframe_strength_column,
            animation_mode
        ]
    )

    # Smart strength slider handlers - ONE-WAY SYNC to prevent infinite loops
    # Slider -> Textbox ONLY. Textbox is source of truth.

    def update_slider_step_size(steps_value):
        """Update slider step size based on steps value."""
        step = 1.0 / max(1, steps_value)
        info_text = f"Resolution: 1/{steps_value} = {step:.4f}. Slider sets constant '0:(value)'. Edit textbox for complex schedules."
        return [
            gr.update(step=step, info=info_text),  # normal_strength_slider
            gr.update(step=step, info=info_text),  # keyframe_strength_slider
        ]

    def slider_to_textbox(slider_value):
        """Convert slider value to schedule textbox format."""
        return f"0: ({slider_value:.2f})"

    # Connect steps change to update slider step size
    steps.change(
        fn=update_slider_step_size,
        inputs=[steps],
        outputs=[normal_strength_slider, keyframe_strength_slider]
    )

    # Connect sliders to textboxes (ONE-WAY: slider -> textbox only)
    # This prevents infinite loops. Textbox is the source of truth.
    normal_strength_slider.change(
        fn=slider_to_textbox,
        inputs=[normal_strength_slider],
        outputs=[normal_strength]
    )

    keyframe_strength_slider.change(
        fn=slider_to_textbox,
        inputs=[keyframe_strength_slider],
        outputs=[keyframe_strength]
    )

    # Gradio's Change functions - hiding and renaming elements based on other elements
    show_info_on_ui.change(fn=change_css, inputs=show_info_on_ui, outputs=[gr.HTML()])
    handle_change_functions(locals())

    # Wire up audio upload to use actual FPS and update max_frames
    if 'audio_upload' in locals() and 'soundtrack_path' in locals():
        def handle_audio_upload_with_fps(audio_filepath, current_fps):
            """Save uploaded audio, calculate duration, and suggest max_frames."""
            if audio_filepath is None:
                return None, "File", "", gr.update()

            import os
            import shutil
            from pathlib import Path

            # Create output/audio directory
            output_dir = Path("output/audio")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded file
            filename = Path(audio_filepath).name
            dest_path = output_dir / filename
            shutil.copy2(audio_filepath, dest_path)
            abs_path = str(dest_path.absolute())

            # Calculate audio duration and suggested max_frames
            try:
                import librosa
                duration = librosa.get_duration(path=abs_path)
                fps_val = current_fps if current_fps and current_fps > 0 else 24
                suggested_max_frames = int(duration * fps_val)
                info_text = f"Duration: {duration:.2f}s | Suggested max_frames @ {fps_val} FPS: {suggested_max_frames}"

                # Auto-update max_frames
                max_frames_update = gr.update(value=suggested_max_frames)
            except Exception as e:
                info_text = f"Could not calculate duration: {str(e)}"
                max_frames_update = gr.update()

            return abs_path, "File", info_text, max_frames_update

        # Re-wire the upload event with actual FPS and max_frames
        if 'max_frames' in locals():
            locals()['audio_upload'].upload(
                fn=handle_audio_upload_with_fps,
                inputs=[locals()['audio_upload'], fps],
                outputs=[
                    locals()['soundtrack_path'],
                    locals()['add_soundtrack'],
                    locals()['audio_info_display'],
                    locals()['max_frames']
                ]
            )

    # Wire up audio sync button to synchronize prompts with audio events
    if 'audio_sync_button' in locals() and 'audio_sync_status' in locals():
        def synchronize_prompts_to_audio(
            soundtrack_path_val,
            audio_sync_prompts_val,
            distribution_mode,
            target_count,
            detection_method,
            frequency_band,
            sensitivity,
            intensity_threshold,
            min_spacing_frames,
            current_fps
        ):
            """Detect audio events and distribute prompts across them."""
            try:
                from pathlib import Path
                import json

                # Validate soundtrack path
                if not soundtrack_path_val or not Path(soundtrack_path_val).exists():
                    return gr.update(), "✗ Error: Please upload an audio file first"

                # Parse user prompts
                from deforum.audio import parse_prompt_list
                user_prompts = parse_prompt_list(audio_sync_prompts_val)

                if not user_prompts:
                    return gr.update(), "✗ Error: Please enter at least one prompt"

                # Load and analyze audio
                from deforum.audio import (
                    process_audio_for_detection,
                    detect_events,
                    generate_keyframes_from_events,
                    distribute_prompts_across_keyframes,
                    suggest_keyframe_count_from_audio
                )
                import librosa

                # Load audio file
                y, sr = librosa.load(soundtrack_path_val, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                fps_val = current_fps if current_fps and current_fps > 0 else 24

                # Process audio for detection
                y_processed = process_audio_for_detection(
                    y, sr,
                    frequency_band=frequency_band,
                    lowpass_cutoff=4000,
                    distortion_gain=10.0
                )

                # Detect events
                event_times = detect_events(
                    y_processed, sr,
                    method=detection_method,
                    sensitivity=sensitivity / 100.0  # Convert 0-100 to 0-1
                )

                if not event_times:
                    return gr.update(), "✗ Error: No audio events detected. Try adjusting sensitivity."

                # Determine keyframe count
                if target_count and target_count > 0:
                    num_keyframes = int(target_count)
                else:
                    num_keyframes = suggest_keyframe_count_from_audio(
                        duration, fps_val,
                        desired_prompts=len(user_prompts)
                    )

                # Limit to detected events
                num_keyframes = min(num_keyframes, len(event_times))

                # Generate keyframes from events
                keyframes = generate_keyframes_from_events(
                    event_times[:num_keyframes],
                    fps=fps_val,
                    min_spacing_frames=min_spacing_frames,
                    intensity_threshold=intensity_threshold / 100.0
                )

                if not keyframes:
                    return gr.update(), "✗ Error: No keyframes generated after filtering. Try reducing min spacing."

                # Distribute prompts across keyframes
                animation_prompts_json = distribute_prompts_across_keyframes(
                    keyframes, user_prompts, mode=distribution_mode
                )

                # Parse back to add formatting
                schedule = json.loads(animation_prompts_json)
                formatted_schedule = json.dumps(schedule, indent=2)

                status_msg = (
                    f"✓ Successfully synchronized!\n"
                    f"• Audio: {duration:.1f}s @ {fps_val} FPS\n"
                    f"• Events detected: {len(event_times)}\n"
                    f"• Keyframes created: {len(keyframes)}\n"
                    f"• Prompts used: {len(user_prompts)} (mode: {distribution_mode})"
                )

                return formatted_schedule, status_msg

            except Exception as e:
                import traceback
                traceback.print_exc()
                return gr.update(), f"✗ Error: {str(e)}"

        # Get all required inputs
        audio_sync_inputs = []
        required_components = [
            'soundtrack_path',
            'audio_sync_prompts',
            'audio_prompt_distribution_mode',
            'audio_target_keyframe_count',
            'audio_detection_method',
            'audio_frequency_band',
            'audio_sensitivity',
            'audio_intensity_threshold',
            'audio_min_spacing_frames',
            'fps'
        ]

        for comp_name in required_components:
            if comp_name in locals():
                audio_sync_inputs.append(locals()[comp_name])
            else:
                print(f"⚠️ Warning: Audio sync component '{comp_name}' not found")

        if len(audio_sync_inputs) == len(required_components):
            locals()['audio_sync_button'].click(
                fn=synchronize_prompts_to_audio,
                inputs=audio_sync_inputs,
                outputs=[
                    locals()['animation_prompts'],  # Update prompts in Prompts tab
                    locals()['audio_sync_status']
                ]
            )
            print("🎵 Audio sync button connected successfully")
        else:
            print(f"⚠️ Could not connect audio sync button: missing components")

    # Wire up AI prompt generation button
    if 'audio_ai_generate_button' in locals():
        def generate_prompts_with_ai(count, theme):
            """Generate escalating prompts using Qwen."""
            try:
                from deforum.wan.qwen_prompt_expander import QwenPromptExpander

                # Initialize Qwen (will auto-select model based on VRAM)
                qwen = QwenPromptExpander()

                # Create prompt for Qwen to generate escalating prompts
                generation_prompt = f"""Generate {int(count)} creative and escalating prompts for an animated video featuring a {theme}.

Requirements:
- Each prompt should be a simple, descriptive phrase
- Start with calm/static scenes and progressively increase action/intensity
- Focus on varied actions, poses, and settings
- Keep each prompt concise (5-10 words)
- Return ONLY the prompts, one per line
- No numbering, no explanations

Example for "bunny":
bunny in forest
bunny hopping gently
bunny sitting by tree
bunny looking around curiously
bunny jumping energetically

Now generate {int(count)} prompts for: {theme}"""

                # Generate with Qwen
                print(f"🤖 Generating {count} prompts for theme: {theme}")
                result = qwen.enhance_prompt(generation_prompt)

                # Clean up the result (remove any numbering or extra formatting)
                lines = [line.strip() for line in result.split('\n') if line.strip()]
                # Filter out lines that look like numbering or explanations
                prompts = []
                for line in lines:
                    # Skip lines with numbering like "1.", "1)", etc.
                    if line[0].isdigit() and (line[1] == '.' or line[1] == ')'):
                        line = line[2:].strip()
                    if line and not line.startswith('#') and not line.startswith('//'):
                        prompts.append(line)

                # Take only the requested count
                prompts = prompts[:int(count)]

                # Join with newlines
                prompts_text = '\n'.join(prompts)

                print(f"✓ Generated {len(prompts)} prompts")
                return prompts_text

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = f"Error generating prompts: {str(e)}\n\nUsing fallback prompts..."
                # Fallback to simple template-based generation
                actions = ["resting", "moving slowly", "looking around", "moving quickly", "jumping"]
                fallback = '\n'.join([f"{theme} {action}" for action in actions[:int(count)]])
                print(f"⚠️ {error_msg}")
                return fallback

        if 'audio_ai_prompt_count' in locals() and 'audio_ai_prompt_theme' in locals() and 'audio_sync_prompts' in locals():
            locals()['audio_ai_generate_button'].click(
                fn=generate_prompts_with_ai,
                inputs=[
                    locals()['audio_ai_prompt_count'],
                    locals()['audio_ai_prompt_theme']
                ],
                outputs=[locals()['audio_sync_prompts']]
            )
            print("✨ AI prompt generation button connected successfully")

    # Set up Wan Generate button if it exists - with better error handling
    if 'wan_generate_button' in locals() and 'wan_generation_status' in locals():
        try:
            print("🔗 Connecting Wan generate button...")
            
            # Import the real Wan generation function from ui_elements
            from .ui_elements import wan_generate_video as wan_generate_video_main
            
            # Get all component values to pass to the Wan generation function
            from deforum.config.args import get_component_names
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

    # Set up Auto-Assign Keyframe Types button
    if 'auto_assign_keyframe_types_btn' in locals() and 'keyframe_type_schedule' in locals():
        try:
            print("🔗 Connecting auto-assign keyframe types button...")

            from .ui_elements import auto_assign_keyframe_types_handler

            # Check if animation_prompts and wan_flf2v_chunk_size are available
            if 'animation_prompts' in locals() and 'wan_flf2v_chunk_size' in locals():
                locals()['auto_assign_keyframe_types_btn'].click(
                    fn=auto_assign_keyframe_types_handler,
                    inputs=[
                        locals()['animation_prompts'],
                        locals()['wan_flf2v_chunk_size']
                    ],
                    outputs=[locals()['keyframe_type_schedule']]
                )
                print("✅ Auto-assign keyframe types button connected successfully")
            else:
                print("⚠️ Warning: animation_prompts or wan_flf2v_chunk_size not found in locals()")
        except Exception as e:
            print(f"⚠️ Warning: Failed to connect auto-assign keyframe types button: {e}")
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
        from deforum.integrations.wan.wan_model_validator import WanModelValidator
        
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
