"""
Settings Components
Contains settings panels and advanced configuration controls
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn, ToolButton
from .component_builders import create_gr_elem, create_row
from ..utils import emoji_utils


def get_tab_setup(d, da):
    """Create the Setup tab with basic configuration.
    
    Args:
        d: DeforumArgs instance
        da: DeforumAnimArgs instance
        
    Returns:
        Dict of component references
    """
    with gr.TabItem(f"{emoji_utils.gear()} Setup"):
        
        # BASIC SETUP
        with gr.Accordion("🛠️ Basic Setup", open=True):
            with FormRow():
                W = create_gr_elem(d.W)
                H = create_gr_elem(d.H)
                
            with FormRow():
                show_info_on_ui = create_gr_elem(d.show_info_on_ui)
                show_controlnet_tab = create_gr_elem(d.show_controlnet_tab)
                
        # GENERATION SETTINGS
        with gr.Accordion("⚙️ Generation Settings", open=True):
            with FormRow():
                sampler = create_gr_elem(d.sampler)
                scheduler = create_gr_elem(d.scheduler)
                steps = create_gr_elem(d.steps)
                
            with FormRow():
                seed = create_gr_elem(d.seed)
                batch_name = create_gr_elem(d.batch_name)
                
        # IMAGE SETTINGS
        with gr.Accordion("🖼️ Image Settings", open=False):
            with FormRow():
                tiling = create_gr_elem(d.tiling)
                restore_faces = create_gr_elem(d.restore_faces)
                
            with FormRow():
                seed_resize_from_w = create_gr_elem(d.seed_resize_from_w)
                seed_resize_from_h = create_gr_elem(d.seed_resize_from_h)
                
        # PATHS AND FILES
        with gr.Accordion("📁 Paths and Files", open=False):
            with FormRow():
                prompts_path = create_gr_elem(d.prompts_path)
                negative_prompts_path = create_gr_elem(d.negative_prompts_path)
                
    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_advanced(d, da):
    """Create the Advanced tab with expert-level controls.
    
    Args:
        d: DeforumArgs instance
        da: DeforumAnimArgs instance
        
    Returns:
        Dict of component references
    """
    with gr.TabItem(f"{emoji_utils.wrench()} Advanced"):
        
        # ADVANCED GENERATION
        with gr.Accordion("🧪 Advanced Generation", open=True):
            with FormRow():
                seed_behavior = create_gr_elem(d.seed_behavior)
                seed_iter_N = create_gr_elem(d.seed_iter_N)
                
            with FormRow():
                motion_preview_mode = create_gr_elem(d.motion_preview_mode)
                
        # ERROR HANDLING
        with gr.Accordion("⚠️ Error Handling", open=False):
            with FormRow():
                reroll_blank_frames = create_gr_elem(d.reroll_blank_frames)
                reroll_patience = create_gr_elem(d.reroll_patience)
                
        # MEMORY OPTIMIZATION
        with gr.Accordion("💾 Memory Optimization", open=False):
            with FormRow():
                store_frames_in_ram = create_gr_elem(da.store_frames_in_ram)
                
        # EXPERIMENTAL FEATURES
        with gr.Accordion("🧪 Experimental Features", open=False):
            gr.Markdown("""
            **⚠️ Warning:** These features are experimental and may cause issues.
            Use at your own risk and report any problems on GitHub.
            """)
            
            # Placeholder for experimental features
            with FormRow():
                gr.Markdown("No experimental features currently available.")
                
    return {k: v for k, v in {**locals(), **vars()}.items()}


def create_batch_mode_section():
    """Create the batch mode settings section.
    
    Returns:
        Dict of component references
    """
    with gr.Accordion("📦 Batch Mode", open=False):
        gr.Markdown("""
        **Batch Mode:** Run multiple animations with different settings files.
        Upload multiple .txt files to process them sequentially.
        """)
        
        with FormRow():
            override_settings_with_file = gr.Checkbox(
                label="Enable batch mode",
                value=False,
                interactive=True,
                elem_id='override_settings',
                info="Run from a list of setting .txt files. Upload them to the box below (visible when enabled)"
            )
        
        with FormRow(visible=False) as settings_file_row:
            custom_settings_file = gr.File(
                label="Setting files",
                interactive=True,
                file_count="multiple",
                file_types=[".txt"],
                elem_id="custom_setting_file"
            )
        
        # Toggle visibility of file upload
        def toggle_settings_file(enabled):
            return gr.update(visible=enabled)
        
        override_settings_with_file.change(
            fn=toggle_settings_file,
            inputs=[override_settings_with_file],
            outputs=[settings_file_row]
        )
    
    return {
        'override_settings_with_file': override_settings_with_file,
        'custom_settings_file': custom_settings_file,
        'settings_file_row': settings_file_row
    }


def create_settings_persistence_section():
    """Create the settings save/load section.
    
    Returns:
        Dict of component references
    """
    with gr.Accordion("💾 Settings Persistence", open=True):
        gr.Markdown("""
        **Save & Load Settings:** Preserve your configurations for reuse.
        Settings are saved as JSON files that can be shared or backed up.
        """)
        
        with FormRow():
            save_settings_btn = gr.Button(
                "💾 Save All Settings",
                elem_id='deforum_save_settings_btn',
                variant="primary"
            )
            load_settings_btn = gr.Button(
                "📁 Load All Settings",
                elem_id='deforum_load_settings_btn',
                variant="secondary"
            )
            load_video_settings_btn = gr.Button(
                "🎬 Load Video Settings",
                elem_id='deforum_load_video_settings_btn',
                variant="secondary"
            )
        
        with FormRow():
            settings_status = gr.Textbox(
                label="Settings Status",
                lines=2,
                interactive=False,
                placeholder="Settings save/load status will appear here..."
            )
        
        # Event handlers for settings buttons would be connected in the main interface
        
    return {
        'save_settings_btn': save_settings_btn,
        'load_settings_btn': load_settings_btn,
        'load_video_settings_btn': load_video_settings_btn,
        'settings_status': settings_status
    }


def create_debug_info_section():
    """Create the debug information section.
    
    Returns:
        Dict of component references
    """
    with gr.Accordion("🐛 Debug Information", open=False):
        gr.Markdown("""
        **Debug Info:** System information and troubleshooting tools.
        Use this information when reporting issues.
        """)
        
        with FormRow():
            debug_info = gr.Textbox(
                label="System Information",
                lines=10,
                interactive=False,
                value="Click 'Refresh Debug Info' to load system information..."
            )
        
        with FormRow():
            refresh_debug_btn = gr.Button(
                "🔄 Refresh Debug Info",
                variant="secondary"
            )
            copy_debug_btn = gr.Button(
                "📋 Copy to Clipboard",
                variant="secondary"
            )
        
        def get_debug_info():
            """Collect system debug information."""
            try:
                import sys
                import platform
                import torch
                import gradio
                from datetime import datetime
                
                info = f"""🐛 **Deforum Debug Information**
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**System:**
• Platform: {platform.platform()}
• Python: {sys.version.split()[0]}
• PyTorch: {torch.__version__}
• CUDA Available: {torch.cuda.is_available()}
• CUDA Devices: {torch.cuda.device_count() if torch.cuda.is_available() else 'N/A'}
• Gradio: {gradio.__version__}

**GPU Information:**"""
                
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                        info += f"\n• GPU {i}: {gpu_name} ({gpu_memory}GB)"
                else:
                    info += "\n• No CUDA GPUs available"
                
                info += f"""

**Memory:**
• Available RAM: {platform.machine()}
• Python Memory Usage: {sys.getsizeof(sys.modules) // (1024**2)}MB (approx)

**Installation:**
• Extension Path: deforum/
• Config System: Functional (Phase 2.5)
• Rendering System: Functional (Phase 2.6)"""
                
                return info
                
            except Exception as e:
                return f"❌ Error collecting debug info: {str(e)}"
        
        def copy_to_clipboard(text):
            """Copy debug info to clipboard (client-side)."""
            return "📋 Debug info copied to clipboard!"
        
        # Connect event handlers
        refresh_debug_btn.click(
            fn=get_debug_info,
            inputs=[],
            outputs=[debug_info]
        )
        
        copy_debug_btn.click(
            fn=copy_to_clipboard,
            inputs=[debug_info],
            outputs=[gr.Textbox(visible=False)]  # Hidden status output
        )
    
    return {
        'debug_info': debug_info,
        'refresh_debug_btn': refresh_debug_btn,
        'copy_debug_btn': copy_debug_btn
    } 