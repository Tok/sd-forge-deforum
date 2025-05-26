def get_tab_wan(dw: SimpleNamespace):
    """WAN 2.1 Video Generation Tab - Integrated with Deforum Schedules"""
    with gr.TabItem(f"{emoji_utils.wan_video()} WAN Video"):
        # WAN Info Accordion
        with gr.Accordion("WAN 2.1 Video Generation Info & Setup", open=False):
            gr.HTML(value=get_gradio_html('wan_video'))
        
        # Auto-Discovery Section - NEW
        with gr.Accordion("🔍 Auto-Discovery (No Manual Paths Needed!)", open=True):
            gr.Markdown("""
            **🚀 NEW: Smart Model Discovery**
            
            WAN models are now **automatically discovered** from common locations:
            - `models/wan/`
            - `models/WAN/` 
            - `models/Wan/`
            - HuggingFace cache
            - Downloads folder
            
            **No manual path configuration required!** Just download your model and it will be found automatically.
            """)
            
            # Model size preference
            wan_model_size = create_row(dw.wan_model_size)
            
            # Information about model sizes
            gr.Markdown("""
            **Model Size Information:**
            - **1.3B (Recommended)**: ~17GB download, faster generation, lower VRAM usage, more stable
            - **14B (High Quality)**: ~75GB download, slower generation, higher VRAM usage, better quality
            
            💡 **Tip**: Start with 1.3B to test if WAN works on your system before downloading the larger model.
            """)
        
        # Hidden model path for compatibility (auto-populated by discovery)
        wan_model_path = gr.Textbox(visible=False, value="auto-discovery")
        
        with gr.Accordion("Basic WAN Settings", open=True):
            gr.Markdown("""
            **🔗 Integrated with Deforum Schedules:**
            - **Seed**: Now uses the Deforum seed schedule from the Keyframes tab
            - **FPS**: Uses the FPS setting from the Output tab slider
            - **Prompts**: Uses your prompt schedule from the Prompts tab
            """)
            
            wan_resolution = create_row(dw.wan_resolution)
        
            with FormRow():
                wan_inference_steps = create_gr_elem(dw.wan_inference_steps)
                wan_guidance_scale = create_gr_elem(dw.wan_guidance_scale)
        
        with gr.Accordion("Advanced WAN Settings", open=False):
            with FormRow():
                wan_frame_overlap = create_gr_elem(dw.wan_frame_overlap)
                wan_motion_strength = create_gr_elem(dw.wan_motion_strength)
                
            with FormRow():
                wan_enable_interpolation = create_gr_elem(dw.wan_enable_interpolation)
                wan_interpolation_strength = create_gr_elem(dw.wan_interpolation_strength)
        
        # Dedicated WAN Generate Button
        with gr.Accordion("Generate WAN Video", open=True):
            gr.Markdown("""
            **🎯 Fully Integrated with Deforum:**
            - **Prompts**: Uses your prompt schedule from the **Prompts tab**
            - **Seed**: Uses the seed schedule from the **Keyframes → Seed & SubSeed tab**
            - **FPS**: Uses the FPS setting from the **Output tab**
            - **Duration**: Calculated automatically from your prompt schedule
            
            **✨ New Features:**
            - No redundant checkbox - just click Generate!
            - Seamless integration with all Deforum settings
            - Auto-discovery finds your models automatically
            - Smart model selection based on your preference
            """)
            
            with FormRow():
                wan_generate_button = gr.Button(
                    "🎬 Generate WAN Video",
                    variant="primary", 
                    size="lg",
                    elem_id="wan_generate_button"
                )
                
            # Status output for WAN generation
            wan_generation_status = gr.Textbox(
                label="Generation Status",
                interactive=False,
                lines=2,
                placeholder="Ready to generate WAN video using Deforum schedules..."
            )
            
            # Don't connect the button here - it will be connected in ui_left.py
            # with access to all components
            pass
        
        with gr.Accordion("📥 Easy Model Download (AUTO-DISCOVERY)", open=False):
            gr.Markdown("""
            ### 🚀 SUPER EASY SETUP - Just Download & Go!
            
            **NEW**: Models are automatically discovered - no path configuration needed!
            
            #### 🎯 Quick Start (Recommended)
            ```bash
            # Install HuggingFace CLI (if not already installed)
            pip install huggingface_hub
            
            # Download 1.3B model (recommended for testing)
            huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan
            ```
            
            #### 🏆 High Quality Option
            ```bash
            # Download 14B model (higher quality, needs more VRAM)
            huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan
            ```
            
            ### 📁 Where to Place Models
            The auto-discovery will find models in any of these locations:
            - `models/wan/` ← **Recommended location**
            - `models/WAN/`
            - `models/Wan/`
            - HuggingFace cache (automatic)
            - Downloads folder
            
            ### 🔗 Integration Features
            
            #### ✅ What's New:
            - **🔍 Auto-Discovery**: No more manual path configuration!
            - **🎯 Seed Integration**: Uses Deforum's seed schedule automatically
            - **📊 FPS Integration**: Uses FPS from Output tab instead of separate slider
            - **📝 Prompt Integration**: Works seamlessly with your Deforum prompt schedule
            - **🚀 Direct Integration**: Uses official WAN repository
            - **📦 Easy Download**: Simple one-command setup
            - **💪 Better Stability**: Much more reliable than previous versions
            
            #### 🆘 Troubleshooting
            If generation fails:
            1. **Check models**: Run `python scripts/deforum_helpers/wan_direct_integration.py`
            2. **Download missing models**: Use commands above
            3. **Verify placement**: Models should be in `models/wan/` directory
            4. **Check logs**: Look for auto-discovery messages in console
            5. **Verify schedules**: Make sure you have prompts in the Prompts tab
            6. **Check seed behavior**: Set seed behavior to 'schedule' if you want custom seed scheduling
            """)
        
        with gr.Accordion("🔗 Deforum Schedule Integration", open=False):
            gr.Markdown("""
            ### 🎯 How WAN Integrates with Deforum Schedules
            
            WAN video generation now uses Deforum's scheduling system for perfect integration:
            
            #### 📝 Prompt Schedule Integration
            - WAN reads your prompts from the **Prompts tab**
            - Each prompt with a frame number becomes a video clip
            - Duration is calculated from the frame differences
            - Example: `{"0": "beach sunset", "120": "forest morning"}` creates two clips
            
            #### 🎲 Seed Schedule Integration  
            - WAN uses the **seed schedule** from Keyframes → Seed & SubSeed
            - Set **Seed behavior** to 'schedule' to enable custom seed scheduling
            - Example: `0:(12345), 60:(67890)` uses different seeds for different clips
            - Leave as 'iter' or 'random' for automatic seed management
            
            #### 🎬 FPS Integration
            - WAN uses the **FPS setting** from the Output tab
            - No separate FPS slider needed - one setting controls everything
            - Ensures video timing matches your intended frame rate
            
            #### ⏱️ Duration Calculation
            - Video duration = (frame_difference / fps) seconds per clip
            - Example: Frames 0→120 at 30fps = 4 second clip
            - WAN automatically pads to optimal frame counts (4n+1 format)
            
            ### 🛠️ Setup Guide
            
            #### Step 1: Configure Prompts
            ```json
            {
                "0": "a serene beach at sunset",
                "90": "a misty forest in the morning", 
                "180": "a bustling city street at night"
            }
            ```
            
            #### Step 2: Set FPS (Output Tab)
            - Choose your desired FPS (e.g., 30 or 60)
            - This affects both timing and video quality
            
            #### Step 3: Configure Seeds (Optional)
            - **For consistent seeds**: Set seed behavior to 'schedule'
            - **For variety**: Leave as 'iter' or 'random'
            
            #### Step 4: Generate
            - Click "Generate WAN Video" button
            - WAN reads all settings from Deforum automatically
            - Each prompt becomes a seamless video clip
            
            ### 🎯 Benefits of Integration
            - **Consistency**: All timing controlled by one FPS setting
            - **Flexibility**: Full power of Deforum's scheduling system
            - **Simplicity**: No duplicate settings or confusion
            - **Precision**: Exact frame timing for audio synchronization
            - **Power**: Complex animations possible through scheduling
            """)
            
    return {k: v for k, v in {**locals(), **vars()}.items()}
