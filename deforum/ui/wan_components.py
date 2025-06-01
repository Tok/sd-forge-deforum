"""
WAN Components
Contains WAN-specific interfaces and event handlers
"""

import gradio as gr
from types import SimpleNamespace
from modules.ui_components import FormRow, FormColumn
from .component_builders import create_gr_elem, create_row
from .rendering.util import emoji_utils


def get_tab_wan(dw: SimpleNamespace):
    """Create the WAN AI tab with advanced AI video generation controls.
    
    Args:
        dw: WanArgs instance
        
    Returns:
        Dict of component references
    """
    with gr.TabItem(f"🤖 Wan AI"):
        
        # ESSENTIAL PROMPTS SECTION - TOP PRIORITY
        with gr.Accordion("📝 Wan Video Prompts (REQUIRED)", open=True):
            gr.Markdown("""
            **🎯 Essential for Wan Generation:** These prompts define what video clips will be generated.
            
            **Quick Setup:** Load → Analyze Movement → Enhance → Generate
            """)
            
            # Prompt Loading Buttons
            with FormRow():
                load_deforum_to_wan_btn = gr.Button(
                    "📋 Load from Deforum Prompts",
                    variant="primary",
                    size="lg",
                    elem_id="load_deforum_to_wan_btn"
                )
                load_wan_defaults_btn = gr.Button(
                    "📝 Load Default Wan Prompts",
                    variant="secondary", 
                    size="lg",
                    elem_id="load_wan_defaults_btn"
                )
            
            # Wan Prompts Display - ALWAYS VISIBLE AND PROMINENT
            wan_enhanced_prompts = gr.Textbox(
                label="Wan Video Prompts (JSON Format)",
                lines=10,
                interactive=True,
                placeholder='REQUIRED: Load prompts first! Click "Load from Deforum Prompts" or "Load Default Wan Prompts" above.',
                info="🎯 ESSENTIAL: These prompts will be used for Wan video generation. Edit manually or use buttons below to enhance.",
                elem_id="wan_enhanced_prompts_textbox"
            )
            
            # Prompt Enhancement Actions
            with FormRow():
                analyze_movement_btn = gr.Button(
                    "📐 Add Movement Descriptions",
                    variant="secondary",
                    size="lg",
                    elem_id="wan_analyze_movement_btn"
                )
                enhance_prompts_btn = gr.Button(
                    "🎨 AI Prompt Enhancement",
                    variant="secondary",
                    size="lg",
                    elem_id="wan_enhance_prompts_btn"
                )
            
            # Camera Shakify Integration Control
            with FormRow():
                wan_enable_shakify = gr.Checkbox(
                    label="🎬 Include Camera Shakify with Movement Analysis",
                    value=True,
                    info="Enable Camera Shakify integration for movement analysis (uses settings from Keyframes → Motion → Shakify tab)",
                    elem_id="wan_enable_shakify_checkbox"
                )
                wan_movement_sensitivity_override = gr.Checkbox(
                    label="Manual Sensitivity Override",
                    value=False,
                    info="Override auto-calculated sensitivity (normally auto-calculated from movement magnitude)",
                    elem_id="wan_sensitivity_override_checkbox"
                )
            
            # Manual Sensitivity Control (hidden by default)
            with FormRow(visible=False) as manual_sensitivity_row:
                wan_manual_sensitivity = gr.Slider(
                    label="Manual Movement Sensitivity",
                    minimum=0.1,
                    maximum=5.0,
                    step=0.1,
                    value=1.0,
                    info="Higher values detect subtler movements (0.1: only large movements, 5.0: very sensitive)",
                    elem_id="wan_manual_sensitivity_slider"
                )
            
            # Movement Analysis Results
            wan_movement_description = gr.Textbox(
                label="Movement Analysis Results",
                lines=6,
                interactive=False,
                placeholder="Movement analysis results will appear here...\n\n💡 TIP: This shows frame-by-frame movement detection with Camera Shakify integration.",
                info="Fine-grained movement descriptions with specific frame ranges and Camera Shakify effects.",
                elem_id="wan_movement_description_textbox",
                visible=True
            )
            
            # Enhancement Progress
            enhancement_progress = gr.Textbox(
                label="AI Enhancement Progress",
                lines=3,
                interactive=False,
                placeholder="AI enhancement progress will show here...",
                info="Shows real-time progress during prompt enhancement.",
                elem_id="wan_enhancement_progress_textbox",
                visible=True
            )
        
        # DEFORUM INTEGRATION
        _create_deforum_integration_section()
        
        # GENERATION SECTION
        wan_generate_button, wan_generation_status = _create_generation_section()
        
        # ESSENTIAL SETTINGS
        wan_resolution = _create_essential_settings_section(dw)
        
        # AI ENHANCEMENT SETTINGS
        _create_ai_enhancement_section(dw)
        
        # MODEL SETTINGS
        _create_model_settings_section(dw)
        
        # OVERRIDES SECTION
        movement_sensitivity_override, wan_movement_sensitivity = _create_overrides_section(dw)
        
        # ADVANCED SETTINGS
        wan_flash_attention_status, check_flash_attention_btn, wan_flash_attention_mode = _create_advanced_settings_section(dw)
        
        # QWEN MODEL MANAGEMENT
        qwen_model_status, check_qwen_models_btn, download_qwen_model_btn, cleanup_qwen_cache_btn = _create_qwen_management_section()
        
        # AUTO-DISCOVERY INFO
        _create_autodiscovery_section()
        
        # Hidden components for compatibility
        wan_model_path = gr.Textbox(visible=False, value="auto-discovery")
        wan_seed = gr.Number(
            precision=dw.wan_seed["precision"], 
            value=dw.wan_seed["value"],
            visible=False
        )
        
        # DOCUMENTATION
        _create_documentation_section()
        
        # CONNECT EVENT HANDLERS
        _connect_wan_event_handlers(
            locals(), wan_enhanced_prompts, analyze_movement_btn, enhance_prompts_btn,
            wan_enable_shakify, wan_movement_sensitivity_override, wan_manual_sensitivity,
            manual_sensitivity_row, wan_movement_description, enhancement_progress,
            wan_generate_button, wan_generation_status, load_deforum_to_wan_btn,
            load_wan_defaults_btn, movement_sensitivity_override, wan_movement_sensitivity,
            check_flash_attention_btn, wan_flash_attention_status, wan_flash_attention_mode,
            check_qwen_models_btn, download_qwen_model_btn, cleanup_qwen_cache_btn,
            qwen_model_status
        )
        
    return {k: v for k, v in {**locals(), **vars()}.items()}


def _create_deforum_integration_section():
    """Create the Deforum integration information section."""
    with gr.Accordion("🔗 Deforum Integration", open=False):
        gr.Markdown("""
        **✅ Wan seamlessly integrates with your Deforum settings:**
        
        - **📝 Prompts:** Uses prompts from Deforum Prompts tab
        - **🎬 Movement:** Uses same movement schedules as normal Deforum renders
        - **🎲 Seed & CFG:** Uses Deforum's seed and CFG schedules
        - **💪 Strength:** Uses Deforum's strength schedule for I2V continuity
        - **🎬 FPS:** Uses Output tab FPS setting
        
        **Movement Integration:**
        - ✅ Translation X/Y/Z, Rotation 3D X/Y/Z, Zoom schedules
        - ✅ **Parseq schedules fully supported**
        - ✅ Movement descriptions automatically calculated and added
        - ✅ Motion intensity dynamically adapts to movement complexity
        """)


def _create_generation_section():
    """Create the generation section."""
    with gr.Accordion("🎬 Generate Wan Video", open=True):
        # Generate Button with Validation
        with FormRow():
            wan_generate_button = gr.Button(
                "🎬 Generate Wan Video (I2V Chaining)",
                variant="primary", 
                size="lg",
                elem_id="wan_generate_button"
            )
            
        # Status output for Wan generation
        wan_generation_status = gr.Textbox(
            label="Generation Status",
            interactive=False,
            lines=5,
            placeholder="⚠️ Prompts required! Load prompts above first, then click Generate.",
            info="Status updates will appear here during generation."
        )
    
    return wan_generate_button, wan_generation_status


def _create_essential_settings_section(dw):
    """Create the essential settings section."""
    with gr.Accordion("⚙️ Essential Settings", open=True):
        with FormRow():
            wan_auto_download = create_gr_elem(dw.wan_auto_download)
            wan_preferred_size = create_gr_elem(dw.wan_preferred_size)
            wan_resolution_elem = create_gr_elem(dw.wan_resolution)
            
            # Update resolution value to handle old format
            def update_resolution_format(current_value):
                """Convert old resolution format to new format with labels"""
                if current_value and '(' not in current_value:
                    if current_value == "864x480":
                        return "864x480 (Landscape)"
                    elif current_value == "480x864":
                        return "480x864 (Portrait)"
                    elif current_value == "1280x720":
                        return "1280x720 (Landscape)"
                    elif current_value == "720x1280":
                        return "720x1280 (Portrait)"
                return current_value
            
            # Apply format update
            if hasattr(wan_resolution_elem, 'value'):
                wan_resolution_elem.value = update_resolution_format(wan_resolution_elem.value)
            
            wan_resolution = wan_resolution_elem
            
        with FormRow():
            wan_inference_steps = gr.Slider(
                label="Inference Steps",
                minimum=5,
                maximum=100,
                step=1,
                value=20,
                elem_id="wan_inference_steps_fixed_min_5",
                info="Steps for generation quality (5-15: fast, 20-50: quality)"
            )
    
    return wan_resolution


def _create_ai_enhancement_section(dw):
    """Create the AI enhancement settings section."""
    with gr.Accordion("🧠 AI Prompt Enhancement (Optional)", open=False):
        gr.Markdown("""
        **Enhance prompts using Qwen AI models** for better video quality:
        - **🧠 AI Enhancement**: Refines and expands prompts
        - **🎬 Movement Integration**: Uses movement descriptions from analysis
        - **🌍 Multi-Language**: English and Chinese support
        """)
        
        with FormRow():
            wan_qwen_model = create_gr_elem(dw.wan_qwen_model)
            wan_qwen_language = create_gr_elem(dw.wan_qwen_language)
            wan_qwen_auto_download = create_gr_elem(dw.wan_qwen_auto_download)


def _create_model_settings_section(dw):
    """Create the model settings section."""
    with gr.Accordion("🔧 Model Settings", open=False):
        with FormRow():
            wan_t2v_model = create_gr_elem(dw.wan_t2v_model)
            wan_i2v_model = create_gr_elem(dw.wan_i2v_model)
        with FormRow():
            wan_model_path = create_gr_elem(dw.wan_model_path)


def _create_overrides_section(dw):
    """Create the override settings section."""
    with gr.Accordion("🔧 Override Settings (Advanced)", open=False):
        gr.Markdown("""
        **Override automatic calculations with fixed values:**
        
        By default, Wan calculates these values from your Deforum schedules. Enable overrides only if you need manual control.
        """)
        
        with FormRow():
            wan_strength_override = create_gr_elem(dw.wan_strength_override)
            wan_fixed_strength = create_gr_elem(dw.wan_fixed_strength)
            
        with FormRow():
            wan_guidance_override = create_gr_elem(dw.wan_guidance_override) 
            wan_guidance_scale = create_gr_elem(dw.wan_guidance_scale)
            
        with FormRow():
            wan_motion_strength_override = create_gr_elem(dw.wan_motion_strength_override)
            wan_motion_strength = create_gr_elem(dw.wan_motion_strength)
            
        # Movement sensitivity - in overrides since it should be auto-calculated
        with FormRow():
            movement_sensitivity_override = gr.Checkbox(
                label="Movement Sensitivity Override",
                value=False,
                info="Override auto-calculated movement sensitivity from Deforum schedules"
            )
            wan_movement_sensitivity = create_gr_elem(dw.wan_movement_sensitivity)
            wan_movement_sensitivity.interactive = False  # Start disabled
    
    return movement_sensitivity_override, wan_movement_sensitivity


def _create_advanced_settings_section(dw):
    """Create the advanced settings section."""
    with gr.Accordion("⚡ Advanced Settings", open=False):
        with FormRow():
            wan_frame_overlap = create_gr_elem(dw.wan_frame_overlap)
            
        with FormRow():
            wan_enable_interpolation = create_gr_elem(dw.wan_enable_interpolation)
            wan_interpolation_strength = create_gr_elem(dw.wan_interpolation_strength)
            
        # Flash Attention Settings
        with gr.Accordion("⚡ Flash Attention Settings", open=False):
            gr.Markdown("""
            **Flash Attention Performance Control**
            
            Flash Attention provides faster and more memory-efficient attention computation.
            
            **Modes:**
            - **Auto (Recommended)**: Try Flash Attention, fall back to PyTorch if unavailable
            - **Force Flash Attention**: Force Flash Attention (fails if not available)
            - **Force PyTorch Fallback**: Always use PyTorch attention (slower but compatible)
            """)
            
            wan_flash_attention_mode = create_gr_elem(dw.wan_flash_attention_mode)
            
            # Flash Attention Status
            wan_flash_attention_status = gr.HTML(
                label="Flash Attention Status",
                value="⚠️ <span style='color: #FF9800;'>Status check unavailable</span>",
                elem_id="wan_flash_attention_status"
            )
            
            check_flash_attention_btn = gr.Button(
                "🔍 Check Flash Attention Status",
                variant="secondary",
                elem_id="wan_check_flash_attention_btn"
            )
    
    return wan_flash_attention_status, check_flash_attention_btn, wan_flash_attention_mode


def _create_qwen_management_section():
    """Create the Qwen model management section."""
    with gr.Accordion("🧠 Qwen Model Management", open=False):
        gr.Markdown("""
        **Model Information & Auto-Download Status**
        
        Monitor Qwen model availability and manage downloads:
        """)
        
        qwen_model_status = gr.HTML(
            label="Qwen Model Status",
            value="⏳ Checking model availability...",
            elem_id="wan_qwen_model_status"
        )
        
        with FormRow():
            check_qwen_models_btn = gr.Button(
                "🔍 Check Model Status",
                variant="secondary",
                elem_id="wan_check_qwen_models_btn"
            )
            download_qwen_model_btn = gr.Button(
                "📥 Download Selected Model",
                variant="primary",
                elem_id="wan_download_qwen_model_btn"
            )
            cleanup_qwen_cache_btn = gr.Button(
                "🧹 Cleanup Model Cache",
                variant="secondary",
                elem_id="wan_cleanup_qwen_cache_btn"
            )
    
    return qwen_model_status, check_qwen_models_btn, download_qwen_model_btn, cleanup_qwen_cache_btn


def _create_autodiscovery_section():
    """Create the auto-discovery information section."""
    with gr.Accordion("📥 Model Auto-Discovery & Setup", open=False):
        gr.Markdown("""
        **✅ Auto-Discovery System**
        
        Wan automatically finds models in these locations:
        - `models/wan/` (recommended)
        - `models/video/wan/`
        - Custom paths you specify
        
        **✨ VACE Models (Recommended)**
        
        VACE models handle both T2V and I2V in one model:
        - **1.3B VACE**: 480p, 8GB VRAM, fast generation
        - **14B VACE**: 480p+720p, 16GB+ VRAM, highest quality
        
        **📥 Easy Download Commands:**
        ```bash
        # Download 1.3B VACE (recommended default)
        huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/wan/Wan2.1-VACE-1.3B
        
        # Or download 14B VACE (high quality)
        huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan/Wan2.1-VACE-14B
        ```
        
        **⚠️ Legacy Models** (for compatibility):
        - T2V models: Text-to-video only
        - I2V models: Separate models for image-to-video
        - Less convenient than VACE, but still supported
        """)


def _create_documentation_section():
    """Create the detailed documentation section."""
    with gr.Accordion("📚 Detailed Documentation", open=False):
        _create_schedule_integration_docs()
        _create_movement_translation_docs()
        _create_setup_guide()
        _create_troubleshooting_guide()


def _create_schedule_integration_docs():
    """Create the schedule integration documentation."""
    with gr.Accordion("🎯 How Wan Integrates with Deforum Schedules", open=False):
        gr.Markdown("""
        ### Prompt Schedule Integration
        - Wan reads your prompts from the **Prompts tab**
        - Each prompt with a frame number becomes a video clip
        - Duration is calculated from the frame differences
        - Example: `{"0": "beach sunset", "120": "forest morning"}` creates two clips
        
        ### Seed Schedule Integration  
        - Wan uses the **seed schedule** from Keyframes → Seed & SubSeed
        - Set **Seed behavior** to 'schedule' to enable custom seed scheduling
        - Example: `0:(12345), 60:(67890)` uses different seeds for different clips
        - Leave as 'iter' or 'random' for automatic seed management
        
        ### Strength Schedule Integration
        - Wan I2V chaining supports **Deforum's strength schedule**!
        - Controls how much the previous frame influences the next clip generation
        - Found in **Keyframes → Strength tab** as "Strength schedule"
        - Higher values (0.7-0.9): Strong continuity, smoother transitions
        - Lower values (0.3-0.6): More creative freedom, less continuity
        - Example: `0:(0.85), 120:(0.6)` - strong continuity at start, more freedom later
        
        ### CFG Scale Schedule Integration
        - Wan supports **Deforum's CFG scale schedule**!
        - Controls how closely generation follows the prompt across clips
        - Found in **Keyframes → CFG tab** as "CFG scale schedule"
        - Higher values (7.5-12): Strong prompt adherence, less creative interpretation
        - Lower values (3-6): More creative interpretation, looser prompt following
        - Example: `0:(7.5), 120:(10.0)` - moderate adherence at start, stronger later
        
        ### FPS Integration
        - Wan uses the **FPS setting** from the Output tab
        - No separate FPS slider needed - one setting controls everything
        - Ensures video timing matches your intended frame rate
        
        ### Duration Calculation & Frame Management
        - Video duration = (frame_difference / fps) seconds per clip
        - Example: Frames 0→120 at 30fps = 4 second clip
        - **Wan 4n+1 Requirement**: Wan requires frame counts to follow 4n+1 format (5, 9, 13, 17, 21, etc.)
        - **Automatic Calculation**: System calculates the nearest 4n+1 value ≥ your requested frames
        - **Frame Discarding**: Extra frames are discarded from the middle to match your exact timing
        - **Display Info**: Console shows exactly which frames will be discarded before generation
        """)


def _create_movement_translation_docs():
    """Create the movement translation documentation."""
    with gr.Accordion("🎬 Movement Translation: From Deforum Schedules to Prompt Descriptions", open=False):
        gr.Markdown("""
        ### ✨ NEW: Frame-Specific Movement Analysis
        
        Wan now provides **unique movement descriptions for each prompt** based on its exact position in the video timeline, eliminating generic repetitive text.
        
        **🎯 Key Improvements:**
        - **Frame-Specific Analysis**: Each prompt analyzes movement at its specific frame range
        - **Directional Specificity**: "panning left", "tilting down", "dolly forward" instead of generic text
        - **Camera Shakify Integration**: Analyzes actual shake patterns at each frame offset
        - **Varied Descriptions**: No more identical "investigative handheld" text across all prompts
        
        ### 🔄 How Frame-Specific Analysis Works
        
        **Traditional Approach (OLD):**
        ```json
        All prompts: "camera movement with investigative handheld camera movement"
        ```
        
        **Frame-Specific Approach (NEW):**
        ```json
        {
          "0": "...with subtle panning left (sustained) and gentle moving down (extended)",
          "43": "...with moderate panning right (brief) and subtle rotating left (sustained)",
          "106": "...with gentle dolly forward (extended) and subtle rolling clockwise (brief)",
          "210": "...with subtle tilting down (extended) and moderate panning left (brief)",
          "324": "...with gentle rotating right (sustained) and subtle dolly backward (extended)"
        }
        ```
        
        ### 📊 Movement Detection & Classification
        
        **Translation Movements:**
        - **Translation X**: 
          - Increasing → "panning right"
          - Decreasing → "panning left"
        - **Translation Y**: 
          - Increasing → "moving up"
          - Decreasing → "moving down"
        - **Translation Z**: 
          - Increasing → "dolly forward"
          - Decreasing → "dolly backward"
        
        **Rotation Movements:**
        - **Rotation 3D X**: 
          - Increasing → "tilting up"
          - Decreasing → "tilting down"
        - **Rotation 3D Y**: 
          - Increasing → "rotating right"
          - Decreasing → "rotating left"
        - **Rotation 3D Z**: 
          - Increasing → "rolling clockwise"
          - Decreasing → "rolling counter-clockwise"
        
        **Zoom & Effects:**
        - **Zoom**: 
          - Increasing → "zooming in"
          - Decreasing → "zooming out"
        
        ### 🎨 Intensity & Duration Modifiers
        
        **Movement Intensity:**
        - **Subtle**: Very small movements (< 1.0 units)
        - **Gentle**: Small movements (1.0 - 10.0 units)
        - **Moderate**: Medium movements (10.0 - 50.0 units)
        - **Strong**: Large movements (> 50.0 units)
        
        **Duration Descriptions:**
        - **Brief**: Short duration (< 20% of total frames)
        - **Extended**: Medium duration (20% - 50% of total frames)
        - **Sustained**: Long duration (> 50% of total frames)
        
        ### 🎬 Camera Shakify Integration
        
        When Camera Shakify is enabled, the system:
        1. **Generates frame-specific shake data** based on the prompt's frame position
        2. **Overlays shake on Deforum schedules** (like experimental render core)
        3. **Analyzes combined movement** for each prompt's timeframe
        4. **Provides varied descriptions** that reflect actual camera behavior
        
        **Example with Camera Shakify INVESTIGATION:**
        ```
        Frame 0 prompt → Analyzes shake pattern frames 0-17
        Frame 43 prompt → Analyzes shake pattern frames 43-60
        Frame 106 prompt → Analyzes shake pattern frames 106-123
        ```
        
        ### 🔧 Smart Motion Analysis
        
        **Sensitivity Auto-Calculation:**
        The system automatically calculates optimal sensitivity based on movement magnitude:
        - **Very subtle** (< 5 units): High sensitivity (3.0)
        - **Subtle** (5-15 units): High sensitivity (2.0)
        - **Normal** (15-50 units): Standard sensitivity (1.0)
        - **Large** (50-200 units): Reduced sensitivity (0.7)
        - **Very large** (> 200 units): Low sensitivity (0.5)
        
        **Segment Grouping:**
        - Groups similar movements that occur close together
        - Reduces redundancy while preserving directional specificity
        - Creates readable, varied descriptions
        
        ### 📈 Results Comparison
        
        **Before Frame-Specific Analysis:**
        ```json
        {
          "0": "...complex camera movement with complex panning movement with 5 phases",
          "43": "...complex camera movement with complex panning movement with 5 phases",
          "106": "...complex camera movement with complex panning movement with 5 phases"
        }
        ```
        
        **After Frame-Specific Analysis:**
        ```json
        {
          "0": "...camera movement with subtle panning left (sustained) and gentle moving down (extended)",
          "43": "...camera movement with moderate panning right (brief) and subtle rotating left (sustained)",
          "106": "...camera movement with gentle dolly forward (extended) and subtle rolling clockwise (brief)"
        }
        ```
        
        ### 🚀 Practical Usage
        
        1. **Set up movement** in Keyframes → Motion tab or enable Camera Shakify
        2. **Configure prompts** in Prompts tab with frame numbers
        3. **Click "Enhance Prompts with Movement Analysis"**
        4. **Review frame-specific descriptions** - each prompt gets unique analysis
        5. **Generate video** with varied, specific movement context for better results
        
        This frame-specific system ensures each video clip gets movement descriptions that accurately reflect what's happening during its specific timeframe!
        """)


def _create_setup_guide():
    """Create the setup guide."""
    with gr.Accordion("🛠️ Setup Guide", open=False):
        gr.Markdown("""
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
        
        #### Step 3: Configure Strength Schedule (Optional but Recommended)
        - Go to **Keyframes → Strength tab**
        - Set "Strength schedule" to control I2V continuity
        - Example: `0:(0.85), 60:(0.7), 120:(0.5)` for gradual creative freedom
        
        #### Step 4: Configure CFG Scale Schedule (Optional but Recommended)
        - Go to **Keyframes → CFG tab**
        - Set "CFG scale schedule" to control prompt adherence
        - Example: `0:(7.5), 60:(9.0), 120:(6.0)` for varying prompt adherence
        
        #### Step 5: Configure Seeds (Optional)
        - **For consistent seeds**: Set seed behavior to 'schedule'
        - **For variety**: Leave as 'iter' or 'random'
        
        #### Step 6: Generate
        - Click "Generate Wan Video" button
        - Wan reads all settings from Deforum automatically
        - Each prompt becomes a seamless video clip with strength-controlled transitions
        """)


def _create_troubleshooting_guide():
    """Create the troubleshooting guide."""
    with gr.Accordion("🆘 Troubleshooting", open=False):
        gr.Markdown("""
        If generation fails:
        1. **Check models**: Run `python -m deforum.integrations.wan.direct_integration`
        2. **Download missing models**: Use commands in Auto-Discovery section
        3. **Verify placement**: Models should be in `models/wan/` directory
        4. **Check logs**: Look for auto-discovery messages in console
        5. **Verify schedules**: Make sure you have prompts in the Prompts tab
        6. **Check seed behavior**: Set seed behavior to 'schedule' if you want custom seed scheduling
        """)


def _connect_wan_event_handlers(
    local_vars, wan_enhanced_prompts, analyze_movement_btn, enhance_prompts_btn,
    wan_enable_shakify, wan_movement_sensitivity_override, wan_manual_sensitivity,
    manual_sensitivity_row, wan_movement_description, enhancement_progress,
    wan_generate_button, wan_generation_status, load_deforum_to_wan_btn,
    load_wan_defaults_btn, movement_sensitivity_override, wan_movement_sensitivity,
    check_flash_attention_btn, wan_flash_attention_status, wan_flash_attention_mode,
    check_qwen_models_btn, download_qwen_model_btn, cleanup_qwen_cache_btn,
    qwen_model_status
):
    """Connect all WAN event handlers."""
    
    # Movement sensitivity override toggle
    def toggle_movement_sensitivity_override(override_enabled):
        return gr.update(interactive=override_enabled)
    
    movement_sensitivity_override.change(
        fn=toggle_movement_sensitivity_override,
        inputs=[movement_sensitivity_override],
        outputs=[wan_movement_sensitivity]
    )
    
    # Flash attention handlers
    def check_flash_attention_status():
        """Check flash attention availability and return status"""
        try:
            from ..integrations.wan.wan_flash_attention_patch import get_flash_attention_status_html
            return get_flash_attention_status_html()
        except Exception as e:
            return f"❌ <span style='color: #f44336;'>Error checking status: {e}</span>"
    
    def update_flash_attention_mode(mode):
        """Update flash attention mode and return updated status"""
        try:
            from ..integrations.wan.wan_flash_attention_patch import update_patched_flash_attention_mode, get_flash_attention_status_html
            update_patched_flash_attention_mode(mode)
            status = get_flash_attention_status_html()
            return f"{status} - Mode: {mode}"
        except Exception as e:
            return f"❌ <span style='color: #f44336;'>Error updating mode: {e}</span>"
    
    # Connect flash attention handlers
    check_flash_attention_btn.click(
        fn=check_flash_attention_status,
        inputs=[],
        outputs=[wan_flash_attention_status]
    )
    
    wan_flash_attention_mode.change(
        fn=update_flash_attention_mode,
        inputs=[wan_flash_attention_mode],
        outputs=[wan_flash_attention_status]
    )
    
    # Initialize flash attention status
    try:
        from ..integrations.wan.wan_flash_attention_patch import get_flash_attention_status_html
        wan_flash_attention_status.value = get_flash_attention_status_html()
    except Exception:
        wan_flash_attention_status.value = "⚠️ <span style='color: #FF9800;'>Status check unavailable</span>"
    
    # Movement analysis handlers
    from .wan_event_handlers import (
        analyze_movement_handler, validate_wan_generation,
        load_deforum_to_wan_prompts_handler, load_wan_defaults_handler,
        check_qwen_models_handler, download_qwen_model_handler,
        cleanup_qwen_cache_handler
    )
    
    # Connect movement analysis
    analyze_movement_btn.click(
        fn=analyze_movement_handler,
        inputs=[wan_enhanced_prompts, wan_enable_shakify, wan_movement_sensitivity_override, wan_manual_sensitivity],
        outputs=[wan_enhanced_prompts, wan_movement_description]
    )
    
    # Connect sensitivity override toggle
    wan_movement_sensitivity_override.change(
        fn=lambda override_enabled: gr.update(visible=override_enabled),
        inputs=[wan_movement_sensitivity_override],
        outputs=[manual_sensitivity_row]
    )
    
    # Connect generation validation
    wan_generate_button.click(
        fn=validate_wan_generation,
        inputs=[wan_enhanced_prompts],
        outputs=[wan_generation_status]
    )
    
    # Auto-validate when prompts change
    wan_enhanced_prompts.change(
        fn=validate_wan_generation,
        inputs=[wan_enhanced_prompts],
        outputs=[wan_generation_status]
    )
    
    # Connect prompt loading buttons
    load_deforum_to_wan_btn.click(
        fn=load_deforum_to_wan_prompts_handler,
        inputs=[],
        outputs=[wan_enhanced_prompts]
    )
    
    load_wan_defaults_btn.click(
        fn=load_wan_defaults_handler,
        inputs=[],
        outputs=[wan_enhanced_prompts]
    )
    
    # Connect Qwen model management
    check_qwen_models_btn.click(
        fn=check_qwen_models_handler,
        inputs=[local_vars.get('wan_qwen_model')],
        outputs=[qwen_model_status]
    )
    
    download_qwen_model_btn.click(
        fn=download_qwen_model_handler,
        inputs=[local_vars.get('wan_qwen_model'), local_vars.get('wan_qwen_auto_download')],
        outputs=[qwen_model_status]
    )
    
    cleanup_qwen_cache_btn.click(
        fn=cleanup_qwen_cache_handler,
        inputs=[],
        outputs=[qwen_model_status]
    )
    
    # Auto-update model status when selection changes
    if 'wan_qwen_model' in local_vars:
        local_vars['wan_qwen_model'].change(
            fn=check_qwen_models_handler,
            inputs=[local_vars['wan_qwen_model']],
            outputs=[qwen_model_status]
        ) 