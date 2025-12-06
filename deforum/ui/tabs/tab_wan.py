"""Wan interpolation tab for Deforum UI.

Large complex tab containing Wan/RIFE/FILM interpolation settings, model management,
VRAM optimization, generation settings, FLF2V configuration, and extensive documentation.

Note: This is a very large tab (750+ lines) that could benefit from further modularization
into sub-components in the future.
"""

import gradio as gr
from types import SimpleNamespace
from modules.ui_components import FormRow
from deforum.utils.system.logging import emoji as emoji_utils, emoji_if_enabled
from deforum.utils.ui.builders import create_gr_elem, create_row


def get_tab_wan(dw: SimpleNamespace, da: SimpleNamespace = None, skip_tabitem=False):
    """Interpolation Settings Tab - Multi-method interpolation (Wan/RIFE/FILM)

    Args:
        dw: DeforumWanArgs namespace
        da: DeforumAnimArgs namespace (optional, needed for FLF2V tween settings)
        skip_tabitem: If True, don't create TabItem wrapper
    """

    gr.Markdown(f"""
    ## {emoji_if_enabled('🎬')} Interpolation Methods

    **Choose your interpolation method for smooth transitions between keyframes:**

    - **Wan FLF2V:** AI-generated video with semantic understanding (requires FLF2V model download)
    - **FILM:** Smearcore - sharp motion mixing like dragging paint, Google's ML interpolation (works out of the box)

    **Note:** RIFE is available in the post-processing tab for framerate doubling/tripling on completed videos.

    ---
    """)

    # INTERPOLATION METHOD SELECTOR - ALWAYS VISIBLE AT TOP
    gr.Markdown(f"### {emoji_if_enabled('🎯')} Select Interpolation Method")
    with gr.Row():
        flux_flf2v_interpolation_method = create_gr_elem(dw.flux_flf2v_interpolation_method)

    # DA3-3DGS SETTINGS - Conditionally visible accordion
    with gr.Accordion(f"{emoji_if_enabled('🔍')} DA3-3DGS Settings", open=True, visible=False) as da3_3dgs_accordion:
        gr.Markdown("**3D Gaussian Splatting Interpolation Settings**")
        with gr.Row():
            da3_3dgs_model = create_gr_elem(dw.da3_3dgs_model)
            da3_3dgs_num_keyframes = create_gr_elem(dw.da3_3dgs_num_keyframes)
        with gr.Row():
            da3_3dgs_render_keyframes = create_gr_elem(dw.da3_3dgs_render_keyframes)

    # CRITICAL: Immediately capture DA3-3DGS components in locals() after creation
    # Python's locals() dict doesn't auto-update, so we must explicitly assign
    locals()['da3_3dgs_model'] = da3_3dgs_model
    locals()['da3_3dgs_num_keyframes'] = da3_3dgs_num_keyframes
    locals()['da3_3dgs_render_keyframes'] = da3_3dgs_render_keyframes

    gr.Markdown("---")

    # Deforum Integration Info - Shows what settings are used
    link = emoji_utils.link()
    with gr.Accordion(f"{link} Deforum Integration Details", open=False):
        check = emoji_utils.maybe_check()
        memo = emoji_utils.memo()
        dice = emoji_utils.dice()
        strength_emoji = emoji_utils.strength()
        gr.Markdown(f"""
        **{check} Wan seamlessly integrates with your Deforum settings:**

        - **{memo} Prompts:** Uses prompts from Deforum Prompts tab
        - **{movie_camera} Movement:** Uses same movement schedules as normal Deforum renders
        - **{dice} Seed & CFG:** Uses Deforum's seed and CFG schedules
        - **{strength_emoji} Strength:** Uses Deforum's strength schedule for I2V continuity
        - **{movie_camera} FPS:** Uses Output tab FPS setting

        **Movement Integration:**
        - {check} Translation X/Y/Z, Rotation 3D X/Y/Z, Zoom schedules
        - {check} **Parseq schedules fully supported**
        - {check} Movement descriptions automatically calculated and added
        - {check} Motion intensity dynamically adapts to movement complexity
        """)

    # DEPRECATED SECTION - Standalone Wan Workflow no longer used
    # Components moved outside hidden accordion to remain accessible
    if False:  # Dead code kept for reference
        pass
    with gr.Accordion(f"{emoji_utils.warn} DEPRECATED: Standalone Wan Workflow (Reference Only)", open=False, visible=False):
        target = emoji_utils.target()
        gr.Markdown(f"""
        **{target} Essential for Wan Generation:** These prompts define what video clips will be generated.

        **Quick Setup:** Load → Analyze Movement → Enhance → Generate
        """)

        # Prompt Loading Buttons
        clipboard = emoji_utils.clipboard()
        with FormRow():
            load_deforum_to_wan_btn = gr.Button(
                f"{clipboard} Load from Deforum Prompts",
                variant="primary",
                size="lg",
                elem_id="load_deforum_to_wan_btn"
            )
            load_wan_defaults_btn = gr.Button(
                f"{memo} Load Default Wan Prompts",
                variant="secondary",
                size="lg",
                elem_id="load_wan_defaults_btn"
            )
        
        # Wan Prompts Display - ALWAYS VISIBLE AND PROMINENT
        wan_enhanced_prompts = gr.Textbox(
            label="Flux/Wan Prompts (JSON Format)",
            lines=10,
            interactive=True,
            placeholder='REQUIRED: Load prompts first! Click "Load from Deforum Prompts" or "Load Default Wan Prompts" above.',
            info=f"{emoji_utils.target()} ESSENTIAL: These prompts will be used for Wan video generation. Edit manually or use buttons below to enhance.",
            elem_id="wan_enhanced_prompts_textbox"
        )
        
        # Prompt Enhancement Actions
        ruler = emoji_utils.ruler()
        palette = emoji_utils.palette()
        with FormRow():
            analyze_movement_btn = gr.Button(
                f"{ruler} Add Movement Descriptions",
                variant="secondary",
                size="lg",
                elem_id="wan_analyze_movement_btn"
            )
            enhance_prompts_btn = gr.Button(
                f"{palette} AI Prompt Enhancement",
                variant="secondary",
                size="lg",
                elem_id="wan_enhance_prompts_btn"
            )
        
        # Camera Shakify Integration Control
        with FormRow():
            wan_enable_shakify = gr.Checkbox(
                label=f"{emoji_utils.movie_camera()} Include Camera Shakify with Movement Analysis",
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
        
        # Movement Analysis Results - Enhanced with frame-by-frame details
        bulb = emoji_utils.bulb()
        wan_movement_description = gr.Textbox(
            label="Movement Analysis Results",
            lines=6,
            interactive=False,
            placeholder=f"Movement analysis results will appear here...\n\n{bulb} TIP: This shows frame-by-frame movement detection with Camera Shakify integration.",
            info="Fine-grained movement descriptions with specific frame ranges and Camera Shakify effects.",
            elem_id="wan_movement_description_textbox",
            visible=True  # Always visible for immediate feedback
        )
        
        # Enhancement Progress - Shows during AI enhancement
        enhancement_progress = gr.Textbox(
            label="AI Enhancement Progress",
            lines=3,
            interactive=False,
            placeholder="AI enhancement progress will show here...",
            info="Shows real-time progress during prompt enhancement.",
            elem_id="wan_enhancement_progress_textbox",
            visible=True
        )
        gr.Markdown("---")
        gr.Markdown(f"### {emoji_utils.gear()} Essential Settings")

        with FormRow():
            wan_auto_download = create_gr_elem(dw.wan_auto_download)
            wan_resolution = gr.Dropdown(
                label="Wan Resolution",
                choices=["1280x736 (Landscape, 720p)", "736x1280 (Portrait, 720p)", "1024x1024 (Square, 1K)", "1280x704 (Landscape, Letterbox)"],
                value="1280x736 (Landscape, 720p)",
                info="Wan 2.2 TI2V models require resolutions divisible by 32 (VAE=16x * Patch=2x). These are optimized for 720p."
            )

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
    # END DEPRECATED SECTION

    # GENERATION SECTION - Moved outside deprecated accordion for accessibility
    gr.Markdown("---")
    gr.Markdown(f"### {emoji_utils.movie_camera()} Generate Flux/Wan")

    # Generate Button with Validation
    with FormRow():
        wan_generate_button = gr.Button(
            f"{emoji_utils.movie_camera()} Generate Flux/Wan (I2V Chaining)",
            variant="primary",
            size="lg",
            elem_id="wan_generate_button"
        )

    # Status output for Wan generation
    wan_generation_status = gr.Textbox(
        label="Generation Status",
        interactive=False,
        lines=5,
        placeholder=f"{emoji_utils.warn} Prompts required! Load prompts above first, then click Generate.",
        info="Status updates will appear here during generation."
    )

    # MODEL SETTINGS - Collapsed by default
    download = emoji_utils.download()
    with gr.Accordion(f"{emoji_utils.wrench()} Model Settings", open=False):
        gr.Markdown(f"""
        **{download} One-Click Model Download**: Download official Wan 2.2 models from Hugging Face!
        - **TI2V-5B** (Recommended): Works with 16GB VRAM using automatic CPU offload
        - **TI2V-A14B** (Advanced): Highest quality, requires 24GB+ VRAM
        """)

        # Model Download Buttons
        with gr.Accordion(f"{download} Download Models", open=True):
            check = emoji_utils.maybe_check()
            gr.Markdown(f"**{check} Recommended for Most Users (16GB+ VRAM)**")
            with FormRow():
                download_ti2v_5b = gr.Button(f"{download} TI2V-5B (30GB Download, ~16GB VRAM with offload)", variant="primary", size="sm")

            gr.Markdown(f"**{emoji_utils.rocket()} Advanced / High-End GPUs (24GB+ VRAM)**")
            with FormRow():
                download_ti2v_a14b = gr.Button(f"{download} TI2V-A14B (60GB Download, ~32GB VRAM)", size="sm")

            download_status = gr.Textbox(
                label="Download Status",
                interactive=False,
                lines=4,
                placeholder="Click a download button above to start downloading a model..."
            )

        # Model Selection
        with FormRow():
            wan_t2v_model = gr.Dropdown(
                label="TI2V Model (Wan 2.2)",
                choices=["Auto-Detect", "TI2V-5B", "TI2V-A14B", "Custom Path"],
                value="Auto-Detect",
                info="Wan 2.2 unified text/image-to-video model. TI2V-5B auto-enables CPU offload for 16GB VRAM."
            )

        wan_model_path = create_gr_elem(dw.wan_model_path)

        with gr.Row():
            wan_model_info = gr.Textbox(
                label="Detected Model Info",
                interactive=False,
                placeholder="Model information will appear here after loading",
                lines=2
            )

    # VRAM OPTIMIZATION SETTINGS
    save = emoji_utils.save()
    warning = emoji_utils.maybe_warning()
    with gr.Accordion(f"{save} VRAM Optimization", open=False):
        gr.Markdown(f"""
        **Reduce VRAM usage for 16GB GPUs:**

        These settings can help run larger models on GPUs with limited VRAM.
        All settings are OFF by default for maximum compatibility.

        **{warning} Trade-offs:**
        - T5 CPU Offload: Slightly slower text encoding, saves ~3-4GB VRAM
        - Gradient Checkpointing: Slower inference (~15-20%), saves ~2-3GB VRAM
        - Both combined: Can reduce peak VRAM by ~5-7GB

        **Recommended for 16GB VRAM:**
        - Try T5 CPU Offload first
        - Add Gradient Checkpointing if still getting OOM errors
        """)

        with FormRow():
            wan_t5_cpu_offload = create_gr_elem(dw.wan_t5_cpu_offload)
            wan_gradient_checkpointing = create_gr_elem(dw.wan_gradient_checkpointing)

        bulb = emoji_utils.bulb()
        distribution = emoji_utils.distribution()
        vram_optimization_info = gr.HTML(
            value=f"""
            <div style='padding: 10px; background: #1a1a1a; border-radius: 5px; margin-top: 10px;'>
                <p style='margin: 0; color: #aaa;'>
                    {bulb} <strong>Tip:</strong> Enable these if you see "CUDA out of memory" errors on 16GB GPUs.
                    <br/>{distribution} Current setup enables automatic CPU offload based on model size (5B vs A14B).
                </p>
            </div>
            """,
            elem_id="wan_vram_optimization_info"
        )

    # GENERATION SETTINGS - More prominent and open by default
    gr.Markdown("---")
    gr.Markdown(f"### {emoji_utils.gear()} Generation Settings")
    
    palette = emoji_utils.palette()
    with gr.Accordion(f"{palette} T2V / Keyframe Generation", open=True):
        with FormRow():
            wan_strength_override = create_gr_elem(dw.wan_strength_override)
            wan_fixed_strength = create_gr_elem(dw.wan_fixed_strength)

        with FormRow():
            wan_guidance_override = create_gr_elem(dw.wan_guidance_override)
            wan_guidance_scale = create_gr_elem(dw.wan_guidance_scale)

    warning = emoji_utils.maybe_warning()
    with gr.Accordion(f"{emoji_utils.frames()} Wan FLF2V Settings", open=False):
        gr.Markdown(f"""
        **{warning} These settings only apply when Wan is selected as interpolation method above!**

        **MODEL REQUIREMENT:** You MUST use a FLF2V-specific model!
        - **TI2V models (e.g., Wan2.2-TI2V-5B) CANNOT do FLF2V** - they will extend the first frame
        - **Use:** Wan2.1-FLF2V-14B (only FLF2V model available)
        - TI2V models were not trained on first-last-frame data, so they ignore `last_image` parameter

        **IMPORTANT:** FLF2V needs semantic guidance to interpolate correctly!
        - **Guidance Scale:** 3.5 (default) = smooth morphing, 2.5-3.0 = even smoother. **Avoid 5.5+** (causes "mode collapse" - frames stick to first image with sudden transition at end)
        - **Prompt Mode:** **'blend' (RECOMMENDED)** - combines keyframe prompts for semantic guidance
        - **{warning} NEVER use guidance_scale=0.0** (breaks last_image conditioning)
        - **{warning} 'none' mode may not work** (empty prompts often cause first-frame extension)
        """)
        with FormRow():
            wan_flf2v_guidance_scale = create_gr_elem(dw.wan_flf2v_guidance_scale)
            wan_flf2v_prompt_mode = create_gr_elem(dw.wan_flf2v_prompt_mode)

        # Advanced FLF2V settings for 3D mode tween interpolation
        # Only show if da (DeforumAnimArgs) is provided
        if da is not None:
            gr.Markdown("---")
            gr.Markdown(f"### {emoji_utils.target()} Advanced FLF2V Control (For 3D Mode Tween Interpolation)")
            gr.Markdown("""
            **These settings control Wan FLF2V interpolation in 3D modes when "Enable FLF2V Tween Mode" is checked in the 3D Depth tab.**

            - **Chunk Size:** Maximum frames per FLF2V clip (must be 4n+1, e.g., 13, 81)
            - **Keyframe Type Schedule:** Per-keyframe control of interpolation method
            """)

            with FormRow():
                wan_flf2v_chunk_size = create_row(da.wan_flf2v_chunk_size)

            gr.Markdown("**Per-Keyframe Type Control (Advanced):**")
            keyframe_type_schedule = create_row(da.keyframe_type_schedule)

            robot = emoji_utils.robot()
            with FormRow():
                auto_assign_keyframe_types_btn = gr.Button(
                    f"{robot} Auto-Assign Types",
                    variant="secondary",
                    size="sm",
                    elem_id="auto_assign_keyframe_types_btn"
                )
                gr.Markdown("*Analyzes tween distances and suggests optimal types based on chunk size*")

    with gr.Accordion(f"{emoji_utils.wrench()} Advanced Generation", open=False):

        # Advanced Generation Settings
        with FormRow():
            wan_negative_prompt = create_gr_elem(dw.wan_negative_prompt)

        with FormRow():
            wan_sampler = create_gr_elem(dw.wan_sampler)
            wan_scheduler = create_gr_elem(dw.wan_scheduler)

        with FormRow():
            wan_motion_strength_override = create_gr_elem(dw.wan_motion_strength_override)
            wan_motion_strength = create_gr_elem(dw.wan_motion_strength)
            
        # Movement sensitivity - now in overrides since it should be auto-calculated
        with FormRow():
            movement_sensitivity_override = gr.Checkbox(
                label="Movement Sensitivity Override",
                value=False,
                info="Override auto-calculated movement sensitivity from Deforum schedules"
            )
            wan_movement_sensitivity = create_gr_elem(dw.wan_movement_sensitivity)
            wan_movement_sensitivity.interactive = False  # Start disabled
    
    lightning = emoji_utils.lightning()
    with gr.Accordion(f"{lightning} Timing & Interpolation", open=False):
        with FormRow():
            wan_frame_overlap = create_gr_elem(dw.wan_frame_overlap)

        with FormRow():
            wan_enable_interpolation = create_gr_elem(dw.wan_enable_interpolation)
            wan_interpolation_strength = create_gr_elem(dw.wan_interpolation_strength)

        # Flash Attention Settings Section
        with gr.Accordion(f"{lightning} Flash Attention Settings", open=False):
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
            warning = emoji_utils.maybe_warning()
            magnifying_glass = emoji_utils.magnifying_glass()
            wan_flash_attention_status = gr.HTML(
                label="Flash Attention Status",
                value=f"{warning} <span style='color: #FF9800;'>Status check unavailable</span>",
                elem_id="wan_flash_attention_status"
            )

            check_flash_attention_btn = gr.Button(
                f"{magnifying_glass} Check Flash Attention Status",
                variant="secondary",
                elem_id="wan_check_flash_attention_btn"
            )
    
    # QWEN MODEL MANAGEMENT - Collapsed by default
    brain = emoji_utils.brain()
    with gr.Accordion(f"{brain} Qwen Model Management", open=False):
        gr.Markdown("""
        **Model Information & Auto-Download Status**

        Monitor Qwen model availability and manage downloads:
        """)

        hourglass = emoji_utils.hourglass()
        magnifying_glass = emoji_utils.magnifying_glass()
        download = emoji_utils.download()
        broom = emoji_utils.broom()

        qwen_model_status = gr.HTML(
            label="Qwen Model Status",
            value=f"{hourglass} Checking model availability...",
            elem_id="wan_qwen_model_status"
        )

        with FormRow():
            check_qwen_models_btn = gr.Button(
                f"{magnifying_glass} Check Model Status",
                variant="secondary",
                elem_id="wan_check_qwen_models_btn"
            )
            download_qwen_model_btn = gr.Button(
                f"{download} Download Selected Model",
                variant="primary",
                elem_id="wan_download_qwen_model_btn"
            )
            cleanup_qwen_cache_btn = gr.Button(
                f"{broom} Cleanup Model Cache",
                variant="secondary",
                elem_id="wan_cleanup_qwen_cache_btn"
            )

    # Auto-Discovery and Setup Information
    download = emoji_utils.download()
    with gr.Accordion(f"{download} Model Auto-Discovery & Setup", open=False):
        check = emoji_utils.maybe_check()
        sparkles = emoji_utils.sparkles()
        gr.Markdown(f"""
        **{check} Auto-Discovery System**

        Wan automatically finds models in these locations:
        - `models/Deforum/wan/` (recommended)
        - `models/video/wan/`
        - Custom paths you specify

        **{sparkles} Wan 2.2 TI2V Models (Recommended)**

        TI2V models are unified text/image-to-video with diffusers format:
        - **TI2V-5B**: 720p@24fps, 24GB VRAM, RTX 4090 compatible (recommended)
        - **TI2V-A14B**: Mixture-of-Experts, 32GB+ VRAM, highest quality

        **{download} Easy Download Commands:**
        ```bash
        # Download TI2V-5B (recommended default)
        huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir models/Deforum/wan/Wan2.2-TI2V-5B

        # Or download TI2V-A14B (highest quality)
        huggingface-cli download Wan-AI/Wan2.2-TI2V-A14B-Diffusers --local-dir models/Deforum/wan/Wan2.2-TI2V-A14B
        ```

        **Note**: This extension supports Wan 2.2 TI2V models only.
        Legacy Wan 2.1 models (T2V, I2V, VACE) are no longer supported.
        """)
    
    # Hidden model path for compatibility (auto-populated by discovery)
    wan_model_path = gr.Textbox(visible=False, value="auto-discovery")
    
    # Hidden wan_seed for compatibility (integrated with Deforum schedules)
    wan_seed = gr.Number(
        precision=dw.wan_seed["precision"], 
        value=dw.wan_seed["value"],
        visible=False
    )
    
    # Detailed Documentation - Collapsed by default
    books = emoji_utils.books()
    with gr.Accordion(f"{books} Detailed Documentation", open=False):
        target = emoji_utils.target()
        with gr.Accordion(f"{target} How Wan Integrates with Deforum Schedules", open=False):
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
            
            with gr.Accordion(f"{emoji_utils.movie_camera()} Movement Translation: From Deforum Schedules to Prompt Descriptions", open=False):
                sparkles = emoji_utils.sparkles()
                target = emoji_utils.target()
                refresh = emoji_utils.refresh_icon()
                distribution = emoji_utils.distribution()
                palette = emoji_utils.palette()
                movie_camera = emoji_utils.movie_camera()
                wrench = emoji_utils.wrench()
                chart_increasing = emoji_utils.chart_increasing()
                rocket = emoji_utils.rocket()
                gr.Markdown(f"""
                ### {sparkles} NEW: Frame-Specific Movement Analysis

                Wan now provides **unique movement descriptions for each prompt** based on its exact position in the video timeline, eliminating generic repetitive text.

                **{target} Key Improvements:**
                - **Frame-Specific Analysis**: Each prompt analyzes movement at its specific frame range
                - **Directional Specificity**: "panning left", "tilting down", "dolly forward" instead of generic text
                - **Camera Shakify Integration**: Analyzes actual shake patterns at each frame offset
                - **Varied Descriptions**: No more identical "investigative handheld" text across all prompts

                ### {refresh} How Frame-Specific Analysis Works
                
                **Traditional Approach (OLD):**
                ```json
                All prompts: "camera movement with investigative handheld camera movement"
                ```
                
                **Frame-Specific Approach (NEW):**
                ```json
                {{
                  "0": "...with subtle panning left (sustained) and gentle moving down (extended)",
                  "43": "...with moderate panning right (brief) and subtle rotating left (sustained)",
                  "106": "...with gentle dolly forward (extended) and subtle rolling clockwise (brief)",
                  "210": "...with subtle tilting down (extended) and moderate panning left (brief)",
                  "324": "...with gentle rotating right (sustained) and subtle dolly backward (extended)"
                }}}}
                ```

                ### {distribution} Movement Detection & Classification

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

                ### {palette} Intensity & Duration Modifiers
                
                **Movement Intensity:**
                - **Subtle**: Very small movements (< 1.0 units)
                - **Gentle**: Small movements (1.0 - 10.0 units)
                - **Moderate**: Medium movements (10.0 - 50.0 units)
                - **Strong**: Large movements (> 50.0 units)
                
                **Duration Descriptions:**
                - **Brief**: Short duration (< 20% of total frames)
                - **Extended**: Medium duration (20% - 50% of total frames)
                - **Sustained**: Long duration (> 50% of total frames)

                ### {movie_camera} Camera Shakify Integration
                
                When Camera Shakify is enabled, the system:
                1. **Generates frame-specific shake data** based on the prompt's frame position
                2. **Overlays shake on Deforum schedules** (like render core)
                3. **Analyzes combined movement** for each prompt's timeframe
                4. **Provides varied descriptions** that reflect actual camera behavior
                
                **Example with Camera Shakify INVESTIGATION:**
                ```
                Frame 0 prompt → Analyzes shake pattern frames 0-17
                Frame 43 prompt → Analyzes shake pattern frames 43-60
                Frame 106 prompt → Analyzes shake pattern frames 106-123
                ```

                ### {wrench} Smart Motion Analysis
                
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

                ### {chart_increasing} Results Comparison
                
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
                {{
                  "0": "...camera movement with subtle panning left (sustained) and gentle moving down (extended)",
                  "43": "...camera movement with moderate panning right (brief) and subtle rotating left (sustained)",
                  "106": "...camera movement with gentle dolly forward (extended) and subtle rolling clockwise (brief)"
                }}}}
                ```

                ### {rocket} Practical Usage
                
                1. **Set up movement** in Keyframes → Motion tab or enable Camera Shakify
                2. **Configure prompts** in Prompts tab with frame numbers
                3. **Click "Enhance Prompts with Movement Analysis"**
                4. **Review frame-specific descriptions** - each prompt gets unique analysis
                5. **Generate video** with varied, specific movement context for better results
                
                This frame-specific system ensures each video clip gets movement descriptions that accurately reflect what's happening during its specific timeframe!
                """)
            
        with gr.Accordion(f"{emoji_utils.tools()} Setup Guide", open=False):
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
            - Click "Generate Flux/Wan" button
            - Wan reads all settings from Deforum automatically
            - Each prompt becomes a seamless video clip with strength-controlled transitions
            """)
            
        with gr.Accordion("🆘 Troubleshooting", open=False):
            gr.Markdown("""
            If generation fails:
            1. **Check models**: Run `python scripts/deforum_helpers/wan_direct_integration.py`
            2. **Download missing models**: Use commands in Auto-Discovery section
            3. **Verify placement**: Models should be in `models/Deforum/wan/` directory
            4. **Check logs**: Look for auto-discovery messages in console
            5. **Verify schedules**: Make sure you have prompts in the Prompts tab
            6. **Check seed behavior**: Set seed behavior to 'schedule' if you want custom seed scheduling
            """)

    # Connect DA3-3DGS accordion visibility to interpolation method selection
    def toggle_da3_3dgs_settings(method):
        return gr.update(visible=(method == "DA3-3DGS"))

    flux_flf2v_interpolation_method.change(
        fn=toggle_da3_3dgs_settings,
        inputs=[flux_flf2v_interpolation_method],
        outputs=[da3_3dgs_accordion]
    )

    # Connect movement sensitivity override toggle
    def toggle_movement_sensitivity_override(override_enabled):
        return gr.update(interactive=override_enabled)

    movement_sensitivity_override.change(
        fn=toggle_movement_sensitivity_override,
        inputs=[movement_sensitivity_override],
        outputs=[wan_movement_sensitivity]
    )
        
    # Ensure wan_inference_steps is properly captured
    locals()['wan_inference_steps'] = wan_inference_steps
    
    # Button handlers for flash attention
    def check_flash_attention_status():
        """Check flash attention availability and return status"""
        try:
            from deforum.integrations.wan.wan_flash_attention_patch import get_flash_attention_status_html
            return get_flash_attention_status_html()
        except Exception as e:
            cross = emoji_utils.maybe_cross()
            return f"{cross} <span style='color: #f44336;'>Error checking status: {e}</span>"

    def update_flash_attention_mode(mode):
        """Update flash attention mode and return updated status"""
        try:
            from deforum.integrations.wan.wan_flash_attention_patch import update_patched_flash_attention_mode, get_flash_attention_status_html
            update_patched_flash_attention_mode(mode)
            status = get_flash_attention_status_html()
            return f"{status} - Mode: {mode}"
        except Exception as e:
            cross = emoji_utils.maybe_cross()
            return f"{cross} <span style='color: #f44336;'>Error updating mode: {e}</span>"
    
    # Connect button click to status check
    check_flash_attention_btn.click(
        fn=check_flash_attention_status,
        inputs=[],
        outputs=[wan_flash_attention_status]
    )

    # Connect mode change to status update
    wan_flash_attention_mode.change(
        fn=update_flash_attention_mode,
        inputs=[wan_flash_attention_mode],
        outputs=[wan_flash_attention_status]
    )

    # Initialize status on load
    try:
        from deforum.integrations.wan.wan_flash_attention_patch import get_flash_attention_status_html
        wan_flash_attention_status.value = get_flash_attention_status_html()
    except Exception:
        warning = emoji_utils.maybe_warning()
        wan_flash_attention_status.value = f"{warning} <span style='color: #FF9800;'>Status check unavailable</span>"
    
    # Get all component names for the handlers
    from deforum.config.args import get_component_names
    component_names = get_component_names()

    # DEPRECATED EVENT HANDLERS - Commented out for hidden standalone workflow
    # These are no longer needed as Wan is now integrated with main Deforum workflow

    # # Connect sensitivity override toggle to show/hide manual sensitivity slider
    # wan_movement_sensitivity_override.change(
    #     fn=lambda override_enabled: gr.update(visible=override_enabled),
    #     inputs=[wan_movement_sensitivity_override],
    #     outputs=[manual_sensitivity_row]
    # )
    #
    # # Connect generate button with validation
    # wan_generate_button.click(
    #     fn=validate_wan_generation,
    #     inputs=[wan_enhanced_prompts],
    #     outputs=[wan_generation_status]
    # )
    #
    # # Add automatic validation status updates when prompts change
    # wan_enhanced_prompts.change(
    #     fn=validate_wan_generation,
    #     inputs=[wan_enhanced_prompts],
    #     outputs=[wan_generation_status]
    # )

    # Connect model download buttons
    from deforum.integrations.wan.wan_model_downloader import download_wan_model

    download_ti2v_5b.click(
    fn=lambda: download_wan_model("TI2V-5B"),
    outputs=[download_status]
    )

    download_ti2v_a14b.click(
    fn=lambda: download_wan_model("TI2V-A14B"),
    outputs=[download_status]
    )

    # DEPRECATED: Standalone prompt loading buttons (now use main Prompts tab)
    # load_deforum_to_wan_btn.click(
    #     fn=load_deforum_to_wan_prompts_handler,
    #     inputs=[],
    #     outputs=[wan_enhanced_prompts]
    # )
    #
    # load_wan_defaults_btn.click(
    #     fn=load_wan_defaults_handler,
    #     inputs=[],
    #     outputs=[wan_enhanced_prompts]
    # )

    # Connect prompt template loading buttons
    # NOTE: These will be properly connected in ui_left.py where animation_prompts is accessible

    # Store button references for connection in ui_left.py
    if 'load_wan_prompts_btn' in locals():
        locals()['load_wan_prompts_btn']._handler = load_wan_prompts_handler
    if 'load_deforum_prompts_btn' in locals():
        locals()['load_deforum_prompts_btn']._handler = load_deforum_prompts_handler

    return {k: v for k, v in {**locals(), **vars()}.items()}
