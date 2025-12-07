"""Tuning UI for automated parameter optimization.

This module provides an interactive UI for running parameter sweeps
and visualizing quality metrics in real-time.
"""

import gradio as gr
from pathlib import Path
from typing import Dict, List, Any
import json

from deforum.utils.system.logging import get_logger, emoji as emoji_utils

logger = get_logger()


def create_tuning_tab() -> tuple:
    """Create the Tuning tab UI.

    Returns:
        Tuple of (interface, title, id) for WebUI tab registration
    """
    with gr.Blocks() as tuning_interface:
        # Theme-aware emojis for buttons and status
        rocket = emoji_utils.rocket()
        refresh = emoji_utils.refresh_icon()
        folder = emoji_utils.folder()
        sparkles = emoji_utils.sparkles()
        microscope = emoji_utils.microscope()
        stop = emoji_utils.stop()
        check = emoji_utils.maybe_check()
        cross = emoji_utils.maybe_cross()
        gr.Markdown(f"""
        # {microscope} Deforum Parameter Tuning Lab

        Automated quality assessment for finding optimal Deforum parameters.
        """)

        # Separate subtabs for different test types
        with gr.Tabs():
            # I2V Chaining Tests Tab
            with gr.Tab("I2V Chaining Tests"):
                gr.Markdown("""
                **Tests for image-to-video chaining workflows:**

                These tests measure BOTH color preservation AND temporal consistency together:
                - **Color Preservation**: How long before colors degrade to grayscale?
                - **Temporal Consistency**: How stable are frame-to-frame transitions (SSIM)?
                - **Degradation Rate**: How fast quality decays per iteration

                **How it works:**
                1. Start with a colorful rainbow gradient image
                2. Run img2img repeatedly (output → input chaining)
                3. Measure color saturation and frame similarity over 20 iterations
                4. Find optimal strength values that maximize stability

                **Leverages fractional strength:** 0.01 step precision for fine-grained tuning
                """)

                with gr.Row():
                    # Left column: Parameter selection
                    with gr.Column(scale=1):
                        gr.Markdown("## Test Configuration")

                        test_type = gr.Dropdown(
                            label="Test Type",
                            choices=[
                                "Color Preservation (I2V Chaining)",
                                "Temporal Consistency (Frame Stability)",
                                "Flux Parameter Sweep",
                            ],
                            value="Color Preservation (I2V Chaining)",
                        )

                gr.Markdown("### Parameters to Test")

                # Steps configuration
                with gr.Group():
                    gr.Markdown("**Sampling Steps**")
                    test_steps = gr.CheckboxGroup(
                        label="Steps to test",
                        choices=["4 (Schnell)", "8", "12", "16", "20 (Dev)"],
                        value=["20 (Dev)"],
                    )

                # Strength configuration
                with gr.Group():
                    gr.Markdown("**Normal/Tween Strength**")
                    strength_min = gr.Slider(
                        label="Min strength",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.80,
                        step=0.01,
                        info="Cadence frame stability (higher = more preservation)",
                    )
                    strength_max = gr.Slider(
                        label="Max strength",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.01,
                        info="Upper bound for sweep",
                    )
                    strength_step = gr.Slider(
                        label="Step size",
                        minimum=0.01,
                        maximum=0.1,
                        value=0.01,
                        step=0.01,
                        info="Uses fractional strength (0.01 = 1% precision)",
                    )

                # Keyframe strength configuration
                with gr.Group():
                    gr.Markdown("**Keyframe Strength**")
                    kf_strength_min = gr.Slider(
                        label="Min keyframe strength",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.10,
                        step=0.01,
                        info="Keyframe change amount (lower = more diffusion)",
                    )
                    kf_strength_max = gr.Slider(
                        label="Max keyframe strength",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.25,
                        step=0.01,
                        info="Upper bound for sweep",
                    )
                    kf_strength_step = gr.Slider(
                        label="Step size",
                        minimum=0.01,
                        maximum=0.1,
                        value=0.01,
                        step=0.01,
                        info="Uses fractional strength (0.01 = 1% precision)",
                    )

                # Test limits (for I2V chaining tests)
                with gr.Group():
                    gr.Markdown("**Test Limits**")
                    max_iterations = gr.Slider(
                        label="Max iterations (I2V chaining)",
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=1,
                        info="Stop after N iterations or grayscale threshold",
                    )
                    grayscale_threshold = gr.Slider(
                        label="Grayscale threshold",
                        minimum=0,
                        maximum=50,
                        value=20,
                        step=1,
                        info="Color score below this = considered grayscale",
                    )

                # Action buttons
                with gr.Row():
                    run_tests_btn = gr.Button(f"{rocket} Run Tests", variant="primary", size="lg")
                    stop_tests_btn = gr.Button(f"{stop} Stop", variant="stop")

                with gr.Row():
                    refresh_btn = gr.Button(f"{refresh} Refresh Results", size="sm")
                    open_tuning_dir_btn = gr.Button(f"{folder} Open Tuning Directory", size="sm")

                # Status
                status_box = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    lines=3,
                )

                # Progress tracking
                progress_bar = gr.Progress()

                # Results components (updated via event handlers)
                results_table = gr.DataFrame(
                    headers=[
                        "Steps",
                        "Normal Strength",
                        "KF Strength",
                        "Iterations",
                        "Final Color",
                        "Avg Temporal",
                        "Overall Score",
                    ],
                    label="Test Results",
                    interactive=False,
                )

                best_params_json = gr.JSON(
                    label="Optimal Settings",
                    value={},
                )

                apply_best_btn = gr.Button(
                    f"{sparkles} Apply to Deforum Defaults",
                    variant="secondary",
                )

                metrics_plot = gr.Plot(
                    label="Quality Metrics",
                )

                heatmap_plot = gr.Plot(
                    label="Parameter Heatmap (Overall Score)",
                )

            # Orbit Tests Tab
            with gr.Tab("Orbit Tests"):
                gr.Markdown("""
                **Depth warping orbital camera path tests:**
                - Find optimal rotation_factor for stable orbital movement
                - Test different aspect ratios (16:9, 9:16, 1:1)
                - Measure iterations until subject goes off-screen
                - Pure depth warping (NO diffusion) for clean metrics

                **Process:**
                1. Select aspect ratios to test
                2. Define rotation factor range
                3. Set orbit parameters (radius, iterations)
                4. Run automated orbit tests
                5. View plotly graph showing optimal factors
                """)

                with gr.Row():
                    # Left column: Orbit test configuration
                    with gr.Column(scale=1):
                        gr.Markdown("## Orbit Test Configuration")

                        gr.Markdown("### Aspect Ratios")
                        orbit_aspect_ratios = gr.CheckboxGroup(
                            label="Aspect ratios to test",
                            choices=["16:9 (Landscape)", "9:16 (Portrait)", "1:1 (Square)"],
                            value=["16:9 (Landscape)"],
                        )

                        gr.Markdown("### Rotation Factor Sweep")
                        gr.Markdown(
                            "_Find optimal ratio between translation_x and rotation_3d_y for stable orbits_"
                        )
                        orbit_rotation_factor_min = gr.Slider(
                            label="Min rotation factor (translation_x / rotation_3d_y)",
                            minimum=-50.0,
                            maximum=-1.0,
                            value=-10.0,
                            step=0.05,
                            info="More negative = stronger counter-rotation (empirical optimal: -8.0)",
                        )
                        orbit_rotation_factor_max = gr.Slider(
                            label="Max rotation factor (translation_x / rotation_3d_y)",
                            minimum=-50.0,
                            maximum=-1.0,
                            value=-6.0,
                            step=0.05,
                            info="Less negative = weaker counter-rotation (empirical range: -6 to -9, theory: -1.0)",
                        )
                        orbit_rotation_factor_step = gr.Slider(
                            label="Step size",
                            minimum=0.01,
                            maximum=2.0,
                            value=0.05,
                            step=0.01,
                            info="0.05 = fine (31 tests), 0.1 = coarse (16 tests), 0.01 = very fine (151 tests)",
                        )

                        gr.Markdown("### Orbit Parameters")
                        orbit_orbit_radius = gr.Slider(
                            label="Movement scale (translation amount)",
                            minimum=1.0,
                            maximum=20.0,
                            value=5.0,
                            step=0.5,
                            info="How much to move per orbit (2-3 = very gentle, 5 = moderate, 10+ = aggressive, may exit frame)",
                        )
                        orbit_orbit_iterations = gr.Slider(
                            label="Depth warp iterations per test",
                            minimum=10,
                            maximum=200,
                            value=50,
                            step=5,
                            info="How many frames to generate per test configuration",
                        )

                        # Orbit test action buttons
                        with gr.Row():
                            orbit_run_btn = gr.Button(f"{rocket} Run Orbit Tests", variant="primary", size="lg")
                            orbit_stop_btn = gr.Button(f"{stop} Stop", variant="stop")

                        with gr.Row():
                            orbit_refresh_btn = gr.Button(f"{refresh} Refresh Results", size="sm")
                            orbit_open_dir_btn = gr.Button(f"{folder} Open Results Directory", size="sm")

                        # Orbit test status
                        orbit_status_box = gr.Textbox(
                            label="Status",
                            value="Ready",
                            interactive=False,
                            lines=3,
                        )

                    # Right column: Orbit test results
                    with gr.Column(scale=2):
                        gr.Markdown("## Orbit Test Results")

                        orbit_progress_bar = gr.Progress()

                        with gr.Tabs():
                            with gr.Tab("Metrics Plot"):
                                gr.Markdown("""
                                ### Rotation Factor vs Sphere Visibility

                                **Y-axis:** Iterations until sphere goes off-screen (higher = better)
                                **X-axis:** Rotation factor (translation/rotation ratio)

                                **Goal:** Find the factor where the sphere stays in frame longest for each aspect ratio.
                                """)

                                orbit_metrics_plot = gr.Plot(
                                    label="Stability Metrics",
                                )

                            with gr.Tab("Heatmap"):
                                gr.Markdown("""
                                ### Parameter Heatmap

                                Visual heatmap showing iterations until off-screen for all tested configurations.
                                Darker colors indicate better stability (sphere stayed in frame longer).
                                """)

                                orbit_heatmap_plot = gr.Plot(
                                    label="Parameter Heatmap",
                                )

                            with gr.Tab("Results Table"):
                                gr.Markdown("""
                                ### All Test Configurations

                                **Visual Metrics:**
                                - Iterations Until Off-Screen: Primary stability metric
                                - Max Drift: Maximum pixel displacement from initial position

                                **Depth Analysis Metrics:**
                                - Sphere Depth: Mean depth value of sphere (0=far, 1=near)
                                - Depth Range: Depth variation across sphere surface
                                - Depth Separation: How well sphere is distinguished from background
                                - Depth Stability: Frame-to-frame depth consistency (1.0=perfect)
                                - Gradient Quality: Smoothness of depth gradients (1.0=perfect)
                                """)

                                orbit_results_table = gr.DataFrame(
                                    headers=[
                                        "Aspect Ratio",
                                        "Width×Height",
                                        "Rotation Factor",
                                        "Orbit Radius",
                                        "Iterations Until Off-Screen",
                                        "Max Drift (px)",
                                        "Sphere Depth",
                                        "Depth Range",
                                        "Depth Separation",
                                        "Depth Stability",
                                        "Gradient Quality",
                                    ],
                                    label="Orbit Test Results",
                                    interactive=False,
                                )

                            with gr.Tab("Best Parameters"):
                                gr.Markdown("### Optimal Rotation Factors")

                                orbit_best_params = gr.JSON(
                                    label="Best Factor for Each Aspect Ratio",
                                    value={},
                                )

            # RAFT Optical Flow Tuning Tab
            with gr.Tab("RAFT Tuning"):
                gr.Markdown("""
                **RAFT optical flow parameter tuning for depth warping:**
                - Test different RAFT configurations on orbital camera paths
                - Compare depth-only vs depth+RAFT warping quality
                - Find optimal flow_factor for RAFT guidance strength
                - Measure flow consistency and temporal stability

                **What RAFT Does:**
                1. Calculates optical flow vectors between keyframes
                2. Warps the flow field using 3D transforms
                3. Applies warped flow to depth-warped image
                4. Flow factor controls how much to trust RAFT vs pure depth

                **Goal:** Find RAFT settings that improve depth warping without artifacts.
                """)

                with gr.Row():
                    # Left column: RAFT test configuration
                    with gr.Column(scale=1):
                        gr.Markdown("## RAFT Test Configuration")

                        gr.Markdown("### Test Setup")
                        raft_aspect_ratios = gr.CheckboxGroup(
                            label="Aspect ratios to test",
                            choices=["16:9 (Landscape)", "9:16 (Portrait)", "1:1 (Square)"],
                            value=["16:9 (Landscape)"],
                        )
                        raft_rotation_factor = gr.Slider(
                            label="Rotation factor (fixed)",
                            minimum=-50.0,
                            maximum=-1.0,
                            value=-8.0,
                            step=0.05,
                            info="Empirically validated optimal from orbit tests (range: -6 to -9)",
                        )
                        raft_orbit_radius = gr.Slider(
                            label="Movement scale (translation amount)",
                            minimum=1.0,
                            maximum=20.0,
                            value=5.0,
                            step=0.5,
                            info="Translation per orbit (2-3 = gentle, 5 = moderate, 10+ = aggressive)",
                        )
                        raft_orbit_iterations = gr.Slider(
                            label="Depth warp iterations per test",
                            minimum=10,
                            maximum=500,
                            value=200,
                            step=5,
                            info="How many frames to generate per test (200 = standard, 300-500 = extended)",
                        )

                        gr.Markdown("### RAFT Parameters to Sweep")
                        raft_model_sizes = gr.CheckboxGroup(
                            label="RAFT model sizes",
                            choices=["Small", "Large"],
                            value=["Small"],
                            info="Small = faster, Large = potentially better quality",
                        )
                        raft_flow_iterations_min = gr.Slider(
                            label="Min flow iterations",
                            minimum=6,
                            maximum=50,
                            value=14,
                            step=2,
                            info="RAFT refinement iterations (empirical optimal: 16)",
                        )
                        raft_flow_iterations_max = gr.Slider(
                            label="Max flow iterations",
                            minimum=6,
                            maximum=50,
                            value=18,
                            step=2,
                            info="Narrow sweep around optimal 16 (14, 16, 18)",
                        )
                        raft_flow_iterations_step = gr.Slider(
                            label="Iterations step size",
                            minimum=2,
                            maximum=10,
                            value=2,
                            step=2,
                            info="Step between iteration values (14, 16, 18)",
                        )
                        raft_flow_factor_min = gr.Slider(
                            label="Min flow factor",
                            minimum=0.0,
                            maximum=2.0,
                            value=1.4,
                            step=0.1,
                            info="Fine sweep around optimal 1.5 (today: test with rotation_factor=-8.0)",
                        )
                        raft_flow_factor_max = gr.Slider(
                            label="Max flow factor",
                            minimum=0.0,
                            maximum=2.0,
                            value=1.8,
                            step=0.1,
                            info="Explore beyond 1.5 to find upper limit (1.4, 1.5, 1.6, 1.7, 1.8)",
                        )
                        raft_flow_factor_step = gr.Slider(
                            label="Flow factor step size",
                            minimum=0.1,
                            maximum=0.5,
                            value=0.1,
                            step=0.05,
                            info="0.1 = fine sweep around optimal (5 values)",
                        )

                        # RAFT test action buttons
                        with gr.Row():
                            raft_run_btn = gr.Button(f"{rocket} Run RAFT Tests", variant="primary", size="lg")
                            raft_stop_btn = gr.Button(f"{stop} Stop", variant="stop")

                        with gr.Row():
                            raft_refresh_btn = gr.Button(f"{refresh} Refresh Results", size="sm")
                            raft_open_dir_btn = gr.Button(f"{folder} Open Results Directory", size="sm")

                        raft_status_box = gr.Textbox(
                            label="Status",
                            value="Ready",
                            interactive=False,
                            lines=3,
                        )

                    # Right column: RAFT test results
                    with gr.Column(scale=2):
                        gr.Markdown("## RAFT Test Results")
                        raft_progress_bar = gr.Progress()

                        with gr.Tabs():
                            with gr.Tab("Graph"):
                                gr.Markdown("""
                                ### Flow Factor vs Depth Warping Quality

                                **Y-axis:** Iterations until sphere goes off-screen
                                **X-axis:** Flow factor (RAFT guidance strength)

                                Compare different RAFT configurations to find optimal settings.
                                """)

                                raft_graph_html = gr.HTML(
                                    label="RAFT Tuning Graph",
                                    value="<p>Run tests to generate graph...</p>",
                                )

                            with gr.Tab("Results Table"):
                                gr.Markdown("""
                                ### All RAFT Test Configurations

                                **RAFT Parameters:**
                                - Model Size: Small (fast) or Large (quality)
                                - Flow Iterations: RAFT refinement iterations
                                - Flow Factor: How much to trust RAFT vs depth-only

                                **Quality Metrics:**
                                - Iterations Until Off-Screen: Primary stability metric
                                - Flow Consistency: Are motion vectors coherent? (1.0=perfect)
                                - Depth+RAFT Improvement: Quality gain over depth-only
                                """)

                                raft_results_table = gr.DataFrame(
                                    headers=[
                                        "Aspect Ratio",
                                        "Model Size",
                                        "Flow Iterations",
                                        "Flow Factor",
                                        "Iterations Until Off-Screen",
                                        "Max Drift (px)",
                                        "Flow Consistency",
                                        "Improvement vs Depth-Only (%)",
                                    ],
                                    label="RAFT Test Results",
                                    interactive=False,
                                )

                            with gr.Tab("Best Parameters"):
                                gr.Markdown("### Optimal RAFT Configuration")

                                raft_best_params = gr.JSON(
                                    label="Best RAFT Settings for Each Aspect Ratio",
                                    value={},
                                )

            # DA3-3DGS Tuning Tab
            with gr.Tab("DA3-3DGS Tuning"):
                gr.Markdown("""
                **DA3-3DGS (Depth Anything V3 + 3D Gaussian Splatting) parameter tuning:**
                - Test different scene strategies (per-segment, per-prompt, rolling-window)
                - Tune neighbor segments and densification factors
                - Optimize near-clip distance for quality
                - Find best balance between keyframe count and splat density

                **✨ REAL 3DGS Testing with Actual Depth Estimation & Rendering**
                - Tests use **gradient sphere images** as input (no ZIT diffusion needed)
                - Runs **ACTUAL DA3 depth estimation** on keyframes
                - Builds **REAL 3DGS splat scenes** from depth maps
                - Renders **novel views** with actual splat rendering
                - Measures **real VRAM usage**, render times, and quality (SSIM)
                - **All rendered frames saved** to output directory
                - Generates **markdown report** for easy copy-paste to Claude

                **What DA3-3DGS Does (in real rendering):**
                1. Collects N consecutive keyframes around each segment
                2. Estimates camera poses using DA3 GIANT model
                3. Builds 3D Gaussian Splatting scene (~705k base splats)
                4. Renders novel views via camera pose interpolation
                5. Applies densification to increase splat count (1-8x)

                **Quality Tradeoffs:**
                - **More keyframes** = Better geometry coverage, slower, more VRAM
                - **Higher densification** = Finer detail per scene, more VRAM
                - **Per-segment** = Fast, minimal VRAM, but coordinate drift
                - **Per-prompt** = Semantic coherence, eliminates drift, VRAM scales with prompt length
                - **Rolling window** = Predictable VRAM, fixed window size

                **Goal:** Find optimal parameters for your use case (speed vs quality vs VRAM).
                """)

                with gr.Row():
                    # Left column: 3DGS test configuration
                    with gr.Column(scale=1):
                        gr.Markdown("## DA3-3DGS Test Configuration")

                        gr.Markdown("### Test Setup")
                        dgs_aspect_ratios = gr.CheckboxGroup(
                            label="Aspect ratios to test",
                            choices=["16:9 (Landscape)", "9:16 (Portrait)", "1:1 (Square)"],
                            value=["16:9 (Landscape)"],
                        )
                        dgs_rotation_factor = gr.Slider(
                            label="Rotation factor (fixed)",
                            minimum=-50.0,
                            maximum=-1.0,
                            value=-8.0,
                            step=0.05,
                            info="Empirically validated optimal from orbit tests (range: -6 to -9)",
                        )
                        dgs_orbit_radius = gr.Slider(
                            label="Movement scale (translation amount)",
                            minimum=1.0,
                            maximum=20.0,
                            value=5.0,
                            step=0.5,
                            info="Translation per orbit (2-3 = gentle, 5 = moderate, 10+ = aggressive)",
                        )
                        dgs_test_iterations = gr.Slider(
                            label="Test iterations (frames to generate)",
                            minimum=10,
                            maximum=200,
                            value=50,
                            step=5,
                            info="How many frames to generate per test configuration",
                        )

                        gr.Markdown("### Scene Strategy")
                        dgs_scene_strategies = gr.CheckboxGroup(
                            label="Scene strategies to test",
                            choices=["per_segment", "per_prompt", "rolling_window"],
                            value=["per_segment"],
                            info="per_segment = fast, per_prompt = semantic coherence, rolling_window = fixed VRAM",
                        )
                        dgs_rolling_window_size = gr.Slider(
                            label="Rolling window size (keyframes)",
                            minimum=10,
                            maximum=100,
                            step=5,
                            value=30,
                            info="For rolling_window mode only",
                        )
                        dgs_max_prompt_keyframes = gr.Slider(
                            label="Max keyframes per prompt scene",
                            minimum=10,
                            maximum=200,
                            step=10,
                            value=50,
                            info="For per_prompt mode only: Split large prompt segments if they exceed this",
                        )

                        gr.Markdown("### Quality Parameters to Sweep")
                        dgs_models = gr.CheckboxGroup(
                            label="DA3 models",
                            choices=["DA3-GIANT", "DA3NESTED-GIANT-LARGE"],
                            value=["DA3-GIANT", "DA3NESTED-GIANT-LARGE"],
                            info="GIANT = 4GB VRAM, LARGE = 4.5GB VRAM + better quality (compare both)",
                        )
                        dgs_neighbor_segments_min = gr.Slider(
                            label="Min neighbor segments",
                            minimum=2,
                            maximum=10,
                            value=4,
                            step=1,
                            info="Minimum keyframes to include around each segment",
                        )
                        dgs_neighbor_segments_max = gr.Slider(
                            label="Max neighbor segments",
                            minimum=2,
                            maximum=10,
                            value=8,
                            step=1,
                            info="Maximum keyframes to include (sweep 4, 6, 8)",
                        )
                        dgs_neighbor_segments_step = gr.Slider(
                            label="Neighbor segments step",
                            minimum=1,
                            maximum=4,
                            value=2,
                            step=1,
                            info="Step between values (4, 6, 8)",
                        )
                        dgs_densification_min = gr.Slider(
                            label="Min densification factor",
                            minimum=1,
                            maximum=8,
                            value=2,
                            step=1,
                            info="1x = 705k splats, 2x = 1.4M splats (sweep 2, 4, 6)",
                        )
                        dgs_densification_max = gr.Slider(
                            label="Max densification factor",
                            minimum=1,
                            maximum=8,
                            value=6,
                            step=1,
                            info="8x = 5.6M splats (may OOM on 24GB GPUs)",
                        )
                        dgs_densification_step = gr.Slider(
                            label="Densification step",
                            minimum=1,
                            maximum=4,
                            value=2,
                            step=1,
                            info="Step between values (2, 4, 6)",
                        )
                        dgs_nearclip_min = gr.Slider(
                            label="Min near-clip distance",
                            minimum=0.00,
                            maximum=1.0,
                            value=0.05,
                            step=0.01,
                            info="Filter out splats too close (0.00 = disabled, reduces 'straw' artifacts)",
                        )
                        dgs_nearclip_max = gr.Slider(
                            label="Max near-clip distance",
                            minimum=0.00,
                            maximum=1.0,
                            value=0.15,
                            step=0.01,
                            info="Test range: 0.05, 0.10, 0.15 (0.00 = disabled)",
                        )
                        dgs_nearclip_step = gr.Slider(
                            label="Near-clip step",
                            minimum=0.01,
                            maximum=0.5,
                            value=0.05,
                            step=0.01,
                            info="Step between values",
                        )

                        # 3DGS test action buttons
                        with gr.Row():
                            dgs_run_btn = gr.Button(f"{rocket} Run 3DGS Tests", variant="primary", size="lg")
                            dgs_stop_btn = gr.Button(f"{stop} Stop", variant="stop")

                        with gr.Row():
                            dgs_refresh_btn = gr.Button(f"{refresh} Refresh Results", size="sm")
                            dgs_open_dir_btn = gr.Button(f"{folder} Open Results Directory", size="sm")

                        dgs_status_box = gr.Textbox(
                            label="Status",
                            value="Ready",
                            interactive=False,
                            lines=3,
                        )

                    # Right column: 3DGS test results
                    with gr.Column(scale=2):
                        gr.Markdown("## DA3-3DGS Test Results")
                        dgs_progress_bar = gr.Progress()

                        with gr.Tabs():
                            with gr.Tab("Graph"):
                                gr.Markdown("""
                                ### Quality vs Parameters

                                **Y-axis:** Quality metric (iterations until off-screen / temporal consistency)
                                **X-axis:** Parameter being swept

                                Compare different DA3-3DGS configurations to find optimal settings.
                                """)

                                dgs_graph_html = gr.HTML(
                                    label="3DGS Tuning Graph",
                                    value="<p>Run tests to generate graph...</p>",
                                )

                            with gr.Tab("Results Table"):
                                gr.Markdown("""
                                ### All DA3-3DGS Test Configurations

                                **DA3-3DGS Parameters:**
                                - Scene Strategy: How scenes are built (per-segment/per-prompt/rolling-window)
                                - Model: DA3-GIANT (4GB) or DA3NESTED-GIANT-LARGE (4.5GB)
                                - Neighbor Segments: Keyframes around each segment (4-10)
                                - Densification: Splat count multiplier (1-8x)
                                - Near-Clip: Distance filtering threshold (0.01-1.0)

                                **Quality Metrics:**
                                - Visual Quality: Subjective quality score (1-10)
                                - Temporal Consistency: Frame-to-frame stability (SSIM)
                                - Coordinate Drift: How much scenes drift between segments
                                - VRAM Usage: Peak memory usage in GB
                                - Render Time: Time per frame in seconds
                                """)

                                dgs_results_table = gr.DataFrame(
                                    headers=[
                                        "Aspect Ratio",
                                        "Scene Strategy",
                                        "Model",
                                        "Neighbor Segments",
                                        "Densification",
                                        "Near-Clip",
                                        "Visual Quality",
                                        "Temporal Consistency",
                                        "Coordinate Drift",
                                        "VRAM (GB)",
                                        "Render Time (s/frame)",
                                    ],
                                    label="DA3-3DGS Test Results",
                                    interactive=False,
                                )

                            with gr.Tab("Best Parameters"):
                                gr.Markdown("### Optimal DA3-3DGS Configuration")

                                dgs_best_params = gr.JSON(
                                    label="Best 3DGS Settings for Each Aspect Ratio",
                                    value={},
                                )

                            with gr.Tab("VRAM Analysis"):
                                gr.Markdown("""
                                ### VRAM Usage Analysis

                                **Key Insights:**
                                - Base model: DA3-GIANT ~4GB, DA3NESTED-GIANT-LARGE ~4.5GB
                                - Per keyframe overhead: ~200MB
                                - Splat memory: (705k × densification) splats × ~2.5MB per million
                                - Resolution overhead: (W×H / 1024²) × 500MB

                                **Examples:**
                                - 10 keyframes @ 8x density @ 512×512: ~20GB VRAM
                                - 50 keyframes @ 2x density @ 512×512: ~17GB VRAM
                                - 30 keyframes @ 4x density @ 1024×1024: ~24GB VRAM

                                **Recommendations:**
                                - 16GB GPU: max 4x densification, 30 keyframes
                                - 24GB GPU: max 6x densification, 50 keyframes
                                - 40GB GPU: max 8x densification, 100 keyframes
                                """)

                                dgs_vram_plot = gr.HTML(
                                    label="VRAM Usage vs Parameters",
                                    value="<p>Run tests to generate VRAM analysis...</p>",
                                )

        # Wire up event handlers
        import requests
        import time

        # Store current test ID
        current_test_id = {"id": None}

        def on_run_tests(
            test_type_val,
            test_steps_val,
            strength_min_val,
            strength_max_val,
            strength_step_val,
            kf_strength_min_val,
            kf_strength_max_val,
            kf_strength_step_val,
            max_iterations_val,
            grayscale_threshold_val,
            aspect_ratios_val,
            rotation_factor_min_val,
            rotation_factor_max_val,
            rotation_factor_step_val,
            orbit_radius_val,
            orbit_iterations_val,
        ):
            """Start tuning tests via API."""
            try:
                # Parse step values
                steps_mapping = {
                    "4 (Schnell)": 4,
                    "8": 8,
                    "12": 12,
                    "16": 16,
                    "20 (Dev)": 20,
                }
                steps = [steps_mapping[s] for s in test_steps_val]

                # Map test type to API enum
                test_type_mapping = {
                    "Color Preservation (I2V Chaining)": "color_preservation",
                    "Temporal Consistency (Frame Stability)": "temporal_consistency",
                    "Flux Parameter Sweep": "flux_parameter_sweep",
                    "Depth Warping Orbit (Translation/Rotation Factor)": "depth_warping_orbit",
                }

                # Build API request
                config = {
                    "test_type": test_type_mapping[test_type_val],
                    "steps": steps,
                    "strength_min": strength_min_val,
                    "strength_max": strength_max_val,
                    "strength_step": strength_step_val,
                    "kf_strength_min": kf_strength_min_val,
                    "kf_strength_max": kf_strength_max_val,
                    "kf_strength_step": kf_strength_step_val,
                    "max_iterations": int(max_iterations_val),
                    "grayscale_threshold": grayscale_threshold_val,
                }

                # Add orbit-specific parameters if testing depth warping orbit
                if test_type_val == "Depth Warping Orbit (Translation/Rotation Factor)":
                    # Parse aspect ratios
                    aspect_mapping = {
                        "16:9 (Landscape)": (16/9, 512, 288),
                        "9:16 (Portrait)": (9/16, 288, 512),
                        "1:1 (Square)": (1.0, 512, 512),
                    }
                    aspect_configs = [aspect_mapping[ar] for ar in aspect_ratios_val]

                    config.update({
                        "aspect_ratios": aspect_configs,
                        "rotation_factor_min": rotation_factor_min_val,
                        "rotation_factor_max": rotation_factor_max_val,
                        "rotation_factor_step": rotation_factor_step_val,
                        "orbit_radius": orbit_radius_val,
                        "orbit_iterations": int(orbit_iterations_val),
                    })

                # Submit to API
                logger.debug(f"Submitting tuning config: {json.dumps(config, indent=2)}")
                response = requests.post(
                    "http://localhost:7860/deforum_api/tuning/start",
                    json=config,
                )

                # Log detailed error for 422 validation failures
                if response.status_code == 422:
                    try:
                        error_detail = response.json()
                        logger.error(f"API validation error (422): {json.dumps(error_detail, indent=2)}")
                    except:
                        logger.error(f"API validation error (422): {response.text}")

                response.raise_for_status()

                result = response.json()
                current_test_id["id"] = result["test_id"]

                logger.info(f"Started tuning test: {current_test_id['id']}")
                return f"{check} Test started: {current_test_id['id']}\nStatus: {result['status']}"

            except Exception as e:
                logger.error(f"Failed to start tuning test: {e}", exc_info=True)
                return f"{cross} Error: {str(e)}"

        def on_stop_tests():
            """Stop running tests via API."""
            if not current_test_id["id"]:
                return "No test running"

            try:
                response = requests.post(
                    f"http://localhost:7860/deforum_api/tuning/{current_test_id['id']}/cancel"
                )
                response.raise_for_status()
                logger.info(f"Cancelled test: {current_test_id['id']}")
                return f"{stop} Test cancelled"
            except Exception as e:
                logger.error(f"Failed to cancel test: {e}", exc_info=True)
                return f"{cross} Error: {str(e)}"

        def on_apply_best():
            """Apply best parameters to defaults."""
            logger.info("Applying best parameters...")
            return f"{sparkles} Applied! (Feature coming soon)"

        def on_open_tuning_dir():
            """Open the tuning output directory in file browser."""
            from pathlib import Path
            from modules.util import open_folder
            import os

            # Use Forge's standard output directory (same as normal generations)
            forge_root = Path(os.getcwd())
            tuning_dir = forge_root / "output" / "deforum-tuning"
            tuning_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Opening tuning directory: {tuning_dir}")
            open_folder(str(tuning_dir))
            return f"{folder} Opened: {tuning_dir}"

        def poll_test_status():
            """Poll for test status updates."""
            if not current_test_id["id"]:
                return None, None, None, None, None

            try:
                response = requests.get(
                    f"http://localhost:7860/deforum_api/tuning/{current_test_id['id']}"
                )
                response.raise_for_status()
                status = response.json()

                # Format status message
                status_msg = (
                    f"Test ID: {status['test_id']}\n"
                    f"Status: {status['status']}\n"
                    f"Progress: {status['progress']*100:.1f}%"
                )

                # Format results table and charts
                if status["results"]:
                    import pandas as pd
                    from deforum.ui.tuning_charts import (
                        create_metrics_plot,
                        create_heatmap_plot,
                        find_best_configuration,
                        create_orbit_metrics_plot,
                        create_orbit_heatmap,
                        find_best_orbit_configuration,
                        create_3dgs_plotly_metrics,
                        create_3dgs_plotly_heatmap,
                        find_best_3dgs_configuration,
                    )

                    df = pd.DataFrame(status["results"])

                    # Detect test type from results structure
                    first_result = status["results"][0]
                    is_orbit = 'rotation_factor' in first_result
                    # 3DGS tests (both synthetic and real) have 'densification' and 'neighbor_segments'
                    is_3dgs_test = 'densification' in first_result and 'neighbor_segments' in first_result

                    if is_orbit:
                        # Orbit test visualization
                        best_config = find_best_orbit_configuration(status["results"])
                        metrics_fig = create_orbit_metrics_plot(status["results"])
                        heatmap_fig = create_orbit_heatmap(status["results"], 'iterations_until_offscreen')
                    elif is_3dgs_test:
                        # 3DGS test visualization (both synthetic and real) - plotly for interactivity
                        best_config = find_best_3dgs_configuration(status["results"])
                        metrics_fig = create_3dgs_plotly_metrics(status["results"])  # Returns HTML string
                        heatmap_fig = create_3dgs_plotly_heatmap(status["results"], 'overall_score')  # Returns HTML string
                    else:
                        # Standard I2V chaining test visualization
                        best_config = find_best_configuration(status["results"])
                        metrics_fig = create_metrics_plot(status["results"])
                        heatmap_fig = create_heatmap_plot(status["results"], 'overall_score')

                    return status_msg, df, best_config, metrics_fig, heatmap_fig

                return status_msg, None, None, None, None

            except Exception as e:
                import traceback
                logger.error(f"Failed to poll test status: {e}")
                logger.error(traceback.format_exc())
                return f"Error polling status: {e}", None, None, None, None

        # I2V Chaining Tests button handlers
        run_tests_btn.click(
            fn=on_run_tests,
            inputs=[
                test_type,
                test_steps,
                strength_min,
                strength_max,
                strength_step,
                kf_strength_min,
                kf_strength_max,
                kf_strength_step,
                max_iterations,
                grayscale_threshold,
                gr.State([]),  # Dummy orbit params (not used for I2V tests)
                gr.State(-7.0),
                gr.State(-3.0),
                gr.State(0.05),
                gr.State(2.0),
                gr.State(50),
            ],
            outputs=[status_box],
        )

        # Orbit Tests button handlers
        orbit_run_btn.click(
            fn=on_run_tests,
            inputs=[
                gr.State("Depth Warping Orbit (Translation/Rotation Factor)"),  # Force orbit test type
                gr.State(["20 (Dev)"]),  # Dummy steps (not used for orbit)
                gr.State(0.80),  # Dummy I2V params (not used for orbit)
                gr.State(0.95),
                gr.State(0.05),
                gr.State(0.10),
                gr.State(0.25),
                gr.State(0.05),
                gr.State(20),
                gr.State(20.0),
                orbit_aspect_ratios,  # Actual orbit params
                orbit_rotation_factor_min,
                orbit_rotation_factor_max,
                orbit_rotation_factor_step,
                orbit_orbit_radius,
                orbit_orbit_iterations,
            ],
            outputs=[orbit_status_box],
        )

        stop_tests_btn.click(
            fn=on_stop_tests,
            inputs=[],
            outputs=[status_box],
        )

        orbit_stop_btn.click(
            fn=on_stop_tests,
            inputs=[],
            outputs=[orbit_status_box],
        )

        apply_best_btn.click(
            fn=on_apply_best,
            inputs=[],
            outputs=[status_box],
        )

        open_tuning_dir_btn.click(
            fn=on_open_tuning_dir,
            inputs=[],
            outputs=[status_box],
        )

        orbit_open_dir_btn.click(
            fn=on_open_tuning_dir,
            inputs=[],
            outputs=[orbit_status_box],
        )

        refresh_btn.click(
            fn=poll_test_status,
            inputs=[],
            outputs=[status_box, results_table, best_params_json, metrics_plot, heatmap_plot],
        )

        orbit_refresh_btn.click(
            fn=poll_test_status,
            inputs=[],
            outputs=[orbit_status_box, orbit_results_table, orbit_best_params, orbit_metrics_plot, orbit_heatmap_plot],
        )

        # RAFT Tests button handlers
        def on_run_raft_tests(
            raft_aspect_ratios_val,
            raft_rotation_factor_val,
            raft_orbit_radius_val,
            raft_orbit_iterations_val,
            raft_model_sizes_val,
            raft_flow_iterations_min_val,
            raft_flow_iterations_max_val,
            raft_flow_iterations_step_val,
            raft_flow_factor_min_val,
            raft_flow_factor_max_val,
            raft_flow_factor_step_val,
        ):
            """Start RAFT tuning tests via API."""
            try:
                # Parse aspect ratios
                aspect_configs = []
                for aspect_str in raft_aspect_ratios_val:
                    if "16:9" in aspect_str:
                        aspect_configs.append([1.78, 512, 288])
                    elif "9:16" in aspect_str:
                        aspect_configs.append([0.56, 288, 512])
                    elif "1:1" in aspect_str:
                        aspect_configs.append([1.0, 512, 512])

                # Create RAFT test config
                config = {
                    "test_type": "raft_tuning",
                    "aspect_ratios": aspect_configs,
                    "raft_rotation_factor": raft_rotation_factor_val,
                    "orbit_radius": raft_orbit_radius_val,
                    "orbit_iterations": int(raft_orbit_iterations_val),
                    "raft_model_sizes": raft_model_sizes_val,
                    "raft_flow_iterations_min": int(raft_flow_iterations_min_val),
                    "raft_flow_iterations_max": int(raft_flow_iterations_max_val),
                    "raft_flow_iterations_step": int(raft_flow_iterations_step_val),
                    "raft_flow_factor_min": float(raft_flow_factor_min_val),
                    "raft_flow_factor_max": float(raft_flow_factor_max_val),
                    "raft_flow_factor_step": float(raft_flow_factor_step_val),
                }

                # Submit test
                response = requests.post(
                    "http://localhost:7860/deforum_api/tuning/start",
                    json=config
                )
                response.raise_for_status()
                data = response.json()

                # Store test ID
                current_test_id["id"] = data["test_id"]

                return f"RAFT tests started. Test ID: {data['test_id']}\nRunning..."

            except Exception as e:
                logger.error(f"Failed to start RAFT tests: {e}")
                return f"Error starting RAFT tests: {e}"

        raft_run_btn.click(
            fn=on_run_raft_tests,
            inputs=[
                raft_aspect_ratios,
                raft_rotation_factor,
                raft_orbit_radius,
                raft_orbit_iterations,
                raft_model_sizes,
                raft_flow_iterations_min,
                raft_flow_iterations_max,
                raft_flow_iterations_step,
                raft_flow_factor_min,
                raft_flow_factor_max,
                raft_flow_factor_step,
            ],
            outputs=[raft_status_box],
        )

        raft_stop_btn.click(
            fn=on_stop_tests,
            inputs=[],
            outputs=[raft_status_box],
        )

        raft_open_dir_btn.click(
            fn=on_open_tuning_dir,
            inputs=[],
            outputs=[raft_status_box],
        )

        raft_refresh_btn.click(
            fn=poll_test_status,
            inputs=[],
            outputs=[raft_status_box, raft_results_table, raft_best_params, raft_graph_html, gr.State(None)],
        )

        # DA3-3DGS Tests button handlers
        def on_run_dgs_tests(
            dgs_aspect_ratios_val,
            dgs_rotation_factor_val,
            dgs_orbit_radius_val,
            dgs_test_iterations_val,
            dgs_scene_strategies_val,
            dgs_rolling_window_size_val,
            dgs_max_prompt_keyframes_val,
            dgs_models_val,
            dgs_neighbor_segments_min_val,
            dgs_neighbor_segments_max_val,
            dgs_neighbor_segments_step_val,
            dgs_densification_min_val,
            dgs_densification_max_val,
            dgs_densification_step_val,
            dgs_nearclip_min_val,
            dgs_nearclip_max_val,
            dgs_nearclip_step_val,
        ):
            """Start DA3-3DGS tuning tests via API."""
            try:
                # DEBUG: Log received values to diagnose Gradio caching issue
                logger.info(f"[UI DEBUG] Received DA3-3DGS test parameters:")
                logger.info(f"  models: {dgs_models_val}")
                logger.info(f"  neighbor_segments: {dgs_neighbor_segments_min_val}-{dgs_neighbor_segments_max_val} step {dgs_neighbor_segments_step_val}")
                logger.info(f"  densification: {dgs_densification_min_val}-{dgs_densification_max_val} step {dgs_densification_step_val}")
                logger.info(f"  nearclip: {dgs_nearclip_min_val}-{dgs_nearclip_max_val} step {dgs_nearclip_step_val}")

                # Parse aspect ratios
                aspect_configs = []
                for aspect_str in dgs_aspect_ratios_val:
                    if "16:9" in aspect_str:
                        aspect_configs.append([1.78, 512, 288])
                    elif "9:16" in aspect_str:
                        aspect_configs.append([0.56, 288, 512])
                    elif "1:1" in aspect_str:
                        aspect_configs.append([1.0, 512, 512])

                # Create DA3-3DGS test config (using synthetic test images, no diffusion)
                # IMPORTANT: Keys must match TuningTestConfig field names (with dgs_ prefix)
                config = {
                    "test_type": "da3_3dgs_synthetic",
                    "aspect_ratios": aspect_configs,
                    "rotation_factor": dgs_rotation_factor_val,
                    "orbit_radius": dgs_orbit_radius_val,
                    "test_iterations": int(dgs_test_iterations_val),
                    "dgs_scene_strategies": dgs_scene_strategies_val,
                    "dgs_rolling_window_size": int(dgs_rolling_window_size_val),
                    "dgs_max_prompt_keyframes": int(dgs_max_prompt_keyframes_val),
                    "dgs_models": dgs_models_val,
                    "dgs_neighbor_segments_min": int(dgs_neighbor_segments_min_val),
                    "dgs_neighbor_segments_max": int(dgs_neighbor_segments_max_val),
                    "dgs_neighbor_segments_step": int(dgs_neighbor_segments_step_val),
                    "dgs_densification_min": int(dgs_densification_min_val),
                    "dgs_densification_max": int(dgs_densification_max_val),
                    "dgs_densification_step": int(dgs_densification_step_val),
                    "dgs_nearclip_min": float(dgs_nearclip_min_val),
                    "dgs_nearclip_max": float(dgs_nearclip_max_val),
                    "dgs_nearclip_step": float(dgs_nearclip_step_val),
                }

                # Submit test
                response = requests.post(
                    "http://localhost:7860/deforum_api/tuning/start",
                    json=config
                )
                response.raise_for_status()
                data = response.json()

                # Store test ID
                current_test_id["id"] = data["test_id"]

                return f"{check} DA3-3DGS tests started. Test ID: {data['test_id']}\nRunning..."

            except Exception as e:
                logger.error(f"Failed to start DA3-3DGS tests: {e}")
                return f"{cross} Error starting DA3-3DGS tests: {e}"

        dgs_run_btn.click(
            fn=on_run_dgs_tests,
            inputs=[
                dgs_aspect_ratios,
                dgs_rotation_factor,
                dgs_orbit_radius,
                dgs_test_iterations,
                dgs_scene_strategies,
                dgs_rolling_window_size,
                dgs_max_prompt_keyframes,
                dgs_models,
                dgs_neighbor_segments_min,
                dgs_neighbor_segments_max,
                dgs_neighbor_segments_step,
                dgs_densification_min,
                dgs_densification_max,
                dgs_densification_step,
                dgs_nearclip_min,
                dgs_nearclip_max,
                dgs_nearclip_step,
            ],
            outputs=[dgs_status_box],
        )

        dgs_stop_btn.click(
            fn=on_stop_tests,
            inputs=[],
            outputs=[dgs_status_box],
        )

        dgs_open_dir_btn.click(
            fn=on_open_tuning_dir,
            inputs=[],
            outputs=[dgs_status_box],
        )

        dgs_refresh_btn.click(
            fn=poll_test_status,
            inputs=[],
            outputs=[dgs_status_box, dgs_results_table, dgs_best_params, dgs_graph_html, dgs_vram_plot],
        )

    return tuning_interface, "Deforum Tuning", "deforum_tuning"


def should_show_tuning_tab() -> bool:
    """Check if tuning tab should be shown based on CLI flags.

    Returns:
        True if --deforum-run-tuning flag is set
    """
    try:
        from modules.shared import cmd_opts
        return getattr(cmd_opts, 'deforum_run_tuning', False)
    except:
        return False
