"""Tuning UI for automated parameter optimization.

This module provides an interactive UI for running parameter sweeps
and visualizing quality metrics in real-time.
"""

import gradio as gr
from pathlib import Path
from typing import Dict, List, Any
import json

from deforum.utils.system.logging import get_logger

logger = get_logger()


def create_tuning_tab() -> tuple:
    """Create the Tuning tab UI.

    Returns:
        Tuple of (interface, title, id) for WebUI tab registration
    """
    with gr.Blocks() as tuning_interface:
        gr.Markdown("""
        # 🔬 Deforum Parameter Tuning Lab

        Automated quality assessment for finding optimal Deforum parameters.

        **How it works:**
        1. Select parameters to test (strength, steps, etc.)
        2. Define parameter ranges
        3. Run automated tests
        4. View quality metrics and comparisons
        5. Apply best settings to Deforum defaults
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
                        step=0.05,
                    )
                    strength_max = gr.Slider(
                        label="Max strength",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                    )
                    strength_step = gr.Slider(
                        label="Step size",
                        minimum=0.01,
                        maximum=0.1,
                        value=0.05,
                        step=0.01,
                    )

                # Keyframe strength configuration
                with gr.Group():
                    gr.Markdown("**Keyframe Strength**")
                    kf_strength_min = gr.Slider(
                        label="Min keyframe strength",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.10,
                        step=0.05,
                    )
                    kf_strength_max = gr.Slider(
                        label="Max keyframe strength",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.25,
                        step=0.05,
                    )
                    kf_strength_step = gr.Slider(
                        label="Step size",
                        minimum=0.01,
                        maximum=0.1,
                        value=0.05,
                        step=0.01,
                    )

                # Test limits
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
                    run_tests_btn = gr.Button("🚀 Run Tests", variant="primary", size="lg")
                    stop_tests_btn = gr.Button("⏹️ Stop", variant="stop")

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Results", size="sm")
                    open_tuning_dir_btn = gr.Button("📁 Open Tuning Directory", size="sm")

                # Status
                status_box = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    lines=3,
                )

            # Right column: Results and visualization
            with gr.Column(scale=2):
                gr.Markdown("## Results")

                # Progress tracking
                progress_bar = gr.Progress()

                with gr.Tabs():
                    # Summary tab
                    with gr.Tab("Summary"):
                        gr.Markdown("### Best Parameters Found")

                        best_params_json = gr.JSON(
                            label="Optimal Settings",
                            value={},
                        )

                        apply_best_btn = gr.Button(
                            "✨ Apply to Deforum Defaults",
                            variant="secondary",
                        )

                        gr.Markdown("### All Results")

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

                    # Charts tab
                    with gr.Tab("Charts"):
                        gr.Markdown("### Quality Metrics Visualization")

                        # Placeholder for charts
                        metrics_plot = gr.Plot(
                            label="Quality Metrics",
                        )

                        # Heatmap for parameter combinations
                        heatmap_plot = gr.Plot(
                            label="Parameter Heatmap (Overall Score)",
                        )

                    # Image Comparison tab
                    with gr.Tab("Image Comparison"):
                        gr.Markdown("### Visual Comparison")

                        comparison_select = gr.Dropdown(
                            label="Select test configuration",
                            choices=[],
                            value=None,
                        )

                        with gr.Row():
                            iteration_slider = gr.Slider(
                                label="Iteration",
                                minimum=0,
                                maximum=20,
                                value=0,
                                step=1,
                            )

                        with gr.Row():
                            comparison_image = gr.Image(
                                label="Generated Frame",
                                type="filepath",
                            )

                            with gr.Column():
                                color_score_display = gr.Textbox(
                                    label="Color Score",
                                    value="--",
                                    interactive=False,
                                )
                                temporal_score_display = gr.Textbox(
                                    label="Temporal Consistency",
                                    value="--",
                                    interactive=False,
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

                # Submit to API
                response = requests.post(
                    "http://localhost:7860/deforum_api/tuning/start",
                    json=config,
                )
                response.raise_for_status()

                result = response.json()
                current_test_id["id"] = result["test_id"]

                logger.info(f"Started tuning test: {current_test_id['id']}")
                return f"✅ Test started: {current_test_id['id']}\nStatus: {result['status']}"

            except Exception as e:
                logger.error(f"Failed to start tuning test: {e}", exc_info=True)
                return f"❌ Error: {str(e)}"

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
                return "⏹️ Test cancelled"
            except Exception as e:
                logger.error(f"Failed to cancel test: {e}", exc_info=True)
                return f"❌ Error: {str(e)}"

        def on_apply_best():
            """Apply best parameters to defaults."""
            logger.info("Applying best parameters...")
            return "✨ Applied! (Feature coming soon)"

        def on_open_tuning_dir():
            """Open the tuning output directory in file browser."""
            from pathlib import Path
            from modules.util import open_folder
            import os

            # Use Forge's standard outputs directory (same as normal generations)
            forge_root = Path(os.getcwd())
            tuning_dir = forge_root / "outputs" / "deforum-tuning"
            tuning_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Opening tuning directory: {tuning_dir}")
            open_folder(str(tuning_dir))
            return f"📁 Opened: {tuning_dir}"

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
                    )

                    df = pd.DataFrame(status["results"])
                    best_config = find_best_configuration(status["results"])

                    # Generate charts
                    metrics_fig = create_metrics_plot(status["results"])
                    heatmap_fig = create_heatmap_plot(status["results"], 'overall_score')

                    return status_msg, df, best_config, metrics_fig, heatmap_fig

                return status_msg, None, None, None, None

            except Exception as e:
                logger.error(f"Failed to poll test status: {e}")
                return f"Error polling status: {e}", None, None, None, None

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
            ],
            outputs=[status_box],
        )

        stop_tests_btn.click(
            fn=on_stop_tests,
            inputs=[],
            outputs=[status_box],
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

        refresh_btn.click(
            fn=poll_test_status,
            inputs=[],
            outputs=[status_box, results_table, best_params_json, metrics_plot, heatmap_plot],
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
