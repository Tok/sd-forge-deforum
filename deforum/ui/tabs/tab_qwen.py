"""Qwen AI Enhancement tab for Deforum UI.

Standalone tab for Qwen AI prompt enhancement with model management.
Note: This is currently integrated into the Prompts tab accordion,
but extracted here for future standalone use.
"""

import gradio as gr
from types import SimpleNamespace
from modules.ui_components import FormRow
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.ui.builders import create_gr_elem


def get_tab_qwen(dw: SimpleNamespace):
    """Create the AI Prompt Enhancement (Qwen) tab.

    Args:
        dw: DeforumWanArgs namespace

    Returns:
        dict: Component dictionary for event binding
    """
    # Import handlers from ui_elements (temporary - will be extracted later)
    from deforum.ui.ui_elements import (
        check_qwen_models_handler,
        download_qwen_model_handler,
        cleanup_qwen_cache_handler
    )

    with gr.TabItem(f"🧠 AI Enhancement"):
        gr.Markdown("""
        ## AI Prompt Enhancement with Qwen

        **Enhance your prompts using Qwen AI models** for better generation quality:
        - Refines and expands prompt descriptions
        - Analyzes Deforum movement schedules
        - Translates technical motion into descriptive language
        - Supports English and Chinese

        **Usage:**
        1. Write your base prompts in the Prompts tab above
        2. Configure Qwen settings below
        3. Enhancement integrates automatically when generating

        **Note:** Qwen models are lazy-loaded only when needed and auto-cleanup before generation to free VRAM.
        """)

        # Qwen Settings
        with gr.Accordion(f"{emoji_utils.gear()} Qwen Settings", open=True):
            with FormRow():
                wan_qwen_model = create_gr_elem(dw.wan_qwen_model)
                wan_qwen_language = create_gr_elem(dw.wan_qwen_language)
                wan_qwen_auto_download = create_gr_elem(dw.wan_qwen_auto_download)

        # Model Management
        with gr.Accordion(f"{emoji_utils.wrench()} Model Management", open=False):
            gr.Markdown("""
            **Model Information & Status**

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

        # Connect event handlers for Qwen model management
        check_qwen_models_btn.click(
            fn=check_qwen_models_handler,
            inputs=[wan_qwen_model],
            outputs=[qwen_model_status]
        )

        download_qwen_model_btn.click(
            fn=download_qwen_model_handler,
            inputs=[wan_qwen_model, wan_qwen_auto_download],
            outputs=[qwen_model_status]
        )

        cleanup_qwen_cache_btn.click(
            fn=cleanup_qwen_cache_handler,
            inputs=[],
            outputs=[qwen_model_status]
        )

        # Auto-update model status when model selection changes
        wan_qwen_model.change(
            fn=check_qwen_models_handler,
            inputs=[wan_qwen_model],
            outputs=[qwen_model_status]
        )

    return {k: v for k, v in {**locals(), **vars()}.items()}
