"""Prompts tab for Deforum UI.

Contains animation prompts, positive/negative prompt templates, prompt timing,
AI enhancement (Qwen), and FPS converter utilities.
"""

import gradio as gr
from types import SimpleNamespace
from modules.ui_components import FormRow
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.ui.builders import create_gr_elem, create_row
from deforum.config.defaults import get_gradio_html, DeforumAnimPrompts


def get_tab_prompts(da, dw, dv=None):
    """Create the Prompts tab with animation prompts and AI enhancement.

    Args:
        da: DeforumAnimArgs namespace
        dw: DeforumWanArgs namespace
        dv: DeforumOutputArgs namespace (optional)

    Returns:
        dict: Component dictionary for event binding
    """
    # Import dv if not provided
    if dv is None:
        from deforum.config.args import DeforumOutputArgs
        dv = SimpleNamespace(**DeforumOutputArgs())

    # Import handlers from ui_elements (temporary - will be extracted later)
    from deforum.ui.ui_elements import (
        check_qwen_models_handler,
        download_qwen_model_handler,
        cleanup_qwen_cache_handler,
        convert_fps_handler
    )

    with gr.TabItem(f"{emoji_utils.prompts()} Prompts"):
        # PROMPTS INFO ACCORD
        with gr.Accordion(
            label='*Important* notes on Prompts',
            elem_id='prompts_info_accord',
            open=False
        ) as prompts_info_accord:
            gr.HTML(value=get_gradio_html('prompts'))

        animation_prompts = create_row(
            gr.Textbox(
                label="Prompts",
                lines=8,
                interactive=True,
                value=DeforumAnimPrompts(),
                info="""Full prompts list in a JSON format. The value on left side is the frame number and
                     its presence also defines the frame as a keyframe if a 'keyframe distribution' mode
                     is active. Duplicating the same prompt multiple times to define keyframes
                     is therefore expected and fine."""
            )
        )

        animation_prompts_positive = create_row(
            gr.Textbox(
                label="Prompts positive",
                lines=1,
                interactive=True,
                placeholder="words in here will be added to the start of all positive prompts"
            )
        )

        animation_prompts_negative = create_row(
            gr.Textbox(
                label="Prompts negative",
                value="nsfw, nude",
                lines=1,
                interactive=True,
                placeholder="words here will be added to the end of all negative prompts.  ignored with Flux."
            )
        )

        # PROMPT TIMING SETTINGS
        stopwatch = emoji_utils.stopwatch()
        with gr.Accordion(f"{stopwatch} Prompt Timing", open=False):
            gr.Markdown("""
            **Prompt Authored FPS:** If you authored prompts at a different FPS (e.g., 60 FPS) but want to render at another (e.g., 24 FPS), set this to auto-convert frame numbers.

            **Audio settings** have been moved to Init → Audio Sync tab.
            """)

            with FormRow() as prompt_fps_row:
                prompt_authored_fps = create_gr_elem(dv.prompt_authored_fps)

        # AI PROMPT ENHANCEMENT - Qwen integration
        brain = emoji_utils.brain()
        with gr.Accordion(f"{brain} AI Prompt Enhancement (Qwen)", open=False):
            gr.Markdown("""
            **Enhance your prompts using Qwen AI models** for better generation quality:
            - Refines and expands prompt descriptions
            - Analyzes Deforum movement schedules (translation, rotation, zoom)
            - Translates technical motion into descriptive language
            - Supports English and Chinese output

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

        # FPS CONVERTER
        with gr.Accordion(f"{emoji_utils.stopwatch()} FPS Converter", open=False):
            gr.Markdown("""
            **Convert prompt frame numbers between different FPS settings**

            Use this when you need to:
            - Convert prompts synced to 60 FPS for use at 24 FPS (or any other FPS)
            - Adjust timing when changing video output FPS
            - Rescale animation timing to different frame rates

            **Example:** Prompts synced to amen break at 60 FPS → Convert to 24 FPS for Wan video generation

            **Formula:** `new_frame = old_frame × (target_fps / source_fps)`
            """)

            with FormRow():
                fps_converter_source = gr.Number(
                    label="Source FPS",
                    value=60,
                    minimum=1,
                    maximum=240,
                    step=1,
                    info="Current FPS that prompts are synced to"
                )
                fps_converter_target = gr.Number(
                    label="Target FPS",
                    value=24,
                    minimum=1,
                    maximum=240,
                    step=1,
                    info="Desired FPS for prompt conversion"
                )

            refresh = emoji_utils.refresh_icon()
            with FormRow():
                fps_converter_btn = gr.Button(
                    f"{refresh} Convert Prompt Frame Numbers",
                    variant="primary",
                    elem_id="fps_converter_btn"
                )
                fps_converter_preview = gr.Checkbox(
                    label="Preview Only",
                    value=False,
                    info="Show conversion preview without updating prompts"
                )

            fps_converter_output = gr.HTML(
                label="Conversion Result",
                value="",
                elem_id="fps_converter_output"
            )

            # Connect event handler
            fps_converter_btn.click(
                fn=convert_fps_handler,
                inputs=[animation_prompts, fps_converter_source, fps_converter_target, fps_converter_preview],
                outputs=[animation_prompts, fps_converter_output]
            )

        # NOTE: Composable mask scheduling moved to dedicated Masking tab

    return {k: v for k, v in {**locals(), **vars()}.items()}
