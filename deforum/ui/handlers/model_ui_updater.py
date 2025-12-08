"""Model-based UI updater handler.

Updates UI component interactivity based on currently loaded model capabilities.
"""

import gradio as gr
from deforum.utils.system.logging import get_logger, emoji_if_enabled

logger = get_logger()


def update_ui_for_model():
    """Check current model and return gr.update() dicts for UI components.

    Returns:
        tuple: (negative_prompt_update, cfg_update, distilled_cfg_update, status_msg)
    """
    try:
        from deforum.utils.model_detection import (
            is_flux_model,
            is_lumina_model,
            is_zimage_model,
            get_model_name
        )

        model_name = get_model_name()
        logger.debug(f"Updating UI for model: {model_name}")

        # Determine which models ignore which parameters
        ignores_negative = is_flux_model() or is_lumina_model() or is_zimage_model()
        ignores_cfg = is_zimage_model()  # Z-Image has no CFG at all
        uses_distilled_cfg = is_flux_model()

        # Create update dicts for Gradio components
        negative_prompt_update = gr.update(
            interactive=not ignores_negative,
            placeholder=(
                "Negative prompts ignored by this model" if ignores_negative
                else "Words to avoid in generation"
            )
        )

        cfg_update = gr.update(
            interactive=not ignores_cfg,
            info=(
                "CFG not used by this model (distilled)" if ignores_cfg
                else "Guidance strength for traditional CFG"
            )
        )

        distilled_cfg_update = gr.update(
            visible=uses_distilled_cfg,
            interactive=uses_distilled_cfg
        )

        # Status message
        check = emoji_if_enabled("✓") or "OK"
        warning = emoji_if_enabled("⚠️") or "!"

        if ignores_negative and ignores_cfg:
            status = f"{warning} {model_name} is a distilled model - negative prompts and CFG disabled"
        elif ignores_negative:
            status = f"{warning} {model_name} ignores negative prompts (use distilled CFG instead)"
        elif ignores_cfg:
            status = f"{warning} {model_name} doesn't use CFG (distilled model)"
        else:
            status = f"{check} {model_name} - all parameters available"

        logger.info(f"UI updated for {model_name}")
        return negative_prompt_update, cfg_update, distilled_cfg_update, status

    except Exception as e:
        logger.error(f"Failed to update UI for model: {e}")
        warning = emoji_if_enabled("❌") or "ERROR"
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            f"{warning} Failed to detect model - UI not updated"
        )
