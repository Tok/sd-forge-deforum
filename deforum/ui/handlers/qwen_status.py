"""Qwen model status checking helpers.

Extracted from ui_elements.py to reduce complexity.
"""

from typing import Dict, List, Tuple, Optional, Any


def _load_qwen_emojis() -> Dict[str, str]:
    """Load all emoji symbols for Qwen status display."""
    from deforum.utils.system.logging import emoji as emoji_utils

    return {
        'check': emoji_utils.maybe_check(),
        'cross': emoji_utils.maybe_cross(),
        'warning': emoji_utils.maybe_warning(),
        'palette': emoji_utils.palette(),
        'hourglass': emoji_utils.hourglass(),
        'refresh': emoji_utils.refresh_icon(),
        'zzz': emoji_utils.sleeping(),
        'fire': emoji_utils.fire(),
    }


def _build_model_selection_status(
    qwen_model: str,
    qwen_manager: Any,
    available_vram: float,
    emojis: Dict[str, str]
) -> Tuple[str, Dict[str, Any]]:
    """Build model selection status section.

    Args:
        qwen_model: Selected model name
        qwen_manager: Qwen manager instance
        available_vram: Available VRAM in GB
        emojis: Emoji symbol dictionary

    Returns:
        Tuple of (actual_model_name, model_info_dict)
    """
    status_parts = [f"<strong style='color: #333;'>Selected Model:</strong> {qwen_model}"]

    if qwen_model == "Auto-Select":
        auto_selected = qwen_manager.auto_select_model()
        status_parts.append(f"<strong style='color: #333;'>Auto-Selected:</strong> {auto_selected}")
        status_parts.append(
            f"<strong style='color: #333;'>Reason:</strong> "
            f"Best fit for {available_vram:.1f}GB VRAM"
        )
        qwen_model = auto_selected

    model_info = qwen_manager.get_model_info(qwen_model)
    return qwen_model, model_info, status_parts


def _build_model_info_status(
    model_info: Optional[Dict[str, Any]],
    available_vram: float,
    emojis: Dict[str, str]
) -> List[str]:
    """Build model information status section."""
    if not model_info:
        return []

    status_parts = [
        f"<strong style='color: #333;'>Description:</strong> {model_info.get('description', 'N/A')}",
        f"<strong style='color: #333;'>VRAM Required:</strong> {model_info.get('vram_gb', 'Unknown')}GB",
        f"<strong style='color: #333;'>Available VRAM:</strong> {available_vram:.1f}GB",
    ]

    vram_required = model_info.get('vram_gb', 0)
    if vram_required <= available_vram:
        status_parts.append(
            f"{emojis['check']} <span style='color: #4CAF50;'>VRAM requirement met</span>"
        )
    else:
        status_parts.append(
            f"{emojis['warning']} <span style='color: #FF9800;'>May exceed available VRAM</span>"
        )

    return status_parts


def _build_download_status(
    is_downloaded: bool,
    model_info: Optional[Dict[str, Any]],
    emojis: Dict[str, str]
) -> List[str]:
    """Build download status section."""
    if is_downloaded:
        return [
            f"{emojis['check']} <span style='color: #4CAF50;'>Model downloaded and available</span>"
        ]

    status_parts = [
        f"{emojis['cross']} <span style='color: #f44336;'>Model not downloaded</span>"
    ]

    if model_info and 'hf_name' in model_info:
        status_parts.append(
            f"<strong style='color: #333;'>HuggingFace ID:</strong> {model_info['hf_name']}"
        )

    return status_parts


def _build_loading_status(
    is_loaded: bool,
    loaded_info: Optional[Dict[str, Any]],
    qwen_model: str,
    emojis: Dict[str, str]
) -> List[str]:
    """Build loading status section."""
    if not is_loaded:
        return [f"{emojis['zzz']} <span style='color: #333;'>No model currently loaded</span>"]

    if loaded_info and loaded_info['name'] == qwen_model:
        status_parts = [
            f"{emojis['fire']} <span style='color: #4CAF50;'>Model currently loaded and ready</span>"
        ]

        estimated_vram = loaded_info.get('vram_usage', 0)
        if estimated_vram > 0:
            status_parts.append(
                f"<strong style='color: #333;'>Estimated VRAM usage:</strong> {estimated_vram:.1f}GB"
            )

        return status_parts

    current_model = loaded_info['name'] if loaded_info else "Unknown"
    return [
        f"{emojis['refresh']} <span style='color: #FF9800;'>Different model loaded: {current_model}</span>",
        "<span style='color: #333;'>Will switch on next enhancement</span>",
    ]


def _build_quick_setup_instructions(
    is_downloaded: bool,
    is_loaded: bool,
    emojis: Dict[str, str]
) -> List[str]:
    """Build quick setup instructions section."""
    if not is_downloaded:
        return [
            "<br><strong style='color: #333;'>Quick Setup:</strong>",
            f"1. {emojis['check']} Enable 'Auto-Download Qwen Models' above",
            f"2. {emojis['palette']} Click 'AI Prompt Enhancement' for auto-download",
            f"3. {emojis['hourglass']} Wait for download to complete",
        ]

    if not is_loaded:
        return [
            "<br><strong style='color: #333;'>Ready to Use:</strong>",
            f"{emojis['palette']} Click 'AI Prompt Enhancement' to load and use this model",
        ]

    return ["<br><strong style='color: #333;'>Status:</strong> Ready for prompt enhancement!"]
