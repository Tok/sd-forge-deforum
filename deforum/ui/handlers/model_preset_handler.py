"""UI handler for model preset detection and application."""

from typing import Dict, Any, Optional, Tuple
import gradio as gr

from deforum.config.model_presets import (
    detect_model_type,
    get_preset_for_model,
    create_preset_message,
    get_settings_dict_from_preset,
    ModelType,
    ModelPreset
)
from deforum.utils.system.logging import get_logger, emoji_if_enabled

logger = get_logger()


# Global state for tracking model changes
_last_detected_model: Optional[str] = None
_last_model_type: ModelType = ModelType.UNKNOWN


def detect_loaded_model() -> Optional[str]:
    """Detect currently loaded model from Forge.

    Returns:
        Model name/path if detected, None otherwise
    """
    try:
        import modules.shared as shared

        # Try to get current checkpoint info
        if hasattr(shared, 'sd_model') and shared.sd_model is not None:
            if hasattr(shared.sd_model, 'sd_checkpoint_info'):
                checkpoint_info = shared.sd_model.sd_checkpoint_info
                if hasattr(checkpoint_info, 'filename'):
                    return checkpoint_info.filename
                elif hasattr(checkpoint_info, 'name'):
                    return checkpoint_info.name

        # Fallback: check opts for selected checkpoint
        if hasattr(shared, 'opts') and hasattr(shared.opts, 'sd_model_checkpoint'):
            return shared.opts.sd_model_checkpoint

    except Exception as e:
        logger.debug(f"Could not detect loaded model: {e}")

    return None


def check_model_changed() -> Tuple[bool, Optional[str], Optional[ModelType]]:
    """Check if model has changed since last detection.

    Returns:
        Tuple of (changed, model_name, model_type)
    """
    global _last_detected_model, _last_model_type

    current_model = detect_loaded_model()

    if current_model is None:
        return False, None, ModelType.UNKNOWN

    # Detect model type
    current_type = detect_model_type(current_model)

    # Check if changed
    changed = (current_model != _last_detected_model) or (current_type != _last_model_type)

    # Update tracking
    _last_detected_model = current_model
    _last_model_type = current_type

    return changed, current_model, current_type


def get_preset_for_current_model() -> Tuple[Optional[ModelPreset], str]:
    """Get preset for currently loaded model.

    Returns:
        Tuple of (preset, model_name_or_error)
    """
    model_name = detect_loaded_model()

    if model_name is None:
        return None, "No model loaded"

    preset = get_preset_for_model(model_name)

    if preset is None:
        model_type = detect_model_type(model_name)
        if model_type == ModelType.UNKNOWN:
            return None, f"Unknown model: {model_name}"
        else:
            return None, f"No preset for {model_type.value}"

    return preset, model_name


def handle_apply_model_defaults(render_mode: str, *current_values) -> Tuple[str, Dict[str, Any]]:
    """Apply model defaults based on currently loaded model.

    Args:
        render_mode: Current render mode
        *current_values: Current UI component values (for comparison)

    Returns:
        Tuple of (status_message, settings_dict)
    """
    preset, model_name = get_preset_for_current_model()

    if preset is None:
        cross = emoji_if_enabled("❌")
        return f"{cross} {model_name}", {}

    # Get settings dict for this preset and render mode
    settings = get_settings_dict_from_preset(preset, render_mode)

    # Create confirmation message
    message = create_preset_message(preset, render_mode)

    logger.info(f"Applied {preset.model_type.value} preset for {render_mode} mode")

    check = emoji_if_enabled("✅")
    return f"{check} {preset.model_type.value.upper()} defaults applied", settings


def create_preset_status_message(
    model_name: Optional[str],
    model_type: ModelType,
    render_mode: str
) -> str:
    """Create status message showing current model and preset availability.

    Args:
        model_name: Name of loaded model
        model_type: Detected model type
        render_mode: Current render mode

    Returns:
        Formatted status string
    """
    circle = emoji_if_enabled("⚪")
    target = emoji_if_enabled("🎯")

    if model_name is None:
        return f"{circle} No model loaded"

    if model_type == ModelType.UNKNOWN:
        return f"{circle} Unknown model: {model_name[:50]}"

    preset = get_preset_for_model(model_name)

    if preset is None:
        return f"{circle} {model_type.value.upper()} (no preset available)"

    return f"{target} {model_type.value.upper()} | {preset.steps} steps, {preset.scheduler}"


def handle_model_change_notification(render_mode: str) -> str:
    """Check for model changes and create notification.

    Args:
        render_mode: Current render mode

    Returns:
        Notification HTML or empty string
    """
    changed, model_name, model_type = check_model_changed()

    if not changed:
        return ""

    if model_type == ModelType.UNKNOWN:
        return ""

    preset = get_preset_for_model(model_name)

    if preset is None:
        return ""

    # Create notification
    target = emoji_if_enabled("🎯")
    message = f"""
<div style="padding: 15px; background: rgba(23, 167, 254, 0.1); border-left: 4px solid #17A7FE; border-radius: 4px; margin: 10px 0;">
    <div style="font-weight: bold; color: #17A7FE; margin-bottom: 8px;">
        {target} Model Changed: {model_type.value.upper()}
    </div>
    <div style="color: #CBD5E1; font-size: 13px; margin-bottom: 8px;">
        Optimal settings available for this model.
    </div>
    <div style="color: #94A3B8; font-size: 12px;">
        Click "Apply Model Defaults" to use recommended settings.
    </div>
</div>
""".strip()

    return message


# ============================================================================
# Component Update Helpers
# ============================================================================

def apply_preset_to_components(
    settings: Dict[str, Any],
    components: Dict[str, Any]
) -> Dict[Any, Any]:
    """Convert settings dict to Gradio component updates.

    Args:
        settings: Settings dict from get_settings_dict_from_preset()
        components: Dict of Gradio components by name

    Returns:
        Dict mapping components to their new values
    """
    updates = {}

    for key, value in settings.items():
        if key in components:
            component = components[key]
            updates[component] = value

    return updates


def create_model_info_html(preset: Optional[ModelPreset], model_name: str) -> str:
    """Create HTML info box showing current model preset.

    Args:
        preset: Model preset if available
        model_name: Name of loaded model

    Returns:
        HTML string
    """
    if preset is None:
        return f"""
<div style="padding: 10px; background: rgba(100, 116, 139, 0.1); border-radius: 4px;">
    <div style="color: #94A3B8; font-size: 12px;">
        Model: {model_name[:60]}
    </div>
    <div style="color: #64748B; font-size: 11px; margin-top: 4px;">
        No preset available
    </div>
</div>
""".strip()

    return f"""
<div style="padding: 10px; background: rgba(23, 167, 254, 0.05); border-radius: 4px;">
    <div style="color: #17A7FE; font-weight: bold; font-size: 13px; margin-bottom: 6px;">
        {preset.model_type.value.upper()}
    </div>
    <div style="color: #CBD5E1; font-size: 11px; margin-bottom: 4px;">
        {preset.steps} steps • {preset.scheduler} • {preset.width}x{preset.height}
    </div>
    <div style="color: #94A3B8; font-size: 10px;">
        {preset.notes[:100]}...
    </div>
</div>
""".strip()
