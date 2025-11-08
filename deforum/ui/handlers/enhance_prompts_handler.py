"""Helper functions for enhance_prompts_handler.

Extracted from ui_elements.py to reduce complexity.
"""

from typing import Dict, Tuple, Any, Optional
import json
from deforum.utils.system.logging import get_logger

logger = get_logger()


def load_enhance_prompts_emojis() -> Dict[str, str]:
    """Load all emoji symbols for prompt enhancement UI.

    Returns:
        Dict of emoji symbols
    """
    from deforum.utils.system.logging import emoji as emoji_utils
    return {
        'check': emoji_utils.maybe_check(),
        'cross': emoji_utils.maybe_cross(),
        'warning': emoji_utils.maybe_warning(),
        'palette': emoji_utils.palette(),
        'memo': emoji_utils.memo(),
        'wrench': emoji_utils.wrench(),
        'hourglass': emoji_utils.hourglass(),
        'download': emoji_utils.download(),
        'bulb': emoji_utils.bulb(),
        'refresh_icon': emoji_utils.refresh_icon(),
        'robot': emoji_utils.robot(),
        'magnifying_glass': emoji_utils.magnifying_glass(),
        'pencil': emoji_utils.pencil(),
        'chart_increasing': emoji_utils.chart_increasing(),
        'save': emoji_utils.save(),
        'sparkles': emoji_utils.sparkles(),
        'ruler': emoji_utils.ruler(),
        'party': emoji_utils.party(),
    }


def validate_auto_download(
    qwen_manager: Any,
    qwen_model: str,
    auto_download: bool,
    emojis: Dict[str, str],
    progress_update: str
) -> Tuple[bool, str, str]:
    """Validate auto-download setting for model availability.

    Args:
        qwen_manager: QwenModelManager instance
        qwen_model: Model name to check
        auto_download: Whether auto-download is enabled
        emojis: Dict of emoji symbols
        progress_update: Progress string to append to

    Returns:
        Tuple of (is_valid, error_message, progress_update)
    """
    if not auto_download and not qwen_manager.is_model_downloaded(qwen_model):
        error_msg = f"""{emojis['cross']} Qwen model not available: {qwen_model}

{emojis['wrench']} **Model Download Required:**
1. {emojis['check']} Enable "Auto-Download Qwen Models" checkbox
2. {emojis['palette']} Click "AI Prompt Enhancement" again to auto-download
3. {emojis['hourglass']} Wait for download to complete

{emojis['download']} **Manual Download Alternative:**
1. Use HuggingFace CLI: `huggingface-cli download {qwen_manager.get_model_info(qwen_model).get('huggingface_id', 'model-id')}`
2. {emojis['check']} Enable auto-download for easier setup

{emojis['bulb']} **Auto-download is recommended** for seamless model management."""
        return False, error_msg, progress_update + f"{emojis['cross']} Model not available - enable auto-download!"

    return True, "", progress_update


def handle_model_switching(
    qwen_manager: Any,
    qwen_model: str,
    emojis: Dict[str, str],
    progress_update: str
) -> str:
    """Handle model switching if different model is requested.

    Args:
        qwen_manager: QwenModelManager instance
        qwen_model: Requested model name
        emojis: Dict of emoji symbols
        progress_update: Progress string to append to

    Returns:
        Updated progress string
    """
    if qwen_manager.is_model_loaded():
        loaded_info = qwen_manager.get_loaded_model_info()
        current_model = loaded_info['name'] if loaded_info else "Unknown"

        # If different model requested, cleanup first
        if qwen_model != "Auto-Select" and current_model != qwen_model:
            logger.info(f"Switching from {current_model} to {qwen_model}", emoji='refresh')
            progress_update += f"{emojis['refresh_icon']} Switching from {current_model} to {qwen_model}...\n"
            qwen_manager.cleanup_cache()

    return progress_update


def log_model_selection(
    qwen_manager: Any,
    qwen_model: str,
    emojis: Dict[str, str],
    progress_update: str
) -> str:
    """Log model selection if model is not loaded.

    Args:
        qwen_manager: QwenModelManager instance
        qwen_model: Requested model name
        emojis: Dict of emoji symbols
        progress_update: Progress string to append to

    Returns:
        Updated progress string
    """
    if not qwen_manager.is_model_loaded():
        if qwen_model == "Auto-Select":
            selected_model = qwen_manager.auto_select_model()
            logger.info(f"{emojis['robot']} Auto-selected model: {selected_model}")
            progress_update += f"{emojis['robot']} Auto-selected model: {selected_model}\n"
        else:
            logger.info(f"{emojis['download']} Loading Qwen model: {qwen_model}")
            progress_update += f"{emojis['download']} Loading Qwen model: {qwen_model}...\n"

    return progress_update


def parse_wan_prompts(
    current_prompts: str,
    emojis: Dict[str, str],
    progress_update: str
) -> Tuple[Optional[Dict[str, str]], str, str]:
    """Parse Wan prompts from JSON or readable format.

    Args:
        current_prompts: Prompts string (JSON or readable format)
        emojis: Dict of emoji symbols
        progress_update: Progress string to append to

    Returns:
        Tuple of (prompts_dict, error_message, progress_update)
    """
    from deforum.utils.system.logging import emoji as emoji_utils

    if not current_prompts or not current_prompts.strip():
        logger.warning(f"{emojis['warning']} Empty Wan prompts")
        return None, "", progress_update

    # Try JSON format first
    try:
        animation_prompts = json.loads(current_prompts)
        logger.info(f"{emoji_utils.maybe_check()} Successfully parsed {len(animation_prompts)} Wan prompts as JSON")
        progress_update += f"{emojis['check']} Parsed {len(animation_prompts)} prompts successfully\n"
        return animation_prompts, "", progress_update
    except json.JSONDecodeError:
        pass

    # Try readable format (Frame X: prompt)
    try:
        animation_prompts = {}
        for line in current_prompts.strip().split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                frame_part = parts[0].strip()
                prompt_part = parts[1].strip()

                # Extract frame number
                if frame_part.lower().startswith('frame '):
                    frame_num = frame_part[6:].strip()
                else:
                    frame_num = frame_part

                animation_prompts[frame_num] = prompt_part

        if animation_prompts:
            logger.info(f"{emoji_utils.maybe_check()} Successfully parsed {len(animation_prompts)} Wan prompts as readable format")
            progress_update += f"{emojis['check']} Parsed {len(animation_prompts)} prompts from readable format\n"
            return animation_prompts, "", progress_update
        else:
            raise ValueError("No valid prompts found")
    except Exception as e:
        logger.error(f"Could not parse Wan prompts: {e}", emoji='off')
        error_msg = f"{emojis['cross']} Invalid format in Wan prompts. Expected JSON format like:\n{{\n  \"0\": \"prompt text\",\n  \"60\": \"another prompt\"\n}}\n\nOr readable format like:\nFrame 0: prompt text\nFrame 60: another prompt"
        return None, error_msg, progress_update + f"{emojis['cross']} Failed to parse prompts!"


def validate_prompts_content(
    animation_prompts: Optional[Dict[str, str]],
    emojis: Dict[str, str],
    progress_update: str
) -> Tuple[bool, str, str]:
    """Validate prompts content (not empty, not default).

    Args:
        animation_prompts: Dict of prompts to validate
        emojis: Dict of emoji symbols
        progress_update: Progress string to append to

    Returns:
        Tuple of (is_valid, error_message, progress_update)
    """
    # Check if prompts exist
    if not animation_prompts:
        error_msg = f"""{emojis['cross']} No Wan prompts found!

{emojis['wrench']} **Setup Required:**
1. {emojis['memo']} Load prompts using "Load from Deforum Prompts" or "Load Default Wan Prompts"
2. {emojis['memo']} Make sure your prompts are in proper JSON format like:
   {{
     "0": "prompt text",
     "60": "another prompt",
     "120": "a cyberpunk environment with glowing elements"
   }}
3. {emojis['palette']} Click **AI Prompt Enhancement** again after setting up prompts

{emojis['bulb']} **Quick Start:**
Click "Load Default Wan Prompts" to start with example prompts!"""
        return False, error_msg, progress_update + f"{emojis['cross']} No prompts to enhance!"

    # Check for default prompts
    if len(animation_prompts) == 1 and "0" in animation_prompts and "beautiful landscape" in animation_prompts["0"]:
        error_msg = f"""{emojis['cross']} Default prompts detected!

{emojis['wrench']} **Please configure your actual animation prompts:**
1. {emojis['memo']} Load your real prompts using the load buttons above
2. {emojis['pencil']} Or manually edit the Wan prompts field
3. {emojis['palette']} Click **AI Prompt Enhancement** again

{emojis['bulb']} **For your animation sequence:**
Set up prompts like:
{{{{
  "0": "A peaceful scene, photorealistic",
  "18": "A scene with glowing effects, neon colors, synthwave aesthetic",
  "36": "A cyberpunk scene with LED patterns, digital environment"
}}}}"""
        return False, error_msg, progress_update + f"{emojis['cross']} Default prompts detected!"

    return True, "", progress_update


def create_prompt_expander(
    qwen_manager: Any,
    qwen_model: str,
    auto_download: bool,
    emojis: Dict[str, str],
    progress_update: str
) -> Tuple[Any | None, str, str]:
    """Create Qwen prompt expander with error handling.

    Args:
        qwen_manager: QwenModelManager instance
        qwen_model: Model name to create
        auto_download: Whether auto-download is enabled
        emojis: Dict of emoji symbols
        progress_update: Progress string to append to

    Returns:
        Tuple of (prompt_expander, error_message, progress_update)
    """
    try:
        progress_update += f"{emojis['download']} Creating AI model instance...\n"
        prompt_expander = qwen_manager.create_prompt_expander(qwen_model, auto_download)

        if not prompt_expander:
            if auto_download:
                error_msg = f"""{emojis['hourglass']} Downloading {qwen_model} model...

{emojis['refresh_icon']} **Download in Progress:**
Model download started automatically. This may take a few minutes.

{emojis['download']} **Please wait** and try clicking "AI Prompt Enhancement" again in 30-60 seconds.

{emojis['bulb']} **Status**: Check console for download progress."""
                return None, error_msg, progress_update + f"{emojis['hourglass']} Downloading {qwen_model}..."
            else:
                error_msg = f"""{emojis['cross']} Failed to create Qwen prompt expander: {qwen_model}

{emojis['wrench']} **Solutions:**
1. {emojis['check']} Enable "Auto-Download Qwen Models" and try again
2. {emojis['download']} Manual download: Check console for HuggingFace CLI commands
3. {emojis['refresh_icon']} Restart WebUI after downloading

{emojis['chart_increasing']} **Model Info**: {qwen_manager.get_model_info(qwen_model).get('description', 'N/A')}"""
                return None, error_msg, progress_update + f"{emojis['cross']} Failed to create AI model!"

        return prompt_expander, "", progress_update
    except Exception as e:
        error_msg = f"""{emojis['cross']} Error creating Qwen prompt expander: {str(e)}

{emojis['wrench']} **Troubleshooting:**
1. {emojis['check']} Enable auto-download and try again
2. {emojis['refresh_icon']} Restart WebUI if models were just downloaded
3. {emojis['save']} Check available disk space ({qwen_manager.get_model_info(qwen_model).get('vram_gb', 'Unknown')}GB VRAM required)

{emojis['bulb']} **Tip**: Try selecting "Auto-Select" for automatic model choice."""
        return None, error_msg, progress_update + f"{emojis['cross']} Error: {str(e)}"


def enhance_prompts_with_movement(
    qwen_manager: Any,
    animation_prompts: Dict[str, str],
    qwen_model: str,
    language: str,
    auto_download: bool,
    handler_function: Any,
    emojis: Dict[str, str],
    progress_update: str
) -> Tuple[Dict[str, str], str]:
    """Enhance prompts and append movement descriptions.

    Args:
        qwen_manager: QwenModelManager instance
        animation_prompts: Dict of prompts to enhance
        qwen_model: Model name
        language: Target language
        auto_download: Whether auto-download is enabled
        handler_function: Handler function (to access _movement_description)
        emojis: Dict of emoji symbols
        progress_update: Progress string to append to

    Returns:
        Tuple of (enhanced_prompts_dict, progress_update)
    """
    from deforum.utils.system.logging import emoji as emoji_utils

    progress_update += f"{emojis['sparkles']} Enhancing prompts with AI...\n"
    enhanced_prompts_dict = qwen_manager.enhance_prompts(
        prompts=animation_prompts,
        model_name=qwen_model,
        language=language,
        auto_download=auto_download
    )

    # Check if movement descriptions are available and append them
    movement_description = ""
    if hasattr(handler_function, '_movement_description'):
        movement_description = handler_function._movement_description
        logger.info(f"{emojis['ruler']} Found movement description to append: {movement_description}")
        progress_update += f"{emojis['ruler']} Adding movement descriptions...\n"

    # Append movement descriptions to enhanced prompts if available
    if movement_description and movement_description.strip():
        for frame_key in enhanced_prompts_dict:
            original_prompt = enhanced_prompts_dict[frame_key]
            enhanced_prompts_dict[frame_key] = f"{original_prompt}. {movement_description}"
        logger.info(f"{emoji_utils.maybe_check()} Appended movement description to {len(enhanced_prompts_dict)} enhanced prompts")
        progress_update += f"{emojis['check']} Added movement to {len(enhanced_prompts_dict)} prompts\n"

    return enhanced_prompts_dict, progress_update
