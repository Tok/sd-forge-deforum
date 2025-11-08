"""Helper functions for wan_generate_video button handler.

Extracted from ui_elements.py to reduce complexity.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from deforum.utils.system.logging import get_logger

logger = get_logger()


def load_wan_emojis() -> Dict[str, str]:
    """Load all emoji symbols for Wan generation UI.

    Returns:
        Dict of emoji symbols
    """
    from deforum.utils.system.logging import emoji as emoji_utils
    return {
        'check': emoji_utils.maybe_check(),
        'cross': emoji_utils.maybe_cross(),
        'warning': emoji_utils.maybe_warning(),
        'download': emoji_utils.download(),
        'trash': emoji_utils.trash(),
        'wrench': emoji_utils.wrench(),
        'bulb': emoji_utils.bulb(),
        'signal': emoji_utils.signal(),
        'save': emoji_utils.save(),
        'refresh_icon': emoji_utils.refresh_icon(),
        'memo': emoji_utils.memo(),
        'movie_camera': emoji_utils.movie_camera(),
        'target': emoji_utils.target(),
        'rocket': emoji_utils.rocket(),
        'chart_increasing': emoji_utils.chart_increasing(),
    }


def get_wan_auto_download_setting(component_args: tuple, component_names: list) -> bool:
    """Extract wan_auto_download setting from component arguments.

    Args:
        component_args: Tuple of component values
        component_names: List of component names

    Returns:
        Boolean indicating if auto-download is enabled
    """
    wan_auto_download = True  # Default value
    emojis = load_wan_emojis()

    try:
        auto_download_index = component_names.index('wan_auto_download')
        if auto_download_index < len(component_args):
            wan_auto_download = component_args[auto_download_index]
    except (ValueError, IndexError):
        logger.error(f"{emojis['warning']} Could not find wan_auto_download setting, using default: True")

    return wan_auto_download


def attempt_model_download(integration, emojis: Dict[str, str]) -> List[Dict]:
    """Attempt to download recommended Wan models.

    Args:
        integration: WanSimpleIntegration instance
        emojis: Dict of emoji symbols

    Returns:
        List of discovered models after download attempt
    """
    try:
        from deforum.integrations.wan.wan_model_downloader import WanModelDownloader
        downloader = WanModelDownloader()

        # Try TI2V-5B first (recommended)
        logger.info(f"{emojis['download']} Downloading Wan 2.2 TI2V-5B model (recommended: 24GB VRAM, RTX 4090)...")
        if downloader.download_model("TI2V-5B"):
            logger.info(f"{emojis['check']} TI2V-5B model download completed!")
            return integration.discover_models()

        # Fallback to A14B
        logger.error("TI2V-5B download failed, trying A14B (MoE)...", emoji='off')
        if downloader.download_model("A14B"):
            logger.info(f"{emojis['check']} A14B MoE model download completed!")
            return integration.discover_models()

        logger.error("All model downloads failed", emoji='off')
        return []

    except Exception as e:
        logger.error(f"Auto-download failed: {e}", emoji='off')
        return []


def is_model_valid(model: Dict[str, Any], emojis: Dict[str, str]) -> bool:
    """Check if a Wan model is valid and not corrupted.

    Args:
        model: Model info dict
        emojis: Dict of emoji symbols

    Returns:
        True if model is valid
    """
    model_path = Path(model['path'])

    # Check TI2V/T2V/I2V models
    if model['type'] in ['TI2V', 'T2V', 'I2V']:
        if (model_path / "model_index.json").exists():
            logger.debug(f"{emojis['check']} {model['name']}: Valid {model['type']} model")
            return True
        logger.debug(f"{model['name']}: Incomplete {model['type']} model", emoji='off')
        return False

    # Check unknown/legacy model types
    has_valid_structure = (
        (model_path / "model_index.json").exists() or
        (model_path / "transformer").exists() or
        any(f.name.startswith("wan") for f in model_path.rglob("*.pth")) or
        any(f.name.startswith("wan") for f in model_path.rglob("*.safetensors"))
    )

    if has_valid_structure:
        logger.debug(f"{emojis['check']} {model['name']}: Valid legacy model")
        return True

    logger.debug(f"{model['name']}: Invalid/leftover files (not a proper Wan model)", emoji='off')
    return False


def validate_discovered_models(models: List[Dict], emojis: Dict[str, str]) -> Tuple[List[Dict], List[Dict]]:
    """Validate discovered models and separate valid from corrupted.

    Args:
        models: List of discovered model dicts
        emojis: Dict of emoji symbols

    Returns:
        Tuple of (valid_models, corrupted_models)
    """
    logger.debug(f"{emojis['check']} Validating discovered models...")
    valid_models = []
    corrupted_models = []

    for model in models:
        if is_model_valid(model, emojis):
            valid_models.append(model)
        else:
            corrupted_models.append(model)

    return valid_models, corrupted_models


def log_corrupted_model_cleanup_instructions(corrupted_models: List[Dict], emojis: Dict[str, str]):
    """Log manual cleanup instructions for corrupted models.

    Args:
        corrupted_models: List of corrupted model dicts
        emojis: Dict of emoji symbols
    """
    logger.warning(f"{emojis['warning']} Found {len(corrupted_models)} corrupted model(s)")
    logger.info("MANUAL CLEANUP INSTRUCTIONS:", emoji='tools')
    logger.info("For safety, corrupted models are NOT automatically deleted.")
    logger.info("If you want to remove them, please:")
    logger.info("")

    for corrupted_model in corrupted_models:
        logger.info(f"{corrupted_model['name']}: {corrupted_model['path']}", emoji='off')

    logger.info("")
    logger.info(f"{emojis['trash']} To manually remove corrupted models:")
    for corrupted_model in corrupted_models:
        logger.info(f'   rm -rf "{corrupted_model["path"]}"')

    logger.info("")
    logger.info(f"{emojis['download']} To re-download models:")
    for corrupted_model in corrupted_models:
        model_name = corrupted_model['name'].lower()
        if 'ti2v' in model_name and '5b' in model_name:
            logger.info("   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir models/Deforum/wan/Wan2.2-TI2V-5B")
        elif 'a14b' in model_name or '14b' in model_name:
            logger.info("   huggingface-cli download Wan-AI/Wan2.2-TI2V-A14B-Diffusers --local-dir models/Deforum/wan/Wan2.2-TI2V-A14B")

    logger.info("")
    logger.info("TIP: Enable 'Auto-Download Models' for automatic downloading of missing models", emoji='bulb')
    logger.warning(f"{emojis['warning']} SAFETY: Always verify corruption before deleting - some errors may be temporary")


def build_no_models_error_message(wan_auto_download: bool, emojis: Dict[str, str]) -> str:
    """Build error message when no models are found.

    Args:
        wan_auto_download: Whether auto-download is enabled
        emojis: Dict of emoji symbols

    Returns:
        Formatted error message string
    """
    if not wan_auto_download:
        auto_download_help = f"""

{emojis['wrench']} AUTO-DOWNLOAD OPTIONS:
1. {emojis['check']} Enable "Auto-Download Models" in the Wan tab (recommended)
2. {emojis['download']} Manual download with HuggingFace CLI:

   **For TI2V-5B (Recommended - Wan 2.2, 24GB VRAM, RTX 4090):**
   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir models/Deforum/wan/Wan2.2-TI2V-5B

   **For TI2V-A14B (Highest Quality - Wan 2.2 MoE, 32GB+ VRAM):**
   huggingface-cli download Wan-AI/Wan2.2-TI2V-A14B-Diffusers --local-dir models/Deforum/wan/Wan2.2-TI2V-A14B

3. {emojis['check']} Restart generation after downloading

{emojis['wrench']} AUTO-REPAIR: Corrupted models are automatically detected and re-downloaded!"""
    else:
        auto_download_help = f"""

{emojis['wrench']} TROUBLESHOOTING:
1. {emojis['signal']} Check internet connection for downloads
2. {emojis['save']} Ensure enough disk space (TI2V-5B: ~30GB, TI2V-A14B: ~60GB)
3. {emojis['refresh_icon']} Try manual download with HuggingFace CLI (see Auto-Discovery tab)
4. {emojis['wrench']} Corrupted models are detected - follow manual cleanup instructions"""

    return f"""{emojis['cross']} No Wan models found!

{emojis['bulb']} QUICK SETUP:
TI2V models are unified text/image-to-video (Wan 2.2) - recommended!

• **TI2V-5B**: 24GB VRAM, 720P@24fps, RTX 4090 compatible (best for most users)
• **TI2V-A14B**: 32GB+ VRAM, Mixture-of-Experts, highest quality (for power users)

{auto_download_help}

{emojis['bulb']} TI2V models handle both text-to-video and image-to-video in one unified model!"""


def extract_animation_prompts_from_args(component_args: tuple, component_names: list, emojis: Dict[str, str]) -> str:
    """Extract animation_prompts from component arguments.

    Args:
        component_args: Tuple of component values
        component_names: List of component names
        emojis: Dict of emoji symbols

    Returns:
        Animation prompts JSON string
    """
    animation_prompts = '{"0": "a beautiful landscape"}'  # Default

    try:
        animation_prompts_index = component_names.index('animation_prompts')
        if animation_prompts_index < len(component_args):
            animation_prompts = component_args[animation_prompts_index]
            logger.info(f"{emojis['memo']} Found animation_prompts at index {animation_prompts_index}")
        else:
            logger.warning(f"{emojis['warning']} animation_prompts index {animation_prompts_index} out of range (have {len(component_args)} args)")
    except ValueError:
        logger.error(f"{emojis['warning']} Could not find animation_prompts in component names")

    return animation_prompts


def discover_and_prepare_models(integration, wan_auto_download: bool, emojis: Dict[str, str]) -> Optional[List[Dict]]:
    """Discover Wan models, auto-download if needed, and validate.

    Args:
        integration: WanSimpleIntegration instance
        wan_auto_download: Whether to attempt auto-download
        emojis: Dict of emoji symbols

    Returns:
        List of valid models or None if no models available
    """
    models = integration.discover_models()

    # If no models and auto-download enabled, try downloading
    if not models and wan_auto_download:
        logger.info(f"{emojis['download']} No models found and auto-download enabled. Downloading recommended model...")
        models = attempt_model_download(integration, emojis)

    # Validate discovered models
    if models:
        valid_models, corrupted_models = validate_discovered_models(models, emojis)

        # Log cleanup instructions for corrupted models
        if corrupted_models and wan_auto_download:
            log_corrupted_model_cleanup_instructions(corrupted_models, emojis)

        return valid_models if valid_models else None

    return None
