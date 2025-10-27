"""Deforum startup banner with slopcore purple gradient styling."""

from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


def print_startup_banner():
    """Print Deforum initialization banner with slopcore purple gradient."""
    import os
    from pathlib import Path

    # ANSI color codes for 7-shade slopcore purple gradient
    from deforum.utils.system.logging.themes import (
        HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4,
        HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7
    )
    from deforum.utils.image.color import hex_to_ansi_foreground

    # Convert hex colors to ANSI
    SLOPCORE_1 = hex_to_ansi_foreground(HEX_SLOPCORE_1)  # Bright blue
    SLOPCORE_2 = hex_to_ansi_foreground(HEX_SLOPCORE_2)  # Blue-purple
    SLOPCORE_3 = hex_to_ansi_foreground(HEX_SLOPCORE_3)  # Light purple
    SLOPCORE_4 = hex_to_ansi_foreground(HEX_SLOPCORE_4)  # Mid purple
    SLOPCORE_5 = hex_to_ansi_foreground(HEX_SLOPCORE_5)  # Purple
    SLOPCORE_6 = hex_to_ansi_foreground(HEX_SLOPCORE_6)  # Deep purple
    SLOPCORE_7 = hex_to_ansi_foreground(HEX_SLOPCORE_7)  # Darkest purple

    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Detect Forge Neo vs classic Forge
    # Neo has text_encoder/ directory, classic has text encoders in VAE/
    try:
        import modules.paths as ph
        models_dir = Path(ph.models_path)
        is_forge_neo = (models_dir / "text_encoder").exists()
    except:
        is_forge_neo = False

    # Create gradient border using all 7 shades (12-13 chars per shade for 90 total)
    # Text line is 90 chars (was miscounted), border reduced by 5
    border_top = (
        f"{SLOPCORE_1}#############"   # 13 chars - Bright blue
        f"{SLOPCORE_2}############"    # 12 chars - Blue-purple (reduced)
        f"{SLOPCORE_3}#############"   # 13 chars - Light purple (reduced)
        f"{SLOPCORE_4}#############"   # 13 chars - Mid purple (center, reduced)
        f"{SLOPCORE_5}#############"   # 13 chars - Purple (reduced)
        f"{SLOPCORE_6}############"    # 12 chars - Deep purple (reduced)
        f"{SLOPCORE_7}#############"   # 13 chars - Darkest purple
        f"{RESET}"
    )
    # Reverse gradient for bottom border
    border_bot = (
        f"{SLOPCORE_7}#############"
        f"{SLOPCORE_6}############"
        f"{SLOPCORE_5}#############"
        f"{SLOPCORE_4}#############"
        f"{SLOPCORE_3}#############"
        f"{SLOPCORE_2}############"
        f"{SLOPCORE_1}#############"
        f"{RESET}"
    )

    # Different messages for Neo vs classic Forge
    if is_forge_neo:
        # Forge Neo: Built-in Flux/Wan support, minimal patching needed
        features_text = f"""{WHITE}Forge Neo Enhancements:
  - Leveraging built-in Flux.1 and Wan 2.1/2.2 support
  - Slopcore UI theme with checkbox-style buttons
  - Flux ControlNet V2 + FLF2V interpolation workflows{RESET}"""
    else:
        # Classic Forge: Needs compatibility patches
        features_text = f"""{WHITE}Applying compatibility patches for Flux.1 ControlNet V2 + Wan 2.1/2.2 AI Video:
  - Flux ControlNet V2 support (patching Forge's IntegratedFluxTransformer2DModel)
  - FlowMatchEulerDiscreteScheduler compatibility (diffusers git main + Forge)
  - Wan 2.1 FLF2V + Wan 2.2 TI2V pipeline integration{RESET}"""

    # Text and border both 95 chars - perfect match
    banner = f"""
{border_top}
{SLOPCORE_4}{BOLD}Stable Diffusion WebUI Forge Enhanced By Zirteq's Fluxabled Fork of the Deforum Extension{RESET}
{border_bot}
{features_text}
{BOLD}Primary Target:{RESET} Forge Neo (fully tested and supported)
{BOLD}Other Forge versions:{RESET} May work but remain untested
{BOLD}Note:{RESET} Optimized for Flux/Wan workflows in dedicated Forge Neo instance
{BOLD}More Info:{RESET} https://github.com/Tok/sd-forge-deforum
{border_top}
"""

    logger.info(banner)
