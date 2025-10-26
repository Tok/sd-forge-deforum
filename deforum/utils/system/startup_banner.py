"""Deforum startup banner with slopcore purple gradient styling."""

from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


def print_startup_banner():
    """Print Deforum initialization banner with slopcore purple gradient."""

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

    # Create gradient border using all 7 shades (14-15 chars per shade for 101 total)
    # Text line is 99 chars, so border is 101 for slight overflow
    border_top = (
        f"{SLOPCORE_1}##############"   # 14 chars - Bright blue
        f"{SLOPCORE_2}##############"   # 14 chars - Blue-purple
        f"{SLOPCORE_3}###############"  # 15 chars - Light purple
        f"{SLOPCORE_4}###############"  # 15 chars - Mid purple (center)
        f"{SLOPCORE_5}###############"  # 15 chars - Purple
        f"{SLOPCORE_6}##############"   # 14 chars - Deep purple
        f"{SLOPCORE_7}##############"   # 14 chars - Darkest purple
        f"{RESET}"
    )
    # Reverse gradient for bottom border
    border_bot = (
        f"{SLOPCORE_7}##############"
        f"{SLOPCORE_6}##############"
        f"{SLOPCORE_5}###############"
        f"{SLOPCORE_4}###############"
        f"{SLOPCORE_3}###############"
        f"{SLOPCORE_2}##############"
        f"{SLOPCORE_1}##############"
        f"{RESET}"
    )

    banner = f"""
{border_top}
{SLOPCORE_4}{BOLD}⚡ Stable Diffusion WebUI Forge Enhanced By Zirteq's Fluxabled Fork of the Deforum Extension ⚡{RESET}
{border_bot}
{WHITE}Applying compatibility patches for enhanced Flux.1 + Wan 2.1 AI Video integration:
  • FlowMatchEulerDiscreteScheduler patching (deforum/integrations/flux_controlnet/diffusers_compat.py)
  • Wan FLF2V pipeline integration with Forge's Flux backend
  • Unified Flux + Wan workflows with seamless model management{RESET}
{BOLD}💡 Note:{RESET} This fork is optimized for Flux/Wan workflows.
{BOLD}For best results:{RESET} Run in a dedicated Forge instance to avoid interfering
   with other extensions and Forge base functionality.
{BOLD}📚 More Info:{RESET} https://github.com/Tok/sd-forge-deforum
{border_top}
"""

    logger.info(banner)
