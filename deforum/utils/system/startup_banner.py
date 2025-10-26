"""Deforum startup banner with slopcore purple gradient styling."""

from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


def print_startup_banner():
    """Print Deforum initialization banner with slopcore purple gradient."""

    # ANSI color codes for slopcore purple gradient (matching UI #667eea -> #764ba2)
    PURPLE_LIGHT = "\033[38;2;102;126;234m"  # #667eea
    PURPLE_MID = "\033[38;2;110;101;184m"    # Interpolated
    PURPLE_DARK = "\033[38;2;118;75;162m"    # #764ba2
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Create gradient characters for border
    border_top = f"{PURPLE_LIGHT}###############################" + f"{PURPLE_MID}###############################" + f"{PURPLE_DARK}#################################{RESET}"
    border_bot = f"{PURPLE_DARK}###############################" + f"{PURPLE_MID}###############################" + f"{PURPLE_LIGHT}#################################{RESET}"

    banner = f"""
{border_top}
{PURPLE_MID}{BOLD}⚡ Stable Diffusion WebUI Forge Enhanced By Zirteq's Fluxabled Fork of the Deforum Extension ⚡{RESET}
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
