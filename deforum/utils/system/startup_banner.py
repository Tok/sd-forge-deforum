"""Deforum startup banner with slopcore purple gradient styling."""

from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


def print_startup_banner():
    """Print Deforum initialization banner with slopcore 2D diagonal gradient table."""
    import os
    from pathlib import Path
    from rich.table import Table
    from rich.console import Console
    from rich import box

    # ANSI color codes for extended slopcore gradient (more shades for 2D effect)
    from deforum.utils.system.logging.themes import (
        HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4,
        HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7
    )
    from deforum.utils.image.color import hex_to_ansi_foreground

    # Create extended gradient with interpolated colors for smoother diagonal effect
    def interpolate_color(hex1, hex2, ratio):
        """Interpolate between two hex colors."""
        r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
        r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        return f"#{r:02x}{g:02x}{b:02x}"

    # Generate smooth gradient with more shades (14 shades for smoother diagonal)
    gradient_colors = []
    base_colors = [HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3, HEX_SLOPCORE_4,
                   HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7]

    # Interpolate between each pair
    for i in range(len(base_colors) - 1):
        gradient_colors.append(base_colors[i])
        # Add one interpolated color between each pair
        mid_color = interpolate_color(base_colors[i], base_colors[i + 1], 0.5)
        gradient_colors.append(mid_color)
    gradient_colors.append(base_colors[-1])

    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Detect Forge Neo vs classic Forge
    try:
        import modules.paths as ph
        models_dir = Path(ph.models_path)
        is_forge_neo = (models_dir / "text_encoder").exists()
    except:
        is_forge_neo = False

    # Create Rich console for table rendering
    console = Console()

    # Create table with diagonal gradient border
    table = Table(
        show_header=False,
        box=box.DOUBLE,
        padding=(0, 1),
        border_style=f"rgb({int(HEX_SLOPCORE_4[1:3], 16)},{int(HEX_SLOPCORE_4[3:5], 16)},{int(HEX_SLOPCORE_4[5:7], 16)})",
        style=f"rgb({int(HEX_SLOPCORE_4[1:3], 16)},{int(HEX_SLOPCORE_4[3:5], 16)},{int(HEX_SLOPCORE_4[5:7], 16)})"
    )

    # Title with diagonal gradient effect and bolt emojis (comically long fork-ception)
    title_text = "⚡ Zirteq's Fluxabled Fork of the Deforum Extension for Forge Neo Fork of Forge WebUI Fork of Automatic1111 ⚡"

    # Create diagonal gradient for title (each character gets color based on position)
    gradient_title = ""
    for i, char in enumerate(title_text):
        color_idx = min(int((i / len(title_text)) * len(gradient_colors)), len(gradient_colors) - 1)
        hex_color = gradient_colors[color_idx]
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        gradient_title += f"[rgb({r},{g},{b})]{char}[/]"

    table.add_row(f"[bold]{gradient_title}[/bold]")

    # Additional info (removed feature lists - just essential info)
    table.add_row("")  # Separator
    table.add_row("[bold]Primary Target:[/bold] Forge Neo (fully tested and supported)")
    table.add_row("[bold]Other Forge Versions:[/bold] May work but remain untested")
    table.add_row("[bold]More Info:[/bold] https://github.com/Tok/sd-forge-deforum")

    # Print with newlines for spacing
    print("")
    console.print(table)
    print("")
