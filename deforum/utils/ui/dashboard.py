"""Terminal dashboard for Deforum rendering.

Provides a fixed-position dashboard display using Rich Layout and Live updates.
Shows frame info, parameters, progress bars, and memory usage in a clean,
organized format while routing essential logs to a scrolling area.
"""

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.console import Console, Group
from rich.text import Text
from rich import box
import re
from collections import deque
from typing import Optional, Dict, Any

from deforum.rendering import options as opt_utils
from deforum.utils.system.logging.themes import get_tqdm_color_for_theme, HEX_CLASSIC_BLUE, HEX_CLASSIC_GREEN, HEX_CLASSIC_ORANGE, HEX_CLASSIC_RED, HEX_CLASSIC_PURPLE, HEX_CLASSIC_YELLOW

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
    Image = None


class MemoryStats:
    """Tracks GPU memory statistics from Forge output."""

    def __init__(self):
        self.target_model = ""
        self.free_gpu_mb = 0.0
        self.total_gpu_mb = 0.0  # We'll infer this from max seen
        self.model_require_mb = 0.0
        self.remaining_mb = 0.0
        self.last_update_time = 0.0

    def update_from_forge_message(self, message: str):
        """Parse Forge memory management message.

        Example:
        "[Memory Management] Target: IntegratedAutoencoderKL, Free GPU: 8238.74 MB,
         Model Require: 159.87 MB, Previously Loaded: 0.00 MB,
         Inference Require: 2038.40 MB, Remaining: 6040.47 MB,
         Moving model(s) has taken 0.36 seconds"
        """
        # Extract target model
        target_match = re.search(r'Target:\s*([^,]+)', message)
        if target_match:
            self.target_model = target_match.group(1).strip()

        # Extract Free GPU
        free_match = re.search(r'Free GPU:\s*([\d.]+)\s*MB', message)
        if free_match:
            self.free_gpu_mb = float(free_match.group(1))
            # Infer total from max seen
            if self.free_gpu_mb > self.total_gpu_mb:
                self.total_gpu_mb = self.free_gpu_mb

        # Extract Model Require
        require_match = re.search(r'Model Require:\s*([\d.]+)\s*MB', message)
        if require_match:
            self.model_require_mb = float(require_match.group(1))

        # Extract Remaining
        remaining_match = re.search(r'Remaining:\s*([\d.]+)\s*MB', message)
        if remaining_match:
            self.remaining_mb = float(remaining_match.group(1))

        # Extract time
        time_match = re.search(r'taken\s*([\d.]+)\s*seconds', message)
        if time_match:
            self.last_update_time = float(time_match.group(1))

    def get_usage_percentage(self) -> float:
        """Calculate VRAM usage percentage."""
        if self.total_gpu_mb == 0:
            return 0.0
        used = self.total_gpu_mb - self.free_gpu_mb
        return (used / self.total_gpu_mb) * 100.0

    def get_free_percentage(self) -> float:
        """Calculate free VRAM percentage."""
        if self.total_gpu_mb == 0:
            return 100.0
        return (self.free_gpu_mb / self.total_gpu_mb) * 100.0


def image_to_ascii_art(image, width: int = 32, height: int = 18, use_color: bool = True) -> str:
    """Convert PIL Image to colored ASCII art grid using 2-space blocks.

    Args:
        image: PIL Image object or numpy array
        width: Number of PIXELS wide (not chars - will be doubled for 2-space pixels)
        height: Number of PIXELS tall
        use_color: If True, use ANSI color codes

    Returns:
        String with ASCII art (colored if use_color=True)

    Note:
        Each pixel is represented by 2 spaces ("  ") with background color,
        maintaining proper aspect ratio in terminal display.
    """
    if image is None or Image is None or np is None:
        return ""

    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Preserve aspect ratio - calculate target size based on image aspect
    img_width, img_height = image.size
    img_aspect = img_width / img_height

    # Target aspect (width * 2 since we use 2 spaces per pixel)
    target_aspect = width / height

    # Adjust dimensions to preserve aspect ratio
    if img_aspect > target_aspect:
        # Image is wider - limit by width
        final_width = width
        final_height = int(width / img_aspect)
    else:
        # Image is taller - limit by height
        final_height = height
        final_width = int(height * img_aspect)

    # Resize to final dimensions
    image = image.resize((final_width, final_height), Image.Resampling.LANCZOS)
    pixels = np.array(image)

    lines = []
    for y in range(final_height):
        line = ""
        for x in range(final_width):
            # Get RGB values
            if len(pixels.shape) == 3 and pixels.shape[2] >= 3:
                r, g, b = pixels[y, x, :3]
            else:
                # Grayscale
                gray = pixels[y, x] if pixels.ndim == 2 else pixels[y, x, 0]
                r = g = b = gray

            # Add pixel as 2 spaces with background color
            if use_color:
                # Use ANSI 24-bit background color (true color)
                line += f"\033[48;2;{r};{g};{b}m  \033[0m"
            else:
                line += "  "

        lines.append(line)

    return "\n".join(lines)


class RenderDashboard:
    """Terminal dashboard for Deforum rendering."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize dashboard.

        Args:
            console: Rich console instance (uses default if None)
        """
        self.console = console or Console()
        self.layout = Layout()
        self.memory = MemoryStats()
        self.log_buffer = deque(maxlen=20)  # Last 20 log messages (larger for main scrolling area)
        self.live = None

        # Get theme and emoji settings
        self.theme = opt_utils.get_log_theme()
        self.use_emojis = opt_utils.is_emojis_enabled()
        self.use_ascii_preview = opt_utils.is_dashboard_ascii_preview_enabled()
        self.use_ascii_to_log = opt_utils.is_dashboard_ascii_to_log_enabled()

        # Store last frame image for ASCII preview
        self.last_frame_image = None

        # Dashboard state
        self.frame_info = {
            'current': 0,
            'total': 0,
            'type': 'KEYFRAME',
            'seed': 0,
            'color_rgb': None,
            'movement': '',
            'prompt': ''
        }

        self.table_data = {
            'steps': '0/20',
            'cfg': '1.0',
            'dist_cfg': '3.5',
            'denoise': '0.85',
            'tr_x': '0',
            'tr_y': '0',
            'tr_z': '0',
            'ro_x': '0',
            'ro_y': '0',
            'ro_z': '0'
        }

        self.progress_data = {
            'diffusion_frames': (0, 100),
            'total_steps': (0, 100),
            'current_step': (0, 20)
        }

        self._setup_layout()

    def _setup_layout(self):
        """Configure dashboard layout structure."""
        # Split into header (frame info + ASCII preview), progress, log (main scrolling area)
        # Header size depends on whether ASCII preview is enabled
        header_size = 24 if self.use_ascii_preview else 6

        self.layout.split(
            Layout(name="header", size=header_size),
            Layout(name="progress", size=8),
            Layout(name="log")  # Takes remaining space - main scrolling log area
        )

    def _render_header(self) -> Panel:
        """Render frame info and parameters header."""
        from deforum.orchestration.generate import _rgb_to_ansi_color_block, _RESET_BG

        # Frame info line
        frame_type_color = "green" if self.frame_info['type'] == 'KEYFRAME' else "yellow"  # Exception colors (success/warn)
        frame_line = Text()
        frame_line.append(f"Animation frame: ", style=self._themed_style('blue', bold=True))
        frame_line.append(f"{self.frame_info['current']}/{self.frame_info['total']} ", style="bold")
        frame_line.append(f"[{self.frame_info['type']}]", style=frame_type_color)

        # Seed and color line
        seed_line = Text()
        seed_line.append(f"Seed: {self.frame_info['seed']}", style=self._themed_style('cyan'))

        # Add color block if available
        if self.frame_info['color_rgb']:
            r, g, b = self.frame_info['color_rgb']
            color_block = _rgb_to_ansi_color_block((r, g, b))
            seed_line.append(f", Color: {color_block}██{_RESET_BG}")

        # Add movement (already includes "Move: " prefix from orchestrator)
        if self.frame_info['movement']:
            seed_line.append(self.frame_info['movement'], style=self._themed_style('purple'))

        # Prompt line
        prompt_line = Text()
        prompt_line.append("Prompt: ", style=self._themed_style('blue'))
        prompt_line.append(self.frame_info['prompt'], style="white")

        # Parameter table
        table = Table(padding=0, box=box.SIMPLE, show_header=True, expand=True)
        table.add_column("Steps", style=self._themed_style('cyan'))
        table.add_column("CFG", style=self._themed_style('cyan'))
        table.add_column("Dist.CFG", style=self._themed_style('cyan'))
        table.add_column("Denoise", style=self._themed_style('cyan'))
        table.add_column("Tr X", style=self._themed_style('blue'))
        table.add_column("Tr Y", style=self._themed_style('blue'))
        table.add_column("Tr Z", style=self._themed_style('blue'))
        table.add_column("Ro X", style=self._themed_style('purple'))
        table.add_column("Ro Y", style=self._themed_style('purple'))
        table.add_column("Ro Z", style=self._themed_style('purple'))

        table.add_row(
            self.table_data['steps'],
            self.table_data['cfg'],
            self.table_data['dist_cfg'],
            self.table_data['denoise'],
            self.table_data['tr_x'],
            self.table_data['tr_y'],
            self.table_data['tr_z'],
            self.table_data['ro_x'],
            self.table_data['ro_y'],
            self.table_data['ro_z']
        )

        # Combine text info
        content = Text()
        content.append_text(frame_line)
        content.append("\n")
        content.append_text(seed_line)
        content.append("\n")
        content.append_text(prompt_line)

        # Add ASCII preview if enabled and image available
        elements = [content, table]
        if self.use_ascii_preview and self.last_frame_image is not None:
            ascii_art = image_to_ascii_art(
                self.last_frame_image,
                width=32,
                height=18,
                use_color=(self.theme != 'simple')
            )
            if ascii_art:
                preview_text = Text()
                preview_text.append(ascii_art)
                elements.append(preview_text)

        # Get themed border color (use blue, but allow simple theme to disable)
        border_color = self._themed_style('blue').replace(' bold', '') if self._themed_style('blue') else "dim"

        return Panel(
            Group(*elements),
            title="Frame Info",
            border_style=border_color,
            padding=(0, 1)
        )

    def _render_progress(self) -> Panel:
        """Render progress bars and memory info."""
        content = Text()

        # Diffusion frames progress
        df_current, df_total = self.progress_data['diffusion_frames']
        df_pct = (df_current / df_total * 100) if df_total > 0 else 0
        df_bar = self._make_bar(df_pct, 30, "purple")
        content.append(f"Diffusion Frames  {df_bar} {df_current}/{df_total}\n")

        # Total steps progress
        ts_current, ts_total = self.progress_data['total_steps']
        ts_pct = (ts_current / ts_total * 100) if ts_total > 0 else 0
        ts_bar = self._make_bar(ts_pct, 30, "red")
        content.append(f"Total Steps       {ts_bar} {ts_current}/{ts_total}\n")

        # Current step progress
        cs_current, cs_total = self.progress_data['current_step']
        cs_pct = (cs_current / cs_total * 100) if cs_total > 0 else 0
        cs_bar = self._make_bar(cs_pct, 30, "orange")
        content.append(f"Current Step      {cs_bar} {cs_current}/{cs_total}\n")

        content.append("\n")

        # Memory info
        mem_pct = self.memory.get_usage_percentage()
        mem_bar = self._make_bar(mem_pct, 30, "cyan")
        content.append(f"VRAM Usage        {mem_bar} ", style=self._themed_style('cyan', bold=True))
        content.append(f"{self.memory.free_gpu_mb/1024:.1f}GB free\n")

        if self.memory.target_model:
            content.append(f"Loading: {self.memory.target_model}\n", style="yellow")  # Exception color (warn)

        # Get themed border color
        border_color = self._themed_style('purple').replace(' bold', '') if self._themed_style('purple') else "dim"

        return Panel(content, title="Progress", border_style=border_color, padding=(0, 1))

    def _render_log(self) -> Panel:
        """Render scrolling log area."""
        content = Text()
        if len(self.log_buffer) == 0:
            content.append("Waiting for generation to start...", style="dim")
        else:
            for msg in self.log_buffer:
                content.append(msg + "\n")

        # Get themed border color
        border_color = self._themed_style('cyan').replace(' bold', '') if self._themed_style('cyan') else "dim"

        return Panel(
            content,
            title="Console Log",
            border_style=border_color,
            padding=(0, 1)
        )

    def _get_themed_color(self, classic_color_name: str) -> str:
        """Get themed color for dashboard elements.

        Args:
            classic_color_name: Classic color name (purple, red, orange, cyan, etc.)

        Returns:
            Hex color code appropriate for current theme
        """
        # Map classic color names to hex codes
        color_hex_map = {
            'purple': HEX_CLASSIC_PURPLE,
            'red': HEX_CLASSIC_RED,
            'orange': HEX_CLASSIC_ORANGE,
            'blue': HEX_CLASSIC_BLUE,
            'green': HEX_CLASSIC_GREEN,
            'yellow': HEX_CLASSIC_YELLOW,
            'cyan': HEX_CLASSIC_BLUE,  # Use blue for cyan
        }

        classic_hex = color_hex_map.get(classic_color_name, HEX_CLASSIC_BLUE)
        themed_hex = get_tqdm_color_for_theme(classic_hex, self.theme)

        # If simple theme (None), return empty string
        if themed_hex is None:
            return ''

        return themed_hex

    def _themed_style(self, classic_color_name: str, bold: bool = False) -> str:
        """Get Rich style string with themed color.

        Args:
            classic_color_name: Classic color name (purple, red, orange, cyan, etc.)
            bold: If True, add bold styling

        Returns:
            Rich style string (e.g., "#764BA2 bold" or "" for simple theme)
        """
        # Exception colors that always use original (warn/error/success)
        exception_colors = {'yellow', 'red', 'green'}

        if classic_color_name in exception_colors:
            # Use original color for essential info
            style = classic_color_name
        else:
            # Get themed color
            themed_hex = self._get_themed_color(classic_color_name)
            if not themed_hex:
                return "bold" if bold else ""
            style = f"#{themed_hex.lstrip('#')}"

        if bold:
            style += " bold"

        return style

    def _make_bar(self, percentage: float, width: int, color_name: str) -> str:
        """Create a simple ASCII progress bar with themed colors.

        Args:
            percentage: 0-100
            width: Character width
            color_name: Classic color name (purple, red, orange, cyan, etc.)

        Returns:
            Colored bar string
        """
        filled = int(width * percentage / 100)
        empty = width - filled

        # Use emoji or ASCII based on settings
        if self.use_emojis:
            filled_char = "█"
            empty_char = "░"
        else:
            filled_char = "█"
            empty_char = "░"

        bar = filled_char * filled + empty_char * empty

        # Get themed color
        color_hex = self._get_themed_color(color_name)
        if color_hex:
            # Remove # prefix if present (Rich doesn't need it)
            color_hex = color_hex.lstrip('#')
            return f"[#{color_hex}]{bar}[/#{color_hex}]"
        else:
            return bar  # No color for simple theme

    def update(self):
        """Update the dashboard display."""
        try:
            self.layout["header"].update(self._render_header())
            self.layout["progress"].update(self._render_progress())
            self.layout["log"].update(self._render_log())

            # Don't call refresh - let Live handle it automatically
            # Calling refresh() can block the main thread
        except Exception as e:
            # Silently ignore render errors to avoid blocking generation
            pass

    def add_log(self, message: str):
        """Add a message to the scrolling log."""
        self.log_buffer.append(message)

    def add_ascii_art_to_log(self, image, frame_idx: int):
        """Add ASCII art of image to the scrolling log if enabled.

        Args:
            image: PIL Image or numpy array
            frame_idx: Current frame index
        """
        if not self.use_ascii_to_log or image is None:
            return

        # Generate small ASCII art for log (smaller than header preview)
        ascii_art = image_to_ascii_art(
            image,
            width=16,  # Smaller for log
            height=9,
            use_color=(self.theme != 'simple')
        )

        if ascii_art:
            # Add frame header
            self.add_log(f"Frame {frame_idx}:")
            # Add each line of ASCII art
            for line in ascii_art.split('\n'):
                self.add_log(line)

    def start(self):
        """Start live dashboard display."""
        # Initialize all layout sections before starting Live
        self.layout["header"].update(self._render_header())
        self.layout["progress"].update(self._render_progress())
        self.layout["log"].update(self._render_log())

        # Start Live display with reduced refresh rate to minimize CPU usage
        self.live = Live(self.layout, console=self.console, refresh_per_second=2)
        self.live.start()

    def stop(self):
        """Stop live dashboard display."""
        if self.live:
            self.live.stop()
            self.live = None
