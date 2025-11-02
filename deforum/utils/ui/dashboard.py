"""Fixed-position terminal dashboard using ANSI escape codes.

Similar to Claude Code CLI - logs scroll up, dashboard stays at bottom.
Uses raw ANSI cursor positioning instead of Rich Live to avoid threading issues.
"""

import sys
import time
import shutil
import atexit
import signal
from typing import Optional

from deforum.rendering import options as opt_utils

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
    Image = None


def image_to_ascii_art(image, width: int = 32, height: int = 18, use_color: bool = True) -> str:
    """Convert PIL Image to colored ASCII art using 2-space blocks."""
    if image is None or Image is None or np is None:
        return ""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Preserve aspect ratio
    img_width, img_height = image.size
    img_aspect = img_width / img_height
    target_aspect = width / height

    if img_aspect > target_aspect:
        final_width = width
        final_height = int(width / img_aspect)
    else:
        final_height = height
        final_width = int(height * img_aspect)

    image = image.resize((final_width, final_height), Image.Resampling.LANCZOS)
    pixels = np.array(image)

    lines = []
    for y in range(final_height):
        line = ""
        for x in range(final_width):
            if len(pixels.shape) == 3 and pixels.shape[2] >= 3:
                r, g, b = pixels[y, x, :3]
            else:
                gray = pixels[y, x] if pixels.ndim == 2 else pixels[y, x, 0]
                r = g = b = gray

            if use_color:
                line += f"\033[48;2;{r};{g};{b}m  \033[0m"
            else:
                line += "  "

        lines.append(line)

    return "\n".join(lines)


class MemoryStats:
    """Tracks GPU memory statistics from Forge output."""

    def __init__(self):
        self.target_model = ""
        self.free_gpu_mb = 0.0
        self.total_gpu_mb = 0.0

    def update_from_forge_message(self, message: str):
        """Parse memory stats from Forge console message."""
        import re
        # Example: "[Memory Management] Target: VAE, Free GPU: 8238.74 MB..."
        target_match = re.search(r'Target:\s*([^,]+)', message)
        if target_match:
            self.target_model = target_match.group(1).strip()

        free_match = re.search(r'Free GPU:\s*([\d.]+)\s*MB', message)
        if free_match:
            self.free_gpu_mb = float(free_match.group(1))

        total_match = re.search(r'Total GPU:\s*([\d.]+)\s*MB', message)
        if total_match:
            self.total_gpu_mb = float(total_match.group(1))


class FixedDashboard:
    """Fixed-position dashboard at bottom of terminal.

    Logs scroll up normally, dashboard stays fixed at bottom.
    Uses ANSI escape codes for cursor positioning.
    """

    def __init__(self):
        """Initialize dashboard."""
        self.memory = MemoryStats()
        self._last_update_time = 0
        self._dashboard_height = 0  # Number of lines dashboard occupies
        self._terminal_height = 0
        self._terminal_width = 0
        self._is_active = False  # Track if dashboard is currently active

        # Resize handling
        self._resize_pending = False
        self._old_sigwinch_handler = None

        # Settings
        self.theme = opt_utils.get_log_theme()
        self.use_ascii_preview = opt_utils.is_dashboard_ascii_preview_enabled()

        # Register cleanup handler to run on exit (even if Ctrl+C)
        atexit.register(self._cleanup_terminal)

        # State
        self.last_frame_image = None
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

        # Register cleanup handler
        atexit.register(self._cleanup_terminal)

    def start(self):
        """Start dashboard - set up scrolling region and fixed dashboard."""
        # Get terminal size
        self._terminal_height, self._terminal_width = self._get_terminal_size()

        # Calculate dashboard height (fixed status only, no ASCII preview)
        self._dashboard_height = 8  # Separator + 1 status + 1 prompt + 2 current bars + 1 models + 3 total bars

        # Set up scrolling region (reserve bottom lines for dashboard)
        # ANSI: \033[{top};{bottom}r sets scrolling region
        scroll_bottom = self._terminal_height - self._dashboard_height
        sys.stdout.write(f"\033[1;{scroll_bottom}r")

        # Move cursor to top of scrolling region
        sys.stdout.write("\033[1;1H")

        # Don't clear screen - preserve previous output (subtitle info etc.)
        sys.stdout.flush()

        # Mark as active
        self._is_active = True

        # Register resize signal handler
        try:
            self._old_sigwinch_handler = signal.signal(signal.SIGWINCH, self._handle_resize)
        except (ValueError, OSError):
            # signal.signal() can fail on some platforms or contexts (like threading)
            pass

        # Initial render
        self._render()

    def stop(self):
        """Stop dashboard - clean up and restore normal scrolling."""
        if not self._is_active:
            return

        self._cleanup_terminal()

        # Print completion message
        print("\n" + "=" * 80)
        print("RENDER COMPLETE")
        print("=" * 80)

    def _cleanup_terminal(self):
        """Clean up terminal state - restore normal scrolling."""
        if not self._is_active:
            return

        try:
            # Restore old signal handler
            if self._old_sigwinch_handler is not None:
                try:
                    signal.signal(signal.SIGWINCH, self._old_sigwinch_handler)
                except (ValueError, OSError):
                    pass
                self._old_sigwinch_handler = None

            # CRITICAL: Reset scrolling region FIRST (most important for terminal restore)
            sys.stdout.write("\033[r")  # Reset to default scrolling region
            sys.stdout.flush()

            # Get current terminal size (might have changed)
            term_height, term_width = self._get_terminal_size()

            # Clear dashboard area
            dashboard_start = term_height - self._dashboard_height + 1
            for row in range(dashboard_start, term_height + 1):
                sys.stdout.write(f"\033[{row};1H")  # Move to row
                sys.stdout.write("\033[K")  # Clear line from cursor to end
                sys.stdout.flush()  # Flush after each line clear

            # Move cursor to safe position (below where dashboard was)
            sys.stdout.write(f"\033[{dashboard_start};1H")
            sys.stdout.flush()

            # Show cursor (in case it was hidden)
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

            # Final newline for good measure
            sys.stdout.write("\n")
            sys.stdout.flush()

        except Exception as e:
            # If cleanup fails, force terminal reset with more aggressive approach
            try:
                sys.stdout.write("\033[r")  # Reset scrolling region
                sys.stdout.flush()
                sys.stdout.write("\033[?25h")  # Show cursor
                sys.stdout.flush()
                sys.stdout.write("\033[H")  # Move to top
                sys.stdout.flush()
            except:
                pass

        # Mark as inactive
        self._is_active = False

    def update(self):
        """Update dashboard (throttled to ~10fps for smooth tqdm updates)."""
        # Check for pending resize first
        self._check_and_handle_resize()

        current_time = time.time()
        if current_time - self._last_update_time < 0.1:  # 100ms = ~10fps
            return

        self._last_update_time = current_time
        self._update_vram()
        self._render()

    def _handle_resize(self, signum, frame):
        """Handle terminal resize signal (SIGWINCH).

        Args:
            signum: Signal number
            frame: Current stack frame (unused)
        """
        # Just mark that resize is pending - don't do heavy work in signal handler
        self._resize_pending = True

    def _check_and_handle_resize(self):
        """Check if resize is pending and handle it."""
        if not self._resize_pending:
            return

        self._resize_pending = False

        # Get new terminal size
        old_width = self._terminal_width
        old_height = self._terminal_height
        self._terminal_height, self._terminal_width = self._get_terminal_size()

        # Only reconfigure if size actually changed
        if old_width == self._terminal_width and old_height == self._terminal_height:
            return

        # Recalculate scrolling region with new terminal size
        scroll_bottom = self._terminal_height - self._dashboard_height
        sys.stdout.write(f"\033[1;{scroll_bottom}r")
        sys.stdout.flush()

        # Force immediate re-render with new dimensions
        self._render()

    def _update_vram(self):
        """Update VRAM stats from torch."""
        try:
            import torch
            if torch.cuda.is_available():
                # Get free memory in MB
                free_memory = torch.cuda.mem_get_info()[0] / (1024 ** 2)
                self.memory.free_gpu_mb = free_memory
        except Exception:
            pass  # Ignore errors

    def _render(self):
        """Render dashboard at fixed position below scrolling region."""
        # Calculate dashboard start row (below scrolling region)
        dashboard_start = self._terminal_height - self._dashboard_height + 1

        # Move to dashboard area
        sys.stdout.write(f"\033[{dashboard_start};1H")

        # Clear from cursor to end of screen
        sys.stdout.write("\033[J")

        # Render content
        lines = self._build_dashboard()
        for line in lines:
            sys.stdout.write(line + "\n")

        # Move cursor back to scrolling region (bottom of scroll area)
        scroll_bottom = self._terminal_height - self._dashboard_height
        sys.stdout.write(f"\033[{scroll_bottom};1H")
        sys.stdout.flush()

    def _build_dashboard(self) -> list:
        """Build dashboard content as list of lines."""
        lines = []

        # Separator (full width with slopcore gradient)
        lines.append(self._create_separator())

        # Status line 1: Frame info with color, movement, and VRAM on right
        # Calculate progress from diffusion frames
        import modules.shared as shared
        taqaddum = shared.total_tqdm if hasattr(shared, 'total_tqdm') else None
        if taqaddum and hasattr(taqaddum, 'total_animation_cycles'):
            df_current = taqaddum.total_animation_cycles.n
            df_total = taqaddum.total_animation_cycles.total
            df_pct = int((df_current / df_total * 100) if df_total > 0 else 0)
        else:
            df_pct = 0

        # Colorize "Animation Frame:" label with theme-aware blue
        from deforum.utils.system.logging.log import HEX_BLUE
        from deforum.utils.system.logging.themes import get_tqdm_color_for_theme
        from deforum.utils.image.color import hex_to_ansi_foreground

        themed_blue_hex = get_tqdm_color_for_theme(HEX_BLUE, self.theme)
        animation_frame_color = hex_to_ansi_foreground(themed_blue_hex)

        # Colorize frame type
        frame_type = self.frame_info['type']
        if self.theme == 'slopcore':
            from deforum.utils.system.logging.themes import HEX_SLOPCORE_3, HEX_SLOPCORE_6
            if frame_type == 'KEYFRAME':
                type_color = hex_to_ansi_foreground(HEX_SLOPCORE_6)  # Deep purple
            else:  # CADENCE
                type_color = hex_to_ansi_foreground(HEX_SLOPCORE_3)  # Light purple
            frame_type_colored = f"{type_color}[{frame_type}]\033[0m"
        else:
            frame_type_colored = f"[{frame_type}]"

        line1_left = f"{animation_frame_color}Animation Frame:\033[0m {self.frame_info['current']}/{self.frame_info['total']} {frame_type_colored} | Progress: {df_pct}%"

        # Add color block if available
        if self.frame_info.get('color_rgb'):
            r, g, b = self.frame_info['color_rgb']
            color_block = f"\033[48;2;{r};{g};{b}m  \033[0m"
            line1_left += f" | Color: {color_block}"

        # Add movement indicators if available
        movement = self.frame_info.get('movement', '')
        if movement:
            line1_left += f" | {movement}"

        # Pad line 1 to full width (VRAM moved to last line with models)
        import re
        visible_left = re.sub(r'\033\[[0-9;]*m', '', line1_left)
        padding_needed = max(0, self._terminal_width - len(visible_left))
        line1 = line1_left + (" " * padding_needed)

        # Line 1: Prompt (centered, without "Prompt:" label)
        prompt = self.frame_info.get('prompt', '')
        if prompt:
            # Truncate prompt if too long for terminal width
            max_prompt_len = self._terminal_width - 4  # Leave some padding
            if len(prompt) > max_prompt_len:
                prompt = prompt[:max_prompt_len - 3] + "..."
            # Center the prompt
            visible_prompt = re.sub(r'\033\[[0-9;]*m', '', prompt)
            padding_total = max(0, self._terminal_width - len(visible_prompt))
            padding_left = padding_total // 2
            padding_right = padding_total - padding_left
            line1 = (" " * padding_left) + prompt + (" " * padding_right)
        else:
            line1 = " " * self._terminal_width
        lines.append(line1)

        # Line 2: Animation Frame (was line 1)
        # Pad to full width (VRAM moved to last line with models)
        visible_left = re.sub(r'\033\[[0-9;]*m', '', line1_left)
        padding_needed = max(0, self._terminal_width - len(visible_left))
        line2 = line1_left + (" " * padding_needed)
        lines.append(line2)

        # Lines 3-7: Reorganized progress bars
        # 3. Current Tweens
        # 4. Current Diffusion Steps
        # 5. Total Diffusion Steps (with reverse gradient if slopcore)
        # 6. Total Frames (with gradient as-is)
        # 7. Models + VRAM (left-aligned, last line)
        lines.extend(self._render_tqdm_bars_reorganized())

        return lines

    def _format_vram_bar(self) -> str:
        """Format VRAM usage as a colored progress bar.

        Returns:
            Formatted VRAM bar string with color (green/yellow/red based on usage)
        """
        try:
            import torch
            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info()
                free_gb = free_mem / (1024 ** 3)
                total_gb = total_mem / (1024 ** 3)
                used_gb = total_gb - free_gb
                used_pct = int((used_gb / total_gb * 100) if total_gb > 0 else 0)

                # Determine color based on usage
                if used_pct >= 95:
                    color = "\033[38;2;255;100;100m"  # Red
                elif used_pct >= 80:
                    color = "\033[38;2;255;220;100m"  # Yellow
                else:
                    color = "\033[38;2;100;255;100m"  # Green

                # Create mini bar (20 chars)
                bar_width = 20
                filled = int(bar_width * used_pct / 100)
                bar = "█" * filled + "░" * (bar_width - filled)

                return f"VRAM: {color}{bar}\033[0m {used_gb:.1f}/{total_gb:.1f}GB ({used_pct}%)"
            else:
                return "VRAM: N/A"
        except Exception:
            return "VRAM: N/A"

    def _format_loaded_models_detailed(self) -> str:
        """Format loaded models info with detailed information on dedicated line.

        Returns:
            String showing which models are loaded with sizes, devices, and status indicators
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return ""

            # Try to detect loaded models from Forge's model management
            import modules.shared as shared
            loaded_models = []

            # Detect main model (Flux/Lumina/SD)
            main_model_on_gpu = False
            if hasattr(shared, 'sd_model') and shared.sd_model is not None:
                model_name = "Unknown"
                model_size_gb = 0

                # Try to get model name from config or checkpoint info
                if hasattr(shared.sd_model, 'sd_checkpoint_info'):
                    checkpoint_info = shared.sd_model.sd_checkpoint_info
                    if hasattr(checkpoint_info, 'model_name'):
                        model_name = checkpoint_info.model_name
                    elif hasattr(checkpoint_info, 'title'):
                        model_name = checkpoint_info.title

                # Simplify model name
                if 'flux' in model_name.lower():
                    model_name = "Flux"
                elif 'lumina' in model_name.lower():
                    model_name = "Lumina"
                elif 'sd' in model_name.lower() or 'stable' in model_name.lower():
                    model_name = "SD"

                # Try to estimate size from parameters
                if hasattr(shared.sd_model, 'parameters'):
                    try:
                        param_count = sum(p.numel() for p in shared.sd_model.parameters())
                        # Rough estimate: 4 bytes per parameter (fp32) or 2 bytes (fp16)
                        model_size_gb = (param_count * 2) / (1024 ** 3)  # Assume fp16
                    except:
                        pass

                # Check if model is on GPU
                try:
                    if hasattr(shared.sd_model, 'device'):
                        main_model_on_gpu = str(shared.sd_model.device).startswith('cuda')
                except:
                    main_model_on_gpu = True  # Assume on GPU if can't determine

                if model_size_gb > 0:
                    loaded_models.append(f"{model_name} ({model_size_gb:.1f}GB)")
                else:
                    loaded_models.append(model_name)

            # Detect depth model using singleton instance
            depth_model_on_gpu = False
            try:
                from deforum.depth.depth import DepthModel
                if DepthModel._instance is not None and not DepthModel._instance.should_delete:
                    depth_algo = DepthModel._instance.depth_algorithm

                    # Extract size from algorithm name (Small/Base/Large)
                    model_size = depth_algo.lower().split('-')[-1]

                    # Approximate sizes for Depth-Anything-V2 (fp16)
                    size_map = {
                        'small': 0.1,   # ~25M params
                        'base': 0.4,    # ~97M params
                        'large': 1.3    # ~335M params
                    }
                    depth_size_gb = size_map.get(model_size, 0.1)

                    # Check if on GPU
                    try:
                        depth_device = str(DepthModel._instance.device)
                        depth_model_on_gpu = depth_device.startswith('cuda')
                    except:
                        depth_model_on_gpu = False

                    if depth_model_on_gpu:
                        loaded_models.append(f"Depth-{model_size.capitalize()} ({depth_size_gb:.1f}GB)")
            except:
                pass  # Depth model not available

            if not loaded_models:
                return ""

            # Build detailed info string
            parts = []

            # Add diffusion model info
            if len(loaded_models) >= 1:
                parts.append(f"Diffusion: {loaded_models[0]}")

            # Add depth model info if present
            if len(loaded_models) >= 2:
                parts.append(f"Depth: {loaded_models[1]}")

            # Add status indicator
            if len(loaded_models) == 2:
                # Both loaded simultaneously
                status = "✓ Both in VRAM"
            elif len(loaded_models) == 1 and main_model_on_gpu:
                # Only diffusion loaded
                status = "⇄ Swapping" if depth_model_on_gpu else ""
            else:
                status = ""

            result = " | ".join(parts)
            if status:
                result += f" | {status}"

            return result

        except Exception:
            return ""  # Silently fail if can't detect

    def _create_separator(self) -> str:
        """Create separator line with slopcore gradient if enabled.

        Returns:
            Separator line string (gradient or plain based on theme)
        """
        separator_char = "─"
        separator_text = separator_char * self._terminal_width

        if self.theme == 'slopcore':
            # Use lightest blue for slopcore separator
            from deforum.utils.system.logging.themes import HEX_SLOPCORE_1
            from deforum.utils.image.color import hex_to_ansi_foreground

            color = hex_to_ansi_foreground(HEX_SLOPCORE_1)
            return f"{color}{separator_text}\033[0m"
        else:
            # Plain separator for classic/simple themes
            return separator_text

    def _render_tqdm_bars(self) -> list:
        """Render tqdm progress bars as text lines.

        Returns:
            List of formatted progress bar strings
        """
        lines = []

        # Use progress_data that's updated by callbacks (works even when tqdm disabled)
        try:
            from deforum.utils.system.logging.log import (
                HEX_BLUE, HEX_GREEN, HEX_ORANGE, HEX_RED, HEX_PURPLE
            )

            # Access shared tqdm instance for totals
            import modules.shared as shared
            taqaddum = shared.total_tqdm

            if taqaddum and hasattr(taqaddum, 'tweens'):
                # Calculate max description length for alignment
                desc5 = getattr(taqaddum.total_animation_cycles, 'desc', None) or "Diffusion Frames"
                max_desc_len = max(
                    len("Current Tweens"),
                    len("Total Frames"),
                    len("Current Diffusion Steps"),  # Longest at 23 chars
                    len("Total Diffusion Steps"),
                    len(desc5)
                )

                # Read from progress_data (updated via callbacks even when tqdm disabled)
                # Bar 1: Current Tweens (blue)
                tw_current, tw_total = self.progress_data.get('current_tweens', (taqaddum._tweens_n, taqaddum.tweens.total))
                lines.append(self._format_tqdm_bar(
                    "Current Tweens",
                    tw_current,
                    tw_total,
                    "tween",
                    HEX_BLUE,
                    max_desc_len
                ))

                # Bar 2: Total Frames (green)
                tf_current, tf_total = self.progress_data.get('total_frames', (taqaddum._total_frames_n, taqaddum.total_frames.total))
                lines.append(self._format_tqdm_bar(
                    "Total Frames",
                    tf_current,
                    tf_total,
                    "frame",
                    HEX_GREEN,
                    max_desc_len
                ))

                # Bar 3: Current Diffusion Steps (orange)
                cs_current, cs_total = self.progress_data.get('current_step', (taqaddum._steps_n, taqaddum.steps.total))
                lines.append(self._format_tqdm_bar(
                    "Current Diffusion Steps",
                    cs_current,
                    cs_total,
                    "step",
                    HEX_ORANGE,
                    max_desc_len
                ))

                # Bar 4: Total Diffusion Steps (red)
                ts_current, ts_total = self.progress_data.get('total_steps', (taqaddum._total_steps_n, taqaddum.total_steps.total))
                lines.append(self._format_tqdm_bar(
                    "Total Diffusion Steps",
                    ts_current,
                    ts_total,
                    "step",
                    HEX_RED,
                    max_desc_len
                ))

                # Bar 5: Diffusion Frames (purple)
                df_current, df_total = self.progress_data.get('diffusion_frames', (taqaddum._animation_cycles_n, taqaddum.total_animation_cycles.total))
                lines.append(self._format_tqdm_bar(
                    desc5,
                    df_current,
                    df_total,
                    "frame",
                    HEX_PURPLE,
                    max_desc_len
                ))
            else:
                # Fallback if tqdm not available
                lines.extend([
                    "Current Tweens: 0/0",
                    "Total Frames: 0/0",
                    "Current Diffusion Steps: 0/0",
                    "Total Diffusion Steps: 0/0",
                    "Diffusion Frames: 0/0"
                ])
        except Exception:
            # Fallback on error
            lines.extend([
                "Current Tweens: 0/0",
                "Total Frames: 0/0",
                "Current Diffusion Steps: 0/0",
                "Total Diffusion Steps: 0/0",
                "Diffusion Frames: 0/0"
            ])

        return lines

    def _render_tqdm_bars_reorganized(self) -> list:
        """Render reorganized progress bars with models line at the end.

        New order:
        1. Current Tweens
        2. Current Diffusion Steps
        3. Total Diffusion Steps
        4. Total Diffusion Frames
        5. Total Frames
        6. Models + VRAM (left-aligned, last line)

        Returns:
            List of formatted progress bar strings
        """
        lines = []

        try:
            from deforum.utils.system.logging.log import (
                HEX_BLUE, HEX_GREEN, HEX_ORANGE, HEX_RED, HEX_PURPLE
            )

            import modules.shared as shared
            taqaddum = shared.total_tqdm

            if taqaddum and hasattr(taqaddum, 'tweens'):
                # Calculate max description length for alignment
                max_desc_len = max(
                    len("Current Tweens"),
                    len("Current Diffusion Steps"),
                    len("Total Diffusion Steps"),
                    len("Total Diffusion Frames"),
                    len("Total Frames")
                )

                # Line 1: Current Tweens (purple - fast/high frequency)
                tw_current, tw_total = self.progress_data.get('current_tweens', (taqaddum._tweens_n, taqaddum.tweens.total))
                lines.append(self._format_tqdm_bar(
                    "Current Tweens",
                    tw_current,
                    tw_total,
                    "tween",
                    HEX_PURPLE,
                    max_desc_len
                ))

                # Line 2: Current Diffusion Steps (blue - fast/high frequency)
                cs_current, cs_total = self.progress_data.get('current_step', (taqaddum._steps_n, taqaddum.steps.total))
                lines.append(self._format_tqdm_bar(
                    "Current Diffusion Steps",
                    cs_current,
                    cs_total,
                    "step",
                    HEX_BLUE,
                    max_desc_len
                ))

                # Line 3: Total Diffusion Steps (green - medium-low frequency)
                ts_current, ts_total = self.progress_data.get('total_steps', (taqaddum._total_steps_n, taqaddum.total_steps.total))
                lines.append(self._format_tqdm_bar(
                    "Total Diffusion Steps",
                    ts_current,
                    ts_total,
                    "step",
                    HEX_GREEN,
                    max_desc_len,
                    reverse_gradient=True  # NEW: reverse gradient
                ))

                # Line 4: Total Diffusion Frames (orange - low frequency)
                df_current, df_total = self.progress_data.get('diffusion_frames', (taqaddum._animation_cycles_n, taqaddum.total_animation_cycles.total))
                lines.append(self._format_tqdm_bar(
                    "Total Diffusion Frames",
                    df_current,
                    df_total,
                    "frame",
                    HEX_ORANGE,
                    max_desc_len
                ))

                # Line 5: Total Frames (red - lowest frequency)
                tf_current, tf_total = self.progress_data.get('total_frames', (taqaddum._total_frames_n, taqaddum.total_frames.total))
                lines.append(self._format_tqdm_bar(
                    "Total Frames",
                    tf_current,
                    tf_total,
                    "frame",
                    HEX_RED,
                    max_desc_len
                ))

                # Line 6: Models (left) + VRAM (right-aligned)
                models_str = self._format_loaded_models_detailed()
                vram_str = self._format_vram_bar()

                # Calculate spacing to right-align VRAM
                import re
                models_visible = re.sub(r'\033\[[0-9;]*m', '', models_str) if models_str else ""
                vram_visible = re.sub(r'\033\[[0-9;]*m', '', vram_str)

                # Space between models and VRAM to fill terminal width
                total_content_len = len(models_visible) + len(vram_visible)
                if models_str:
                    total_content_len += 3  # Account for " | " separator

                padding_needed = max(1, self._terminal_width - total_content_len)

                if models_str:
                    models_vram_line = f"{models_str}{' ' * padding_needed}{vram_str}"
                else:
                    # If no models, still right-align VRAM
                    models_vram_line = f"{' ' * padding_needed}{vram_str}"

                lines.append(models_vram_line)
            else:
                # Fallback if tqdm not available
                lines.extend([
                    "Current Tweens: 0/0",
                    "Current Diffusion Steps: 0/0",
                    "Total Diffusion Steps: 0/0",
                    "Total Diffusion Frames: 0/0",
                    "Total Frames: 0/0",
                    "Models: N/A | VRAM: N/A"
                ])
        except Exception:
            # Fallback on error
            lines.extend([
                "Current Tweens: 0/0",
                "Current Diffusion Steps: 0/0",
                "Total Diffusion Steps: 0/0",
                "Total Diffusion Frames: 0/0",
                "Total Frames: 0/0",
                "Models: N/A | VRAM: N/A"
            ])

        return lines

    def _format_tqdm_bar(self, desc: str, current: int, total: int, unit: str, color_hex: str = None, max_desc_len: int = None, reverse_gradient: bool = False) -> str:
        """Format a single tqdm bar as text with theme colors.

        Args:
            desc: Bar description
            current: Current progress value
            total: Total progress value
            unit: Unit name (tween, frame, step)
            color_hex: Classic color hex for theme mapping
            max_desc_len: Maximum description length for alignment (optional)
            reverse_gradient: If True, reverse the gradient direction (dark→light instead of light→dark)

        Returns:
            Formatted bar string with ANSI colors
        """
        from deforum.utils.system.logging.themes import get_tqdm_color_for_theme
        from deforum.utils.image.color import hex_to_ansi_foreground

        pct = int((current / total * 100) if total > 0 else 0)

        # Pad description for alignment
        if max_desc_len:
            desc_padded = desc.ljust(max_desc_len)
        else:
            desc_padded = desc

        # Calculate dynamic bar width to fill terminal
        # Format: "Description: [BAR] current/total units (pct%)"
        # Reserve space for desc, colons, spaces, numbers, and percentage
        desc_len = max_desc_len if max_desc_len else len(desc)
        suffix = f" {current}/{total} {unit}s ({pct}%)"
        reserved = desc_len + 2 + len(suffix)  # +2 for ": "
        bar_width = max(20, self._terminal_width - reserved - 1)  # -1 for safety margin

        filled = int(bar_width * current / total) if total > 0 else 0

        # Always use solid color bars (gradients were confusing)
        bar_filled = "█" * filled
        bar_empty = "░" * (bar_width - filled)

        # Apply theme color if provided
        if color_hex and self.theme != 'simple':
            themed_hex = get_tqdm_color_for_theme(color_hex, self.theme)
            if themed_hex:
                color_code = hex_to_ansi_foreground(themed_hex)
                bar = f"{color_code}{bar_filled}\033[0m{bar_empty}"
            else:
                bar = bar_filled + bar_empty
        else:
            bar = bar_filled + bar_empty

        # Build line (don't pad here - ANSI codes mess up ljust)
        line = f"{desc_padded}: {bar} {current}/{total} {unit}s ({pct}%)"

        # Calculate visible length (excluding ANSI codes)
        import re
        visible_line = re.sub(r'\033\[[0-9;]*m', '', line)
        padding_needed = max(0, self._terminal_width - len(visible_line))

        return line + (" " * padding_needed)

    def _create_gradient_bar(self, filled: int, empty: int, color_hex: str, reverse: bool = False) -> str:
        """Create a gradient progress bar for slopcore theme.

        Args:
            filled: Number of filled characters
            empty: Number of empty characters
            color_hex: Base color hex for gradient mapping
            reverse: If True, reverse gradient direction (dark→light instead of light→dark)

        Returns:
            Gradient-colored bar string
        """
        from deforum.utils.system.logging.themes import (
            HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3,
            HEX_SLOPCORE_4, HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7
        )
        from deforum.utils.image.color import hex_to_ansi_foreground

        # Use full 7-shade slopcore gradient
        gradient_colors = [
            HEX_SLOPCORE_1, HEX_SLOPCORE_2, HEX_SLOPCORE_3,
            HEX_SLOPCORE_4, HEX_SLOPCORE_5, HEX_SLOPCORE_6, HEX_SLOPCORE_7
        ]

        # Reverse gradient if requested (dark→light instead of light→dark)
        if reverse:
            gradient_colors = list(reversed(gradient_colors))

        result = ""

        # Gradient filled portion
        if filled > 0:
            chars_per_color = filled / len(gradient_colors)
            for i in range(filled):
                color_idx = min(int(i / chars_per_color), len(gradient_colors) - 1)
                color = hex_to_ansi_foreground(gradient_colors[color_idx])
                result += f"{color}█"
            result += "\033[0m"  # Reset after filled

        # Empty portion (no color)
        result += "░" * empty

        return result

    def _get_terminal_size(self):
        """Get terminal size (height, width)."""
        try:
            size = shutil.get_terminal_size()
            return size.lines, size.columns
        except:
            return 40, 80  # Default fallback

    def _move_cursor_below_dashboard(self):
        """Move cursor to below dashboard area."""
        row = self._terminal_height + 1
        sys.stdout.write(f"\033[{row};1H")
        sys.stdout.flush()

    def add_log(self, message: str):
        """Add log message to scrolling region."""
        # Check for memory management messages
        if "[Memory Management]" in message:
            self.memory.update_from_forge_message(message)
            self.update()  # Refresh dashboard with new VRAM info

        # Cursor should already be in scrolling region
        # Just print normally - scrolling region will handle it
        print(message)

    def add_ascii_art_to_log(self, image, frame_idx: int):
        """Add ASCII art to scrolling log if enabled.

        Args:
            image: PIL Image or numpy array to convert
            frame_idx: Current frame number
        """
        if not opt_utils.is_dashboard_ascii_to_log_enabled():
            return

        if image is None:
            return

        width, height = opt_utils.get_dashboard_ascii_size()
        ascii_art = image_to_ascii_art(
            image,
            width=width,
            height=height,
            use_color=(self.theme != 'simple')
        )

        if ascii_art:
            # Integrate frame number into first line of ASCII art
            lines = ascii_art.split('\n')
            if lines:
                frame_text = f"[Frame {frame_idx}]"
                # Overlay text on first line (white text on colored background)
                first_line = lines[0]
                # Strip ANSI codes to calculate visible length
                import re
                visible_first = re.sub(r'\033\[[0-9;]*m', '', first_line)

                # If we have enough space, overlay the text at the start
                if len(visible_first) >= len(frame_text):
                    # Create white text on background: each char replaces 2-space block
                    # White foreground + preserve background color
                    overlay = ""
                    # Parse first N blocks to get background colors
                    pos = 0
                    for char in frame_text:
                        # Find next background color code in first_line
                        bg_match = re.search(r'\033\[48;2;(\d+);(\d+);(\d+)m', first_line[pos:])
                        if bg_match:
                            r, g, b = bg_match.groups()
                            # White text (255,255,255) on extracted background
                            overlay += f"\033[38;2;255;255;255m\033[48;2;{r};{g};{b}m{char} \033[0m"
                            # Move position past this block (bg code + 2 spaces + reset)
                            pos += bg_match.end() + 3  # Skip " \033[0m"
                        else:
                            # Fallback: white on black
                            overlay += f"\033[38;2;255;255;255m{char} \033[0m"

                    # Replace start of first line with overlay
                    # Count how many complete blocks (each "  " = one block) to replace
                    blocks_to_replace = len(frame_text)
                    # Each block is: \033[48;2;r;g;bm  \033[0m (approx 20-25 chars)
                    # Find position after N blocks
                    block_count = 0
                    cut_pos = 0
                    for match in re.finditer(r'\033\[48;2;\d+;\d+;\d+m  \033\[0m', first_line):
                        if block_count >= blocks_to_replace:
                            cut_pos = match.start()
                            break
                        block_count += 1

                    lines[0] = overlay + first_line[cut_pos:]

                print("\n".join(lines))
