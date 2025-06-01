"""
Wan Progress Utilities - Styled progress indicators matching experimental render core
"""

import sys
from tqdm import tqdm
from typing import Optional, Any

# Import the same color constants as experimental render core
from ....utils.color_constants import (
    HEX_BLUE, HEX_GREEN, HEX_ORANGE, HEX_RED, HEX_PURPLE, HEX_YELLOW,
    BLUE, GREEN, ORANGE, RED, PURPLE, YELLOW, RESET_COLOR, BOLD
)

# Import shared WebUI progress handling
try:
    import modules.shared as shared
    WEBUI_AVAILABLE = True
except ImportError:
    WEBUI_AVAILABLE = False
    shared = None


class WanProgressBar:
    """Styled progress bar for Wan operations using experimental render core colors"""
    
    # Progress bar formats matching experimental render core
    NO_ETA_RBAR = "| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
    NO_ETA_BAR_FORMAT = "{l_bar}{bar}" + f"{NO_ETA_RBAR}"
    DEFAULT_BAR_FORMAT = "{l_bar}{bar}{r_bar}"
    
    def __init__(self, total: int, description: str, color: str = HEX_BLUE, 
                 unit: str = "step", position: int = 0, 
                 bar_format: str = None):
        """
        Create a styled progress bar for Wan operations
        
        Args:
            total: Total number of items
            description: Progress bar description
            color: Hex color (use HEX_* constants)
            unit: Unit name for progress
            position: Position for multi-bar displays
            bar_format: Custom bar format (defaults to NO_ETA_BAR_FORMAT)
        """
        self.total = total
        self.description = description
        self.color = color
        self.unit = unit
        self.position = position
        self.bar_format = bar_format or self.NO_ETA_BAR_FORMAT
        self.pbar: Optional[tqdm] = None
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def start(self):
        """Start the progress bar"""
        if self.pbar is not None:
            return
            
        # Use WebUI's progress output if available
        file_output = shared.progress_print_out if WEBUI_AVAILABLE and shared else sys.stdout
        
        # Check if console progress bars are disabled
        disable_console = (WEBUI_AVAILABLE and 
                          hasattr(shared, 'cmd_opts') and 
                          getattr(shared.cmd_opts, 'disable_console_progressbars', False))
        
        self.pbar = tqdm(
            total=self.total,
            desc=self.description,
            unit=self.unit,
            position=self.position,
            dynamic_ncols=True,
            file=file_output,
            bar_format=self.bar_format,
            disable=disable_console,
            colour=self.color
        )
        
    def update(self, n: int = 1, postfix: str = None):
        """Update progress bar"""
        if self.pbar:
            if postfix:
                self.pbar.set_postfix_str(postfix)
            self.pbar.update(n)
            
    def set_description(self, desc: str):
        """Update description"""
        if self.pbar:
            self.pbar.set_description(desc)
            
    def set_postfix(self, **kwargs):
        """Set postfix with key-value pairs"""
        if self.pbar:
            self.pbar.set_postfix(**kwargs)
            
    def close(self):
        """Close the progress bar"""
        if self.pbar:
            self.pbar.close()
            self.pbar = None


def print_wan_info(message: str, color: str = BLUE):
    """Print Wan info message with styling"""
    print(f"{color}{BOLD}Wan: {RESET_COLOR}{message}")

    
def print_wan_success(message: str):
    """Print Wan success message"""
    print_wan_info(f"✅ {message}", GREEN)

    
def print_wan_warning(message: str):
    """Print Wan warning message"""
    print_wan_info(f"⚠️ {message}", ORANGE)

    
def print_wan_error(message: str):
    """Print Wan error message"""
    print_wan_info(f"❌ {message}", RED)

    
def print_wan_progress(message: str):
    """Print Wan progress message"""
    print_wan_info(f"🎬 {message}", PURPLE)


def create_wan_model_loader_progress(model_name: str) -> WanProgressBar:
    """Create progress bar for model loading"""
    return WanProgressBar(
        total=100,  # Percentage-based
        description=f"Loading {model_name}",
        color=HEX_BLUE,
        unit="%"
    )


def create_wan_clip_progress(total_clips: int) -> WanProgressBar:
    """Create progress bar for clip generation"""
    return WanProgressBar(
        total=total_clips,
        description="Generating Clips",
        color=HEX_PURPLE,
        unit="clip",
        position=0
    )


def create_wan_frame_progress(total_frames: int, clip_idx: int) -> WanProgressBar:
    """Create progress bar for frame generation within a clip"""
    return WanProgressBar(
        total=total_frames,
        description=f"Clip {clip_idx + 1} Frames",
        color=HEX_GREEN,
        unit="frame",
        position=1
    )


def create_wan_inference_progress(total_steps: int) -> WanProgressBar:
    """Create progress bar for inference steps"""
    return WanProgressBar(
        total=total_steps,
        description="Inference Steps",
        color=HEX_ORANGE,
        unit="step",
        position=2
    )


def create_wan_video_processing_progress(total_frames: int) -> WanProgressBar:
    """Create progress bar for video post-processing"""
    return WanProgressBar(
        total=total_frames,
        description="Processing Video",
        color=HEX_RED,
        unit="frame"
    )


# Context managers for common Wan operations
class WanModelLoadingContext:
    """Context manager for model loading with progress"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pbar = None
        
    def __enter__(self):
        print_wan_progress(f"Loading model: {self.model_name}")
        self.pbar = create_wan_model_loader_progress(self.model_name)
        self.pbar.start()
        return self.pbar
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            if exc_type is None:
                self.pbar.update(100)  # Complete
                print_wan_success(f"Model loaded: {self.model_name}")
            else:
                print_wan_error(f"Failed to load model: {self.model_name}")
            self.pbar.close()


class WanGenerationContext:
    """Context manager for video generation with progress"""
    
    def __init__(self, total_clips: int):
        self.total_clips = total_clips
        self.clip_pbar = None
        
    def __enter__(self):
        print_wan_progress(f"Starting generation of {self.total_clips} clips")
        self.clip_pbar = create_wan_clip_progress(self.total_clips)
        self.clip_pbar.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.clip_pbar:
            if exc_type is None:
                print_wan_success(f"Generated {self.total_clips} clips successfully")
            else:
                print_wan_error(f"Generation failed: {exc_val}")
            self.clip_pbar.close()
            
    def update_clip(self, clip_idx: int, clip_description: str = ""):
        """Update clip progress"""
        if self.clip_pbar:
            postfix = f"Clip {clip_idx + 1}/{self.total_clips}"
            if clip_description:
                postfix += f" - {clip_description[:30]}"
            self.clip_pbar.update(1)
            self.clip_pbar.set_postfix_str(postfix)


# Utility functions for common print replacements
def replace_wan_prints_with_styled(text: str) -> str:
    """Replace common Wan print patterns with styled versions"""
    replacements = {
        "🎬 Wan": f"{PURPLE}{BOLD}🎬 Wan{RESET_COLOR}",
        "✅": f"{GREEN}✅{RESET_COLOR}",
        "❌": f"{RED}❌{RESET_COLOR}",
        "⚠️": f"{ORANGE}⚠️{RESET_COLOR}",
        "🔧": f"{BLUE}🔧{RESET_COLOR}",
        "🔍": f"{YELLOW}🔍{RESET_COLOR}",
        "📁": f"{GREEN}📁{RESET_COLOR}",
        "📝": f"{BLUE}📝{RESET_COLOR}",
        "🎯": f"{PURPLE}🎯{RESET_COLOR}",
    }
    
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result 