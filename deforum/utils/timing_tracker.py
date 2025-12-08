"""Timing tracker for performance profiling during Deforum generation.

Tracks time spent in major operations and generates a summary report with ASCII graph.
"""

import time
from typing import Dict, Optional
from contextlib import contextmanager
from deforum.utils.system.logging import get_logger

logger = get_logger()


class TimingTracker:
    """Singleton timing tracker for accumulating operation durations.

    Tracks time spent in major categories:
    - diffusion: Actual diffusion/sampling (img2img/txt2img)
    - depth_estimation: Depth map generation (DA2/DA3)
    - depth_warping: 2D/3D transformations (anim_frame_warp)
    - wan_interpolation: Wan FLF2V AI video interpolation
    - preprocessing: Noise, masks, preparation before diffusion
    - postprocessing: Saving frames, color correction after diffusion
    - video_stitching: FFmpeg video/audio stitching
    - other: Everything else (setup, teardown, misc)
    """

    _instance: Optional['TimingTracker'] = None

    def __new__(cls):
        """Singleton pattern to ensure single global tracker."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize timing categories (only on first instantiation)."""
        if self._initialized:
            return

        self.categories: Dict[str, float] = {
            'diffusion': 0.0,
            'depth_estimation': 0.0,
            'depth_warping': 0.0,
            'wan_interpolation': 0.0,
            'preprocessing': 0.0,
            'postprocessing': 0.0,
            'video_stitching': 0.0,
            'other': 0.0,
        }
        self.total_time: float = 0.0
        self.start_time: Optional[float] = None
        self._initialized = True

    def reset(self):
        """Reset all timing data (call at start of new generation)."""
        for category in self.categories:
            self.categories[category] = 0.0
        self.total_time = 0.0
        self.start_time = None

    def start_generation(self):
        """Mark the start of generation (for total time tracking)."""
        self.reset()
        self.start_time = time.perf_counter()

    def end_generation(self):
        """Mark the end of generation and calculate total time."""
        if self.start_time is not None:
            self.total_time = time.perf_counter() - self.start_time

    @contextmanager
    def track(self, category: str):
        """Context manager to track time in a specific category.

        Args:
            category: Category name (must exist in self.categories)

        Example:
            with tracker.track('diffusion'):
                # diffusion code here
                pass
        """
        if category not in self.categories:
            logger.warning(f"Unknown timing category: {category}, using 'other'")
            category = 'other'

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.categories[category] += elapsed

    def add_time(self, category: str, seconds: float):
        """Manually add time to a category (for external timing).

        Args:
            category: Category name
            seconds: Time in seconds to add
        """
        if category not in self.categories:
            logger.warning(f"Unknown timing category: {category}, using 'other'")
            category = 'other'
        self.categories[category] += seconds

    def print_report(self):
        """Print a detailed timing report with ASCII bar graph."""
        if self.total_time == 0:
            self.end_generation()  # Auto-calculate total if not done

        if self.total_time == 0:
            logger.info("No timing data available")
            return

        # Calculate actual total from categories
        category_total = sum(self.categories.values())

        # Format time as HH:MM:SS or MM:SS
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            if hours > 0:
                return f"{hours}h {minutes:02d}m {secs:02d}s"
            elif minutes > 0:
                return f"{minutes}m {secs:02d}s"
            else:
                return f"{secs}s"

        # Print header
        print("\n" + "=" * 70)
        print("⏱️  GENERATION TIMING REPORT")
        print("=" * 70)
        print(f"Total Generation Time: {format_time(self.total_time)}")
        print(f"Tracked Operations:    {format_time(category_total)}")

        # Calculate percentages and sort by time (descending)
        categories_with_pct = [
            (name, time_val, (time_val / category_total * 100) if category_total > 0 else 0)
            for name, time_val in self.categories.items()
            if time_val > 0  # Only show non-zero categories
        ]
        categories_with_pct.sort(key=lambda x: x[1], reverse=True)

        if not categories_with_pct:
            print("\nNo operation timing data collected")
            print("=" * 70)
            return

        # Print breakdown with ASCII bars
        print("\nBreakdown by Operation:")
        print("-" * 70)

        # Max bar width (40 characters)
        max_bar_width = 40
        max_time = max(time_val for _, time_val, _ in categories_with_pct)

        for name, time_val, pct in categories_with_pct:
            # Calculate bar length
            if max_time > 0:
                bar_length = int((time_val / max_time) * max_bar_width)
            else:
                bar_length = 0

            # Create bar with gradient characters
            bar = "█" * bar_length

            # Format category name (capitalize and replace underscores)
            display_name = name.replace('_', ' ').title()

            # Print row: category | bar | time | percentage
            print(f"{display_name:20s} │ {bar:40s} {format_time(time_val):>10s} ({pct:5.1f}%)")

        print("-" * 70)

        # Show untracked time if significant
        untracked = self.total_time - category_total
        if untracked > 1.0:  # Only show if > 1 second
            untracked_pct = (untracked / self.total_time * 100) if self.total_time > 0 else 0
            print(f"{'Untracked Overhead':20s} │ {' ':40s} {format_time(untracked):>10s} ({untracked_pct:5.1f}%)")
            print("-" * 70)

        print("=" * 70 + "\n")


# Global singleton instance
_tracker_instance = None


def get_timing_tracker() -> TimingTracker:
    """Get the global timing tracker instance.

    Returns:
        Global TimingTracker singleton
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = TimingTracker()
    return _tracker_instance
