"""Base interface for tween generators.

Defines the common interface that all tween generators must implement.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseTweenGenerator(ABC):
    """Base class for tween frame generators.

    All tween generators must implement the generate_tween() method.
    """

    @abstractmethod
    def generate_tween(self, data: Any, tween_frame: Any, prev_image: np.ndarray,
                       image: np.ndarray, depth: Any = None) -> np.ndarray:
        """Generate a single tween frame.

        Args:
            data: RenderData object with all rendering state
            tween_frame: Tween frame metadata (value, index, etc.)
            prev_image: Previous keyframe image (numpy array, BGR)
            image: Current/next keyframe image (numpy array, BGR)
            depth: Depth map for current frame (optional)

        Returns:
            Generated tween frame as numpy array (BGR format)
        """
        pass

    @abstractmethod
    def supports_mode(self, animation_mode: str) -> bool:
        """Check if this generator supports the given animation mode.

        Args:
            animation_mode: Animation mode string ('2D', '3D', etc.)

        Returns:
            True if this generator can be used with the mode
        """
        pass
