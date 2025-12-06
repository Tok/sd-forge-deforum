"""Tween frame generation strategies for Deforum.

This module provides different strategies for generating tween frames between
diffusion keyframes:

- depth_warp: Traditional depth warping (default, existing pipeline)
- da3_multiview: Multi-view geometry using Depth Anything V3
- da3_gaussian: 3D Gaussian Splatting scene rendering (Phase 3)

Each generator implements a common interface for consistency.
"""

from enum import Enum


class TweenGenerationMode(Enum):
    """Tween generation strategies"""
    DEPTH_WARP = "depth_warp"  # Traditional depth warping (default)
    DA3_MULTIVIEW = "da3_multiview"  # Multi-view geometry (Phase 2)
    DA3_GAUSSIAN = "da3_gaussian"  # 3D Gaussian Splatting (Phase 3)

    @classmethod
    def default(cls):
        return cls.DEPTH_WARP


__all__ = [
    'TweenGenerationMode',
]
