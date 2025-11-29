"""
Render Mode System - Unified mode selection for Deforum animation workflows.

This module defines the 4 core rendering modes that determine how Deforum generates animations.
Each mode has distinct characteristics in terms of keyframe distribution, strength scheduling,
default FPS, and cadence behavior.
"""

from enum import Enum
from typing import NamedTuple
from deforum.rendering.data.frame.key_frame_distribution import KeyFrameDistribution


class ModeConfig(NamedTuple):
    """Configuration for a specific render mode."""
    display_name: str
    keyframe_distribution: KeyFrameDistribution | None
    uses_dual_strength: bool  # True if mode uses both strength schedules
    default_fps: int
    default_cadence: int
    default_steps: int  # Default sampling steps for this mode
    shows_pseudo_cadence: bool  # True if mode displays calculated pseudo-cadence
    description: str


class RenderMode(Enum):
    """
    Unified render mode system combining animation mode and keyframe distribution.

    Each mode represents a distinct workflow with specific characteristics:
    - Keyframe distribution strategy
    - Strength schedule usage (single or dual)
    - Default FPS and cadence settings
    - UI visibility rules
    """

    CLASSIC_3D = "Classic 3D"
    NEW_3D = "New 3D"
    KEYFRAMES_ONLY = "Keyframes Only"
    FLUX_WAN = "Flux + Interpolation"

    @property
    def config(self) -> ModeConfig:
        """Get the configuration for this render mode."""
        return _MODE_CONFIGS[self]

    @staticmethod
    def default() -> "RenderMode":
        """Return the default render mode (New 3D)."""
        return RenderMode.NEW_3D

    @staticmethod
    def from_string(mode_str: str) -> "RenderMode":
        """
        Convert string to RenderMode enum.

        Args:
            mode_str: Mode name from UI or config

        Returns:
            Corresponding RenderMode enum value
        """
        mode_map = {
            "Classic 3D": RenderMode.CLASSIC_3D,
            "New 3D": RenderMode.NEW_3D,
            "Keyframes Only": RenderMode.KEYFRAMES_ONLY,
            "Flux + Interpolation": RenderMode.FLUX_WAN,
            "Flux/Wan": RenderMode.FLUX_WAN,  # Legacy compatibility
        }
        return mode_map.get(mode_str, RenderMode.default())

    def to_legacy_animation_mode(self) -> str:
        """
        Convert RenderMode to legacy animation_mode string.

        Used for backwards compatibility with existing code that checks animation_mode.

        Returns:
            Legacy animation mode string: '3D', 'Interpolation', or 'Flux + Interpolation'
        """
        if self == RenderMode.FLUX_WAN:
            return "Flux + Interpolation"
        else:
            return "3D"

    def get_keyframe_distribution(self) -> KeyFrameDistribution | None:
        """Get the keyframe distribution strategy for this mode."""
        return self.config.keyframe_distribution

    def should_show_3d_tabs(self) -> bool:
        """Return True if 3D-specific tabs (Depth, Shakify, RAFT, ControlNet) should be visible."""
        return self in [RenderMode.CLASSIC_3D, RenderMode.NEW_3D, RenderMode.KEYFRAMES_ONLY]

    def should_show_wan_tab(self) -> bool:
        """Return True if Wan Models tab should be visible."""
        return self == RenderMode.FLUX_WAN

    def should_show_cadence_slider(self) -> bool:
        """Return True if cadence slider should be interactive (not pseudo-cadence display)."""
        return self in [RenderMode.CLASSIC_3D, RenderMode.NEW_3D]

    def should_show_keyframe_strength(self) -> bool:
        """Return True if keyframe strength slider should be visible."""
        # New 3D uses both, Keyframes Only and Flux + Interpolation use keyframe only
        return self in [RenderMode.NEW_3D, RenderMode.KEYFRAMES_ONLY, RenderMode.FLUX_WAN]

    def should_show_normal_strength(self) -> bool:
        """Return True if normal strength slider should be visible."""
        # Classic 3D uses normal only, New 3D uses both
        return self in [RenderMode.CLASSIC_3D, RenderMode.NEW_3D]


# Mode configurations
_MODE_CONFIGS = {
    RenderMode.CLASSIC_3D: ModeConfig(
        display_name="Classic 3D",
        keyframe_distribution=KeyFrameDistribution.OFF,
        uses_dual_strength=False,
        default_fps=24,
        default_cadence=2,
        default_steps=20,  # Flux Dev standard
        shows_pseudo_cadence=False,
        description=(
            "Traditional Deforum rendering with fixed low cadence. "
            "Generates a diffusion frame every N frames (e.g., cadence=2). "
            "Best for use with RAFT and ControlNet. Slower but most stable. "
            "Uses only normal strength schedule. "
            "20 steps = 0.05 strength resolution."
        )
    ),

    RenderMode.NEW_3D: ModeConfig(
        display_name="New 3D (Default)",
        keyframe_distribution=KeyFrameDistribution.REDISTRIBUTED_CADENCE,
        uses_dual_strength=True,
        default_fps=60,
        default_cadence=2,
        default_steps=20,  # Flux Dev standard
        shows_pseudo_cadence=False,
        description=(
            "Modern keyframe redistribution with dual strength schedules. "
            "EXACT keyframe placement at prompt frame numbers (non-negotiable). "
            "Cadence frames distributed BETWEEN keyframes to approximate desired cadence. "
            "Drops cadence frames if too close to keyframes (avoids back-to-back diffusions). "
            "Keyframes use LOW strength (0.15 default) for dramatic changes (17/20 steps). "
            "Cadence frames use HIGH strength (0.85 default) for stability (3/20 steps). "
            "Balances quality, speed, and stability. Works with RAFT and ControlNet. "
            "Uses both strength schedules: keyframe_strength for keyframes, normal strength for cadence frames. "
            "20 steps = 0.05 strength resolution."
        )
    ),

    RenderMode.KEYFRAMES_ONLY: ModeConfig(
        display_name="Keyframes Only",
        keyframe_distribution=KeyFrameDistribution.KEYFRAMES_ONLY,
        uses_dual_strength=False,
        default_fps=60,
        default_cadence=10,  # Not used, but provides pseudo-cadence hint
        default_steps=20,  # Flux Dev standard
        shows_pseudo_cadence=True,
        description=(
            "Pure keyframe diffusion with depth-based tweening between keyframes. "
            "Only generates diffusions at keyframes defined in prompts. "
            "All frames between keyframes are interpolated using depth warping. "
            "Fastest mode, best for slow movements and pure depth transforms. "
            "Cadence is ignored; pseudo-cadence is calculated and displayed. "
            "Not compatible with RAFT/ControlNet (too many non-diffused frames). "
            "Uses only keyframe strength schedule. "
            "20 steps = 0.05 strength resolution."
        )
    ),

    RenderMode.FLUX_WAN: ModeConfig(
        display_name="Flux + Interpolation",
        keyframe_distribution=None,  # Uses separate Flux/Lumina + Interpolation pipeline
        uses_dual_strength=False,
        default_fps=24,
        default_cadence=10,  # Not used for diffusion, but provides pseudo-cadence hint
        default_steps=20,  # Flux Dev for keyframes (Schnell=4, Dev=20, Lumina=30)
        shows_pseudo_cadence=True,
        description=(
            "Flux/Lumina keyframes + choice of interpolation method (Wan/FILM). "
            "Phase 1: Generate keyframes with Flux or Lumina at prompt boundaries. "
            "Phase 2: Interpolate tweens with selected method (Wan FLF2V / FILM). "
            "Phase 3: Stitch final video. "
            "Models: Flux.1 (best quality), Lumina 2.0 (anime, 1024x1024 native). "
            "Best quality for dramatic changes between keyframes. "
            "Hides 3D-specific controls (RAFT, ControlNet, Shakify, Depth). "
            "Shows Wan Models tab (only needed when using Wan method). "
            "Uses only keyframe strength schedule (for Wan I2V chaining). "
            "IMPORTANT: Strength resolution depends on steps. "
            "Flux Dev (20 steps) = 0.05 resolution. Flux Schnell (4 steps) = 0.25 resolution. "
            "Lumina (30 steps) = 0.033 resolution. "
            "Lower steps make strength harder to tune precisely."
        )
    ),
}


def get_mode_choices() -> list[str]:
    """Get list of mode names for UI dropdown."""
    return [mode.value for mode in RenderMode]


def get_mode_descriptions() -> dict[str, str]:
    """Get dictionary mapping mode names to descriptions for UI tooltips."""
    return {mode.value: mode.config.description for mode in RenderMode}
