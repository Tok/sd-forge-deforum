"""Parameter Randomizer - Curated Chaos Engine

Per Qwen's specs: "Rules that FORCE beautiful failures."

This module ONLY affects Zero-HITL generations triggered by "🔥 SLOP IT! 🔥" button.
Normal Deforum usage remains completely unaffected.

Chaos Features:
- Color palettes with 20% RGB inversion chance
- Style combinations with half-frame effects
- Accidental seed = 0 for nostalgic glitch art
- Deliberate parameter mismatches for slopcore aesthetic
"""

import random
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from deforum.utils.audio_generation import SlopLog


class RenderMode(Enum):
    """Available render modes (Flux + Interpolation excluded - too slow)."""
    CLASSIC_3D = "Classic 3D"
    NEW_3D = "New 3D"
    KEYFRAMES_ONLY = "Keyframes Only"


@dataclass(frozen=True)
class SlopcoreParameters:
    """Complete set of randomly selected parameters for slopcore generation.

    All parameters are curated for maximum chaos while staying within
    technically valid ranges (mostly).
    """
    # Video settings
    fps: int
    resolution: Tuple[int, int]  # (width, height)

    # Render mode
    render_mode: str

    # Generation settings
    steps: int
    cfg_scale: float
    sampler: str

    # Animation intensity
    cadence: int
    strength: float  # For tween frames (depth-warped)
    keyframe_strength: float  # For diffusion keyframes (low = more creativity)

    # Camera movement
    preset_type: str
    preset_radius: float
    preset_height: float
    preset_closed_loop: bool

    # Camera intensity
    translation_range: Tuple[float, float]
    rotation_range: Tuple[float, float]
    zoom_range: Tuple[float, float]

    # Shakify (optional chaos)
    shakify_pattern: str | None
    shakify_intensity: float

    # Depth
    depth_model: str
    midas_weight: float

    # Color/style chaos
    color_palette: List[str]
    rgb_inverted: bool
    style: str
    half_frame_effect: bool

    # Effects
    enable_perspective_flip: bool
    noise_type: str
    color_coherence: str

    # Seed
    seed: int
    seed_is_glitch_art: bool  # True if seed=0 (nostalgic broken frames)


class CuratedChaosEngine:
    """Generates parameters that force beautiful failures.

    Per Qwen: "If it's too polished, it's a failure. If it's too chaotic, it's perfect."

    ISOLATION: This ONLY affects Zero-HITL tab. Normal Deforum remains unchanged.
    """

    # Curated ranges (from Qwen's specs)
    FPS_OPTIONS = [24, 30, 60]
    # 16:9 aspect ratio at 720p (HD ready) - landscape or portrait
    RESOLUTION_OPTIONS = [
        (1280, 720),  # Landscape 16:9
        (720, 1280),  # Portrait 9:16
    ]

    STEPS_OPTIONS = [15, 20, 25]
    CFG_SCALE_OPTIONS = [3.5, 5.0, 7.0]
    SAMPLER_OPTIONS = ['euler', 'euler_a', 'dpmpp_2m']

    CADENCE_OPTIONS = [2, 3, 5, 8]
    # Strength values MUST be multiples of 0.05 (for 20-step resolution: 1/20 = 0.05)
    # Each 0.05 = 1 diffusion step at 20 steps
    STRENGTH_OPTIONS = [0.25, 0.30, 0.35]  # For tween frames (5-7 steps) - let new prompts through
    KEYFRAME_STRENGTH_OPTIONS = [0.10, 0.15, 0.20]  # For diffusion keyframes (2-4 steps) - max creativity

    PRESET_TYPES = ['rotate-around', 'spiral', 'street', 'dashcam', 'bodycam', 'figure-eight']
    PRESET_RADIUS_RANGE = (50, 200)
    PRESET_HEIGHT_RANGE = (-50, 50)

    TRANSLATION_RANGES = [(-50, 50), (-100, 100), (-200, 200)]
    ROTATION_RANGES = [(-5, 5), (-10, 10), (-20, 20)]
    ZOOM_RANGES = [(0.95, 1.05), (0.9, 1.1), (0.8, 1.2)]

    # Shakify patterns - use title case values from get_camera_shake_list()
    # All 3D patterns (excluding 2D ones which don't work well with depth warping)
    SHAKIFY_PATTERNS = [
        None,
        'Investigation',
        'The Closeup',
        'The Wedding',
        'Walk to the Store',
        'HandyCam Run',
        'Out Car Window',
    ]
    SHAKIFY_INTENSITY_RANGE = (0.0, 0.7)

    # Depth model - use UI display name (not internal model name)
    DEPTH_MODEL = 'Depth-Anything-V2-Small'  # Small = fastest, good for zero-HITL chaos
    MIDAS_WEIGHT_OPTIONS = [0.2, 0.3, 0.5]  # Legacy parameter, not used with Depth-Anything V2

    NOISE_TYPES = ['perlin', 'uniform']
    COLOR_COHERENCE_OPTIONS = ['Match Frame 0 LAB', 'Match Frame 0 HSV', 'Video Input']

    # Authentic BLANK BANSHEE 0 gradient (7 shades + pink glitch)
    # Exact colors pipetted from BB0 album cover
    # From deforum/utils/system/logging/themes.py
    SLOPCORE_COLORS = [
        '#5606FF',  # Deep purple-blue (SLOPCORE_1 / album top)
        '#4C21FF',  # Purple-blue (SLOPCORE_2)
        '#413CFF',  # Blue-purple (SLOPCORE_3 / banner start)
        '#3757FF',  # Mid blue (SLOPCORE_4)
        '#2C71FE',  # Blue (SLOPCORE_5)
        '#228CFE',  # Bright blue (SLOPCORE_6)
        '#17A7FE',  # Cyan (SLOPCORE_7 / album bottom, banner end)
        '#FF1493',  # Neon pink - the glitch
    ]

    # Style options - expanded for creative freedom
    # Categories: Digital/Retro, Art Movements, Cinematic, Abstract, Experimental
    STYLES = [
        # Digital/Retro
        'glitch art', 'pixel art', 'VHS', 'low-poly', '8-bit', '16-bit', 'vaporwave',
        'Y2K aesthetics', 'analog video', 'CRT scanlines', 'datamosh', 'circuit bent',

        # Art Movements
        'surrealism', 'impressionism', 'cyberpunk', 'steampunk', 'art nouveau',
        'brutalism', 'maximalism', 'minimalism', 'futurism', 'constructivism',

        # Cinematic/Photography
        'film noir', 'neon noir', 'cinematic', 'bokeh', 'long exposure',
        'drone footage', 'timelapse', 'stop motion', 'tilt-shift', 'infrared',

        # Abstract/Experimental
        'fractal', 'kaleidoscope', 'chromatic aberration', 'double exposure',
        'light painting', 'solarization', 'cross-processing', 'bleach bypass',

        # Modern Digital
        'synthwave', 'outrun', 'cyberdelic', 'hyper-saturated', 'matte painting',
        'concept art', 'digital painting', 'photo-realistic', 'cel-shaded',

        # Weird/Experimental
        'found footage', 'security camera', 'microscopic', 'x-ray', 'thermal',
        'satellite imagery', 'holographic', 'nothing'
    ]

    # Chaos probabilities
    PROB_RGB_INVERT = 0.20
    PROB_STYLE_COMBO = 0.25
    PROB_HALF_FRAME_EFFECT = 0.25  # If style combo
    PROB_SEED_ZERO = 0.10

    def __init__(self, random_seed: int = -1):
        """Initialize chaos engine.

        Args:
            random_seed: Seed for reproducible chaos (-1 = truly random)
        """
        if random_seed != -1:
            random.seed(random_seed)

        self.slop_log: List[SlopLog] = []

    def generate_parameters(self, duration_seconds: float, theme: str) -> SlopcoreParameters:
        """Generate complete parameter set for slopcore generation.

        Args:
            duration_seconds: Video duration
            theme: User-provided theme (or empty for pure chaos)

        Returns:
            Complete slopcore parameters
        """
        self.slop_log = []

        # Basic video settings
        fps = random.choice(self.FPS_OPTIONS)
        resolution = random.choice(self.RESOLUTION_OPTIONS)
        render_mode = RenderMode.NEW_3D.value  # Always use New 3D (redistributed mode)

        # Generation settings (balance speed/quality/chaos)
        steps = random.choice(self.STEPS_OPTIONS)
        cfg_scale = random.choice(self.CFG_SCALE_OPTIONS)
        sampler = random.choice(self.SAMPLER_OPTIONS)

        # Animation intensity
        cadence = random.choice(self.CADENCE_OPTIONS)
        strength = random.choice(self.STRENGTH_OPTIONS)
        keyframe_strength = random.choice(self.KEYFRAME_STRENGTH_OPTIONS)

        # Camera path
        preset_type = random.choice(self.PRESET_TYPES)
        preset_radius = random.uniform(*self.PRESET_RADIUS_RANGE)
        preset_height = random.uniform(*self.PRESET_HEIGHT_RANGE)
        preset_closed_loop = random.choice([True, False])

        # Camera intensity
        translation_range = random.choice(self.TRANSLATION_RANGES)
        rotation_range = random.choice(self.ROTATION_RANGES)
        zoom_range = random.choice(self.ZOOM_RANGES)

        # Shakify (30% chance for jitter, 10% chance for 360° spin - via preset selection)
        shakify_pattern = random.choice(self.SHAKIFY_PATTERNS)
        shakify_intensity = random.uniform(*self.SHAKIFY_INTENSITY_RANGE) if shakify_pattern else 0.0

        # Depth
        depth_model = self.DEPTH_MODEL
        midas_weight = random.choice(self.MIDAS_WEIGHT_OPTIONS)

        # Color palette with potential RGB inversion
        color_palette, rgb_inverted = self._select_color_palette()

        # Style with potential combination + half-frame effect
        style, half_frame_effect = self._select_style()

        # Effects
        enable_perspective_flip = random.choice([True, False])
        noise_type = random.choice(self.NOISE_TYPES)
        color_coherence = random.choice(self.COLOR_COHERENCE_OPTIONS)

        # Seed with potential glitch art mode (seed=0)
        seed, seed_is_glitch_art = self._select_seed()

        return SlopcoreParameters(
            fps=fps,
            resolution=resolution,
            render_mode=render_mode,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            cadence=cadence,
            strength=strength,
            keyframe_strength=keyframe_strength,
            preset_type=preset_type,
            preset_radius=preset_radius,
            preset_height=preset_height,
            preset_closed_loop=preset_closed_loop,
            translation_range=translation_range,
            rotation_range=rotation_range,
            zoom_range=zoom_range,
            shakify_pattern=shakify_pattern,
            shakify_intensity=shakify_intensity,
            depth_model=depth_model,
            midas_weight=midas_weight,
            color_palette=color_palette,
            rgb_inverted=rgb_inverted,
            style=style,
            half_frame_effect=half_frame_effect,
            enable_perspective_flip=enable_perspective_flip,
            noise_type=noise_type,
            color_coherence=color_coherence,
            seed=seed,
            seed_is_glitch_art=seed_is_glitch_art,
        )

    def _select_color_palette(self) -> Tuple[List[str], bool]:
        """Select 2-3 slopcore colors with 20% chance of RGB inversion.

        Returns:
            Tuple of (color_palette, rgb_inverted)
        """
        num_colors = random.randint(2, 3)
        palette = random.sample(self.SLOPCORE_COLORS, num_colors)

        # 20% chance to invert RGB channels
        rgb_inverted = random.random() < self.PROB_RGB_INVERT
        if rgb_inverted:
            self._log_decision(
                "RGB channel inversion (blue→red, red→blue)",
                self.PROB_RGB_INVERT,
                True
            )
        else:
            self._log_decision("RGB channel inversion", self.PROB_RGB_INVERT, False)

        return palette, rgb_inverted

    def _select_style(self) -> Tuple[str, bool]:
        """Select style with 25% chance of combination + half-frame effect.

        Returns:
            Tuple of (style_string, half_frame_effect)
        """
        # 25% chance to combine two styles
        if random.random() < self.PROB_STYLE_COMBO:
            styles = random.sample(self.STYLES, 2)
            style = f"{styles[0]} + {styles[1]}"

            # If combo, 25% chance to apply only to half the frame
            half_frame_effect = random.random() < self.PROB_HALF_FRAME_EFFECT
            if half_frame_effect:
                self._log_decision(
                    f"Style combo with half-frame effect: {style} (only left half)",
                    self.PROB_STYLE_COMBO * self.PROB_HALF_FRAME_EFFECT,
                    True
                )
            else:
                self._log_decision(
                    f"Style combo: {style} (full frame)",
                    self.PROB_STYLE_COMBO,
                    True
                )
        else:
            style = random.choice(self.STYLES)
            half_frame_effect = False
            self._log_decision("Style combo", self.PROB_STYLE_COMBO, False)

        return style, half_frame_effect

    def _select_seed(self) -> Tuple[int, bool]:
        """Select seed with 10% chance of seed=0 (glitch art mode).

        Returns:
            Tuple of (seed, is_glitch_art)
        """
        # 10% chance for seed=0 (nostalgic glitch art - same broken frame every time)
        if random.random() < self.PROB_SEED_ZERO:
            self._log_decision(
                "Seed = 0 (nostalgic glitch art: same broken frame every time)",
                self.PROB_SEED_ZERO,
                True
            )
            return 0, True
        else:
            self._log_decision("Seed = 0 (glitch art)", self.PROB_SEED_ZERO, False)
            # Random seed with slopcore offset
            base_seed = random.randint(0, 2**31 - 1)
            slopcore_offset = random.randint(1, 100)
            return base_seed + slopcore_offset, False

    def _log_decision(self, decision: str, probability: float, triggered: bool):
        """Log an intentional parameter choice.

        Args:
            decision: Description of the decision
            probability: Probability of this decision
            triggered: Whether it actually happened
        """
        self.slop_log.append(SlopLog(decision, probability, triggered))

    def get_slop_log(self) -> List[SlopLog]:
        """Get log of all intentional bad decisions.

        Returns:
            List of SlopLog entries
        """
        return self.slop_log


# Public API
def randomize_parameters(
    duration_seconds: float,
    theme: str = "",
    random_seed: int = -1
) -> Tuple[SlopcoreParameters, List[SlopLog]]:
    """Generate curated chaos parameters for slopcore generation.

    ISOLATION: This ONLY affects Zero-HITL tab. Normal Deforum remains unchanged.

    Args:
        duration_seconds: Target video duration
        theme: Optional theme/vibe (or empty for pure chaos)
        random_seed: Seed for reproducible chaos (-1 = truly random)

    Returns:
        Tuple of (parameters, slop_log)

    Example:
        >>> params, log = randomize_parameters(
        ...     duration_seconds=3.0,
        ...     theme="cyberpunk neon city",
        ...     random_seed=42
        ... )
        >>> print(f"FPS: {params.fps}, Resolution: {params.resolution}")
        >>> print(f"Render mode: {params.render_mode}")
        >>> for entry in log:
        ...     if entry.triggered:
        ...         print(f"✓ {entry.decision}")
    """
    engine = CuratedChaosEngine(random_seed)
    params = engine.generate_parameters(duration_seconds, theme)
    return params, engine.get_slop_log()
