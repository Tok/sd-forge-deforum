"""
Movement Pattern Analysis for Motion-Aware FLF2V Prompts

Analyzes Deforum camera movement schedules to generate descriptive
motion prompts for FLF2V interpolation, improving semantic understanding.
"""

from typing import Dict, Tuple
import numpy as np
from deforum.utils.system.logging import get_logger

logger = get_logger()


def analyze_movement_pattern(
    start_frame: int,
    end_frame: int,
    animation_keys
) -> str:
    """
    Analyze Deforum movement schedules between two frames and return motion description.

    Args:
        start_frame: Starting frame index
        end_frame: Ending frame index
        animation_keys: Deforum animation keys with schedule series

    Returns:
        Motion description string (e.g., "forward zoom", "orbit left", "static")

    Examples:
        - translation_z increases → "forward zoom"
        - translation_z decreases → "backward zoom"
        - rotation_3d_y positive → "orbit right"
        - rotation_3d_y negative → "orbit left"
        - translation_x positive → "pan right"
        - translation_x negative → "pan left"
        - translation_y positive → "tilt up"
        - translation_y negative → "tilt down"
    """
    try:
        # Get schedule series
        trans_x = animation_keys.translation_x_series
        trans_y = animation_keys.translation_y_series
        trans_z = animation_keys.translation_z_series
        rot_x = animation_keys.rotation_3d_x_series
        rot_y = animation_keys.rotation_3d_y_series
        rot_z = animation_keys.rotation_3d_z_series

        # Clamp frame indices to valid range
        start_idx = min(start_frame, len(trans_x) - 1)
        end_idx = min(end_frame, len(trans_x) - 1)

        # Calculate deltas
        delta_x = trans_x[end_idx] - trans_x[start_idx]
        delta_y = trans_y[end_idx] - trans_y[start_idx]
        delta_z = trans_z[end_idx] - trans_z[start_idx]
        delta_rot_x = rot_x[end_idx] - rot_x[start_idx]
        delta_rot_y = rot_y[end_idx] - rot_y[start_idx]
        delta_rot_z = rot_z[end_idx] - rot_z[start_idx]

        # Threshold for detecting significant movement (avoid noise)
        threshold_trans = 0.5  # Translation threshold
        threshold_rot = 1.0    # Rotation threshold (degrees)

        movements = []

        # Analyze zoom (translation_z)
        if abs(delta_z) > threshold_trans:
            if delta_z > 0:
                movements.append("forward zoom")
            else:
                movements.append("backward zoom")

        # Analyze horizontal pan (translation_x)
        if abs(delta_x) > threshold_trans:
            if delta_x > 0:
                movements.append("pan right")
            else:
                movements.append("pan left")

        # Analyze vertical pan (translation_y)
        if abs(delta_y) > threshold_trans:
            if delta_y > 0:
                movements.append("tilt up")
            else:
                movements.append("tilt down")

        # Analyze orbital rotation (rotation_3d_y)
        if abs(delta_rot_y) > threshold_rot:
            if delta_rot_y > 0:
                movements.append("orbit right")
            else:
                movements.append("orbit left")

        # Analyze pitch (rotation_3d_x)
        if abs(delta_rot_x) > threshold_rot:
            if delta_rot_x > 0:
                movements.append("look up")
            else:
                movements.append("look down")

        # Analyze roll (rotation_3d_z)
        if abs(delta_rot_z) > threshold_rot:
            if delta_rot_z > 0:
                movements.append("roll clockwise")
            else:
                movements.append("roll counterclockwise")

        # Construct description
        if not movements:
            return "static camera"
        elif len(movements) == 1:
            return f"smooth camera {movements[0]}"
        else:
            # Combine multiple movements
            return f"smooth camera {', '.join(movements[:-1])} and {movements[-1]}"

    except Exception as e:
        logger.warning(f"Error analyzing movement pattern: {e}")
        return "smooth camera movement"


def construct_motion_prompt(
    prev_prompt: str,
    next_prompt: str,
    movement: str,
    prompt_mode: str = "blend"
) -> str:
    """
    Build FLF2V prompt with motion context.

    Args:
        prev_prompt: Starting keyframe prompt
        next_prompt: Ending keyframe prompt
        movement: Motion description from analyze_movement_pattern()
        prompt_mode: Prompt mode ("blend", "last", "first", "none")

    Returns:
        Motion-aware prompt for FLF2V

    Examples:
        prev: "city street daytime"
        next: "highway sunset"
        movement: "forward zoom"
        result: "smooth camera forward zoom transitioning from city street daytime to highway sunset"
    """
    if prompt_mode == "none":
        return ""

    if prompt_mode == "first":
        return f"{movement} through {prev_prompt}"

    if prompt_mode == "last":
        return f"{movement} to {next_prompt}"

    # Default: "blend" mode - describe full transition
    if prev_prompt.strip() == next_prompt.strip():
        # Same prompt - just describe movement through scene
        return f"{movement} through {prev_prompt}"
    else:
        # Different prompts - describe transition
        return f"{movement} transitioning from {prev_prompt} to {next_prompt}"


def adaptive_flf2v_guidance(
    prev_prompt: str,
    next_prompt: str,
    base_guidance: float = 3.5,
    min_guidance: float = 3.0,
    max_guidance: float = 5.5
) -> float:
    """
    Adjust FLF2V guidance scale based on prompt similarity.

    Similar prompts → Lower guidance (smooth morphing priority)
    Different prompts → Higher guidance (stronger prompt control)

    Args:
        prev_prompt: Starting keyframe prompt
        next_prompt: Ending keyframe prompt
        base_guidance: Base guidance scale (fallback)
        min_guidance: Minimum guidance (for very similar prompts)
        max_guidance: Maximum guidance (for very different prompts)

    Returns:
        Adjusted guidance scale

    Note:
        FLF2V guidance scale recommendations:
        - 3.0-3.5: Smooth visual morphing (similar prompts)
        - 4.0-5.0: Balanced (moderate prompt changes)
        - 5.5+: Strong prompt adherence (dramatic changes)
        - Never use 0.0 (breaks last_image conditioning)
    """
    try:
        from deforum.utils.prompt_similarity import calculate_prompt_similarity

        similarity = calculate_prompt_similarity(prev_prompt, next_prompt)

        # Inverse relationship: high similarity → low guidance (smooth morphing)
        # Low similarity → high guidance (stronger control)
        guidance = max_guidance - (max_guidance - min_guidance) * similarity

        logger.debug(
            f"Adaptive FLF2V guidance: similarity={similarity:.3f} → "
            f"guidance={guidance:.2f} (range [{min_guidance:.1f}, {max_guidance:.1f}])"
        )

        return guidance

    except Exception as e:
        logger.warning(f"Error calculating adaptive guidance: {e}")
        return base_guidance


def calculate_movement_magnitude(
    start_frame: int,
    end_frame: int,
    animation_keys
) -> float:
    """
    Calculate overall magnitude of movement between frames (for speed analysis).

    Args:
        start_frame: Starting frame index
        end_frame: Ending frame index
        animation_keys: Deforum animation keys

    Returns:
        Movement magnitude (0.0 = static, higher = more movement)
    """
    try:
        # Get schedule series
        trans_x = animation_keys.translation_x_series
        trans_y = animation_keys.translation_y_series
        trans_z = animation_keys.translation_z_series
        rot_x = animation_keys.rotation_3d_x_series
        rot_y = animation_keys.rotation_3d_y_series
        rot_z = animation_keys.rotation_3d_z_series

        # Clamp frame indices
        start_idx = min(start_frame, len(trans_x) - 1)
        end_idx = min(end_frame, len(trans_x) - 1)

        # Calculate deltas
        delta_x = trans_x[end_idx] - trans_x[start_idx]
        delta_y = trans_y[end_idx] - trans_y[start_idx]
        delta_z = trans_z[end_idx] - trans_z[start_idx]
        delta_rot_x = rot_x[end_idx] - rot_x[start_idx]
        delta_rot_y = rot_y[end_idx] - rot_y[start_idx]
        delta_rot_z = rot_z[end_idx] - rot_z[start_idx]

        # Euclidean distance (weighted combination of translation and rotation)
        translation_mag = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        rotation_mag = np.sqrt(delta_rot_x**2 + delta_rot_y**2 + delta_rot_z**2)

        # Weight rotation less than translation (rotation degrees vs translation units)
        magnitude = translation_mag + (rotation_mag * 0.1)

        return magnitude

    except Exception as e:
        logger.warning(f"Error calculating movement magnitude: {e}")
        return 0.0


def enhance_flf2v_prompts_with_qwen(
    prev_prompt: str,
    next_prompt: str,
    movement_description: str,
    enable_qwen: bool = False
) -> Tuple[str, str]:
    """
    Enhance FLF2V prompts using Qwen AI model with movement context.

    Args:
        prev_prompt: Starting keyframe prompt
        next_prompt: Ending keyframe prompt
        movement_description: Camera movement description from analyze_movement_pattern()
        enable_qwen: Whether to use Qwen enhancement

    Returns:
        Tuple of (enhanced_prev_prompt, enhanced_next_prompt)

    Example:
        Input:
            prev: "city street"
            next: "highway"
            movement: "forward zoom"

        Output:
            prev: "photorealistic city street with detailed buildings, daytime lighting"
            next: "cinematic highway with motion blur, sunset lighting, wide angle view"
    """
    if not enable_qwen:
        return prev_prompt, next_prompt

    try:
        from deforum.integrations.wan.utils.qwen_manager import QwenModelManager

        logger.info(f"{emoji_if_enabled('🤖')} Using Qwen to enhance FLF2V prompts...")

        # Initialize Qwen manager
        qwen_manager = QwenModelManager()

        # Auto-select appropriate model based on VRAM
        selected_model = qwen_manager.auto_select_model()
        logger.info(f"   Selected Qwen model: {selected_model}")

        # Create prompt expander
        prompt_expander = qwen_manager.create_prompt_expander(
            model_name=selected_model,
            auto_download=False  # Don't auto-download during generation (too slow)
        )

        if not prompt_expander:
            logger.warning("Qwen model not available - using original prompts")
            return prev_prompt, next_prompt

        # Enhance both prompts with movement context
        enhanced_prev = prompt_expander.expand_prompt(
            prompt=prev_prompt,
            context=f"Camera movement: {movement_description}. This is the starting frame.",
            style="cinematic, detailed"
        )

        enhanced_next = prompt_expander.expand_prompt(
            prompt=next_prompt,
            context=f"Camera movement: {movement_description}. This is the ending frame.",
            style="cinematic, detailed"
        )

        logger.info(f"{emoji_if_enabled('✅')} Qwen-enhanced FLF2V prompts")
        logger.debug(f"   Start: {prev_prompt} → {enhanced_prev}")
        logger.debug(f"   End: {next_prompt} → {enhanced_next}")

        return enhanced_prev, enhanced_next

    except Exception as e:
        logger.warning(f"Error enhancing FLF2V prompts with Qwen: {e}")
        return prev_prompt, next_prompt
