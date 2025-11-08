"""Helper functions for analyze_movement_handler.

Extracted from ui_elements.py to reduce complexity.
"""

from typing import Dict, Tuple, Any
import json
from types import SimpleNamespace
from deforum.utils.system.logging import get_logger

logger = get_logger()


def load_analyze_movement_emojis() -> Dict[str, str]:
    """Load all emoji symbols for movement analysis UI.

    Returns:
        Dict of emoji symbols
    """
    from deforum.utils.system.logging import emoji as emoji_utils
    return {
        'check': emoji_utils.maybe_check(),
        'cross': emoji_utils.maybe_cross(),
        'warning': emoji_utils.maybe_warning(),
        'wrench': emoji_utils.wrench(),
        'memo': emoji_utils.memo(),
        'ruler': emoji_utils.ruler(),
        'target': emoji_utils.target(),
        'distribution': emoji_utils.distribution(),
        'sparkles': emoji_utils.sparkles(),
        'party': emoji_utils.party(),
        'camera': emoji_utils.camera(),
        'abacus': emoji_utils.abacus(),
        'movie_camera': emoji_utils.movie_camera(),
        'chart_increasing': emoji_utils.chart_increasing(),
        'bulb': emoji_utils.bulb(),
    }


def validate_prompts_for_analysis(
    current_prompts: str,
    emojis: Dict[str, str]
) -> Tuple[bool, Dict[str, str] | None, str]:
    """Validate and parse prompts for movement analysis.

    Args:
        current_prompts: JSON string of prompts
        emojis: Dict of emoji symbols

    Returns:
        Tuple of (is_valid, prompts_dict, error_message)
    """
    # Check if prompts exist
    if not current_prompts or current_prompts.strip() == "":
        error_msg = f"""{emojis['cross']} No prompts to analyze!

{emojis['wrench']} **Load prompts first:**
1. {emojis['memo']} Click "Load from Deforum Prompts" or "Load Default Wan Prompts"
2. {emojis['ruler']} Then click "Add Movement Descriptions" again

Movement descriptions will be added to your existing prompts."""
        return False, None, error_msg

    # Parse JSON
    try:
        prompts_dict = json.loads(current_prompts)
        if not prompts_dict:
            error_msg = f"{emojis['cross']} Empty prompts! Load prompts first before analyzing movement."
            return False, None, error_msg
        return True, prompts_dict, ""
    except json.JSONDecodeError:
        error_msg = f"{emojis['cross']} Invalid JSON format! Please fix the prompts format first."
        return False, None, error_msg


def build_anim_args_from_components(
    handler_function: Any,
    emojis: Dict[str, str]
) -> SimpleNamespace:
    """Build anim_args namespace from stored component references.

    Args:
        handler_function: The handler function (to access _movement_components)
        emojis: Dict of emoji symbols

    Returns:
        SimpleNamespace with animation arguments
    """
    from deforum.utils.system.logging import emoji as emoji_utils

    anim_args = SimpleNamespace()

    if hasattr(handler_function, '_movement_components'):
        components = handler_function._movement_components
        try:
            # Get actual schedule strings from Deforum's animation system
            anim_args.translation_x = components.get('translation_x', "0:(0)")
            anim_args.translation_y = components.get('translation_y', "0:(0)")
            anim_args.translation_z = components.get('translation_z', "0:(0)")
            anim_args.rotation_3d_x = components.get('rotation_3d_x', "0:(0)")
            anim_args.rotation_3d_y = components.get('rotation_3d_y', "0:(0)")
            anim_args.rotation_3d_z = components.get('rotation_3d_z', "0:(0)")
            anim_args.zoom = components.get('zoom', "0:(1.0)")
            anim_args.angle = components.get('angle', "0:(0)")
            anim_args.max_frames = int(components.get('max_frames', 100))

            logger.info(f"{emoji_utils.maybe_check()} Using actual Deforum movement schedules from UI")
            logger.info(f"Translation X: {anim_args.translation_x}", emoji='distribution')
            logger.info(f"Translation Z: {anim_args.translation_z}", emoji='distribution')
            logger.info(f"Rotation Y: {anim_args.rotation_3d_y}", emoji='distribution')
            logger.info(f"Zoom: {anim_args.zoom}", emoji='distribution')
            return anim_args
        except Exception as e:
            logger.error(f"{emojis['warning']} Could not access movement schedules: {e}")
    else:
        logger.warning(f"{emoji_utils.maybe_warning()} No stored movement schedule references found")

    # Use static defaults for testing
    anim_args.translation_x = "0:(0)"
    anim_args.translation_y = "0:(0)"
    anim_args.translation_z = "0:(0)"
    anim_args.rotation_3d_x = "0:(0)"
    anim_args.rotation_3d_y = "0:(0)"
    anim_args.rotation_3d_z = "0:(0)"
    anim_args.zoom = "0:(1.0)"
    anim_args.angle = "0:(0)"
    anim_args.max_frames = 120
    return anim_args


def get_camera_shakify_settings(
    anim_args: SimpleNamespace,
    handler_function: Any,
    enable_shakify: bool,
    emojis: Dict[str, str]
) -> None:
    """Get and apply Camera Shakify settings to anim_args.

    Args:
        anim_args: Namespace to update with shakify settings
        handler_function: The handler function (to access _movement_components)
        enable_shakify: Whether shakify is enabled via checkbox
        emojis: Dict of emoji symbols
    """
    from deforum.utils.system.logging import emoji as emoji_utils

    if enable_shakify:
        try:
            # Try to get Camera Shakify settings from stored component references first
            if hasattr(handler_function, '_movement_components'):
                components = handler_function._movement_components
                anim_args.shake_name = components.get('shake_name', "None")
                anim_args.shake_intensity = float(components.get('shake_intensity', 1.0))
                anim_args.shake_speed = float(components.get('shake_speed', 1.0))
                logger.info(f"{emoji_utils.maybe_check()} Using Camera Shakify settings from UI components")
            else:
                # Fallback to reading from DeforumArgs if component references not available
                from deforum.config.args import DeforumArgs
                current_args = DeforumArgs()
                anim_args.shake_name = getattr(current_args, 'shake_name', "None")
                anim_args.shake_intensity = getattr(current_args, 'shake_intensity', 1.0)
                anim_args.shake_speed = getattr(current_args, 'shake_speed', 1.0)
                logger.info(f"{emoji_utils.maybe_check()} Using Camera Shakify settings from DeforumArgs fallback")

            # Camera Shakify is enabled when shake_name is not "None"
            camera_shake_enabled = anim_args.shake_name and anim_args.shake_name != "None"

            if camera_shake_enabled:
                logger.info(f"Camera Shakify ENABLED:", emoji='movie_camera')
                logger.info(f"   Shake Name: {anim_args.shake_name}")
                logger.info(f"   Intensity: {anim_args.shake_intensity}")
                logger.info(f"   Speed: {anim_args.shake_speed}")
            else:
                logger.info(f"{emojis['camera']} Camera Shakify disabled (shake_name: {anim_args.shake_name})")
        except Exception as e:
            logger.error(f"{emojis['warning']} Could not read Camera Shakify settings: {e}")
            # Disable Shakify on error
            anim_args.shake_name = "None"
            anim_args.shake_intensity = 1.0
            anim_args.shake_speed = 1.0
    else:
        # Disable Camera Shakify when checkbox is unchecked
        anim_args.shake_name = "None"
        anim_args.shake_intensity = 1.0
        anim_args.shake_speed = 1.0
        logger.info(f"Camera Shakify manually disabled via UI checkbox", emoji='movie_camera')


def calculate_movement_sensitivity(
    anim_args: SimpleNamespace,
    sensitivity_override: bool,
    manual_sensitivity: float,
    emojis: Dict[str, str]
) -> Tuple[float, str]:
    """Calculate movement sensitivity (auto or manual).

    Args:
        anim_args: Namespace with movement schedules
        sensitivity_override: Whether to use manual sensitivity
        manual_sensitivity: Manual sensitivity value
        emojis: Dict of emoji symbols

    Returns:
        Tuple of (sensitivity, sensitivity_reason)
    """
    if sensitivity_override:
        sensitivity = manual_sensitivity
        sensitivity_reason = f"manual override ({sensitivity:.1f})"
        logger.info(f"{emojis['target']} Using manual sensitivity: {sensitivity}")
        return sensitivity, sensitivity_reason

    # Auto-calculate movement sensitivity from the schedules
    logger.info(f"{emojis['abacus']} Auto-calculating movement sensitivity from Deforum schedules...")

    try:
        from deforum.integrations.wan.utils.movement_analyzer import parse_schedule_string, interpolate_schedule

        # Parse all movement schedules
        x_keyframes = parse_schedule_string(anim_args.translation_x, anim_args.max_frames)
        y_keyframes = parse_schedule_string(anim_args.translation_y, anim_args.max_frames)
        z_keyframes = parse_schedule_string(anim_args.translation_z, anim_args.max_frames)
        zoom_keyframes = parse_schedule_string(anim_args.zoom, anim_args.max_frames)

        # Interpolate to get value ranges
        x_values = interpolate_schedule(x_keyframes, anim_args.max_frames)
        y_values = interpolate_schedule(y_keyframes, anim_args.max_frames)
        z_values = interpolate_schedule(z_keyframes, anim_args.max_frames)
        zoom_values = interpolate_schedule(zoom_keyframes, anim_args.max_frames)

        # Calculate movement ranges
        x_range = max(x_values) - min(x_values) if x_values else 0
        y_range = max(y_values) - min(y_values) if y_values else 0
        z_range = max(z_values) - min(z_values) if z_values else 0
        zoom_range = max(zoom_values) - min(zoom_values) if zoom_values else 0

        # Calculate total movement magnitude
        total_movement = x_range + y_range + z_range + (zoom_range * 50)  # Zoom weighted higher

        # Auto-calculate optimal sensitivity based on movement magnitude
        if total_movement < 5:
            sensitivity, sensitivity_reason = 3.0, "high sensitivity for very subtle movement"
        elif total_movement < 15:
            sensitivity, sensitivity_reason = 2.0, "high sensitivity for subtle movement"
        elif total_movement < 50:
            sensitivity, sensitivity_reason = 1.0, "standard sensitivity for normal movement"
        elif total_movement < 200:
            sensitivity, sensitivity_reason = 0.7, "reduced sensitivity for large movement"
        else:
            sensitivity, sensitivity_reason = 0.5, "low sensitivity for very large movement"

        logger.info(f"Total movement magnitude: {total_movement:.1f}", emoji='distribution')
        logger.info(f"{emojis['target']} Auto-calculated sensitivity: {sensitivity} ({sensitivity_reason})")
        return sensitivity, sensitivity_reason
    except Exception as e:
        logger.error(f"{emojis['warning']} Could not auto-calculate sensitivity: {e}, using default 2.0")
        return 2.0, "default (calculation failed)"


def update_prompts_with_movement(
    prompts_dict: Dict[str, str],
    movement_desc: str,
    average_motion_strength: float
) -> Dict[str, str]:
    """Update prompts with movement descriptions.

    Args:
        prompts_dict: Dict of frame -> prompt
        movement_desc: Movement description text
        average_motion_strength: Average motion strength value

    Returns:
        Dict of updated prompts
    """
    updated_prompts = {}
    for frame, prompt in prompts_dict.items():
        # Clean up existing movement descriptions
        clean_prompt = prompt.replace(", static camera position", "")
        clean_prompt = clean_prompt.replace("static camera position", "")
        clean_prompt = clean_prompt.replace(", camera movement with", ", ")
        if clean_prompt.startswith("camera movement with "):
            clean_prompt = clean_prompt[21:]  # Remove "camera movement with " prefix

        # Remove existing movement descriptions more thoroughly
        clean_prompt = clean_prompt.split('. camera movement:')[0].split('. Camera movement:')[0].strip()

        # Add new movement description
        if movement_desc and average_motion_strength > 0:
            if not clean_prompt.endswith('.'):
                updated_prompts[frame] = f"{clean_prompt}, {movement_desc}"
            else:
                updated_prompts[frame] = f"{clean_prompt.rstrip('.')} {movement_desc}."
        else:
            updated_prompts[frame] = clean_prompt

    return updated_prompts


def build_camera_shakify_status_message(
    anim_args: SimpleNamespace,
    enable_shakify: bool,
    emojis: Dict[str, str]
) -> str:
    """Build Camera Shakify status section for result message.

    Args:
        anim_args: Namespace with shakify settings
        enable_shakify: Whether shakify is enabled via checkbox
        emojis: Dict of emoji symbols

    Returns:
        Formatted shakify status message
    """
    if enable_shakify and hasattr(anim_args, 'shake_name') and anim_args.shake_name != "None":
        return f"""
{emojis['movie_camera']} **Camera Shakify Integration:**
- Pattern: {anim_args.shake_name}
- Intensity: {anim_args.shake_intensity}
- Speed: {anim_args.shake_speed}
- Status: {emojis['check']} Active and applied to movement schedules"""
    elif enable_shakify:
        return f"""
{emojis['movie_camera']} **Camera Shakify Integration:**
- Status: {emojis['warning']} Enabled but no shake pattern selected
- Go to Keyframes → Motion → Shakify tab to configure"""
    else:
        return f"""
{emojis['movie_camera']} **Camera Shakify Integration:**
- Status: {emojis['cross']} Disabled via checkbox
- Enable checkbox above to include shake effects"""


def build_analysis_result_message(
    movement_desc: str,
    average_motion_strength: float,
    sensitivity: float,
    sensitivity_reason: str,
    motion_intensity_schedule: str,
    camera_shakify_status: str,
    updated_prompts: Dict[str, str],
    emojis: Dict[str, str]
) -> str:
    """Build final analysis result message.

    Args:
        movement_desc: Movement description text
        average_motion_strength: Average motion strength value
        sensitivity: Calculated sensitivity value
        sensitivity_reason: Reason for sensitivity choice
        motion_intensity_schedule: Motion intensity schedule string
        camera_shakify_status: Formatted shakify status
        updated_prompts: Dict of updated prompts
        emojis: Dict of emoji symbols

    Returns:
        Formatted result message
    """
    if average_motion_strength > 0:
        return f"""{emojis['check']} Enhanced fine-grained movement analysis complete!

{emojis['target']} **Movement Detection:**
"{movement_desc}"

{emojis['chart_increasing']} **Analysis Details:**
- Motion strength: {average_motion_strength:.3f}
- Sensitivity: {sensitivity} ({sensitivity_reason})
- Detection method: Frame-by-frame analysis with enhanced thresholds
{camera_shakify_status}

{emojis['ruler']} **Motion Intensity Schedule for Wan:**
{motion_intensity_schedule}

{emojis['bulb']} **Copy the schedule above to Wan's Motion Intensity field for synchronized movement effects!**

{emojis['check']} Movement descriptions applied to {len(updated_prompts)} prompts.
Ready for AI enhancement or video generation."""
    else:
        return f"""{emojis['check']} Enhanced movement analysis complete!

{emojis['chart_increasing']} **Analysis Result:**
"{movement_desc}"

{emojis['chart_increasing']} **Analysis Details:**
- Sensitivity: {sensitivity} ({sensitivity_reason})
- Detection method: Frame-by-frame analysis with enhanced thresholds
{camera_shakify_status}

{emojis['camera']} Camera appears to be static based on current movement schedules. To add movement:
1. Go to Keyframes → Motion tab and configure movement schedules
2. Or enable Camera Shakify in the Keyframes → Motion → Shakify tab
3. Then run movement analysis again

{emojis['check']} Analysis complete for {len(updated_prompts)} prompts."""


def build_analysis_error_message(error: Exception, emojis: Dict[str, str]) -> str:
    """Build error message for analysis failure.

    Args:
        error: Exception that occurred
        emojis: Dict of emoji symbols

    Returns:
        Formatted error message
    """
    return f"""{emojis['cross']} Error in enhanced movement analysis: {str(error)}

{emojis['wrench']} **Try this:**
1. Check that Deforum movement schedules are valid (Keyframes → Motion tab)
2. Verify Camera Shakify settings if using shake effects
3. Ensure prompts are in valid JSON format
4. Try disabling Camera Shakify checkbox if issues persist

Contact support if this persists."""
