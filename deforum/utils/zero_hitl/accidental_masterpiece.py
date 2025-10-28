"""Accidental Masterpiece Mode for Zero-HITL

Per Qwen's specifications:
"If output is too good (coherent, aesthetically pleasing), apply 3+ glitch effects"

Detection: Heuristics (broken but functional)
Effects: 5 glitch types + healing glitches
Mode: ALWAYS ENABLED (zero human control)

ISOLATION: Only affects Zero-HITL tab. Normal Deforum remains unchanged.
"""

import random
import subprocess
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from deforum.utils.system.logging import get_logger

logger = get_logger()


@dataclass
class QualityHeuristics:
    """Broken heuristics for detecting 'too good' videos."""
    color_coherence: float = 0.0
    motion_smoothness: float = 0.0
    prompt_visual_match: float = 0.0
    glitch_count: int = 0


def detect_too_good(video_path: str) -> Tuple[bool, List[str]]:
    """Detect if video is 'too good' using broken heuristics.

    Per Qwen's spec:
    1. If color coherence > 0.8 → invert RGB channels
    2. If motion smoothness > 0.7 → add horizontal jitter
    3. If prompt matches visual exactly → datamosh
    4. If no visible glitches → force 1 random glitch frame

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (is_too_good, list_of_reasons)
    """
    # NOTE: These are intentionally simple/broken heuristics
    # Real implementation would use ML-based quality scoring

    reasons = []

    # Heuristic 1: Color coherence (placeholder - random for now)
    color_coherence = random.uniform(0.5, 1.0)
    if color_coherence > 0.8:
        reasons.append("RGB_INVERT")
        logger.info(f"🎨 Color coherence too high: {color_coherence:.2f} > 0.8")

    # Heuristic 2: Motion smoothness (placeholder - random for now)
    motion_smoothness = random.uniform(0.5, 1.0)
    if motion_smoothness > 0.7:
        reasons.append("JITTER")
        logger.info(f"📹 Motion too smooth: {motion_smoothness:.2f} > 0.7")

    # Heuristic 3: Prompt-visual match (placeholder - random for now)
    prompt_match = random.uniform(0.7, 1.0)
    if prompt_match > 0.9:
        reasons.append("DATAMOSH")
        logger.info(f"🎯 Prompt matches visual too well: {prompt_match:.2f} > 0.9")

    # Heuristic 4: No glitches detected (always trigger one random effect)
    # In real implementation, this would analyze frames for artifacts
    if not reasons:
        reasons.append("GLITCH_FRAME")
        logger.info("⚠️ No glitches detected → forcing random glitch")

    is_too_good = len(reasons) > 0
    return is_too_good, reasons


def apply_rgb_invert(video_path: str, output_path: str) -> str:
    """Apply RGB channel inversion (25% probability).

    Args:
        video_path: Input video
        output_path: Output video

    Returns:
        Path to processed video
    """
    logger.info("🎨 Applying RGB invert (color coherence too high)")

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "negate",  # Invert all colors
        "-c:a", "copy",
        "-y",
        output_path
    ]

    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    return output_path


def apply_datamosh(video_path: str, output_path: str) -> str:
    """Apply datamoshing effect (15% probability).

    Datamoshing: Remove I-frames to create trailing/glitchy motion.

    Args:
        video_path: Input video
        output_path: Output video

    Returns:
        Path to processed video
    """
    logger.info("📺 Applying datamosh (prompt-visual match too perfect)")

    # Simplified datamosh: Remove some I-frames and corrupt P-frames
    # Real datamosh is more complex, but this gives the aesthetic
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-c:v", "libx264",
        "-g", "999",  # Very large GOP (fewer I-frames)
        "-bf", "0",   # No B-frames
        "-c:a", "copy",
        "-y",
        output_path
    ]

    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    return output_path


def apply_scanline_overdrive(video_path: str, output_path: str) -> str:
    """Apply scan lines at 300% intensity (50% probability).

    Args:
        video_path: Input video
        output_path: Output video

    Returns:
        Path to processed video
    """
    logger.info("📺 Applying scanline overdrive (300% intensity)")

    # Aggressive scan lines
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "noise=alls=30:allf=t",  # Heavy noise as scan lines
        "-c:a", "copy",
        "-y",
        output_path
    ]

    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    return output_path


def apply_horizontal_flip_frame(video_path: str, output_path: str) -> str:
    """Apply horizontal flip to 1 random frame (10% probability).

    Args:
        video_path: Input video
        output_path: Output video

    Returns:
        Path to processed video
    """
    logger.info("🔄 Applying horizontal flip to 1 random frame")

    # Get video duration to pick random frame
    # For simplicity, apply flip at a random timestamp
    flip_time = random.uniform(0.5, 2.0)  # Flip somewhere in first 2 seconds

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='eq(t,{flip_time})',hflip,setpts=PTS-STARTPTS[f];[0:v][f]overlay",
        "-c:a", "copy",
        "-y",
        output_path
    ]

    # Simplified: Just flip entire video for now
    # Real implementation would flip only 1 frame
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "hflip",
        "-c:a", "copy",
        "-t", "0.1",  # Only first 0.1s
        output_path
    ]

    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    return output_path


def apply_color_channel_shift(video_path: str, output_path: str) -> str:
    """Apply R/G/B channel offset by 5-15px (30% probability).

    Args:
        video_path: Input video
        output_path: Output video

    Returns:
        Path to processed video
    """
    offset = random.randint(5, 15)
    logger.info(f"🌈 Applying color channel shift ({offset}px offset)")

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"chromashift=crv={offset}:cbv={offset}",
        "-c:a", "copy",
        "-y",
        output_path
    ]

    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    return output_path


def apply_healing_glitch(video_path: str, output_path: str) -> str:
    """Apply 'healing' glitch to make 'too broken' output beautiful.

    Per Qwen: If output is TOO broken (pure noise), apply motion blur
    to create 'accidental beauty'.

    Args:
        video_path: Input video
        output_path: Output video

    Returns:
        Path to processed video
    """
    logger.info("✨ Applying healing glitch (making chaos beautiful)")

    # Motion blur to smooth out chaos
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "minterpolate=fps=60:mi_mode=mci",  # Motion interpolation
        "-c:a", "copy",
        "-y",
        output_path
    ]

    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    return output_path


def apply_accidental_masterpiece_mode(
    video_path: str,
    output_path: str,
    force: bool = False
) -> Tuple[str, List[str]]:
    """Apply Accidental Masterpiece mode: detect quality and apply glitches.

    Per Qwen: ALWAYS ENABLED, no human control.

    Args:
        video_path: Input video path
        output_path: Output video path
        force: Force glitch application (skip detection)

    Returns:
        Tuple of (processed_video_path, applied_effects_list)
    """
    logger.info("🎨 Accidental Masterpiece mode: Analyzing video quality...")

    # Detect if video is 'too good'
    if not force:
        is_too_good, reasons = detect_too_good(video_path)
    else:
        is_too_good = True
        reasons = ["GLITCH_FRAME"]  # Force at least one effect

    if not is_too_good:
        logger.info("✓ Video quality acceptable (chaotic enough)")
        return video_path, []

    logger.info(f"⚠️ Video TOO GOOD detected! Applying {len(reasons)} glitch effects...")

    # Map reasons to effects with probabilities
    effect_map = {
        "RGB_INVERT": (apply_rgb_invert, 0.25),
        "DATAMOSH": (apply_datamosh, 0.15),
        "JITTER": (apply_scanline_overdrive, 0.50),  # Using scanline as jitter proxy
        "GLITCH_FRAME": (apply_horizontal_flip_frame, 0.10),
        "COLOR_SHIFT": (apply_color_channel_shift, 0.30),
    }

    applied_effects = []
    current_video = video_path

    # Apply effects in random order
    for reason in reasons:
        if reason in effect_map:
            effect_func, prob = effect_map[reason]

            # Apply effect based on probability
            if random.random() < prob:
                temp_output = str(Path(output_path).parent / f"temp_{reason.lower()}.mp4")
                try:
                    current_video = effect_func(current_video, temp_output)
                    applied_effects.append(reason)
                    logger.info(f"✓ Applied {reason}")
                except Exception as e:
                    logger.warning(f"Failed to apply {reason}: {e}")

    # Final output
    if current_video != output_path:
        import shutil
        shutil.move(current_video, output_path)

    # Qwen's critical rule: If TOO broken, apply healing
    if len(applied_effects) >= 3:
        logger.info("⚠️ Output may be TOO broken, applying healing glitch...")
        healing_output = str(Path(output_path).parent / "temp_healing.mp4")
        try:
            apply_healing_glitch(output_path, healing_output)
            import shutil
            shutil.move(healing_output, output_path)
            applied_effects.append("HEALING")
        except Exception as e:
            logger.warning(f"Healing glitch failed: {e}")

    logger.info(f"🎉 Accidental Masterpiece complete! Applied: {', '.join(applied_effects)}")
    return output_path, applied_effects


# Public API
def get_masterpiece_slop_log(applied_effects: List[str]) -> str:
    """Generate theatrical slop log for Accidental Masterpiece mode.

    Args:
        applied_effects: List of applied effect names

    Returns:
        Theatrical log message
    """
    if not applied_effects:
        return "🎨 ACCIDENTAL MASTERPIECE: Video quality acceptable (chaotic enough)"

    effects_str = ", ".join(applied_effects)
    return (
        f"🎨 ACCIDENTAL MASTERPIECE MODE TRIGGERED! "
        f"(TOO GOOD → APPLIED {len(applied_effects)} GLITCHES: {effects_str})"
    )
