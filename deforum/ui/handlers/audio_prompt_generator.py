"""Audio prompt generation handler for Deforum UI.

Provides AI-powered prompt generation using Qwen with multiple generation modes
and intensity levels for audio-synchronized animations.
"""

import gradio as gr
from deforum.utils.system.logging import get_logger

logger = get_logger()


# Pure helper functions (complexity ≤ 3 each)

def _get_intensity_instruction(intensity: str) -> str:
    """Get intensity instruction text for prompt generation.

    Args:
        intensity: Intensity level (subtle, normal, crazy, extreme, etc.)

    Returns:
        Instruction text for the given intensity level
    """
    intensity_instructions = {
        "": "",  # Empty - let Qwen decide
        "subtle": "Keep prompts very subtle and minimal. Almost imperceptible changes between frames. Focus on nuance and delicate variations.",
        "normal": "Keep prompts realistic and grounded. Progressive but natural changes. Believable transformations.",
        "crazy": "Make prompts over-the-top and extremely creative! Wild transformations and escalating intensity! Go big with each step!",
        "extreme": "GO ABSOLUTELY BONKERS! Each prompt should be MORE INSANE than the last! Reality-bending, physics-defying, mind-blowing escalation! Maximum chaos and creativity!",
        "chaotic": "Embrace complete chaos and unpredictability! Random, erratic, contradictory elements. No rules, pure creative mayhem!",
        "surreal": "Create dream-like, surreal imagery. Logic-defying, symbolic, metaphorical. Think Salvador Dali meets fever dream."
    }

    # If custom value not in dict, use it directly as instruction
    return intensity_instructions.get(
        intensity,
        f"Creative direction: {intensity}" if intensity else intensity_instructions["crazy"]
    )


# Template builders for each mode (pure functions, complexity ≤ 3 each)

def _template_start_to_end(count_int: int, style_text: str, start_prompt: str,
                          end_prompt: str, intensity_inst: str) -> str:
    """Build start-to-end transition template."""
    return f"""You are creating an animated sequence that transitions from one scene to another.

START PROMPT: {start_prompt}
END PROMPT: {end_prompt}

Generate {count_int} {style_text}prompts that smoothly transition from the start to the end.

INTENSITY: {intensity_inst}

Requirements:
- First prompt should be similar to START
- Last prompt should lead into END
- Middle prompts progressively transform from start to end
- Each step should build on the previous one
- {style_text if style_text else ""}Focus on visual progression
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Generate {count_int} transition prompts:"""


def _template_varied(count_int: int, style_text: str, theme: str,
                    intensity_inst: str, common_reqs: str) -> str:
    """Build varied prompts template."""
    return f"""Create {count_int} wildly varied and creative {style_text}prompts featuring {theme}.

INTENSITY: {intensity_inst}

Requirements:
- Each prompt should be COMPLETELY DIFFERENT
- Mix of scenes, actions, perspectives, moods
- {style_text if style_text else ""}Unexpected combinations and scenarios
- Progressive escalation of creativity
{common_reqs} for {theme}:"""


def _template_thematic(count_int: int, style_text: str, theme: str,
                      intensity_inst: str, common_reqs: str) -> str:
    """Build thematic variations template."""
    return f"""Generate {count_int} {style_text}prompts that are variations on the theme of {theme}.

INTENSITY: {intensity_inst}

Requirements:
- All prompts should relate to {theme}
- Explore different aspects, angles, perspectives
- {style_text if style_text else ""}Maintain thematic coherence
- Variations in composition, lighting, action, mood
{common_reqs[:-9]} thematic {style_text}variations of {theme}:"""


def _template_narrative(count_int: int, style_text: str, theme: str,
                       intensity_inst: str, common_reqs: str) -> str:
    """Build narrative sequence template."""
    return f"""Create {count_int} {style_text}prompts that tell a story about {theme}.

INTENSITY: {intensity_inst}

Requirements:
- Prompts should form a narrative sequence
- Clear beginning, middle, progression toward resolution
- {style_text if style_text else ""}Story should be engaging and coherent
- Each prompt advances the plot or reveals character
{common_reqs} for the story of {theme}:"""


def _template_cyclical(count_int: int, style_text: str, theme: str,
                      intensity_inst: str, common_reqs: str) -> str:
    """Build cyclical looping template."""
    return f"""Generate {count_int} {style_text}prompts that form a cyclical, looping pattern featuring {theme}.

INTENSITY: {intensity_inst}

Requirements:
- Prompts should loop back to the beginning
- Last prompt should connect naturally to first prompt
- {style_text if style_text else ""}Pattern should feel circular/repeating
- Maintain rhythm and flow throughout
{common_reqs} for {theme}:"""


def _template_random_walk(count_int: int, style_text: str, theme: str,
                         intensity_inst: str, common_reqs: str) -> str:
    """Build random walk template."""
    return f"""Create {count_int} {style_text}prompts that drift randomly but stay conceptually related to {theme}.

INTENSITY: {intensity_inst}

Requirements:
- Each prompt should be somewhat related to the previous
- Allow unexpected connections and associations
- {style_text if style_text else ""}Maintain loose thematic thread
- Drift naturally like stream of consciousness
{common_reqs} starting from {theme}:"""


def _template_first_person(count_int: int, style_text: str, theme: str,
                          intensity_inst: str, common_reqs: str) -> str:
    """Build first-person perspective template."""
    return f"""Generate {count_int} {style_text}prompts for a first-person perspective camera movement through {theme}.

INTENSITY: {intensity_inst}

**IMPORTANT - Context Stability Instructions:**
- Background/environment should remain STABLE and CONSISTENT across prompts
- Focus on CAMERA MOVEMENT through a fixed scene, not scene transformation
- Example: "driving through city street" NOT "city transforming around driver"
- Think: POV, dash-cam, body-cam, FPV drone footage
- Maintain spatial coherence - viewer is moving, world is not morphing

Requirements:
- First-person camera perspective throughout
- Progressive movement through stable environment (e.g., {theme})
- {style_text if style_text else ""}Natural camera motion: forward, backward, turning, ascending, descending
- Background details should PERSIST across frames (buildings, landmarks stay put)
- Describe what the camera sees as it moves, not scene changes
- Smooth transitions that maintain spatial continuity
{common_reqs} for {theme}:"""


def _template_escalating(count_int: int, style_text: str, theme: str,
                        intensity_inst: str, common_reqs: str) -> str:
    """Build escalating intensity template."""
    return f"""Generate {count_int} {style_text}prompts that build in intensity for an animated sequence featuring {theme}.

INTENSITY: {intensity_inst}

Requirements:
- START CALM: Begin with simple, peaceful scene (e.g., "cute {theme} in nature")
- ESCALATE DRAMATICALLY: Each prompt MORE intense than the last
- {style_text if style_text else ""}Progressive transformation: calm → active → dynamic → EXTREME → ABSOLUTELY WILD
- Final prompts should be PEAK INSANITY (if crazy/extreme mode)
{common_reqs} for {theme}:"""


def _template_default(count_int: int, style_text: str, theme: str,
                     intensity_inst: str, common_reqs: str) -> str:
    """Build default/empty mode template."""
    intensity_line = f'INTENSITY: {intensity_inst}' if intensity_inst else ''
    return f"""Generate {count_int} {style_text}prompts featuring {theme}.

{intensity_line}

{common_reqs} for {theme}:"""


def _template_custom(count_int: int, style_text: str, theme: str, mode: str,
                    intensity_inst: str, common_reqs: str) -> str:
    """Build custom mode template."""
    intensity_line = f'INTENSITY: {intensity_inst}' if intensity_inst else ''
    return f"""Generate {count_int} {style_text}prompts for an animated sequence featuring {theme}.

GENERATION STYLE: {mode}

{intensity_line}

Requirements:
- Follow the GENERATION STYLE instruction above
{common_reqs} for {theme}:"""


def _build_generation_prompt(
    mode: str,
    count: int,
    style: str,
    theme: str,
    intensity_inst: str,
    start_prompt: str,
    end_prompt: str
) -> str:
    """Build Qwen generation prompt based on mode.

    Args:
        mode: Generation mode
        count: Number of prompts to generate
        style: Visual style (optional)
        theme: Main subject/theme
        intensity_inst: Intensity instruction text
        start_prompt: Start prompt (for start-to-end mode)
        end_prompt: End prompt (for start-to-end mode)

    Returns:
        Complete generation prompt for Qwen
    """
    style_text = f"{style} style " if style and style.strip() else ""
    count_int = int(count)
    common_reqs = f"""Requirements:
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Generate {count_int} {style_text}prompts"""

    # Map modes to template builders
    mode_templates = {
        "start-to-end": lambda: _template_start_to_end(count_int, style_text, start_prompt, end_prompt, intensity_inst),
        "varied": lambda: _template_varied(count_int, style_text, theme, intensity_inst, common_reqs),
        "thematic": lambda: _template_thematic(count_int, style_text, theme, intensity_inst, common_reqs),
        "narrative": lambda: _template_narrative(count_int, style_text, theme, intensity_inst, common_reqs),
        "cyclical": lambda: _template_cyclical(count_int, style_text, theme, intensity_inst, common_reqs),
        "random-walk": lambda: _template_random_walk(count_int, style_text, theme, intensity_inst, common_reqs),
        "first-person-perspective": lambda: _template_first_person(count_int, style_text, theme, intensity_inst, common_reqs),
        "escalating": lambda: _template_escalating(count_int, style_text, theme, intensity_inst, common_reqs),
        "": lambda: _template_default(count_int, style_text, theme, intensity_inst, common_reqs),
    }

    # Return template from map, or custom template for unknown modes
    template_builder = mode_templates.get(mode)
    return template_builder() if template_builder else _template_custom(count_int, style_text, theme, mode, intensity_inst, common_reqs)


def _clean_qwen_output(result_text: str, count: int) -> list[str]:
    """Clean and parse Qwen output into prompts list.

    Args:
        result_text: Raw text output from Qwen
        count: Maximum number of prompts to return

    Returns:
        List of cleaned prompt strings
    """
    import re

    # Split into lines and clean
    lines = [line.strip() for line in result_text.split('\n') if line.strip()]
    prompts = []

    for line in lines:
        # Remove leading numbers and punctuation
        clean_line = line
        if len(line) > 0 and line[0].isdigit():
            clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)

        # Skip comment lines
        if clean_line and not clean_line.startswith('#') and not clean_line.startswith('//'):
            prompts.append(clean_line)

    # Return only requested count
    return prompts[:int(count)]


def _generate_fallback_prompts(
    generation_mode: str,
    style: str,
    theme: str,
    count: int,
    start_prompt: str,
    end_prompt: str
) -> str:
    """Generate fallback prompts when Qwen fails.

    Args:
        generation_mode: Generation mode that was attempted
        style: Visual style (optional)
        theme: Main subject/theme
        count: Number of prompts to generate
        start_prompt: Start prompt (for start-to-end mode)
        end_prompt: End prompt (for start-to-end mode)

    Returns:
        Newline-separated fallback prompts
    """
    style_prefix = f"{style} " if style else ""
    count_int = int(count)

    if generation_mode == "start-to-end":
        middle_count = max(0, count_int - 2)
        prompts = [start_prompt] + [f"{style_prefix}{theme} transforming"] * middle_count + [end_prompt]
        return '\n'.join(prompts)
    else:
        actions = ["resting peacefully", "moving slowly", "actively exploring", "racing dynamically", "GOING WILD"]
        prompts = [f"{style_prefix}{theme} {action}" for action in actions[:count_int]]
        return '\n'.join(prompts)


def generate_prompts_with_ai(generation_mode, intensity, style, theme, count, start_prompt, end_prompt, soundtrack_path=None):
    """Generate prompts using Qwen with multiple modes and intensity levels.

    Args:
        generation_mode: Generation style (escalating, start-to-end, varied, etc.)
        intensity: Intensity level (subtle, normal, crazy, extreme, etc.)
        style: Optional visual style to apply
        theme: Main subject/theme for prompts
        count: Number of prompts to generate (0 = auto-calculate from audio)
        start_prompt: Starting prompt (for start-to-end mode)
        end_prompt: Ending prompt (for start-to-end mode)
        soundtrack_path: Optional audio path for auto-calculation when count=0

    Returns:
        Gradio update with generated prompts (one per line)
    """
    logger.info(f"AI PROMPT GENERATION BUTTON CLICKED!", emoji='palette')

    # Auto-calculate count from audio if set to 0
    if count == 0 or count is None:
        if soundtrack_path:
            logger.info("   Auto-calculating prompt count from audio...", emoji='sound')
            # Call the calculate function directly and extract the value
            try:
                result = calculate_prompt_count_from_audio(soundtrack_path)
                if 'value' in result:
                    count = result['value']
                    logger.info(f"   Auto-calculated count: {count}", emoji='abacus')
                else:
                    count = 5  # Fallback
                    logger.warning("   Failed to auto-calculate, using default count: 5")
            except Exception as e:
                count = 5  # Fallback
                logger.warning(f"   Error auto-calculating count: {str(e)}, using default: 5")
        else:
            count = 5  # Fallback when no audio loaded
            logger.info("   No audio loaded, using default count: 5")

    logger.info(f"   Mode: {generation_mode}, Intensity: {intensity}, Style: {style}, Theme: {theme}, Count: {count}")

    try:
        from deforum.integrations.wan.utils.prompt_extend import QwenPromptExpander

        # Initialize Qwen (will auto-select model based on VRAM)
        qwen = QwenPromptExpander()

        # Build intensity instruction
        intensity_inst = _get_intensity_instruction(intensity)

        # Build generation prompt based on mode
        generation_prompt = _build_generation_prompt(
            generation_mode, count, style, theme, intensity_inst, start_prompt, end_prompt
        )

        # Generate with Qwen
        logger.info(f"Generating {count} AI prompts: {generation_mode}/{intensity} {style or ''} {theme}".strip(), emoji='robot')

        system_prompt = "You are a creative AI assistant helping generate prompts for animated sequences. Return ONLY the prompts, one per line, with no numbering or extra formatting."
        result = qwen(prompt=generation_prompt, system_prompt=system_prompt, tar_lang="en")

        # Validate generation succeeded
        if not result.status:
            raise Exception(f"Qwen generation failed: {result.message}")

        result_text = result.prompt

        # Check for silent failures
        if result_text == generation_prompt:
            raise Exception(f"Qwen returned input prompt unchanged - generation failed: {result.message}")

        if not result_text or not result_text.strip():
            raise Exception(f"Qwen generation failed: {result.message if hasattr(result, 'message') else 'No prompts generated'}")

        # Clean and parse output
        prompts = _clean_qwen_output(result_text, count)
        prompts_text = '\n'.join(prompts)

        from deforum.utils.system.logging import emoji as emoji_utils
        check = emoji_utils.maybe_check()
        logger.info(f"{check} Generated {len(prompts)} prompts")
        return gr.update(value=prompts_text)

    except Exception as e:
        import traceback
        traceback.print_exc()
        from deforum.utils.system.logging import emoji as emoji_utils
        warning = emoji_utils.maybe_warning()
        error_msg = f"Error generating prompts: {str(e)}"
        logger.warning(f"{warning} {error_msg}")

        # Generate fallback prompts
        fallback = _generate_fallback_prompts(generation_mode, style, theme, count, start_prompt, end_prompt)
        return gr.update(value=fallback)


def calculate_prompt_count_from_audio(soundtrack_path: str) -> int:
    """Calculate optimal prompt count based on audio duration and BPM.

    Args:
        soundtrack_path: Path or URL to audio file

    Returns:
        Gradio update with calculated prompt count, or 0 if audio invalid
    """
    if not soundtrack_path or not soundtrack_path.strip():
        from deforum.utils.system.logging import emoji as emoji_utils
        warning = emoji_utils.maybe_warning()
        logger.warning(f"{warning} No audio file loaded - cannot calculate prompt count")
        return gr.update(value=5)  # Return default

    try:
        from deforum.audio.analysis import load_audio_file, detect_beats, get_audio_duration
        from deforum.media.video_audio_utilities import download_audio
        from deforum.utils.audio.sync import calculate_keyframes_per_beat

        # Load audio
        local_path = download_audio(soundtrack_path)
        audio, sr = load_audio_file(local_path, sample_rate=22050)
        duration = get_audio_duration(audio, sr)

        # Detect BPM
        _, bpm = detect_beats(audio, sr)

        # Calculate optimal prompts based on BPM
        # Slow music (60-90 BPM): 1 prompt per beat
        # Medium music (90-140 BPM): 1 prompt per 2 beats
        # Fast music (140+ BPM): 1 prompt per 4 beats
        keyframes_per_beat = calculate_keyframes_per_beat(bpm)
        beats_per_second = bpm / 60.0
        suggested_count = int(duration * beats_per_second * keyframes_per_beat)

        # Cap at reasonable maximum (200 prompts for long tracks)
        suggested_count = min(suggested_count, 200)
        # Ensure minimum of 5
        suggested_count = max(suggested_count, 5)

        from deforum.utils.system.logging import emoji as emoji_utils
        check = emoji_utils.maybe_check()
        logger.info(
            f"{check} Audio analysis: {duration:.1f}s @ {bpm:.1f} BPM → {suggested_count} prompts recommended",
            emoji='sound'
        )

        return gr.update(value=suggested_count)

    except Exception as e:
        import traceback
        traceback.print_exc()
        from deforum.utils.system.logging import emoji as emoji_utils
        warning = emoji_utils.maybe_warning()
        logger.warning(f"{warning} Error analyzing audio: {str(e)}")
        return gr.update(value=5)  # Return default on error
