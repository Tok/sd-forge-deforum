"""Audio prompt generation handler for Deforum UI.

Provides AI-powered prompt generation using Qwen with multiple generation modes
and intensity levels for audio-synchronized animations.
"""

from deforum.utils.system.logging import get_logger

logger = get_logger()


def generate_prompts_with_ai(generation_mode, intensity, style, theme, count, start_prompt, end_prompt):
    """Generate prompts using Qwen with multiple modes and intensity levels.

    Args:
        generation_mode: Generation style (escalating, start-to-end, varied, etc.)
        intensity: Intensity level (subtle, normal, crazy, extreme, etc.)
        style: Optional visual style to apply
        theme: Main subject/theme for prompts
        count: Number of prompts to generate
        start_prompt: Starting prompt (for start-to-end mode)
        end_prompt: Ending prompt (for start-to-end mode)

    Returns:
        str: Generated prompts (one per line)
    """
    logger.info("="*80)
    logger.info(f"AI PROMPT GENERATION BUTTON CLICKED!", emoji='palette')
    logger.info(f"   Mode: {generation_mode}")
    logger.info(f"   Intensity: {intensity}")
    logger.info(f"   Style: {style}")
    logger.info(f"   Theme: {theme}")
    logger.info(f"   Count: {count}")
    logger.info("="*80)

    try:
        from deforum.integrations.wan.utils.prompt_extend import QwenPromptExpander

        # Initialize Qwen (will auto-select model based on VRAM)
        qwen = QwenPromptExpander()

        # Build style descriptor
        style_text = f"{style} style " if style and style.strip() else ""

        # Build intensity descriptor
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
        intensity_inst = intensity_instructions.get(
            intensity,
            f"Creative direction: {intensity}" if intensity else intensity_instructions["crazy"]
        )

        # Mode-specific prompt generation
        if generation_mode == "start-to-end":
            # Interpolation mode - fill between start and end
            generation_prompt = f"""You are creating an animated sequence that transitions from one scene to another.

START PROMPT: {start_prompt}
END PROMPT: {end_prompt}

Generate {int(count)} {style_text}prompts that smoothly transition from the start to the end.

INTENSITY: {intensity_inst}

Requirements:
- First prompt should be similar to START
- Last prompt should lead into END
- Middle prompts progressively transform from start to end
- Each step should build on the previous one
- {style_text if style else ""}Focus on visual progression
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Generate {int(count)} transition prompts:"""

        elif generation_mode == "varied":
            # Random creative variations
            generation_prompt = f"""Create {int(count)} wildly varied and creative {style_text}prompts featuring {theme}.

INTENSITY: {intensity_inst}

Requirements:
- Each prompt should be COMPLETELY DIFFERENT
- Mix of scenes, actions, perspectives, moods
- {style_text if style else ""}Unexpected combinations and scenarios
- Progressive escalation of creativity
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Generate {int(count)} varied {style_text}prompts for {theme}:"""

        elif generation_mode == "thematic":
            # Variations on a theme
            generation_prompt = f"""Generate {int(count)} {style_text}prompts that are variations on the theme of {theme}.

INTENSITY: {intensity_inst}

Requirements:
- All prompts should relate to {theme}
- Explore different aspects, angles, perspectives
- {style_text if style else ""}Maintain thematic coherence
- Variations in composition, lighting, action, mood
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Generate {int(count)} thematic {style_text}variations of {theme}:"""

        elif generation_mode == "narrative":
            # Story progression
            generation_prompt = f"""Create {int(count)} {style_text}prompts that tell a story about {theme}.

INTENSITY: {intensity_inst}

Requirements:
- Prompts should form a narrative sequence
- Clear beginning, middle, progression toward resolution
- {style_text if style else ""}Story should be engaging and coherent
- Each prompt advances the plot or reveals character
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Generate {int(count)} narrative {style_text}prompts for the story of {theme}:"""

        elif generation_mode == "cyclical":
            # Repeating patterns / loops
            generation_prompt = f"""Generate {int(count)} {style_text}prompts that form a cyclical, looping pattern featuring {theme}.

INTENSITY: {intensity_inst}

Requirements:
- Prompts should loop back to the beginning
- Last prompt should connect naturally to first prompt
- {style_text if style else ""}Pattern should feel circular/repeating
- Maintain rhythm and flow throughout
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Generate {int(count)} cyclical {style_text}prompts for {theme}:"""

        elif generation_mode == "random-walk":
            # Random but related progressions
            generation_prompt = f"""Create {int(count)} {style_text}prompts that drift randomly but stay conceptually related to {theme}.

INTENSITY: {intensity_inst}

Requirements:
- Each prompt should be somewhat related to the previous
- Allow unexpected connections and associations
- {style_text if style else ""}Maintain loose thematic thread
- Drift naturally like stream of consciousness
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Generate {int(count)} random-walk {style_text}prompts starting from {theme}:"""

        elif generation_mode == "" or not generation_mode:
            # Empty/minimal mode - let Qwen be creative
            generation_prompt = f"""Generate {int(count)} {style_text}prompts featuring {theme}.

{f'INTENSITY: {intensity_inst}' if intensity_inst else ''}

Requirements:
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Generate {int(count)} {style_text}prompts for {theme}:"""

        elif generation_mode in ["escalating"]:  # escalating mode (explicit)
            # Escalating intensity mode
            generation_prompt = f"""Generate {int(count)} {style_text}prompts that build in intensity for an animated sequence featuring {theme}.

INTENSITY: {intensity_inst}

Requirements:
- START CALM: Begin with simple, peaceful scene (e.g., "cute {theme} in nature")
- ESCALATE DRAMATICALLY: Each prompt MORE intense than the last
- {style_text if style else ""}Progressive transformation: calm → active → dynamic → EXTREME → ABSOLUTELY WILD
- Final prompts should be PEAK INSANITY (if crazy/extreme mode)
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Example progression for "{style_text}bunny" (CRAZY mode):
cute bunny sitting peacefully in grass
bunny hopping through vibrant neon forest
{style_text}bunny leaping over glowing obstacles
bunny racing through laser-filled cityscape
EXTREME {style_text}bunny surfing massive energy wave
INSANE {style_text}bunny commanding lightning storm on motorcycle
ABSOLUTELY BONKERS {style_text}bunny transcending reality in cosmic explosion

Now generate {int(count)} {style_text}prompts for {theme}:"""

        else:
            # Custom generation mode - use user's custom text as instruction
            generation_prompt = f"""Generate {int(count)} {style_text}prompts for an animated sequence featuring {theme}.

GENERATION STYLE: {generation_mode}

{f'INTENSITY: {intensity_inst}' if intensity_inst else ''}

Requirements:
- Follow the GENERATION STYLE instruction above
- Keep prompts concise (5-12 words each)
- Return ONLY the prompts, one per line, NO numbering

Generate {int(count)} {style_text}prompts for {theme}:"""

        # Generate with Qwen
        logger.info(f"🤖 Generating {count} prompts | Mode: {generation_mode} | Intensity: {intensity} | Style: {style or 'none'} | Theme: {theme}")

        # Use a simple system prompt and user prompt format
        system_prompt = "You are a creative AI assistant helping generate prompts for animated sequences. Return ONLY the prompts, one per line, with no numbering or extra formatting."

        result = qwen(prompt=generation_prompt, system_prompt=system_prompt, tar_lang="en")

        # Check if generation succeeded first
        if not result.status:
            raise Exception(f"Qwen generation failed: {result.message}")

        # Extract the prompt text from PromptOutput object
        result_text = result.prompt

        # Check if result is just echoing back the input (means generation failed silently)
        if result_text == generation_prompt:
            raise Exception(f"Qwen returned input prompt unchanged - generation failed: {result.message}")

        # Check if we got actual prompt text
        if not result_text or not result_text.strip():
            raise Exception(f"Qwen generation failed: {result.message if hasattr(result, 'message') else 'No prompts generated'}")

        # Clean up the result (remove any numbering or extra formatting)
        lines = [line.strip() for line in result_text.split('\n') if line.strip()]
        prompts = []
        for line in lines:
            # Skip lines with numbering like "1.", "1)", etc.
            clean_line = line
            if len(line) > 0 and line[0].isdigit():
                # Remove leading numbers and punctuation
                import re
                clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
            if clean_line and not clean_line.startswith('#') and not clean_line.startswith('//'):
                prompts.append(clean_line)

        # Take only the requested count
        prompts = prompts[:int(count)]

        # Join with newlines
        prompts_text = '\n'.join(prompts)

        logger.info(f"{emoji_if_enabled('✓')} Generated {len(prompts)} prompts")
        return prompts_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"Error generating prompts: {str(e)}"
        logger.warning(f"⚠️ {error_msg}")
        # Fallback to template-based generation
        style_prefix = f"{style} " if style else ""
        if generation_mode == "start-to-end":
            fallback = '\n'.join([start_prompt] + [f"{style_prefix}{theme} transforming"] * max(0, int(count)-2) + [end_prompt])
        else:
            actions = ["resting peacefully", "moving slowly", "actively exploring", "racing dynamically", "GOING WILD"]
            fallback = '\n'.join([f"{style_prefix}{theme} {action}" for action in actions[:int(count)]])
        return fallback
