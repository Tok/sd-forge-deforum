"""Qwen Conductor - AI Creative Director

Uses Qwen to make ALL creative decisions for zero-HITL generation.

Per Qwen's specs: "Descriptive but nonsensical prompts with 25% contradictions."

ISOLATION: Only affects Zero-HITL tab. Normal Deforum remains unchanged.
"""

import json
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from deforum.utils.zero_hitl.parameter_randomizer import SlopcoreParameters
from deforum.utils.system.logging import get_logger

logger = get_logger()


# Qwen's orchestration prompt (structured output)
QWEN_ORCHESTRATOR_PROMPT = """You are an AI creative director for an automated video generation system.
Your task is to make ALL creative decisions to generate a complete animated video.

USER INPUT:
- Duration: {duration} seconds
- Theme: {theme} (empty = pure randomness)
- Parameters: {params_summary}

YOUR MISSION:
Generate complete creative direction as JSON. Be BOLD and CREATIVE - this is a slopcore generator!

CONSTRAINTS:
- Audio MUST sync to prompts (beat-driven visual changes)
- Camera movement MUST match audio energy/mood
- All parameters already selected - focus on CREATIVE DIRECTION
- Be WILDLY CREATIVE - embrace chaos and contradictions

OUTPUT JSON SCHEMA:
{{
  "visual_theme": {{
    "style": "cyberpunk/nature/abstract/glitch/etc",
    "mood": "energetic/calm/chaotic/dreamy",
    "color_story": "how colors evolve over time"
  }},
  "prompts": [
    {{
      "frame": 0,
      "prompt": "detailed visual description",
      "reasoning": "why this works for the moment"
    }},
    ...{num_keyframes} total keyframes
  ],
  "creative_philosophy": "brief explanation of artistic choices"
}}

CREATIVE GUIDELINES:
- 25% of prompts should have contradictory elements (e.g., "serene beach with tornado")
- Prompts should be descriptive but embrace nonsense
- Visual evolution should have surprising transitions
- Style: {style_param}

RESPOND WITH ONLY THE JSON - NO MARKDOWN, NO EXPLANATIONS OUTSIDE JSON.
"""


@dataclass(frozen=True)
class QwenCreativeDirection:
    """Complete creative direction from Qwen."""
    visual_theme: Dict[str, str]
    prompts: List[Dict[str, Any]]
    creative_philosophy: str


class QwenConductor:
    """AI creative director using Qwen for decision-making.

    Uses existing Qwen infrastructure from qwen_prompt_expander.py.
    """

    def __init__(self):
        """Initialize Qwen conductor."""
        self._qwen_loaded = False
        self._qwen_model = None
        self._qwen_tokenizer = None

    def generate_creative_direction(
        self,
        params: SlopcoreParameters,
        theme: str,
        duration_seconds: float,
        camera_chaos: dict
    ) -> QwenCreativeDirection:
        """Generate complete creative direction using Qwen (camera-aware).

        Args:
            params: Slopcore parameters (already selected)
            theme: User-provided theme
            duration_seconds: Video duration
            camera_chaos: Applied camera chaos (jitter, spin, etc.)

        Returns:
            Complete creative direction
        """
        try:
            # Calculate number of keyframes
            num_keyframes = max(8, min(20, int(duration_seconds * 3)))

            # Load Qwen if not already loaded
            if not self._qwen_loaded:
                self._load_qwen()

            # Generate prompt for Qwen (include camera chaos context)
            params_summary = self._summarize_parameters(params)
            camera_summary = self._summarize_camera_chaos(camera_chaos)
            qwen_prompt = QWEN_ORCHESTRATOR_PROMPT.format(
                duration=duration_seconds,
                theme=theme if theme else "pure randomness",
                params_summary=params_summary,
                num_keyframes=num_keyframes,
                style_param=params.style
            )

            # Append camera chaos context
            if camera_summary:
                qwen_prompt += f"\n\nCAMERA MOVEMENT CONTEXT:\n{camera_summary}\n"
                qwen_prompt += "IMPORTANT: Your prompts MUST reflect this camera movement! If camera spins wildly, describe spinning/vortex visuals. If camera jitters, describe chaotic/shaky scenes.\n"

            # Call Qwen (structured output / JSON mode)
            response = self._call_qwen_structured(qwen_prompt)

            # Parse response
            direction = self._parse_qwen_response(response, num_keyframes, params, duration_seconds)

            logger.info(f"✓ Qwen generated {len(direction.prompts)} keyframe prompts")
            logger.debug(f"Creative philosophy: {direction.creative_philosophy}")

            return direction

        except Exception as e:
            logger.warning(f"Qwen conductor failed ({e}), falling back to placeholder")
            return self._fallback_creative_direction(params, theme, duration_seconds)

    def _load_qwen(self):
        """Load Qwen model (reuse existing infrastructure).

        Uses the same Qwen loading logic from qwen_prompt_expander.py.
        """
        try:
            logger.info("Loading Qwen model for creative direction...")

            # Import Qwen model manager
            from deforum.integrations.wan.utils.qwen_manager import QwenModelManager
            import torch

            # Initialize Qwen manager and auto-select model
            qwen_manager = QwenModelManager()
            selected_model = qwen_manager.auto_select_model(prefer_vl=False)  # Text-only for prompts

            # Get model specs
            model_specs = qwen_manager.MODEL_SPECS[selected_model]
            model_hf_name = model_specs['hf_name']

            logger.info(f"Selected Qwen model: {selected_model} ({model_hf_name})")

            # Load model via transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            cache_dir = str(qwen_manager.models_dir)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model and tokenizer
            self._qwen_tokenizer = AutoTokenizer.from_pretrained(
                model_hf_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )

            self._qwen_model = AutoModelForCausalLM.from_pretrained(
                model_hf_name,
                cache_dir=cache_dir,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )

            if device == "cpu":
                self._qwen_model = self._qwen_model.to(device)

            self._qwen_loaded = True
            logger.info(f"✓ Qwen loaded on {device}")

        except Exception as e:
            logger.error(f"Failed to load Qwen: {e}")
            raise

    def _call_qwen_structured(self, prompt: str) -> str:
        """Call Qwen with structured output (JSON mode).

        Args:
            prompt: Prompt for Qwen

        Returns:
            JSON response string
        """
        import torch

        # Tokenize
        inputs = self._qwen_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self._qwen_model.device)

        # Generate (with JSON-friendly sampling)
        with torch.no_grad():
            outputs = self._qwen_model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=0.9,  # High creativity
                top_p=0.95,
                do_sample=True,
                pad_token_id=self._qwen_tokenizer.eos_token_id
            )

        # Decode
        response = self._qwen_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _parse_qwen_response(
        self,
        response: str,
        num_keyframes: int,
        params: SlopcoreParameters,
        duration_seconds: float
    ) -> QwenCreativeDirection:
        """Parse Qwen's JSON response.

        Args:
            response: Raw Qwen response
            num_keyframes: Expected number of keyframes
            params: Slopcore parameters
            duration_seconds: Duration

        Returns:
            Parsed creative direction
        """
        try:
            # Try to extract JSON from response (may have extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            # Extract fields
            visual_theme = data.get("visual_theme", {})
            prompts = data.get("prompts", [])
            creative_philosophy = data.get("creative_philosophy", "Embrace the chaos!")

            # Validate and fix frame numbers
            total_frames = int(params.fps * duration_seconds)
            prompts = self._fix_frame_numbers(prompts, total_frames, num_keyframes)

            return QwenCreativeDirection(
                visual_theme=visual_theme,
                prompts=prompts,
                creative_philosophy=creative_philosophy
            )

        except Exception as e:
            logger.warning(f"Failed to parse Qwen response: {e}")
            logger.debug(f"Raw response: {response[:500]}...")
            raise

    def _fix_frame_numbers(
        self,
        prompts: List[Dict[str, Any]],
        total_frames: int,
        num_keyframes: int
    ) -> List[Dict[str, Any]]:
        """Fix/validate frame numbers in prompts.

        Args:
            prompts: Prompts from Qwen
            total_frames: Total frames in video
            num_keyframes: Expected number of keyframes

        Returns:
            Prompts with corrected frame numbers
        """
        # If Qwen provided wrong number of prompts, pad or trim
        if len(prompts) < num_keyframes:
            # Duplicate last prompt to reach num_keyframes
            while len(prompts) < num_keyframes:
                prompts.append(prompts[-1].copy())
        elif len(prompts) > num_keyframes:
            # Trim to num_keyframes
            prompts = prompts[:num_keyframes]

        # Redistribute frame numbers evenly
        for i, prompt in enumerate(prompts):
            frame_number = int(i * total_frames / num_keyframes)
            prompt['frame'] = frame_number

        return prompts

    def _summarize_parameters(self, params: SlopcoreParameters) -> str:
        """Summarize parameters for Qwen prompt.

        Args:
            params: Slopcore parameters

        Returns:
            Human-readable summary
        """
        return (
            f"Render: {params.render_mode}, "
            f"{params.fps}fps, {params.resolution[0]}x{params.resolution[1]}, "
            f"Camera: {params.preset_type}, "
            f"Style: {params.style}"
        )

    def _summarize_camera_chaos(self, camera_chaos: dict) -> str:
        """Summarize camera chaos for Qwen prompt.

        Args:
            camera_chaos: Applied camera chaos

        Returns:
            Human-readable summary
        """
        if not camera_chaos:
            return ""

        summary_parts = []
        if 'jitter' in camera_chaos:
            summary_parts.append(f"Camera jitters randomly (±{camera_chaos['jitter']:.1f}px per frame)")
        if 'spin_360' in camera_chaos:
            summary_parts.append(f"Camera spins 360° at frame {camera_chaos['spin_360']} (sudden chaotic rotation)")

        return " | ".join(summary_parts) if summary_parts else ""

    def _fallback_creative_direction(
        self,
        params: SlopcoreParameters,
        theme: str,
        duration_seconds: float
    ) -> QwenCreativeDirection:
        """Fallback to placeholder prompts if Qwen fails.

        Args:
            params: Slopcore parameters
            theme: User theme
            duration_seconds: Duration

        Returns:
            Simple creative direction
        """
        logger.info("Using fallback creative direction (Qwen unavailable)")

        num_keyframes = max(8, min(20, int(duration_seconds * 3)))
        total_frames = int(params.fps * duration_seconds)

        # Base themes
        base_themes = [
            "cyberpunk city with neon jellyfish buildings",
            "underwater dreamscape with floating geometric crystals",
            "glitch art chaos with corrupted reality fragments",
            "pixel art void with cascading data streams",
            "VHS corrupted memories of a future that never was",
            "serene beach with neon glitter tornadoes",
            "abstract dimensions folding into themselves",
            "retro synthwave sunset over impossible architecture"
        ]

        if theme:
            base_prompt = theme
        else:
            base_prompt = random.choice(base_themes)

        # Generate prompts with contradictions
        prompts = []
        for i in range(num_keyframes):
            frame_idx = int(i * total_frames / num_keyframes)

            # 25% contradiction chance
            if random.random() < 0.25:
                contradictions = [
                    "but everything is made of liquid metal",
                    "but the sky is raining geometric shapes",
                    "but gravity works sideways",
                    "but colors are inverted and pulsing",
                    "but time flows backwards in slow motion",
                    "but reality is glitching through multiple dimensions"
                ]
                prompt_text = f"{base_prompt}, {random.choice(contradictions)}"
            else:
                prompt_text = base_prompt

            # Add style
            prompt_text = f"{prompt_text}. Style: {params.style}"

            prompts.append({
                "frame": frame_idx,
                "prompt": prompt_text,
                "reasoning": "Embracing beautiful chaos"
            })

        return QwenCreativeDirection(
            visual_theme={
                "style": params.style,
                "mood": "chaotic slopcore energy",
                "color_story": "Colors pulse and shift unpredictably"
            },
            prompts=prompts,
            creative_philosophy="When Qwen is unavailable, embrace pure randomness and contradiction."
        )


# Public API
def conduct_creative_direction(
    params: SlopcoreParameters,
    theme: str,
    duration_seconds: float,
    camera_chaos: dict = None
) -> QwenCreativeDirection:
    """Generate creative direction using Qwen (camera-aware).

    ISOLATION: Only affects Zero-HITL tab. Normal Deforum unchanged.

    Args:
        params: Slopcore parameters (already selected)
        theme: User-provided theme (or empty for chaos)
        duration_seconds: Video duration
        camera_chaos: Applied camera chaos (jitter, spin, etc.)

    Returns:
        Complete creative direction with prompts

    Example:
        >>> from deforum.utils.zero_hitl import randomize_parameters
        >>> params, _ = randomize_parameters(3.0, "cyberpunk", 42)
        >>> camera_chaos = {'jitter': 10.5, 'spin_360': 45}
        >>> direction = conduct_creative_direction(params, "cyberpunk neon city", 3.0, camera_chaos)
        >>> print(direction.visual_theme)
        >>> for prompt in direction.prompts:
        ...     print(f"Frame {prompt['frame']}: {prompt['prompt']}")
    """
    conductor = QwenConductor()
    return conductor.generate_creative_direction(params, theme, duration_seconds, camera_chaos or {})
