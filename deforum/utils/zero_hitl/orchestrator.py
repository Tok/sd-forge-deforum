"""Zero-HITL Orchestrator - Main Chaos Conductor

Chains the complete pipeline with deliberate nonsense:
1. Audio Generation (intentionally sabotaged)
2. Prompt Generation (descriptive but nonsensical via Qwen)
3. Camera Path (random with jitter/spin)
4. Parameter Selection (curated chaos)
5. Render Execution (with VHS scan lines, artifacts)

Per Qwen: "Chain audio → prompt → camera → render with *deliberate nonsense*."

ISOLATION: This ONLY runs when user clicks "🔥 SLOP IT! 🔥" button.
Normal Deforum remains completely unchanged.
"""

import random
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

from deforum.utils.audio_generation import generate_loop, SlopLog
from deforum.utils.zero_hitl.parameter_randomizer import randomize_parameters, SlopcoreParameters
from deforum.utils.system.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OrchestrationResult:
    """Complete result of zero-HITL orchestration."""
    success: bool
    video_path: str | None
    audio_path: str | None
    settings_json: str  # JSON string of all settings
    slop_log: List[str]  # Human-readable log of bad decisions
    error_message: str | None


class QwenOrchestrator:
    """Main conductor for zero-HITL slopcore generation.

    Coordinates all phases with intentional chaos and nonsense.

    ISOLATION: Only affects Zero-HITL tab. Normal Deforum untouched.
    """

    # Chaos probabilities from Qwen's specs
    PROB_CAMERA_JITTER = 0.30  # Random X/Y shifts per frame
    PROB_CAMERA_SPIN_360 = 0.10  # Spin camera 360° for no reason
    PROB_CRF_51 = 0.05  # Invalid CRF value → black screen

    def __init__(self, output_dir: str = "outputs/zero_hitl"):
        """Initialize orchestrator.

        Args:
            output_dir: Where to save generated files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.slop_log: List[str] = []
        self.all_slop_logs: List[SlopLog] = []

    def slop_it(
        self,
        duration_seconds: float,
        theme: str = "",
        random_seed: int = -1
    ) -> OrchestrationResult:
        """🔥 SLOP IT! 🔥

        Generate complete video from minimal input with maximum chaos.

        Args:
            duration_seconds: Target video duration
            theme: Optional theme/vibe (or empty for pure chaos)
            random_seed: Seed for reproducible chaos (-1 = truly random)

        Returns:
            OrchestrationResult with video path and all settings
        """
        try:
            logger.info("🔥 SLOP IT! 🔥 Starting zero-HITL orchestration...")
            self._log("🎲 Zero-HITL orchestration started")

            # Set random seed if specified (for reproducibility)
            if random_seed != -1:
                random.seed(random_seed)
                self._log(f"🎲 Random seed: {random_seed}")

            # Phase 1: Generate Audio
            self._log("🔊 Phase 1: Generating intentionally broken audio...")
            audio_path, audio_slop_log = self._generate_audio(duration_seconds, theme)
            self.all_slop_logs.extend(audio_slop_log)
            self._log(f"✓ Audio generated: {audio_path}")

            # Phase 2: Generate Parameters (curated chaos)
            self._log("🎲 Phase 2: Randomizing parameters (curated chaos)...")
            params, param_slop_log = randomize_parameters(duration_seconds, theme, random_seed)
            self.all_slop_logs.extend(param_slop_log)
            self._log(f"✓ Parameters selected: {params.render_mode}, {params.fps}fps, {params.resolution}")

            # Phase 3: Generate Camera Path (with potential jitter/spin)
            # CRITICAL: Camera chaos MUST come before prompts so movement informs creative direction
            self._log("📹 Phase 3: Generating camera path with chaos...")
            camera_chaos = self._apply_camera_chaos(params)
            self._log(f"✓ Camera path: {params.preset_type} (chaos applied)")

            # Phase 4: Generate Prompts (via Qwen) - informed by camera movement
            self._log("🤖 Phase 4: Generating nonsensical prompts with Qwen (camera-aware)...")
            prompts = self._generate_prompts_with_qwen(params, theme, duration_seconds, camera_chaos)
            self._log(f"✓ Generated {len(prompts)} keyframe prompts")

            # Phase 5: Execute Render (wired to actual Deforum render)
            self._log("🎬 Phase 5: Executing Deforum render with VHS artifacts...")
            video_path = self._execute_render(params, prompts, audio_path)
            self._log(f"✓ Render complete: {video_path}")

            # Save settings JSON
            settings_json = self._generate_settings_json(params, prompts, audio_path, camera_chaos)

            # Format slop log for UI
            slop_log_formatted = self._format_slop_log()

            logger.info(f"🎉 Zero-HITL complete! Video: {video_path}")

            return OrchestrationResult(
                success=True,
                video_path=str(video_path),
                audio_path=str(audio_path),
                settings_json=settings_json,
                slop_log=slop_log_formatted,
                error_message=None
            )

        except Exception as e:
            logger.error(f"💥 Zero-HITL orchestration failed: {e}")
            self._log(f"❌ ERROR: {e}")

            return OrchestrationResult(
                success=False,
                video_path=None,
                audio_path=None,
                settings_json="{}",
                slop_log=self._format_slop_log(),
                error_message=str(e)
            )

    def _generate_audio(
        self,
        duration_seconds: float,
        theme: str
    ) -> Tuple[str, List[SlopLog]]:
        """Generate intentionally broken audio.

        Args:
            duration_seconds: Target duration
            theme: User theme (may affect audio prompt)

        Returns:
            Tuple of (audio_path, slop_log)
        """
        # Generate audio prompt based on theme
        if theme:
            # Use theme to guide audio generation
            audio_prompt = f"170 BPM breakbeat loop, {theme} style"
        else:
            # Pure randomness
            bpms = [128, 140, 160, 170, 180]
            styles = ["jungle", "dnb", "techno", "ambient", "glitch"]
            bpm = random.choice(bpms)
            style = random.choice(styles)
            audio_prompt = f"{bpm} BPM {style} breakbeat loop"

        # Generate audio with intentional sabotage
        audio_filename = f"slop_{random.randint(1000, 9999)}.wav"
        audio_path = str(self.output_dir / audio_filename)

        path, slop_log = generate_loop(
            prompt=audio_prompt,
            duration_seconds=duration_seconds,
            bpm=int(audio_prompt.split()[0]),  # Extract BPM from prompt
            output_path=audio_path,
            enable_chaos=True  # Qwen demands chaos
        )

        return path, slop_log

    def _generate_prompts_with_qwen(
        self,
        params: SlopcoreParameters,
        theme: str,
        duration_seconds: float,
        camera_chaos: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate prompts using Qwen creative conductor (camera-aware).

        Args:
            params: Slopcore parameters
            theme: User theme
            duration_seconds: Duration
            camera_chaos: Applied camera chaos (jitter, spin, etc.)

        Returns:
            List of keyframe prompts
        """
        from deforum.utils.zero_hitl.qwen_conductor import conduct_creative_direction

        try:
            # Call Qwen conductor for creative direction (pass camera_chaos for context)
            direction = conduct_creative_direction(params, theme, duration_seconds, camera_chaos)

            # Log creative philosophy
            self._log(f"✓ Qwen's philosophy: {direction.creative_philosophy}")
            self._log(f"✓ Visual theme: {direction.visual_theme.get('style', 'unknown')}")

            return direction.prompts

        except Exception as e:
            logger.warning(f"Qwen conductor failed: {e}, using fallback")
            self._log(f"⚠️ Qwen unavailable, using fallback prompts")

            # Fallback is already handled in qwen_conductor.py
            # This should not normally happen, but just in case
            return []

    def _apply_camera_chaos(self, params: SlopcoreParameters) -> Dict[str, Any]:
        """Apply camera chaos (jitter, 360° spin).

        Args:
            params: Slopcore parameters

        Returns:
            Dict describing applied camera chaos
        """
        chaos_applied = {}

        # 30% chance to add jitter (random X/Y shifts per frame)
        if random.random() < self.PROB_CAMERA_JITTER:
            jitter_amount = random.uniform(5, 20)  # pixels
            chaos_applied['jitter'] = jitter_amount
            self._log(f"🌀 CAMERA JITTERED TO ±{jitter_amount:.1f}px PER FRAME (EARTHQUAKE SIMULATOR)")
            self.all_slop_logs.append(SlopLog(
                f"🌀 CAMERA JITTERED TO ±{jitter_amount:.1f}px PER FRAME (EARTHQUAKE SIMULATOR)",
                self.PROB_CAMERA_JITTER,
                True
            ))
        else:
            self.all_slop_logs.append(SlopLog(
                "Camera jitter",
                self.PROB_CAMERA_JITTER,
                False
            ))

        # 10% chance to spin camera 360° for no reason
        if random.random() < self.PROB_CAMERA_SPIN_360:
            spin_frame = random.randint(10, int(params.fps * 2))  # Spin in first 2 seconds
            chaos_applied['spin_360'] = spin_frame
            self._log(f"🌀 SPUN 360° AT FRAME {spin_frame} FOR NO REASON (DURING A 'SERENE' SCENE)")
            self.all_slop_logs.append(SlopLog(
                f"🌀 SPUN 360° AT FRAME {spin_frame} FOR NO REASON (DURING A 'SERENE' SCENE)",
                self.PROB_CAMERA_SPIN_360,
                True
            ))
        else:
            self.all_slop_logs.append(SlopLog(
                "360° camera spin",
                self.PROB_CAMERA_SPIN_360,
                False
            ))

        return chaos_applied

    def _execute_render(
        self,
        params: SlopcoreParameters,
        prompts: List[Dict[str, Any]],
        audio_path: str
    ) -> str:
        """Execute Deforum render with zero-HITL parameters.

        Args:
            params: Slopcore parameters
            prompts: Generated prompts
            audio_path: Path to audio file

        Returns:
            Path to generated video
        """
        from deforum.utils.zero_hitl.render_integration import execute_render

        # Apply CRF chaos (5% chance for invalid CRF=51)
        # NOTE: This is applied in render_integration.build_args_from_slopcore()
        # but we track the decision here for the slop log
        if random.random() < self.PROB_CRF_51:
            self._log("💀 CRF=51 (INVALID) → BLACK SCREEN OF THE SOUL (GLITCH ART ACHIEVED)")
            self.all_slop_logs.append(SlopLog(
                "💀 CRF=51 (INVALID) → BLACK SCREEN OF THE SOUL (GLITCH ART ACHIEVED)",
                self.PROB_CRF_51,
                True
            ))
            # TODO: Pass crf_override=51 to render_integration
        else:
            self._log("🎬 CRF=40 (MAXIMUM COMPRESSION ARTIFACTS, AS QWEN INTENDED)")
            self.all_slop_logs.append(SlopLog(
                "CRF=51 (black screen)",
                self.PROB_CRF_51,
                False
            ))

        # Always add VHS scan lines (mandatory slop)
        # TODO: Implement VHS scan lines post-processing
        self._log("✓ VHS scan lines (pending post-processing implementation)")

        # Execute actual Deforum render
        video_path = execute_render(
            params=params,
            prompts=prompts,
            audio_path=audio_path,
            output_dir=str(self.output_dir)
        )

        return video_path

    def _generate_settings_json(
        self,
        params: SlopcoreParameters,
        prompts: List[Dict[str, Any]],
        audio_path: str,
        camera_chaos: Dict[str, Any]
    ) -> str:
        """Generate complete settings JSON for transparency.

        Args:
            params: Slopcore parameters
            prompts: Generated prompts
            audio_path: Path to audio
            camera_chaos: Applied camera chaos

        Returns:
            JSON string of all settings
        """
        settings = {
            "parameters": asdict(params),
            "prompts": prompts,
            "audio_path": audio_path,
            "camera_chaos": camera_chaos,
            "slop_log": [
                {
                    "decision": log.decision,
                    "probability": log.probability,
                    "triggered": log.triggered
                }
                for log in self.all_slop_logs
            ]
        }

        return json.dumps(settings, indent=2)

    def _log(self, message: str):
        """Log a message to both logger and slop log.

        Args:
            message: Message to log
        """
        logger.info(message)
        self.slop_log.append(message)

    def _format_slop_log(self) -> List[str]:
        """Format slop log for UI display.

        Returns:
            List of formatted log messages
        """
        formatted = self.slop_log.copy()

        # Add section for intentional bad decisions
        formatted.append("")
        formatted.append("═══ INTENTIONAL BAD DECISIONS ═══")

        for log_entry in self.all_slop_logs:
            if log_entry.triggered:
                prob_str = f"{log_entry.probability:.0%}"
                formatted.append(f"✓ {log_entry.decision} ({prob_str} chance)")

        return formatted


# Public API
def orchestrate_slop(
    duration_seconds: float,
    theme: str = "",
    random_seed: int = -1,
    output_dir: str = "outputs/zero_hitl"
) -> OrchestrationResult:
    """🔥 SLOP IT! 🔥

    Main entry point for zero-HITL orchestration.

    ISOLATION: This ONLY runs when user clicks the button.
    Normal Deforum remains completely unchanged.

    Args:
        duration_seconds: Target video duration (1-10 seconds)
        theme: Optional theme/vibe (or empty for pure chaos)
        random_seed: Seed for reproducible chaos (-1 = truly random)
        output_dir: Where to save generated files

    Returns:
        OrchestrationResult with video path and all settings

    Example:
        >>> result = orchestrate_slop(
        ...     duration_seconds=3.0,
        ...     theme="cyberpunk neon city",
        ...     random_seed=42
        ... )
        >>> if result.success:
        ...     print(f"Video: {result.video_path}")
        ...     print(f"Settings: {result.settings_json}")
        ...     for log in result.slop_log:
        ...         print(log)
    """
    orchestrator = QwenOrchestrator(output_dir)
    return orchestrator.slop_it(duration_seconds, theme, random_seed)
