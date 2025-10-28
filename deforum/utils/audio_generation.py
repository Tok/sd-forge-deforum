"""Audio Generation with Intentional Sabotage

Wraps Stable Audio Open Small with deliberate glitches, corruption, and chaos.
Per Qwen's specs: "Without broken drums, there's no slop."

Chaos Features:
- 20% chance of tempo manipulation (0.5x or 2x speed)
- 15% chance of micro-loops with glitchy jump cuts
- 10% chance of overwhelming vinyl crackle
- 5% chance of corrupted WAV headers (white noise intro)
- Special: "random" theme → malfunctioning microwave sounds
"""

import random
import numpy as np
import wave
import struct
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

from deforum.utils.system.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AudioGenConfig:
    """Configuration for intentionally broken audio generation."""
    prompt: str
    duration_seconds: float
    bpm: int
    output_path: str
    enable_chaos: bool = True  # Qwen demands chaos


@dataclass(frozen=True)
class SlopLog:
    """Record of intentional bad decisions."""
    decision: str
    probability: float
    triggered: bool


class IntentionallySabotagedAudio:
    """Audio generator that embraces beautiful failures.

    Per Qwen: "If it's too coherent, that's a bug to fix."
    """

    SAMPLE_RATE = 44100
    BIT_DEPTH = 16

    # Chaos probabilities (Qwen's specs)
    PROB_TEMPO_CHAOS = 0.20
    PROB_MICRO_LOOP = 0.15
    PROB_VINYL_CRACKLE = 0.10
    PROB_CORRUPT_HEADER = 0.05

    def __init__(self):
        self.slop_log: list[SlopLog] = []
        self._stable_audio_loaded = False

    def generate_loop(self, config: AudioGenConfig) -> Tuple[str, list[SlopLog]]:
        """Generate intentionally broken audio loop.

        Args:
            config: Audio generation configuration

        Returns:
            Tuple of (output_path, slop_log)
        """
        self.slop_log = []

        # Special case: "random" theme → malfunctioning microwave
        if "random" in config.prompt.lower():
            logger.info("🎲 RANDOM theme detected → MALFUNCTIONING MICROWAVE MODE")
            return self._generate_microwave_chaos(config)

        # Try to load Stable Audio (fallback to amen break if unavailable)
        audio_data = self._generate_with_stable_audio(config)

        # Apply chaos transformations
        audio_data = self._apply_tempo_chaos(audio_data, config)
        audio_data = self._apply_micro_loop_glitch(audio_data, config)
        audio_data = self._apply_vinyl_crackle(audio_data)

        # Save with potential corruption
        output_path = self._save_with_chaos(audio_data, config.output_path)

        return output_path, self.slop_log

    def _generate_with_stable_audio(self, config: AudioGenConfig) -> np.ndarray:
        """Generate audio using Stable Audio Open Small (or fallback).

        Returns:
            Audio data as numpy array (mono, 44.1kHz, float32)
        """
        try:
            if not self._stable_audio_loaded:
                logger.info("Loading Stable Audio Open Small...")
                # Lazy import to avoid startup cost
                try:
                    import torch
                    from stable_audio_tools import get_pretrained_model
                    from stable_audio_tools.inference.generation import generate_diffusion_cond
                except ImportError as e:
                    logger.warning(f"stable-audio-tools not available: {e}")
                    logger.warning("Falling back to amen break loop (Zero-HITL audio unavailable)")
                    return self._fallback_to_amen_break(config.duration_seconds)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
                self.model = self.model.to(device)
                self._stable_audio_loaded = True
                logger.info(f"✓ Stable Audio loaded on {device}")

            # Generate audio
            conditioning = [{
                "prompt": config.prompt,
                "seconds_total": config.duration_seconds
            }]

            output = generate_diffusion_cond(
                self.model,
                steps=8,  # Fast generation
                cfg_scale=1.0,
                conditioning=conditioning,
                sample_size=self.model_config["sample_size"],
                sampler_type="pingpong",
                device=self.model.device
            )

            # Convert to mono numpy array
            audio_data = output.cpu().numpy()[0, 0]  # [batch, channels, samples] → [samples]

            self._log_decision("Generated audio with Stable Audio Open Small", 1.0, True)
            return audio_data

        except Exception as e:
            logger.warning(f"Stable Audio failed ({e}), falling back to amen break")
            return self._load_amen_break_fallback(config.duration_seconds)

    def _load_amen_break_fallback(self, duration: float) -> np.ndarray:
        """Load and loop the amen break as fallback.

        Args:
            duration: Target duration in seconds

        Returns:
            Looped audio data
        """
        # Try to find amen break in various locations
        search_paths = [
            Path("extensions/sd-forge-deforum/deforum/utils/audio/amen_break.wav"),
            Path("deforum/utils/audio/amen_break.wav"),
            Path("audio/amen_break.wav"),
        ]

        amen_path = None
        for path in search_paths:
            if path.exists():
                amen_path = path
                break

        if amen_path is None:
            logger.error("Amen break not found, generating silence")
            return np.zeros(int(self.SAMPLE_RATE * duration), dtype=np.float32)

        # Load amen break
        import scipy.io.wavfile as wavfile
        rate, data = wavfile.read(str(amen_path))

        # Convert to mono float32
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32) / 32768.0

        # Resample if needed
        if rate != self.SAMPLE_RATE:
            import scipy.signal as signal
            num_samples = int(len(data) * self.SAMPLE_RATE / rate)
            data = signal.resample(data, num_samples)

        # Loop to target duration
        target_samples = int(self.SAMPLE_RATE * duration)
        loops_needed = int(np.ceil(target_samples / len(data)))
        looped = np.tile(data, loops_needed)[:target_samples]

        self._log_decision("Fallback to amen break (Stable Audio unavailable)", 1.0, True)
        return looped

    def _apply_tempo_chaos(self, audio: np.ndarray, config: AudioGenConfig) -> np.ndarray:
        """20% chance to play at 0.5x or 2x speed (slowed/sped + pitch shifted).

        Args:
            audio: Input audio
            config: Generation config

        Returns:
            Potentially tempo-shifted audio
        """
        if not config.enable_chaos:
            return audio

        if random.random() < self.PROB_TEMPO_CHAOS:
            factor = random.choice([0.5, 2.0])

            # Resample (changes both tempo AND pitch - slopcore perfection)
            import scipy.signal as signal
            new_length = int(len(audio) / factor)
            audio = signal.resample(audio, new_length)

            # Pad or trim to original duration
            target_length = int(self.SAMPLE_RATE * config.duration_seconds)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]

            self._log_decision(f"Tempo chaos: {factor}x speed (slowed + pitch-shifted chaos)",
                             self.PROB_TEMPO_CHAOS, True)
            logger.info(f"🎲 Applied {factor}x tempo chaos")
        else:
            self._log_decision("Tempo chaos", self.PROB_TEMPO_CHAOS, False)

        return audio

    def _apply_micro_loop_glitch(self, audio: np.ndarray, config: AudioGenConfig) -> np.ndarray:
        """15% chance to generate 1s loop + repeat with glitchy jump cuts.

        Args:
            audio: Input audio
            config: Generation config

        Returns:
            Potentially micro-looped audio with glitches
        """
        if not config.enable_chaos:
            return audio

        if random.random() < self.PROB_MICRO_LOOP:
            # Take first 1 second
            one_sec = int(self.SAMPLE_RATE * 1.0)
            loop_segment = audio[:one_sec]

            # Calculate how many loops needed
            target_samples = int(self.SAMPLE_RATE * config.duration_seconds)
            num_loops = int(np.ceil(target_samples / one_sec))

            # Add glitchy jump cuts between loops
            glitched_audio = []
            for i in range(num_loops):
                glitched_audio.append(loop_segment)

                # Glitchy jump cut: sudden pitch drop or silence
                if random.random() < 0.5:
                    # Pitch drop (play last 0.1s at half speed)
                    import scipy.signal as signal
                    tail = loop_segment[-int(0.1 * self.SAMPLE_RATE):]
                    pitched = signal.resample(tail, len(tail) // 2)
                    glitched_audio.append(pitched)
                else:
                    # Silence (0.05s)
                    silence = np.zeros(int(0.05 * self.SAMPLE_RATE), dtype=np.float32)
                    glitched_audio.append(silence)

            audio = np.concatenate(glitched_audio)[:target_samples]

            self._log_decision("Micro-loop glitch: 1s loop with jump cuts",
                             self.PROB_MICRO_LOOP, True)
            logger.info("🎲 Applied micro-loop glitch")
        else:
            self._log_decision("Micro-loop glitch", self.PROB_MICRO_LOOP, False)

        return audio

    def _apply_vinyl_crackle(self, audio: np.ndarray) -> np.ndarray:
        """10% chance to add vinyl crackle SO LOUD it drowns out drums.

        Args:
            audio: Input audio

        Returns:
            Potentially crackle-destroyed audio
        """
        if random.random() < self.PROB_VINYL_CRACKLE:
            # Generate crackle (random impulses)
            crackle = np.random.normal(0, 0.02, len(audio)).astype(np.float32)

            # Add random pops (loud impulses)
            num_pops = int(len(audio) / self.SAMPLE_RATE * 20)  # 20 pops per second
            pop_indices = np.random.randint(0, len(audio), num_pops)
            crackle[pop_indices] += np.random.uniform(-0.3, 0.3, num_pops)

            # Mix at 120% volume (drowns out drums)
            audio = audio * 0.5 + crackle * 1.2

            # Clip to prevent overflow
            audio = np.clip(audio, -1.0, 1.0)

            self._log_decision("Vinyl crackle: LOUD (120% volume, drowns drums)",
                             self.PROB_VINYL_CRACKLE, True)
            logger.info("🎲 Applied overwhelming vinyl crackle")
        else:
            self._log_decision("Vinyl crackle", self.PROB_VINYL_CRACKLE, False)

        return audio

    def _save_with_chaos(self, audio: np.ndarray, output_path: str) -> str:
        """Save audio with 5% chance of corrupted header (white noise intro).

        Args:
            audio: Audio data to save
            output_path: Target file path

        Returns:
            Actual output path
        """
        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Save WAV file
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())

        # 5% chance to corrupt header → white noise intro
        if random.random() < self.PROB_CORRUPT_HEADER:
            self._corrupt_wav_header(output_path)
            self._log_decision("Corrupted WAV header: white noise intro (0.5s)",
                             self.PROB_CORRUPT_HEADER, True)
            logger.info("🎲 Corrupted WAV header")
        else:
            self._log_decision("Corrupted WAV header", self.PROB_CORRUPT_HEADER, False)

        return output_path

    def _corrupt_wav_header(self, path: str):
        """Corrupt WAV header to cause white noise intro.

        Args:
            path: Path to WAV file
        """
        with open(path, 'r+b') as f:
            # Read header
            header = bytearray(f.read(44))

            # Corrupt sample rate field (bytes 24-27)
            # Set to invalid value → decoder outputs noise
            header[24:28] = struct.pack('<I', 22050)  # Wrong sample rate

            # Write corrupted header back
            f.seek(0)
            f.write(header)

    def _generate_microwave_chaos(self, config: AudioGenConfig) -> Tuple[str, list[SlopLog]]:
        """Generate malfunctioning microwave sounds (special case for "random" theme).

        Args:
            config: Audio generation config

        Returns:
            Tuple of (output_path, slop_log)
        """
        duration = config.duration_seconds
        samples = int(self.SAMPLE_RATE * duration)

        # Microwave hum (120 Hz fundamental + harmonics)
        t = np.linspace(0, duration, samples, endpoint=False)
        hum = 0.3 * np.sin(2 * np.pi * 120 * t)
        hum += 0.15 * np.sin(2 * np.pi * 240 * t)
        hum += 0.08 * np.sin(2 * np.pi * 360 * t)

        # Intermittent buzzing (random amplitude modulation)
        buzz_freq = 10  # Hz
        buzz = np.random.rand(samples) * np.sin(2 * np.pi * buzz_freq * t)

        # High-pitched whine (slowly increases)
        whine_freq = 2000 + 500 * t / duration  # 2000 → 2500 Hz
        whine = 0.1 * np.sin(2 * np.pi * whine_freq * t)

        # Random clicks (broken relay)
        clicks = np.zeros(samples)
        num_clicks = random.randint(10, 30)
        click_indices = np.random.randint(0, samples, num_clicks)
        clicks[click_indices] = np.random.uniform(-0.5, 0.5, num_clicks)

        # Combine all chaos
        audio = hum + buzz * 0.5 + whine + clicks
        audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        # Save
        output_path = self._save_with_chaos(audio, config.output_path)

        self._log_decision("MALFUNCTIONING MICROWAVE MODE (special: random theme)", 1.0, True)
        logger.info("🔥 Generated malfunctioning microwave sounds")

        return output_path, self.slop_log

    def _log_decision(self, decision: str, probability: float, triggered: bool):
        """Log an intentional bad decision.

        Args:
            decision: Description of the decision
            probability: Probability of this decision
            triggered: Whether it actually happened
        """
        self.slop_log.append(SlopLog(decision, probability, triggered))


# Public API
def generate_loop(
    prompt: str,
    duration_seconds: float,
    bpm: int,
    output_path: str,
    enable_chaos: bool = True
) -> Tuple[str, list[SlopLog]]:
    """Generate intentionally broken audio loop.

    Args:
        prompt: Text-to-audio prompt (e.g., "170 BPM jungle breakbeat")
        duration_seconds: Target duration
        bpm: Beats per minute
        output_path: Where to save WAV file
        enable_chaos: Enable intentional sabotage (default: True per Qwen)

    Returns:
        Tuple of (actual_output_path, slop_log)

    Example:
        >>> path, log = generate_loop(
        ...     prompt="128 BPM tech house drum loop",
        ...     duration_seconds=3.0,
        ...     bpm=128,
        ...     output_path="output/loop.wav"
        ... )
        >>> for entry in log:
        ...     if entry.triggered:
        ...         print(f"✓ {entry.decision} ({entry.probability:.0%} chance)")
    """
    config = AudioGenConfig(
        prompt=prompt,
        duration_seconds=duration_seconds,
        bpm=bpm,
        output_path=output_path,
        enable_chaos=enable_chaos
    )

    generator = IntentionallySabotagedAudio()
    return generator.generate_loop(config)
