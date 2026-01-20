"""
LTX-2 Audio Integration - CRITICAL FOR AUDIO SYNC

This module implements Audio-Guided Conditioning (Option C from the plan):
- Uses Deforum's audio track as conditioning for LTX-2
- LTX-2 generates video matched to existing audio
- Discards LTX-2's generated audio (preserves Deforum audio)
- Ensures perfect frame-accurate audio synchronization

⚠️ CRITICAL: "Audio sync is KING" - all implementations must preserve
frame-accurate synchronization with Deforum's audio track.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Tuple
from deforum.utils.system.logging import get_logger

logger = get_logger()


class LTX2AudioIntegration:
    """Handles audio synchronization for LTX-2 generation"""

    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize audio integration.

        Args:
            target_sample_rate: Target sample rate for LTX-2 conditioning (16kHz default)
        """
        self.target_sample_rate = target_sample_rate

    def condition_ltx2_on_deforum_audio(
        self,
        audio_path: str,
        segment_start_sec: float,
        segment_duration_sec: float
    ) -> Optional[np.ndarray]:
        """
        Extract audio segment as conditioning for LTX-2.

        This extracts a precise segment from Deforum's audio track to use as
        conditioning for LTX-2 video generation, ensuring temporal alignment.

        Args:
            audio_path: Path to Deforum's audio file
            segment_start_sec: Start time in seconds
            segment_duration_sec: Duration in seconds

        Returns:
            Audio features for LTX-2 conditioning, or None if audio unavailable

        Example:
            For frames 0-60 at 30fps:
            - segment_start_sec = 0.0
            - segment_duration_sec = 2.0
            - Extracts first 2 seconds of audio

        Supported Formats:
            - WAV (best - lossless, no decoding issues)
            - FLAC (best - lossless, no decoding issues)
            - MP3 (good - lossy but widely supported)
            - OGG (good - open format)
            - M4A/AAC (may require ffmpeg)

        Note: If MP3 decoding fails, consider converting to WAV/FLAC for reliability.
        """
        if not audio_path or not Path(audio_path).exists():
            logger.warning("No audio file available for LTX-2 conditioning")
            return None

        audio_path = Path(audio_path)

        # Check audio format and warn if potentially problematic
        audio_format = audio_path.suffix.lower()
        if audio_format not in ['.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac']:
            logger.warning(f"Unsupported audio format: {audio_format} - LTX-2 conditioning may fail")

        try:
            # Load audio segment directly from source (no pre-cutting needed)
            # librosa handles MP3/WAV/FLAC/OGG automatically via ffmpeg/audioread
            audio, sr = librosa.load(
                str(audio_path),
                offset=segment_start_sec,
                duration=segment_duration_sec,
                sr=self.target_sample_rate,
                mono=True  # LTX-2 expects mono audio for conditioning
            )

            # Verify we got audio data
            if audio is None or len(audio) == 0:
                logger.error(f"Failed to load audio segment (empty data)")
                return None

            logger.debug(
                f"Loaded audio segment: {segment_start_sec:.2f}s to "
                f"{segment_start_sec + segment_duration_sec:.2f}s "
                f"(shape: {audio.shape}, sr: {sr}Hz, format: {audio_format})"
            )

            # Extract features for conditioning
            features = self._extract_audio_features(audio, sr)

            return features

        except Exception as e:
            logger.error(f"Error loading audio segment from {audio_path.name}: {e}")

            # Provide helpful troubleshooting info
            if audio_format == '.mp3':
                logger.info("💡 Tip: If MP3 loading fails, try converting to WAV/FLAC:")
                logger.info(f"   ffmpeg -i {audio_path} -ar 16000 -ac 1 {audio_path.with_suffix('.wav')}")

            return None

    def _extract_audio_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract audio features for LTX-2 conditioning.

        Args:
            audio: Raw audio waveform
            sr: Sample rate

        Returns:
            Audio features (mel spectrogram or similar)
        """
        try:
            # Extract mel spectrogram (common for audio-video models)
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=128,  # Standard mel bands
                fmax=8000    # Max frequency
            )

            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            logger.debug(f"Extracted mel spectrogram: shape={mel_spec_db.shape}")

            return mel_spec_db

        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            # Fallback: return raw audio
            return audio

    def synchronize_ltx2_output(
        self,
        generated_frames: list,
        target_frame_count: int,
        source_fps: int,
        target_fps: int
    ) -> list:
        """
        Resample LTX-2 output to match Deforum FPS exactly.

        LTX-2 generates at 50fps native. This resamples to match Deforum's FPS
        to maintain frame-accurate synchronization.

        Args:
            generated_frames: List of PIL Images from LTX-2 (at source_fps)
            target_frame_count: Exact number of frames needed
            source_fps: LTX-2's native FPS (usually 50)
            target_fps: Deforum's target FPS (e.g., 24, 30, 60)

        Returns:
            Resampled list of frames (exactly target_frame_count length)

        Example:
            LTX-2 generates 100 frames at 50fps (2 seconds)
            Deforum needs 60 frames at 30fps (2 seconds)
            → Resample 100→60 to maintain 2-second duration
        """
        if len(generated_frames) == target_frame_count:
            # Already correct length
            return generated_frames

        logger.info(
            f"Resampling LTX-2 output: {len(generated_frames)} frames @ {source_fps}fps "
            f"→ {target_frame_count} frames @ {target_fps}fps"
        )

        try:
            # Calculate resampling indices
            source_indices = np.linspace(0, len(generated_frames) - 1, target_frame_count)

            # Resample frames (using nearest neighbor for simplicity)
            resampled = []
            for idx in source_indices:
                # Round to nearest frame
                frame_idx = int(round(idx))
                frame_idx = min(frame_idx, len(generated_frames) - 1)  # Clamp
                resampled.append(generated_frames[frame_idx])

            logger.info(f"Resampled to {len(resampled)} frames (target: {target_frame_count})")

            return resampled

        except Exception as e:
            logger.error(f"Error resampling LTX-2 output: {e}")
            # Fallback: truncate or pad
            if len(generated_frames) > target_frame_count:
                return generated_frames[:target_frame_count]
            else:
                # Pad with last frame
                return generated_frames + [generated_frames[-1]] * (target_frame_count - len(generated_frames))

    def verify_audio_sync(
        self,
        frame_count: int,
        fps: float,
        audio_duration_sec: float
    ) -> Tuple[bool, float]:
        """
        Verify that frame count and audio duration are synchronized.

        Args:
            frame_count: Number of frames
            fps: Frames per second
            audio_duration_sec: Audio duration in seconds

        Returns:
            Tuple of (is_synced, error_frames)
            - is_synced: True if error < 1 frame
            - error_frames: Absolute error in frames

        Example:
            60 frames @ 30fps = 2.0 seconds
            Audio duration = 2.0 seconds
            → is_synced=True, error_frames=0.0
        """
        expected_duration = frame_count / fps
        actual_duration = audio_duration_sec

        error_sec = abs(expected_duration - actual_duration)
        error_frames = error_sec * fps

        is_synced = error_frames < 1.0  # Less than 1 frame error

        if not is_synced:
            logger.warning(
                f"Audio sync error: {error_frames:.2f} frames "
                f"(expected {expected_duration:.3f}s, got {actual_duration:.3f}s @ {fps}fps)"
            )
        else:
            logger.debug(f"Audio sync verified: error = {error_frames:.3f} frames")

        return is_synced, error_frames

    def calculate_segment_timing(
        self,
        start_frame: int,
        end_frame: int,
        fps: float
    ) -> Tuple[float, float]:
        """
        Calculate audio segment timing for a frame range.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (inclusive)
            fps: Frames per second

        Returns:
            Tuple of (start_time_sec, duration_sec)

        Example:
            Frames 0-59 @ 30fps:
            - start_time = 0.0 / 30 = 0.0s
            - duration = (59 - 0 + 1) / 30 = 2.0s
        """
        frame_count = end_frame - start_frame + 1
        start_time_sec = start_frame / fps
        duration_sec = frame_count / fps

        return start_time_sec, duration_sec
