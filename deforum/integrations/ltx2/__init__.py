"""LTX-2 Audio-Video AI Integration for Deforum.

LTX-2 is the first truly open audio-video AI model (19B params: 14B video + 5B audio).
Provides high-quality video generation with frame-accurate audio synchronization.

Integration approach: Audio-Guided I2V Chaining
- Generate keyframes with diffusion models (Flux/Z-Image/Lumina)
- Chain LTX-2 I2V segments between keyframes with audio conditioning
- Audio drives motion naturally (perfect for music videos)
"""

from .ltx2_pipeline import LTX2Pipeline

__all__ = ['LTX2Pipeline']
