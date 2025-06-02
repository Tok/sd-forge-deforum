"""
Data Models for Prompt Enhancement System

Immutable data structures following functional programming principles.
All models are frozen dataclasses to ensure immutability.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Protocol
from enum import Enum


class PromptLanguage(Enum):
    """Language enumeration for prompt enhancement"""
    ENGLISH = "english"
    CHINESE = "chinese"


class PromptStyle(Enum):
    """Style enumeration for prompt enhancement"""
    PHOTOREALISTIC = "photorealistic"
    ARTISTIC = "artistic"
    CINEMATIC = "cinematic"
    ANIME = "anime"
    VINTAGE = "vintage"
    FUTURISTIC = "futuristic"


class ModelType(Enum):
    """Model type enumeration"""
    TEXT_ONLY = "text_only"
    VISION_LANGUAGE = "vision_language"


@dataclass(frozen=True)
class ModelSpec:
    """Immutable model specification"""
    name: str
    hf_name: str
    vram_gb: int
    model_type: ModelType
    description: str


@dataclass(frozen=True)
class PromptEnhancementRequest:
    """Immutable prompt enhancement request"""
    prompts: Dict[str, str]
    language: PromptLanguage
    style: Optional[PromptStyle] = None
    theme: Optional[str] = None
    custom_style: Optional[str] = None
    custom_theme: Optional[str] = None
    model_name: Optional[str] = None
    max_length: int = 100
    preserve_original_meaning: bool = True


@dataclass(frozen=True)
class PromptEnhancementResult:
    """Immutable result of prompt enhancement"""
    enhanced_prompts: Dict[str, str]
    original_prompts: Dict[str, str]
    model_used: str
    language: PromptLanguage
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    enhancement_count: int = 0


@dataclass(frozen=True)
class EnhancementConfig:
    """Immutable enhancement configuration"""
    temperature: float = 0.7
    max_tokens: int = 512
    preserve_quotes: bool = True
    preserve_original_terms: bool = True
    add_style_modifiers: bool = True


# Model specifications (immutable configuration)
MODEL_SPECS: Dict[str, ModelSpec] = {
    "QwenVL2.5_7B": ModelSpec(
        name="QwenVL2.5_7B",
        hf_name="Qwen/Qwen2.5-VL-7B-Instruct",
        vram_gb=16,
        model_type=ModelType.VISION_LANGUAGE,
        description="7B Vision-Language model, supports image+text input"
    ),
    "QwenVL2.5_3B": ModelSpec(
        name="QwenVL2.5_3B",
        hf_name="Qwen/Qwen2.5-VL-3B-Instruct",
        vram_gb=8,
        model_type=ModelType.VISION_LANGUAGE,
        description="3B Vision-Language model, supports image+text input"
    ),
    "Qwen2.5_14B": ModelSpec(
        name="Qwen2.5_14B",
        hf_name="Qwen/Qwen2.5-14B-Instruct",
        vram_gb=28,
        model_type=ModelType.TEXT_ONLY,
        description="14B Text-only model, high quality prompt enhancement"
    ),
    "Qwen2.5_7B": ModelSpec(
        name="Qwen2.5_7B",
        hf_name="Qwen/Qwen2.5-7B-Instruct",
        vram_gb=14,
        model_type=ModelType.TEXT_ONLY,
        description="7B Text-only model, good balance of quality and speed"
    ),
    "Qwen2.5_3B": ModelSpec(
        name="Qwen2.5_3B",
        hf_name="Qwen/Qwen2.5-3B-Instruct",
        vram_gb=6,
        model_type=ModelType.TEXT_ONLY,
        description="3B Text-only model, fast and memory-efficient"
    )
}
