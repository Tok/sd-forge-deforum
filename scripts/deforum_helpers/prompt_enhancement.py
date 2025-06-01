"""
Functional prompt enhancement system for Deforum.

This module provides pure functions for enhancing prompts using AI models,
following functional programming principles with immutable data structures
and side effect isolation.

Key principles:
- Pure functions with no side effects
- Immutable data structures
- Functional composition using map, filter, reduce
- Small, well-named functions
- Isolated side effects (model inference, I/O)
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union, Protocol
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


# Protocol for external model interfaces (dependency injection)
class ModelInferenceService(Protocol):
    """Protocol for model inference services"""
    
    def enhance_prompt(self, prompt: str, system_prompt: str, config: EnhancementConfig) -> Tuple[bool, str, str]:
        """
        Pure interface for prompt enhancement
        Returns: (success, enhanced_prompt, error_message)
        """
        ...
    
    def is_available(self, model_name: str) -> bool:
        """Check if model is available for inference"""
        ...


# Pure functions for prompt processing
def normalize_language(language: Union[str, PromptLanguage]) -> PromptLanguage:
    """Pure function: language input -> normalized PromptLanguage enum"""
    if isinstance(language, PromptLanguage):
        return language
    
    language_lower = str(language).lower()
    if language_lower in ("english", "en", "eng"):
        return PromptLanguage.ENGLISH
    elif language_lower in ("chinese", "zh", "chn"):
        return PromptLanguage.CHINESE
    else:
        return PromptLanguage.ENGLISH  # Default fallback


def normalize_style(style: Optional[Union[str, PromptStyle]]) -> Optional[PromptStyle]:
    """Pure function: style input -> normalized PromptStyle enum"""
    if style is None:
        return None
    if isinstance(style, PromptStyle):
        return style
    
    style_lower = str(style).lower()
    style_mapping = {
        "photorealistic": PromptStyle.PHOTOREALISTIC,
        "photo": PromptStyle.PHOTOREALISTIC,
        "realistic": PromptStyle.PHOTOREALISTIC,
        "artistic": PromptStyle.ARTISTIC,
        "art": PromptStyle.ARTISTIC,
        "cinematic": PromptStyle.CINEMATIC,
        "cinema": PromptStyle.CINEMATIC,
        "movie": PromptStyle.CINEMATIC,
        "anime": PromptStyle.ANIME,
        "animation": PromptStyle.ANIME,
        "vintage": PromptStyle.VINTAGE,
        "retro": PromptStyle.VINTAGE,
        "futuristic": PromptStyle.FUTURISTIC,
        "sci-fi": PromptStyle.FUTURISTIC,
        "future": PromptStyle.FUTURISTIC,
    }
    
    return style_mapping.get(style_lower)


def validate_prompts_dict(prompts: Any) -> Dict[str, str]:
    """Pure function: any input -> validated prompts dictionary"""
    if not prompts:
        return {}
    
    if isinstance(prompts, str):
        try:
            prompts = json.loads(prompts)
        except json.JSONDecodeError:
            return {}
    
    if not isinstance(prompts, dict):
        return {}
    
    # Validate and clean prompts
    validated = {}
    for key, value in prompts.items():
        if value and isinstance(value, str) and value.strip():
            validated[str(key)] = value.strip()
    
    return validated


def build_style_theme_modifier(style: Optional[PromptStyle], 
                             theme: Optional[str],
                             custom_style: Optional[str], 
                             custom_theme: Optional[str]) -> str:
    """Pure function: style/theme inputs -> style modifier string"""
    modifiers = []
    
    # Add style modifiers
    if style:
        style_modifiers = {
            PromptStyle.PHOTOREALISTIC: "photorealistic, high quality, detailed",
            PromptStyle.ARTISTIC: "artistic, expressive, creative composition",
            PromptStyle.CINEMATIC: "cinematic lighting, dramatic composition, film-like",
            PromptStyle.ANIME: "anime style, vibrant colors, stylized",
            PromptStyle.VINTAGE: "vintage aesthetic, retro styling, classic look",
            PromptStyle.FUTURISTIC: "futuristic design, sci-fi elements, modern technology"
        }
        if style in style_modifiers:
            modifiers.append(style_modifiers[style])
    
    # Add custom style
    if custom_style and custom_style.strip():
        modifiers.append(custom_style.strip())
    
    # Add theme
    if theme and theme.strip():
        modifiers.append(f"with {theme.strip()} theme")
    
    # Add custom theme
    if custom_theme and custom_theme.strip():
        modifiers.append(custom_theme.strip())
    
    return ", ".join(modifiers)


def get_system_prompt(language: PromptLanguage, model_type: ModelType) -> str:
    """Pure function: language + model type -> appropriate system prompt"""
    # Simplified system prompts for our functional implementation
    if language == PromptLanguage.CHINESE:
        if model_type == ModelType.VISION_LANGUAGE:
            return """你是一位Prompt优化师，旨在将用户输入改写为优质Prompt，使其更完整、更具表现力，同时不改变原意。
任务要求：
1. 对于过于简短的用户输入，在不改变原意前提下，合理推断并补充细节；
2. 完善用户描述中出现的主体特征、画面风格、空间关系、镜头景别；
3. 整体中文输出，保留引号中原文以及重要信息；
4. 强调输入中的运动信息和镜头运镜；
5. 改写后的prompt字数控制在80-100字左右；
请直接输出改写后的文本。"""
        else:
            return """你是一位Prompt优化师，旨在将用户输入改写为优质Prompt，使其更完整、更具表现力，同时不改变原意。
任务要求：
1. 对于过于简短的用户输入，在不改变原意前提下，合理推断并补充细节；
2. 完善用户描述中出现的主体特征、画面风格、空间关系、镜头景别；
3. 整体中文输出，保留引号中原文以及重要信息；
4. 强调输入中的运动信息和镜头运镜；
5. 改写后的prompt字数控制在80-100字左右；
请直接输出改写后的文本。"""
    else:  # English
        if model_type == ModelType.VISION_LANGUAGE:
            return """You are a prompt engineer, aiming to rewrite user inputs into high-quality prompts for better video generation without affecting the original meaning.
Task requirements:
1. For overly concise user inputs, reasonably infer and add details to make the video more complete and appealing;
2. Enhance the main features in user descriptions (appearance, expression, posture, etc.), visual style, spatial relationships, and shot scales;
3. Output the entire prompt in English, retaining original text in quotes and preserving key input information;
4. Emphasize motion information and different camera movements present in the input description;
5. The revised prompt should be around 80-100 words long;
Please directly output the rewritten English text."""
        else:
            return """You are a prompt engineer, aiming to rewrite user inputs into high-quality prompts for better video generation without affecting the original meaning.
Task requirements:
1. For overly concise user inputs, reasonably infer and add details to make the video more complete and appealing;
2. Enhance the main features in user descriptions (appearance, expression, posture, etc.), visual style, spatial relationships, and shot scales;
3. Output the entire prompt in English, retaining original text in quotes and preserving key input information;
4. Emphasize motion information and different camera movements present in the input description;
5. The revised prompt should be around 80-100 words long;
Please directly output the rewritten English text."""


def apply_style_to_prompt(prompt: str, style_modifier: str) -> str:
    """Pure function: prompt + style modifier -> styled prompt"""
    if not style_modifier:
        return prompt
    
    # Add style modifier to prompt in a natural way
    if prompt.endswith('.'):
        return f"{prompt[:-1]}, {style_modifier}."
    else:
        return f"{prompt}, {style_modifier}"


def auto_select_model(available_models: List[str], target_vram_gb: float = 16.0) -> str:
    """Pure function: available models + VRAM -> best model choice"""
    if not available_models:
        return "Qwen2.5_7B"  # Default fallback
    
    # Filter models that fit in VRAM and sort by capability
    suitable_models = []
    for model_name in available_models:
        if model_name in MODEL_SPECS:
            spec = MODEL_SPECS[model_name]
            if spec.vram_gb <= target_vram_gb:
                suitable_models.append((model_name, spec.vram_gb, spec.model_type))
    
    if not suitable_models:
        # Return smallest model if none fit
        return min(available_models, key=lambda m: MODEL_SPECS.get(m, MODEL_SPECS["Qwen2.5_3B"]).vram_gb)
    
    # Sort by VRAM (larger is better) then by vision-language preference
    # Use negative VRAM for descending sort, then boolean for VL preference
    suitable_models.sort(key=lambda x: (-x[1], x[2] != ModelType.VISION_LANGUAGE))
    return suitable_models[0][0]


# Core enhancement functions (functional composition)
def enhance_single_prompt(prompt: str, 
                         language: PromptLanguage,
                         model_spec: ModelSpec,
                         style_modifier: str,
                         system_prompt: str,
                         config: EnhancementConfig,
                         model_service: ModelInferenceService) -> Tuple[bool, str, str]:
    """
    Pure function: single prompt enhancement using functional composition
    Returns: (success, enhanced_prompt, error_message)
    """
    if not prompt or not prompt.strip():
        return False, prompt, "Empty prompt"
    
    # Apply style modifier first (pure transformation)
    styled_prompt = apply_style_to_prompt(prompt.strip(), style_modifier)
    
    # Side effect isolation: model inference
    try:
        success, enhanced, error = model_service.enhance_prompt(styled_prompt, system_prompt, config)
        # Return enhanced result regardless of success (model service decides what to return)
        return success, enhanced, error
    except Exception as e:
        return False, prompt, f"Model inference error: {str(e)}"


def enhance_prompts_batch(prompts: Dict[str, str],
                         language: PromptLanguage,
                         model_spec: ModelSpec,
                         style_modifier: str,
                         config: EnhancementConfig,
                         model_service: ModelInferenceService) -> Tuple[Dict[str, str], List[str]]:
    """
    Pure function: batch prompt enhancement using functional composition
    Returns: (enhanced_prompts, error_messages)
    """
    if not prompts:
        return {}, []
    
    system_prompt = get_system_prompt(language, model_spec.model_type)
    enhanced_prompts = {}
    errors = []
    
    # Use functional approach with tuple comprehension and filter
    results = tuple(
        (key, enhance_single_prompt(prompt, language, model_spec, style_modifier, 
                                   system_prompt, config, model_service))
        for key, prompt in prompts.items()
    )
    
    # Separate successful and failed enhancements
    for key, (success, enhanced, error) in results:
        enhanced_prompts[key] = enhanced
        if not success and error:
            errors.append(f"Frame {key}: {error}")
    
    return enhanced_prompts, errors


# High-level enhancement functions
def enhance_prompts(request: PromptEnhancementRequest,
                   model_service: ModelInferenceService,
                   config: Optional[EnhancementConfig] = None) -> PromptEnhancementResult:
    """
    Pure function: enhancement request -> result using functional composition
    Main entry point for prompt enhancement
    """
    start_time = time.time()
    
    if config is None:
        config = EnhancementConfig()
    
    # Validate and normalize inputs (pure transformations)
    validated_prompts = validate_prompts_dict(request.prompts)
    if not validated_prompts:
        return PromptEnhancementResult(
            enhanced_prompts={},
            original_prompts={},
            model_used="none",
            language=request.language,
            processing_time=0.0,
            success=False,
            error_message="No valid prompts provided",
            enhancement_count=0
        )
    
    # Build style modifier (pure transformation)
    style_modifier = build_style_theme_modifier(
        request.style, request.theme, request.custom_style, request.custom_theme
    )
    
    # Select model (pure function)
    model_name = request.model_name or "Qwen2.5_7B"
    if model_name not in MODEL_SPECS:
        return PromptEnhancementResult(
            enhanced_prompts=validated_prompts,
            original_prompts=validated_prompts,
            model_used=model_name,
            language=request.language,
            processing_time=time.time() - start_time,
            success=False,
            error_message=f"Unknown model: {model_name}",
            enhancement_count=0
        )
    
    model_spec = MODEL_SPECS[model_name]
    
    # Check model availability (side effect, but isolated)
    if not model_service.is_available(model_name):
        return PromptEnhancementResult(
            enhanced_prompts=validated_prompts,
            original_prompts=validated_prompts,
            model_used=model_name,
            language=request.language,
            processing_time=time.time() - start_time,
            success=False,
            error_message=f"Model not available: {model_name}",
            enhancement_count=0
        )
    
    # Enhance prompts (functional composition with isolated side effects)
    enhanced_prompts, errors = enhance_prompts_batch(
        validated_prompts, request.language, model_spec, style_modifier, config, model_service
    )
    
    processing_time = time.time() - start_time
    success = len(errors) == 0
    error_message = "; ".join(errors) if errors else None
    
    return PromptEnhancementResult(
        enhanced_prompts=enhanced_prompts,
        original_prompts=validated_prompts,
        model_used=model_name,
        language=request.language,
        processing_time=processing_time,
        success=success,
        error_message=error_message,
        enhancement_count=len(enhanced_prompts)
    )


def enhance_prompts_simple(prompts: Union[Dict[str, str], str],
                          model_service: ModelInferenceService,
                          language: str = "english",
                          style: Optional[str] = None,
                          model_name: Optional[str] = None) -> PromptEnhancementResult:
    """
    Pure function: simplified prompt enhancement interface
    Convenience function for common use cases
    """
    request = PromptEnhancementRequest(
        prompts=validate_prompts_dict(prompts),
        language=normalize_language(language),
        style=normalize_style(style),
        model_name=model_name
    )
    
    return enhance_prompts(request, model_service)


# Statistics and analysis functions
def get_enhancement_statistics(result: PromptEnhancementResult) -> Dict[str, Any]:
    """Pure function: enhancement result -> statistics"""
    if not result.enhanced_prompts:
        return {
            "total_prompts": 0,
            "enhanced_count": 0,
            "success_rate": 0.0,
            "average_length_increase": 0.0,
            "processing_time": result.processing_time,
            "model_used": result.model_used
        }
    
    original_prompts = result.original_prompts
    enhanced_prompts = result.enhanced_prompts
    
    # Calculate length statistics using functional approach
    length_increases = tuple(
        len(enhanced_prompts.get(key, "")) - len(original_prompts.get(key, ""))
        for key in original_prompts.keys()
        if key in enhanced_prompts
    )
    
    average_length_increase = sum(length_increases) / len(length_increases) if length_increases else 0.0
    success_rate = result.enhancement_count / len(original_prompts) if original_prompts else 0.0
    
    return {
        "total_prompts": len(original_prompts),
        "enhanced_count": result.enhancement_count,
        "success_rate": success_rate,
        "average_length_increase": average_length_increase,
        "processing_time": result.processing_time,
        "model_used": result.model_used,
        "language": result.language.value,
        "success": result.success
    }


def format_enhancement_report(result: PromptEnhancementResult) -> str:
    """Pure function: enhancement result -> formatted report"""
    stats = get_enhancement_statistics(result)
    
    if not result.success:
        return f"""❌ Prompt Enhancement Failed
Error: {result.error_message}
Model: {result.model_used}
Processing Time: {result.processing_time:.2f}s"""
    
    return f"""✅ Prompt Enhancement Complete!

📊 Statistics:
- Total Prompts: {stats['total_prompts']}
- Enhanced: {stats['enhanced_count']}
- Success Rate: {stats['success_rate']:.1%}
- Average Length Increase: +{stats['average_length_increase']:.0f} characters
- Model Used: {stats['model_used']}
- Language: {stats['language'].title()}
- Processing Time: {stats['processing_time']:.2f}s

The enhanced prompts are ready for video generation!"""


# Legacy compatibility functions
def create_enhancement_request_from_legacy(prompts: Any, 
                                         model_name: str = "Auto-Select",
                                         language: str = "English") -> PromptEnhancementRequest:
    """Pure function: legacy inputs -> enhancement request (compatibility)"""
    return PromptEnhancementRequest(
        prompts=validate_prompts_dict(prompts),
        language=normalize_language(language),
        model_name=model_name if model_name != "Auto-Select" else None
    )


def enhance_prompts_legacy_interface(prompts: Dict[str, str],
                                   model_service: ModelInferenceService,
                                   model_name: str = "Auto-Select",
                                   language: str = "English") -> Dict[str, str]:
    """Pure function: legacy interface -> enhanced prompts (compatibility)"""
    request = create_enhancement_request_from_legacy(prompts, model_name, language)
    result = enhance_prompts(request, model_service)
    return result.enhanced_prompts 