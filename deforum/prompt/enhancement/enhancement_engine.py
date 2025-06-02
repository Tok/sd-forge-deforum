"""
Enhancement Engine for Prompt Enhancement System

Core enhancement logic and batch processing using functional composition.
All enhancement functions follow functional programming principles.
"""

import time
from typing import Dict, List, Tuple, Optional
from .data_models import (
    PromptEnhancementRequest, PromptEnhancementResult, EnhancementConfig,
    PromptLanguage, ModelSpec, MODEL_SPECS
)
from .model_interface import ModelInferenceService, get_system_prompt
from .core_processing import (
    validate_prompts_dict, build_style_theme_modifier, apply_style_to_prompt,
    extract_quoted_text, restore_quoted_text, clean_prompt_text
)


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
    
    # Extract and preserve quoted text (pure transformation)
    prompt_without_quotes, quoted_segments = extract_quoted_text(prompt.strip())
    
    # Apply style modifier first (pure transformation)
    styled_prompt = apply_style_to_prompt(prompt_without_quotes, style_modifier)
    
    # Side effect isolation: model inference
    try:
        success, enhanced, error = model_service.enhance_prompt(styled_prompt, system_prompt, config)
        
        if success and enhanced:
            # Clean and restore quoted text
            enhanced_clean = clean_prompt_text(enhanced)
            enhanced_with_quotes = restore_quoted_text(enhanced_clean, quoted_segments)
            return True, enhanced_with_quotes, ""
        else:
            # Return original prompt if enhancement failed
            return False, prompt, error or "Enhancement failed"
            
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


def enhance_prompts_main(request: PromptEnhancementRequest,
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


def enhance_prompts_simple(prompts,
                          model_service: ModelInferenceService,
                          language: str = "english",
                          style: Optional[str] = None,
                          model_name: Optional[str] = None) -> PromptEnhancementResult:
    """
    Pure function: simplified prompt enhancement interface
    Convenience function for common use cases
    """
    from .core_processing import normalize_language, normalize_style
    
    request = PromptEnhancementRequest(
        prompts=validate_prompts_dict(prompts),
        language=normalize_language(language),
        style=normalize_style(style),
        model_name=model_name
    )
    
    return enhance_prompts_main(request, model_service)


def enhance_prompts_legacy_interface(prompts: Dict[str, str],
                                   model_service: ModelInferenceService,
                                   model_name: str = "Auto-Select",
                                   language: str = "English") -> Dict[str, str]:
    """Pure function: legacy interface -> enhanced prompts (compatibility)"""
    from .core_processing import create_prompt_request_from_legacy
    
    request = create_prompt_request_from_legacy(prompts, model_name, language)
    result = enhance_prompts_main(request, model_service)
    return result.enhanced_prompts


def validate_enhancement_request(request: PromptEnhancementRequest) -> List[str]:
    """Pure function: enhancement request -> validation errors"""
    errors = []
    
    if not request.prompts:
        errors.append("No prompts provided")
    
    validated_prompts = validate_prompts_dict(request.prompts)
    if not validated_prompts:
        errors.append("No valid prompts found")
    
    if request.model_name and request.model_name not in MODEL_SPECS:
        errors.append(f"Unknown model: {request.model_name}")
    
    if request.max_length <= 0:
        errors.append("Max length must be positive")
    
    return errors


def estimate_processing_time(prompt_count: int, model_name: str) -> float:
    """Pure function: prompt count + model -> estimated processing time"""
    if model_name not in MODEL_SPECS:
        return 0.0
    
    model_spec = MODEL_SPECS[model_name]
    
    # Simple estimation based on model size and prompt count
    base_time_per_prompt = {
        "Qwen2.5_3B": 2.0,
        "Qwen2.5_7B": 3.0,
        "Qwen2.5_14B": 5.0,
        "QwenVL2.5_3B": 3.0,
        "QwenVL2.5_7B": 4.0,
    }
    
    return prompt_count * base_time_per_prompt.get(model_name, 3.0)


def calculate_enhancement_quality_score(original: str, enhanced: str) -> float:
    """Pure function: original + enhanced prompts -> quality score (0.0 to 1.0)"""
    if not original or not enhanced:
        return 0.0
    
    # Simple quality scoring based on improvement factors
    length_improvement = len(enhanced) / max(len(original), 1)
    length_score = min(length_improvement / 2.0, 1.0)  # Cap at 100% improvement
    
    # Word diversity improvement
    original_words = set(original.lower().split())
    enhanced_words = set(enhanced.lower().split())
    
    new_words = enhanced_words - original_words
    diversity_score = min(len(new_words) / max(len(original_words), 1), 1.0)
    
    # Combined score
    quality_score = (length_score * 0.6 + diversity_score * 0.4)
    return min(quality_score, 1.0)


def batch_quality_analysis(results: PromptEnhancementResult) -> Dict[str, float]:
    """Pure function: enhancement results -> quality analysis per prompt"""
    if not results.enhanced_prompts or not results.original_prompts:
        return {}
    
    quality_scores = {}
    
    for key in results.enhanced_prompts:
        if key in results.original_prompts:
            original = results.original_prompts[key]
            enhanced = results.enhanced_prompts[key]
            quality_scores[key] = calculate_enhancement_quality_score(original, enhanced)
    
    return quality_scores
