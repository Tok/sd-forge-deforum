"""
Core Prompt Processing Functions

Pure functions for prompt normalization, validation, and transformation.
All functions are side-effect free and follow functional programming principles.
"""

import json
from typing import Dict, Optional, Union, Any
from .data_models import PromptLanguage, PromptStyle


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


def apply_style_to_prompt(prompt: str, style_modifier: str) -> str:
    """Pure function: prompt + style modifier -> styled prompt"""
    if not style_modifier:
        return prompt
    
    # Add style modifier to prompt in a natural way
    if prompt.endswith('.'):
        return f"{prompt[:-1]}, {style_modifier}."
    else:
        return f"{prompt}, {style_modifier}"


def clean_prompt_text(prompt: str) -> str:
    """Pure function: raw prompt -> cleaned prompt text"""
    if not prompt:
        return ""
    
    # Remove extra whitespace and normalize line endings
    cleaned = " ".join(prompt.strip().split())
    
    # Remove duplicate commas and clean spacing around punctuation
    import re
    cleaned = re.sub(r',\s*,', ',', cleaned)
    cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)
    cleaned = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1\2', cleaned)
    
    return cleaned


def extract_quoted_text(prompt: str) -> tuple[str, list[str]]:
    """Pure function: prompt -> (prompt_without_quotes, quoted_segments)"""
    import re
    
    # Find all quoted segments
    quoted_segments = re.findall(r'"([^"]*)"', prompt)
    
    # Remove quotes from prompt
    prompt_without_quotes = re.sub(r'"[^"]*"', '', prompt)
    prompt_without_quotes = clean_prompt_text(prompt_without_quotes)
    
    return prompt_without_quotes, quoted_segments


def restore_quoted_text(enhanced_prompt: str, quoted_segments: list[str]) -> str:
    """Pure function: enhanced prompt + original quotes -> prompt with quotes restored"""
    if not quoted_segments:
        return enhanced_prompt
    
    # Simple approach: append quoted segments at the end
    # This preserves important quoted information while keeping enhancement
    quoted_text = ', '.join(f'"{segment}"' for segment in quoted_segments if segment.strip())
    
    if quoted_text:
        if enhanced_prompt.endswith('.'):
            return f"{enhanced_prompt[:-1]}, {quoted_text}."
        else:
            return f"{enhanced_prompt}, {quoted_text}"
    
    return enhanced_prompt


def validate_prompt_length(prompt: str, min_length: int = 10, max_length: int = 1000) -> bool:
    """Pure function: prompt + limits -> length validation result"""
    if not prompt:
        return False
    
    prompt_length = len(prompt.strip())
    return min_length <= prompt_length <= max_length


def calculate_prompt_complexity(prompt: str) -> float:
    """Pure function: prompt -> complexity score (0.0 to 1.0)"""
    if not prompt:
        return 0.0
    
    # Simple complexity scoring based on various factors
    words = prompt.split()
    unique_words = set(word.lower().strip('.,!?;:') for word in words)
    
    # Factors for complexity
    word_count_score = min(len(words) / 50.0, 1.0)  # More words = more complex
    vocabulary_diversity = len(unique_words) / max(len(words), 1)  # Unique/total ratio
    punctuation_count = sum(1 for char in prompt if char in '.,!?;:')
    punctuation_score = min(punctuation_count / 10.0, 1.0)
    
    # Weighted average
    complexity = (word_count_score * 0.4 + vocabulary_diversity * 0.4 + punctuation_score * 0.2)
    return min(complexity, 1.0)


def detect_prompt_language_heuristic(prompt: str) -> PromptLanguage:
    """Pure function: prompt text -> detected language (heuristic)"""
    if not prompt:
        return PromptLanguage.ENGLISH
    
    # Simple heuristic: count CJK characters
    cjk_chars = sum(1 for char in prompt if '\u4e00' <= char <= '\u9fff')
    total_chars = len(prompt.replace(' ', ''))
    
    if total_chars > 0 and (cjk_chars / total_chars) > 0.3:
        return PromptLanguage.CHINESE
    else:
        return PromptLanguage.ENGLISH


def merge_prompt_segments(base_prompt: str, *additional_segments: str) -> str:
    """Pure function: base prompt + segments -> merged prompt"""
    all_segments = [base_prompt] + [seg for seg in additional_segments if seg and seg.strip()]
    
    if not all_segments:
        return ""
    
    # Clean and join segments
    cleaned_segments = [clean_prompt_text(seg) for seg in all_segments if seg.strip()]
    return ", ".join(cleaned_segments)


def split_prompt_by_frames(prompt_text: str) -> Dict[str, str]:
    """Pure function: multi-frame prompt text -> frame-specific prompts"""
    if not prompt_text:
        return {}
    
    # Try to parse as JSON first
    try:
        parsed = json.loads(prompt_text)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items() if v}
    except (json.JSONDecodeError, TypeError):
        pass
    
    # If not JSON, treat as single prompt for frame 0
    return {"0": prompt_text.strip()}


def create_prompt_request_from_legacy(prompts: Any, 
                                    model_name: str = "Auto-Select",
                                    language: str = "English"):
    """Pure function: legacy inputs -> enhancement request (compatibility)"""
    from .data_models import PromptEnhancementRequest
    
    return PromptEnhancementRequest(
        prompts=validate_prompts_dict(prompts),
        language=normalize_language(language),
        model_name=model_name if model_name != "Auto-Select" else None
    )
