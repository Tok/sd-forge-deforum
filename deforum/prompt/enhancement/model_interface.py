"""
Model Interface for Prompt Enhancement System

Protocol definitions and model selection logic following
dependency injection and functional programming principles.
"""

from typing import Protocol, Tuple, List
from .data_models import EnhancementConfig, ModelSpec, ModelType, MODEL_SPECS


class ModelInferenceService(Protocol):
    """Protocol for model inference services - dependency injection interface"""
    
    def enhance_prompt(self, prompt: str, system_prompt: str, config: EnhancementConfig) -> Tuple[bool, str, str]:
        """
        Pure interface for prompt enhancement
        Returns: (success, enhanced_prompt, error_message)
        """
        ...
    
    def is_available(self, model_name: str) -> bool:
        """Check if model is available for inference"""
        ...


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


def get_system_prompt(language_enum, model_type: ModelType) -> str:
    """Pure function: language + model type -> appropriate system prompt"""
    from .data_models import PromptLanguage
    
    # Simplified system prompts for our functional implementation
    if language_enum == PromptLanguage.CHINESE:
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


def validate_model_selection(model_name: str) -> bool:
    """Pure function: model name -> validation result"""
    return model_name in MODEL_SPECS


def get_model_spec(model_name: str) -> ModelSpec:
    """Pure function: model name -> model specification"""
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_SPECS[model_name]


def get_available_models() -> List[str]:
    """Pure function: -> list of all available model names"""
    return list(MODEL_SPECS.keys())


def filter_models_by_vram(max_vram_gb: float) -> List[str]:
    """Pure function: VRAM limit -> list of suitable models"""
    return [
        name for name, spec in MODEL_SPECS.items()
        if spec.vram_gb <= max_vram_gb
    ]


def get_models_by_type(model_type: ModelType) -> List[str]:
    """Pure function: model type -> list of models of that type"""
    return [
        name for name, spec in MODEL_SPECS.items()
        if spec.model_type == model_type
    ]
