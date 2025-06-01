"""
Unit tests for the functional prompt enhancement system.

Tests all pure functions, immutable data structures, and functional composition
for prompt enhancement and AI model integration.
"""

import pytest
import json
from typing import Dict, Tuple
from unittest.mock import Mock

from scripts.deforum_helpers.prompt_enhancement import (
    # Data structures and enums
    PromptLanguage, PromptStyle, ModelType,
    ModelSpec, PromptEnhancementRequest, PromptEnhancementResult, EnhancementConfig,
    MODEL_SPECS,
    
    # Pure utility functions
    normalize_language, normalize_style, validate_prompts_dict, 
    build_style_theme_modifier, get_system_prompt, apply_style_to_prompt,
    auto_select_model,
    
    # Core enhancement functions
    enhance_single_prompt, enhance_prompts_batch, enhance_prompts,
    
    # Convenience functions
    enhance_prompts_simple,
    
    # Statistics and analysis
    get_enhancement_statistics, format_enhancement_report,
    
    # Legacy compatibility
    create_enhancement_request_from_legacy, enhance_prompts_legacy_interface
)


class MockModelInferenceService:
    """Mock model inference service for testing"""
    
    def __init__(self, available_models=None, success=True, enhanced_suffix=" enhanced"):
        self.available_models = available_models or ["Qwen2.5_7B", "Qwen2.5_3B"]
        self.success = success
        self.enhanced_suffix = enhanced_suffix
        self.call_count = 0
        self.last_calls = []
    
    def enhance_prompt(self, prompt: str, system_prompt: str, config) -> Tuple[bool, str, str]:
        """Mock enhancement that appends suffix to prompt"""
        self.call_count += 1
        self.last_calls.append((prompt, system_prompt, config))
        
        if self.success:
            return True, prompt + self.enhanced_suffix, ""
        else:
            return False, prompt, "Mock enhancement failed"
    
    def is_available(self, model_name: str) -> bool:
        """Check if model is in available list"""
        return model_name in self.available_models


class TestDataStructures:
    """Test immutable data structures"""
    
    def test_model_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec = ModelSpec(
            name="test_model",
            hf_name="test/model",
            vram_gb=8,
            model_type=ModelType.TEXT_ONLY,
            description="Test model"
        )
        
        with pytest.raises(AttributeError):
            spec.name = "new_name"  # Should be immutable
        
        assert spec.name == "test_model"
        assert spec.model_type == ModelType.TEXT_ONLY
    
    def test_prompt_enhancement_request_immutability(self):
        """Test that PromptEnhancementRequest is immutable"""
        request = PromptEnhancementRequest(
            prompts={"0": "test prompt"},
            language=PromptLanguage.ENGLISH,
            style=PromptStyle.CINEMATIC
        )
        
        with pytest.raises(AttributeError):
            request.language = PromptLanguage.CHINESE
        
        assert request.language == PromptLanguage.ENGLISH
        assert request.style == PromptStyle.CINEMATIC
    
    def test_prompt_enhancement_result_immutability(self):
        """Test that PromptEnhancementResult is immutable"""
        result = PromptEnhancementResult(
            enhanced_prompts={"0": "enhanced"},
            original_prompts={"0": "original"},
            model_used="Qwen2.5_7B",
            language=PromptLanguage.ENGLISH,
            processing_time=1.0,
            success=True
        )
        
        with pytest.raises(AttributeError):
            result.success = False
        
        assert result.success is True
        assert result.model_used == "Qwen2.5_7B"
    
    def test_enhancement_config_immutability(self):
        """Test that EnhancementConfig is immutable"""
        config = EnhancementConfig(temperature=0.8, max_tokens=256)
        
        with pytest.raises(AttributeError):
            config.temperature = 0.5
        
        assert config.temperature == 0.8
        assert config.max_tokens == 256


class TestNormalizationFunctions:
    """Test input normalization functions"""
    
    def test_normalize_language(self):
        """Test language normalization"""
        # Test enum pass-through
        assert normalize_language(PromptLanguage.ENGLISH) == PromptLanguage.ENGLISH
        
        # Test string variants
        assert normalize_language("english") == PromptLanguage.ENGLISH
        assert normalize_language("English") == PromptLanguage.ENGLISH
        assert normalize_language("en") == PromptLanguage.ENGLISH
        assert normalize_language("eng") == PromptLanguage.ENGLISH
        
        assert normalize_language("chinese") == PromptLanguage.CHINESE
        assert normalize_language("zh") == PromptLanguage.CHINESE
        assert normalize_language("chn") == PromptLanguage.CHINESE
        
        # Test default fallback
        assert normalize_language("unknown") == PromptLanguage.ENGLISH
        assert normalize_language("") == PromptLanguage.ENGLISH
        assert normalize_language(None) == PromptLanguage.ENGLISH
    
    def test_normalize_style(self):
        """Test style normalization"""
        # Test None handling
        assert normalize_style(None) is None
        assert normalize_style("") is None
        
        # Test enum pass-through
        assert normalize_style(PromptStyle.CINEMATIC) == PromptStyle.CINEMATIC
        
        # Test string variants
        assert normalize_style("photorealistic") == PromptStyle.PHOTOREALISTIC
        assert normalize_style("photo") == PromptStyle.PHOTOREALISTIC
        assert normalize_style("realistic") == PromptStyle.PHOTOREALISTIC
        
        assert normalize_style("cinematic") == PromptStyle.CINEMATIC
        assert normalize_style("cinema") == PromptStyle.CINEMATIC
        assert normalize_style("movie") == PromptStyle.CINEMATIC
        
        assert normalize_style("anime") == PromptStyle.ANIME
        assert normalize_style("animation") == PromptStyle.ANIME
        
        # Test unknown style
        assert normalize_style("unknown_style") is None
    
    def test_validate_prompts_dict(self):
        """Test prompts dictionary validation"""
        # Test valid dictionary
        valid_prompts = {"0": "prompt 1", "30": "prompt 2"}
        assert validate_prompts_dict(valid_prompts) == valid_prompts
        
        # Test JSON string
        json_prompts = '{"0": "prompt 1", "30": "prompt 2"}'
        assert validate_prompts_dict(json_prompts) == {"0": "prompt 1", "30": "prompt 2"}
        
        # Test invalid JSON
        assert validate_prompts_dict('{"invalid": json}') == {}
        
        # Test empty/None inputs
        assert validate_prompts_dict(None) == {}
        assert validate_prompts_dict("") == {}
        assert validate_prompts_dict({}) == {}
        
        # Test non-dict input
        assert validate_prompts_dict("not a dict") == {}
        assert validate_prompts_dict(123) == {}
        
        # Test filtering empty/invalid values
        mixed_prompts = {"0": "valid", "1": "", "2": "  ", "3": None, "4": "also valid"}
        expected = {"0": "valid", "4": "also valid"}
        assert validate_prompts_dict(mixed_prompts) == expected


class TestStyleAndThemeProcessing:
    """Test style and theme processing functions"""
    
    def test_build_style_theme_modifier(self):
        """Test style/theme modifier building"""
        # Test single style
        modifier = build_style_theme_modifier(PromptStyle.CINEMATIC, None, None, None)
        assert "cinematic lighting" in modifier
        assert "dramatic composition" in modifier
        
        # Test with theme
        modifier = build_style_theme_modifier(None, "nature", None, None)
        assert "with nature theme" in modifier
        
        # Test with custom style and theme
        modifier = build_style_theme_modifier(None, None, "custom style", "custom theme")
        assert "custom style" in modifier
        assert "custom theme" in modifier
        
        # Test combined
        modifier = build_style_theme_modifier(
            PromptStyle.ANIME, "fantasy", "custom", "mythical"
        )
        assert "anime style" in modifier
        assert "with fantasy theme" in modifier
        assert "custom" in modifier
        assert "mythical" in modifier
        
        # Test empty inputs
        modifier = build_style_theme_modifier(None, None, None, None)
        assert modifier == ""
        
        # Test empty strings
        modifier = build_style_theme_modifier(None, "", "  ", None)
        assert modifier == ""
    
    def test_apply_style_to_prompt(self):
        """Test style application to prompts"""
        # Test with period
        result = apply_style_to_prompt("A beautiful landscape.", "cinematic")
        assert result == "A beautiful landscape, cinematic."
        
        # Test without period
        result = apply_style_to_prompt("A beautiful landscape", "cinematic")
        assert result == "A beautiful landscape, cinematic"
        
        # Test empty style
        result = apply_style_to_prompt("A beautiful landscape", "")
        assert result == "A beautiful landscape"
        
        # Test empty prompt
        result = apply_style_to_prompt("", "cinematic")
        assert result == ", cinematic"


class TestSystemPrompts:
    """Test system prompt generation"""
    
    def test_get_system_prompt_english(self):
        """Test English system prompts"""
        # Text-only model
        prompt = get_system_prompt(PromptLanguage.ENGLISH, ModelType.TEXT_ONLY)
        assert "prompt engineer" in prompt.lower()
        assert "english" in prompt.lower()
        assert "80-100 words" in prompt
        
        # Vision-language model
        prompt = get_system_prompt(PromptLanguage.ENGLISH, ModelType.VISION_LANGUAGE)
        assert "prompt engineer" in prompt.lower()
        assert "english" in prompt.lower()
    
    def test_get_system_prompt_chinese(self):
        """Test Chinese system prompts"""
        # Text-only model
        prompt = get_system_prompt(PromptLanguage.CHINESE, ModelType.TEXT_ONLY)
        assert "Prompt优化师" in prompt
        assert "中文输出" in prompt
        assert "80-100字" in prompt
        
        # Vision-language model
        prompt = get_system_prompt(PromptLanguage.CHINESE, ModelType.VISION_LANGUAGE)
        assert "Prompt优化师" in prompt
        assert "中文输出" in prompt


class TestModelSelection:
    """Test model selection functions"""
    
    def test_auto_select_model(self):
        """Test automatic model selection"""
        available_models = ["Qwen2.5_3B", "Qwen2.5_7B", "QwenVL2.5_7B"]
        
        # Test normal selection with adequate VRAM
        selected = auto_select_model(available_models, target_vram_gb=20.0)
        assert selected == "QwenVL2.5_7B"  # Largest VL model that fits
        
        # Test limited VRAM - 7B model needs 14GB, so 10GB should select 3B
        selected = auto_select_model(available_models, target_vram_gb=10.0)
        assert selected == "Qwen2.5_3B"  # Only 3B model fits in 10GB
        
        # Test adequate VRAM for 7B but not VL - should prefer 7B text model
        selected = auto_select_model(available_models, target_vram_gb=15.0)
        assert selected == "Qwen2.5_7B"  # 7B text model fits, VL doesn't
        
        # Test very limited VRAM
        selected = auto_select_model(available_models, target_vram_gb=4.0)
        assert selected == "Qwen2.5_3B"  # Smallest available
        
        # Test empty list
        selected = auto_select_model([])
        assert selected == "Qwen2.5_7B"  # Default fallback
        
        # Test unknown models
        selected = auto_select_model(["unknown_model"], target_vram_gb=10.0)
        assert selected == "unknown_model"  # Returns what's available


class TestCoreEnhancementFunctions:
    """Test core enhancement functions"""
    
    def test_enhance_single_prompt(self):
        """Test single prompt enhancement"""
        mock_service = MockModelInferenceService()
        model_spec = MODEL_SPECS["Qwen2.5_7B"]
        config = EnhancementConfig()
        
        success, enhanced, error = enhance_single_prompt(
            "test prompt",
            PromptLanguage.ENGLISH,
            model_spec,
            "cinematic",
            "system prompt",
            config,
            mock_service
        )
        
        assert success is True
        assert enhanced == "test prompt, cinematic enhanced"
        assert error == ""
        assert mock_service.call_count == 1
    
    def test_enhance_single_prompt_failure(self):
        """Test single prompt enhancement failure"""
        mock_service = MockModelInferenceService(success=False)
        model_spec = MODEL_SPECS["Qwen2.5_7B"]
        config = EnhancementConfig()
        
        success, enhanced, error = enhance_single_prompt(
            "test prompt",
            PromptLanguage.ENGLISH,
            model_spec,
            "",
            "system prompt",
            config,
            mock_service
        )
        
        assert success is False
        assert enhanced == "test prompt"  # Returns original on failure
        assert "Mock enhancement failed" in error
    
    def test_enhance_single_prompt_empty(self):
        """Test enhancement with empty prompt"""
        mock_service = MockModelInferenceService()
        model_spec = MODEL_SPECS["Qwen2.5_7B"]
        config = EnhancementConfig()
        
        success, enhanced, error = enhance_single_prompt(
            "",
            PromptLanguage.ENGLISH,
            model_spec,
            "",
            "system prompt",
            config,
            mock_service
        )
        
        assert success is False
        assert enhanced == ""
        assert error == "Empty prompt"
        assert mock_service.call_count == 0  # Should not call service
    
    def test_enhance_prompts_batch(self):
        """Test batch prompt enhancement"""
        mock_service = MockModelInferenceService()
        model_spec = MODEL_SPECS["Qwen2.5_7B"]
        config = EnhancementConfig()
        
        prompts = {"0": "prompt 1", "30": "prompt 2"}
        
        enhanced_prompts, errors = enhance_prompts_batch(
            prompts, PromptLanguage.ENGLISH, model_spec, "style", config, mock_service
        )
        
        assert len(enhanced_prompts) == 2
        assert enhanced_prompts["0"] == "prompt 1, style enhanced"
        assert enhanced_prompts["30"] == "prompt 2, style enhanced"
        assert len(errors) == 0
        assert mock_service.call_count == 2
    
    def test_enhance_prompts_batch_mixed_results(self):
        """Test batch enhancement with mixed success/failure"""
        # Mock service that fails on second call
        mock_service = Mock()
        mock_service.enhance_prompt.side_effect = [
            (True, "enhanced 1", ""),
            (False, "original 2", "enhancement failed")
        ]
        
        model_spec = MODEL_SPECS["Qwen2.5_7B"]
        config = EnhancementConfig()
        prompts = {"0": "prompt 1", "30": "prompt 2"}
        
        enhanced_prompts, errors = enhance_prompts_batch(
            prompts, PromptLanguage.ENGLISH, model_spec, "", config, mock_service
        )
        
        assert enhanced_prompts["0"] == "enhanced 1"
        assert enhanced_prompts["30"] == "original 2"
        assert len(errors) == 1
        assert "Frame 30" in errors[0]


class TestHighLevelEnhancement:
    """Test high-level enhancement functions"""
    
    def test_enhance_prompts_success(self):
        """Test successful prompt enhancement"""
        mock_service = MockModelInferenceService()
        
        request = PromptEnhancementRequest(
            prompts={"0": "test prompt", "30": "another prompt"},
            language=PromptLanguage.ENGLISH,
            style=PromptStyle.CINEMATIC,
            model_name="Qwen2.5_7B"
        )
        
        result = enhance_prompts(request, mock_service)
        
        assert result.success is True
        assert len(result.enhanced_prompts) == 2
        assert "cinematic" in result.enhanced_prompts["0"]
        assert " enhanced" in result.enhanced_prompts["0"]
        assert result.model_used == "Qwen2.5_7B"
        assert result.language == PromptLanguage.ENGLISH
        assert result.enhancement_count == 2
        assert result.processing_time > 0
    
    def test_enhance_prompts_no_valid_prompts(self):
        """Test enhancement with no valid prompts"""
        mock_service = MockModelInferenceService()
        
        request = PromptEnhancementRequest(
            prompts={},
            language=PromptLanguage.ENGLISH
        )
        
        result = enhance_prompts(request, mock_service)
        
        assert result.success is False
        assert result.error_message == "No valid prompts provided"
        assert len(result.enhanced_prompts) == 0
        assert result.enhancement_count == 0
    
    def test_enhance_prompts_unknown_model(self):
        """Test enhancement with unknown model"""
        mock_service = MockModelInferenceService()
        
        request = PromptEnhancementRequest(
            prompts={"0": "test"},
            language=PromptLanguage.ENGLISH,
            model_name="unknown_model"
        )
        
        result = enhance_prompts(request, mock_service)
        
        assert result.success is False
        assert "Unknown model" in result.error_message
        assert result.enhanced_prompts == {"0": "test"}  # Returns originals
    
    def test_enhance_prompts_model_unavailable(self):
        """Test enhancement with unavailable model"""
        mock_service = MockModelInferenceService(available_models=["Qwen2.5_3B"])  # Only 3B available, not 7B
        
        request = PromptEnhancementRequest(
            prompts={"0": "test"},
            language=PromptLanguage.ENGLISH,
            model_name="Qwen2.5_7B"  # Request 7B which is not available
        )
        
        result = enhance_prompts(request, mock_service)
        
        assert result.success is False
        assert "Model not available" in result.error_message
        assert result.enhanced_prompts == {"0": "test"}
    
    def test_enhance_prompts_simple(self):
        """Test simplified enhancement interface"""
        mock_service = MockModelInferenceService()
        
        result = enhance_prompts_simple(
            prompts={"0": "test prompt"},
            model_service=mock_service,
            language="english",
            style="cinematic"
        )
        
        assert result.success is True
        assert "cinematic" in result.enhanced_prompts["0"]
        assert result.language == PromptLanguage.ENGLISH


class TestStatisticsAndAnalysis:
    """Test statistics and analysis functions"""
    
    def test_get_enhancement_statistics(self):
        """Test enhancement statistics calculation"""
        result = PromptEnhancementResult(
            enhanced_prompts={"0": "enhanced prompt", "30": "another enhanced"},
            original_prompts={"0": "prompt", "30": "another"},
            model_used="Qwen2.5_7B",
            language=PromptLanguage.ENGLISH,
            processing_time=1.5,
            success=True,
            enhancement_count=2
        )
        
        stats = get_enhancement_statistics(result)
        
        assert stats["total_prompts"] == 2
        assert stats["enhanced_count"] == 2
        assert stats["success_rate"] == 1.0
        assert stats["average_length_increase"] > 0  # "enhanced " and " enhanced"
        assert stats["processing_time"] == 1.5
        assert stats["model_used"] == "Qwen2.5_7B"
        assert stats["language"] == "english"
        assert stats["success"] is True
    
    def test_get_enhancement_statistics_empty(self):
        """Test statistics with no enhanced prompts"""
        result = PromptEnhancementResult(
            enhanced_prompts={},
            original_prompts={},
            model_used="none",
            language=PromptLanguage.ENGLISH,
            processing_time=0.0,
            success=False,
            enhancement_count=0
        )
        
        stats = get_enhancement_statistics(result)
        
        assert stats["total_prompts"] == 0
        assert stats["enhanced_count"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["average_length_increase"] == 0.0
    
    def test_format_enhancement_report(self):
        """Test enhancement report formatting"""
        # Success case
        result = PromptEnhancementResult(
            enhanced_prompts={"0": "enhanced"},
            original_prompts={"0": "original"},
            model_used="Qwen2.5_7B",
            language=PromptLanguage.ENGLISH,
            processing_time=1.0,
            success=True,
            enhancement_count=1
        )
        
        report = format_enhancement_report(result)
        assert "✅ Prompt Enhancement Complete!" in report
        assert "Total Prompts: 1" in report
        assert "Enhanced: 1" in report
        assert "Success Rate: 100.0%" in report
        assert "Qwen2.5_7B" in report
        
        # Failure case
        result_fail = PromptEnhancementResult(
            enhanced_prompts={},
            original_prompts={},
            model_used="test",
            language=PromptLanguage.ENGLISH,
            processing_time=0.5,
            success=False,
            error_message="Test error",
            enhancement_count=0
        )
        
        report_fail = format_enhancement_report(result_fail)
        assert "❌ Prompt Enhancement Failed" in report_fail
        assert "Test error" in report_fail


class TestLegacyCompatibility:
    """Test legacy compatibility functions"""
    
    def test_create_enhancement_request_from_legacy(self):
        """Test legacy request creation"""
        prompts = {"0": "test prompt"}
        
        request = create_enhancement_request_from_legacy(
            prompts, model_name="Qwen2.5_7B", language="English"
        )
        
        assert request.prompts == prompts
        assert request.language == PromptLanguage.ENGLISH
        assert request.model_name == "Qwen2.5_7B"
        
        # Test Auto-Select
        request_auto = create_enhancement_request_from_legacy(
            prompts, model_name="Auto-Select", language="chinese"
        )
        
        assert request_auto.model_name is None  # Auto-Select -> None
        assert request_auto.language == PromptLanguage.CHINESE
    
    def test_enhance_prompts_legacy_interface(self):
        """Test legacy enhancement interface"""
        mock_service = MockModelInferenceService()
        prompts = {"0": "test prompt"}
        
        enhanced = enhance_prompts_legacy_interface(
            prompts, mock_service, model_name="Qwen2.5_7B", language="English"
        )
        
        assert enhanced == {"0": "test prompt enhanced"}


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases"""
    
    def test_complex_style_theme_combination(self):
        """Test complex style and theme combinations"""
        mock_service = MockModelInferenceService()
        
        request = PromptEnhancementRequest(
            prompts={"0": "A mountain"},
            language=PromptLanguage.ENGLISH,
            style=PromptStyle.CINEMATIC,
            theme="fantasy",
            custom_style="ethereal lighting",
            custom_theme="mystical atmosphere"
        )
        
        result = enhance_prompts(request, mock_service)
        
        assert result.success is True
        enhanced = result.enhanced_prompts["0"]
        assert "cinematic lighting" in enhanced
        assert "with fantasy theme" in enhanced
        assert "ethereal lighting" in enhanced
        assert "mystical atmosphere" in enhanced
    
    def test_chinese_language_processing(self):
        """Test Chinese language processing"""
        mock_service = MockModelInferenceService()
        
        request = PromptEnhancementRequest(
            prompts={"0": "测试提示"},
            language=PromptLanguage.CHINESE,
            model_name="Qwen2.5_7B"
        )
        
        result = enhance_prompts(request, mock_service)
        
        assert result.success is True
        assert result.language == PromptLanguage.CHINESE
        
        # Check that Chinese system prompt was used
        system_prompt_call = mock_service.last_calls[0][1]
        assert "Prompt优化师" in system_prompt_call
    
    def test_large_batch_processing(self):
        """Test processing of large prompt batches"""
        mock_service = MockModelInferenceService()
        
        # Create large batch
        prompts = {str(i): f"prompt {i}" for i in range(100)}
        
        request = PromptEnhancementRequest(
            prompts=prompts,
            language=PromptLanguage.ENGLISH,
            model_name="Qwen2.5_3B"
        )
        
        result = enhance_prompts(request, mock_service)
        
        assert result.success is True
        assert len(result.enhanced_prompts) == 100
        assert result.enhancement_count == 100
        assert mock_service.call_count == 100
    
    def test_json_string_input_validation(self):
        """Test JSON string input validation and parsing"""
        mock_service = MockModelInferenceService()
        
        # Valid JSON string
        json_prompts = '{"0": "prompt 1", "30": "prompt 2"}'
        
        result = enhance_prompts_simple(
            prompts=json_prompts,
            model_service=mock_service
        )
        
        assert result.success is True
        assert len(result.enhanced_prompts) == 2
        assert "0" in result.enhanced_prompts
        assert "30" in result.enhanced_prompts


class TestFunctionalProgrammingPrinciples:
    """Test that functional programming principles are followed"""
    
    def test_pure_functions_no_side_effects(self):
        """Test that pure functions don't modify input arguments"""
        # Test with mutable inputs
        prompts_dict = {"0": "test prompt", "30": "another"}
        original_prompts = prompts_dict.copy()
        
        # Call validation function multiple times
        result1 = validate_prompts_dict(prompts_dict)
        result2 = validate_prompts_dict(prompts_dict)
        
        # Original should be unchanged
        assert prompts_dict == original_prompts
        # Results should be identical
        assert result1 == result2
        
        # Test style building
        style_result1 = build_style_theme_modifier(PromptStyle.CINEMATIC, "theme", "custom", "custom_theme")
        style_result2 = build_style_theme_modifier(PromptStyle.CINEMATIC, "theme", "custom", "custom_theme")
        assert style_result1 == style_result2
    
    def test_immutability_preserved(self):
        """Test that data structures remain immutable throughout processing"""
        mock_service = MockModelInferenceService()
        
        request = PromptEnhancementRequest(
            prompts={"0": "test"},
            language=PromptLanguage.ENGLISH
        )
        
        result = enhance_prompts(request, mock_service)
        
        # All data structures should be immutable
        with pytest.raises(AttributeError):
            request.language = PromptLanguage.CHINESE
        
        with pytest.raises(AttributeError):
            result.success = False
        
        with pytest.raises(AttributeError):
            result.model_used = "different_model"
    
    def test_functional_composition(self):
        """Test that functions compose well together"""
        mock_service = MockModelInferenceService()
        
        # Create a pipeline using function composition
        prompts_raw = '{"0": "test", "30": "another"}'
        validated_prompts = validate_prompts_dict(prompts_raw)
        language = normalize_language("english")
        style = normalize_style("cinematic")
        style_modifier = build_style_theme_modifier(style, None, None, None)
        
        # Compose into request
        request = PromptEnhancementRequest(
            prompts=validated_prompts,
            language=language,
            style=style
        )
        
        # Should be equivalent to direct creation
        result_composed = enhance_prompts(request, mock_service)
        
        # Also test via convenience function
        result_direct = enhance_prompts_simple(
            prompts_raw, mock_service, language="english", style="cinematic"
        )
        
        # Results should be equivalent
        assert result_composed.enhanced_prompts == result_direct.enhanced_prompts
        assert result_composed.language == result_direct.language
    
    def test_error_isolation(self):
        """Test that errors are properly isolated and don't affect other operations"""
        # Mock service that throws exception
        mock_service = Mock()
        mock_service.enhance_prompt.side_effect = Exception("Model error")
        mock_service.is_available.return_value = True
        
        request = PromptEnhancementRequest(
            prompts={"0": "test"},
            language=PromptLanguage.ENGLISH,
            model_name="Qwen2.5_7B"
        )
        
        # Should handle exception gracefully
        result = enhance_prompts(request, mock_service)
        
        assert result.success is False
        assert "Model inference error" in result.error_message
        assert result.enhanced_prompts["0"] == "test"  # Returns original
    
    def test_tuple_comprehensions_and_functional_operators(self):
        """Test that functional operators are used correctly"""
        # Test that validation uses functional filtering
        input_prompts = {"0": "valid", "1": "", "2": "  also valid  ", "3": None}
        validated = validate_prompts_dict(input_prompts)
        
        # Should filter out empty/invalid and strip whitespace
        expected = {"0": "valid", "2": "also valid"}
        assert validated == expected
        
        # Test batch processing functional composition
        mock_service = MockModelInferenceService()
        model_spec = MODEL_SPECS["Qwen2.5_7B"]
        config = EnhancementConfig()
        
        prompts = {"0": "prompt1", "30": "prompt2", "60": "prompt3"}
        enhanced, errors = enhance_prompts_batch(
            prompts, PromptLanguage.ENGLISH, model_spec, "", config, mock_service
        )
        
        # Should process all prompts functionally
        assert len(enhanced) == 3
        assert all(" enhanced" in prompt for prompt in enhanced.values())
        assert len(errors) == 0 