"""
Test helper utilities for Deforum tests
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Optional, List
from unittest.mock import Mock


def create_test_animation_args(**overrides) -> SimpleNamespace:
    """
    Create test animation arguments with optional overrides
    
    Args:
        **overrides: Any fields to override in the default args
        
    Returns:
        SimpleNamespace with animation arguments
    """
    defaults = {
        'translation_x': "0:(0)",
        'translation_y': "0:(0)",
        'translation_z': "0:(0)",
        'rotation_3d_x': "0:(0)",
        'rotation_3d_y': "0:(0)",
        'rotation_3d_z': "0:(0)",
        'zoom': "0:(1.0)",
        'angle': "0:(0)",
        'max_frames': 120,
        'shake_name': "None",
        'shake_intensity': 1.0,
        'shake_speed': 1.0,
    }
    
    # Apply overrides
    defaults.update(overrides)
    
    # Create SimpleNamespace
    args = SimpleNamespace()
    for key, value in defaults.items():
        setattr(args, key, value)
    
    return args


def create_test_prompts(frame_count: int = 3, base_prompt: str = "test prompt") -> Dict[str, str]:
    """
    Create test prompts for a given number of frames
    
    Args:
        frame_count: Number of prompts to create
        base_prompt: Base prompt text to use
        
    Returns:
        Dictionary of frame -> prompt mappings
    """
    prompts = {}
    for i in range(frame_count):
        frame = i * 30  # Space prompts 30 frames apart
        prompts[str(frame)] = f"{base_prompt} {i + 1}"
    
    return prompts


def create_temp_settings_file(settings: Dict[str, Any], suffix: str = ".txt") -> Path:
    """
    Create a temporary settings file with given settings
    
    Args:
        settings: Dictionary of settings to write
        suffix: File suffix (default: .txt)
        
    Returns:
        Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        json.dump(settings, f, indent=2)
        return Path(f.name)


def assert_schedule_valid(schedule_str: str, expected_frames: Optional[List[int]] = None):
    """
    Assert that a schedule string is valid
    
    Args:
        schedule_str: Schedule string to validate
        expected_frames: Optional list of expected frame numbers
    """
    assert isinstance(schedule_str, str), "Schedule must be a string"
    assert schedule_str.strip(), "Schedule cannot be empty"
    
    # Basic format validation
    if ":" in schedule_str:
        parts = schedule_str.split(",")
        for part in parts:
            part = part.strip()
            if part:  # Skip empty parts
                assert ":" in part, f"Invalid format - missing colon: {part}"
                frame_str, value_str = part.split(":", 1)
                frame_str = frame_str.strip()
                value_str = value_str.strip()
                
                # Validate frame number
                assert frame_str, f"Empty frame number in: {part}"
                assert frame_str.isdigit(), f"Invalid frame number: {frame_str}"
                
                # Validate value format
                assert value_str, f"Empty value in: {part}"
                assert value_str.startswith("(") and value_str.endswith(")"), \
                    f"Invalid value format - must be in parentheses: {value_str}"
                
                # Validate value content
                inner_value = value_str[1:-1].strip()
                assert inner_value, f"Empty value in parentheses: {value_str}"
                try:
                    float(inner_value)
                except ValueError:
                    raise AssertionError(f"Invalid numeric value: {inner_value}")
    else:
        # If no colon, this is invalid
        raise AssertionError(f"Invalid schedule format - no colon found: {schedule_str}")
    
    if expected_frames:
        for frame in expected_frames:
            assert str(frame) in schedule_str, f"Frame {frame} not found in schedule"


def assert_prompts_enhanced(original: Dict[str, str], enhanced: Dict[str, str]):
    """
    Assert that prompts have been properly enhanced
    
    Args:
        original: Original prompts
        enhanced: Enhanced prompts
    """
    assert len(enhanced) == len(original), "Enhanced prompts count mismatch"
    
    for frame, original_prompt in original.items():
        assert frame in enhanced, f"Frame {frame} missing from enhanced prompts"
        enhanced_prompt = enhanced[frame]
        
        # Enhanced prompt should be longer and different
        assert len(enhanced_prompt) > len(original_prompt), \
            f"Enhanced prompt for frame {frame} is not longer"
        assert enhanced_prompt != original_prompt, \
            f"Enhanced prompt for frame {frame} is unchanged"


def mock_qwen_manager(enhance_result: Optional[Dict[str, str]] = None) -> Mock:
    """
    Create a mock Qwen manager for testing
    
    Args:
        enhance_result: Optional result to return from enhance_prompts
        
    Returns:
        Mock Qwen manager
    """
    mock = Mock()
    mock.is_model_loaded.return_value = False
    mock.is_model_downloaded.return_value = True
    mock.auto_select_model.return_value = "Qwen2.5-7B-Instruct"
    mock.get_model_info.return_value = {
        'name': 'Qwen2.5-7B-Instruct',
        'description': 'Test model',
        'vram_gb': 8
    }
    
    if enhance_result:
        mock.enhance_prompts.return_value = enhance_result
    else:
        mock.enhance_prompts.return_value = {
            "0": "enhanced test prompt 1",
            "30": "enhanced test prompt 2"
        }
    
    return mock


def create_movement_test_data() -> Dict[str, Any]:
    """
    Create test data for movement analysis
    
    Returns:
        Dictionary with movement test data
    """
    return {
        'static': {
            'args': create_test_animation_args(),
            'expected_strength': 0.0,
            'expected_description': 'static camera position'
        },
        'simple_translation': {
            'args': create_test_animation_args(
                translation_x="0:(0), 30:(10)",
                max_frames=60
            ),
            'expected_strength': 0.5,  # Approximate
            'expected_keywords': ['panning', 'right']
        },
        'complex_movement': {
            'args': create_test_animation_args(
                translation_x="0:(0), 20:(5), 40:(-5)",
                translation_y="0:(0), 30:(3)",
                rotation_3d_y="0:(0), 15:(2)",
                zoom="0:(1.0), 45:(1.2)",
                max_frames=60
            ),
            'expected_strength': 1.0,  # Approximate
            'expected_keywords': ['panning', 'moving', 'rotating', 'zooming']
        }
    }


def validate_movement_result(result_description: str, result_strength: float, 
                           expected_keywords: Optional[List[str]] = None,
                           min_strength: float = 0.0, max_strength: float = 10.0):
    """
    Validate movement analysis results
    
    Args:
        result_description: Movement description to validate
        result_strength: Movement strength to validate
        expected_keywords: Optional keywords that should be in description
        min_strength: Minimum expected strength
        max_strength: Maximum expected strength
    """
    assert isinstance(result_description, str), "Description must be a string"
    assert result_description.strip(), "Description cannot be empty"
    assert isinstance(result_strength, (int, float)), "Strength must be numeric"
    assert min_strength <= result_strength <= max_strength, \
        f"Strength {result_strength} not in range [{min_strength}, {max_strength}]"
    
    if expected_keywords:
        description_lower = result_description.lower()
        for keyword in expected_keywords:
            assert keyword.lower() in description_lower, \
                f"Keyword '{keyword}' not found in description: {result_description}"


class TestDataManager:
    """Helper class for managing test data files"""
    
    def __init__(self, test_data_dir: Path):
        self.test_data_dir = Path(test_data_dir)
    
    def get_settings_file(self) -> Path:
        """Get the test settings file"""
        return self.test_data_dir / "simple.input_settings.txt"
    
    def get_parseq_file(self) -> Path:
        """Get the test parseq file"""
        return self.test_data_dir / "parseq.json"
    
    def get_video_file(self) -> Path:
        """Get the test video file"""
        return self.test_data_dir / "example_init_vid.mp4"
    
    def load_settings(self) -> Dict[str, Any]:
        """Load test settings from file"""
        settings_file = self.get_settings_file()
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                return json.load(f)
        return {}
    
    def load_parseq(self) -> Dict[str, Any]:
        """Load test parseq data from file"""
        parseq_file = self.get_parseq_file()
        if parseq_file.exists():
            with open(parseq_file, 'r') as f:
                return json.load(f)
        return {}


def skip_if_no_gpu():
    """Decorator to skip tests if no GPU is available"""
    import pytest
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    
    return pytest.mark.skipif(not has_gpu, reason="GPU not available")


def skip_if_no_model(model_path: str):
    """Decorator to skip tests if a specific model is not available"""
    import pytest
    from pathlib import Path
    
    has_model = Path(model_path).exists()
    return pytest.mark.skipif(not has_model, reason=f"Model not available: {model_path}")


def parametrize_movement_scenarios():
    """Decorator for parametrizing movement test scenarios"""
    import pytest
    
    scenarios = [
        ("static", {"translation_x": "0:(0)", "max_frames": 30}),
        ("simple_pan", {"translation_x": "0:(0), 30:(10)", "max_frames": 60}),
        ("complex_movement", {
            "translation_x": "0:(0), 20:(5), 40:(-5)",
            "translation_y": "0:(0), 30:(3)",
            "rotation_3d_y": "0:(0), 15:(2)",
            "max_frames": 60
        }),
    ]
    
    return pytest.mark.parametrize("scenario_name,overrides", scenarios) 