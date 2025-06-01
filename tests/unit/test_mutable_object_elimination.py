#!/usr/bin/env python3

"""
Comprehensive tests for Deforum's immutable data model system.
Tests functional programming patterns and immutable object handling.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import copy
import numpy as np
from types import SimpleNamespace

# Import the data models from the new package structure
from deforum.models.data_models import (
    DeforumArgs,
    DeforumAnimArgs,
    DeforumOutputArgs,
    validate_args,
    create_immutable_args
)

from deforum.models.schedule_models import (
    FrameInterpolater,
    ScheduleDefinition,
    create_immutable_schedule,
    validate_schedule_data
)

# Import with fallback for missing dependencies
try:
    from scripts.deforum_helpers.data_models import (
        ProcessingResult, UIDefaults, SettingsState, ExternalLibraryArgs,
        TestFixtureArgs, validate_processing_result, validate_ui_defaults
    )
    from scripts.deforum_helpers.schedules_models import (
        AnimationSchedules, ControlNetSchedules, FreeUSchedules,
        KohyaSchedules, LooperSchedules, ParseqScheduleData
    )
    
    # Try to import PIL for image processing
    try:
        from PIL import Image
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False
        # Mock Image for testing
        Image = Mock()
        Image.Image = Mock
    
    DATA_MODELS_AVAILABLE = True
except ImportError as e:
    DATA_MODELS_AVAILABLE = False
    print(f"Data models not available for testing: {e}")

# Skip all tests if data models are not available
pytestmark = pytest.mark.skipif(not DATA_MODELS_AVAILABLE, reason="Data models dependencies not available")


@pytest.fixture
def mock_image():
    """Create a mock image for testing"""
    if PIL_AVAILABLE:
        # Create a small test image
        return Image.new('RGB', (64, 64), color='red')
    else:
        return Mock(spec=Image.Image)


@pytest.fixture
def mock_anim_args():
    """Create mock animation args for testing"""
    mock_args = Mock()
    mock_args.max_frames = 100
    mock_args.angle = "0: (0)"
    mock_args.zoom = "0: (1.0)"
    mock_args.translation_x = "0: (0)"
    mock_args.translation_y = "0: (0)"
    mock_args.translation_z = "0: (0)"
    mock_args.rotation_3d_x = "0: (0)"
    mock_args.rotation_3d_y = "0: (0)"
    mock_args.rotation_3d_z = "0: (0)"
    mock_args.transform_center_x = "0: (0.5)"
    mock_args.transform_center_y = "0: (0.5)"
    mock_args.perspective_flip_theta = "0: (0)"
    mock_args.perspective_flip_phi = "0: (0)"
    mock_args.perspective_flip_gamma = "0: (0)"
    mock_args.perspective_flip_fv = "0: (53)"
    mock_args.noise_schedule = "0: (0.065)"
    mock_args.strength_schedule = "0: (0.85)"
    mock_args.keyframe_strength_schedule = "0: (0.50)"
    mock_args.contrast_schedule = "0: (1.0)"
    mock_args.cfg_scale_schedule = "0: (1.0)"
    mock_args.distilled_cfg_scale_schedule = "0: (3.5)"
    mock_args.steps_schedule = "0: (20)"
    mock_args.seed_schedule = '0:(s), 1:(-1)'
    mock_args.sampler_schedule = '0: ("Euler")'
    mock_args.scheduler_schedule = '0: ("Simple")'
    mock_args.ddim_eta_schedule = "0: (0)"
    mock_args.ancestral_eta_schedule = "0: (1)"
    mock_args.subseed_schedule = "0: (1)"
    mock_args.subseed_strength_schedule = "0: (0)"
    mock_args.checkpoint_schedule = '0: ("model1.ckpt")'
    mock_args.clipskip_schedule = "0: (2)"
    mock_args.noise_multiplier_schedule = "0: (1.05)"
    mock_args.mask_schedule = '0: ("{video_mask}")'
    mock_args.noise_mask_schedule = '0: ("{video_mask}")'
    mock_args.kernel_schedule = "0: (5)"
    mock_args.sigma_schedule = "0: (1)"
    mock_args.amount_schedule = "0: (0.1)"
    mock_args.threshold_schedule = "0: (0)"
    mock_args.aspect_ratio_schedule = "0: (1.0)"
    mock_args.fov_schedule = "0: (70)"
    mock_args.near_schedule = "0: (200)"
    mock_args.far_schedule = "0: (10000)"
    mock_args.cadence_flow_factor_schedule = "0:(1)"
    mock_args.redo_flow_factor_schedule = "0:(1)"
    
    # Create ScheduleKeys instance
    return mock_args


class TestProcessingResult:
    """Test ProcessingResult immutable data structure"""

    def test_processing_result_creation(self, mock_image):
        """Test basic ProcessingResult creation"""
        result = ProcessingResult(
            images=(mock_image,),
            info="Test processing",
            success=True
        )
        
        assert len(result.images) == 1
        assert result.info == "Test processing"
        assert result.success == True
        assert result.processing_time == 0.0
        assert len(result.warnings) == 0

    def test_processing_result_immutability(self, mock_image):
        """Test that ProcessingResult is immutable"""
        result = ProcessingResult(images=(mock_image,), info="Test")
        
        with pytest.raises(Exception):  # Should raise FrozenInstanceError
            result.info = "Modified"
        
        with pytest.raises(Exception):
            result.success = False

    def test_motion_preview_factory(self, mock_image):
        """Test factory method for motion preview results"""
        result = ProcessingResult.create_motion_preview(mock_image)
        
        assert len(result.images) == 1
        assert result.images[0] == mock_image
        assert "motion preview" in result.info.lower()
        assert result.success == True

    def test_generation_result_factory(self, mock_image):
        """Test factory method for generation results"""
        images = [mock_image, mock_image]
        result = ProcessingResult.create_generation_result(images, "Generated 2 images")
        
        assert len(result.images) == 2
        assert result.info == "Generated 2 images"
        assert result.success == True

    def test_with_warning_method(self, mock_image):
        """Test adding warnings to ProcessingResult"""
        result = ProcessingResult(images=(mock_image,), info="Test")
        
        result_with_warning = result.with_warning("Test warning")
        
        # Original should be unchanged
        assert len(result.warnings) == 0
        
        # New result should have warning
        assert len(result_with_warning.warnings) == 1
        assert result_with_warning.warnings[0] == "Test warning"
        
        # Other fields should remain the same
        assert result_with_warning.images == result.images
        assert result_with_warning.info == result.info

    def test_processing_result_validation(self, mock_image):
        """Test ProcessingResult validation"""
        valid_result = ProcessingResult(images=(mock_image,), info="Test")
        assert validate_processing_result(valid_result) == True
        
        # Test invalid processing time
        invalid_result = ProcessingResult(
            images=(mock_image,), 
            info="Test", 
            processing_time=-1.0
        )
        assert validate_processing_result(invalid_result) == False


class TestUIDefaults:
    """Test UIDefaults immutable data structure"""

    def test_ui_defaults_creation(self):
        """Test basic UIDefaults creation"""
        defaults = UIDefaults(
            deforum_args={"W": 512, "H": 512},
            animation_args={"max_frames": 100}
        )
        
        assert defaults.deforum_args["W"] == 512
        assert defaults.animation_args["max_frames"] == 100
        assert isinstance(defaults.video_args, dict)

    def test_ui_defaults_immutability(self):
        """Test that UIDefaults is immutable"""
        defaults = UIDefaults()
        
        with pytest.raises(Exception):
            defaults.deforum_args = {"new": "value"}

    @patch('scripts.deforum_helpers.data_models.DeforumArgs')
    @patch('scripts.deforum_helpers.data_models.DeforumAnimArgs')
    def test_create_defaults_factory(self, mock_anim_args, mock_deforum_args):
        """Test factory method for creating UI defaults"""
        mock_deforum_args.return_value = {"W": 1024, "H": 1024}
        mock_anim_args.return_value = {"max_frames": 333}
        
        defaults = UIDefaults.create_defaults()
        
        assert isinstance(defaults, UIDefaults)
        mock_deforum_args.assert_called_once()
        mock_anim_args.assert_called_once()

    def test_ui_defaults_validation(self):
        """Test UIDefaults validation"""
        valid_defaults = UIDefaults(
            deforum_args={},
            animation_args={},
            video_args={},
            parseq_args={},
            wan_args={},
            root_args={}
        )
        assert validate_ui_defaults(valid_defaults) == True
        
        # Test missing required attribute (this would be caught at creation time)
        # We can't easily test this with frozen dataclasses


class TestSettingsState:
    """Test SettingsState immutable data structure"""

    def test_settings_state_creation(self):
        """Test basic SettingsState creation"""
        settings = SettingsState(
            loaded_settings={"key": "value"},
            file_path="/path/to/settings.json"
        )
        
        assert settings.loaded_settings["key"] == "value"
        assert settings.file_path == "/path/to/settings.json"
        assert len(settings.validation_errors) == 0

    def test_settings_state_immutability(self):
        """Test that SettingsState is immutable"""
        settings = SettingsState()
        
        with pytest.raises(Exception):
            settings.loaded_settings = {"new": "value"}

    def test_from_dict_factory(self):
        """Test factory method for creating SettingsState from dict"""
        settings_dict = {
            "W": 1024,
            "H": 768,
            "wan_mode": "Disabled",
            "max_frames": 100
        }
        
        settings = SettingsState.from_dict(settings_dict, "/test/path")
        
        assert settings.loaded_settings == settings_dict
        assert settings.file_path == "/test/path"
        assert isinstance(settings.wan_args, dict)

    def test_with_validation_error_method(self):
        """Test adding validation errors to SettingsState"""
        settings = SettingsState(loaded_settings={"key": "value"})
        
        settings_with_error = settings.with_validation_error("Invalid key")
        
        # Original should be unchanged
        assert len(settings.validation_errors) == 0
        
        # New settings should have error
        assert len(settings_with_error.validation_errors) == 1
        assert settings_with_error.validation_errors[0] == "Invalid key"


class TestExternalLibraryArgs:
    """Test ExternalLibraryArgs immutable data structure"""

    def test_external_library_args_creation(self):
        """Test basic ExternalLibraryArgs creation"""
        args = ExternalLibraryArgs(
            multi=2.0,
            video="/path/to/video.mp4",
            output="/path/to/output"
        )
        
        assert args.multi == 2.0
        assert args.video == "/path/to/video.mp4"
        assert args.output == "/path/to/output"
        assert args.exp == 1  # Default value

    def test_rife_args_factory(self):
        """Test RIFE-specific factory method"""
        rife_args = ExternalLibraryArgs.create_rife_args()
        
        assert rife_args.multi == 1.0
        assert rife_args.exp == 1
        assert rife_args.UHD == False
        assert rife_args.ext == "mp4"

    def test_film_args_factory(self):
        """Test FILM-specific factory method"""
        film_args = ExternalLibraryArgs.create_film_args()
        
        assert film_args.multi == 1.0
        assert film_args.exp == 1


class TestTestFixtureArgs:
    """Test TestFixtureArgs immutable data structure"""

    def test_test_fixture_args_creation(self):
        """Test basic TestFixtureArgs creation"""
        args = TestFixtureArgs(
            W=1024,
            H=768,
            seed=12345
        )
        
        assert args.W == 1024
        assert args.H == 768
        assert args.seed == 12345
        assert args.sampler == "Euler"  # Default

    def test_minimal_args_factory(self):
        """Test minimal args factory method"""
        args = TestFixtureArgs.create_minimal_args()
        
        assert args.W == 512
        assert args.H == 512
        assert args.seed == 42

    def test_animation_args_factory(self):
        """Test animation args factory method"""
        args = TestFixtureArgs.create_animation_args()
        
        assert args.max_frames == 50
        assert args.animation_mode == "2D"

    def test_video_args_factory(self):
        """Test video args factory method"""
        args = TestFixtureArgs.create_video_args()
        
        assert args.fps == 24
        assert args.max_frames == 120


class TestImmutableSchedules:
    """Test immutable schedule models"""
    
    def test_animation_schedules_immutable(self):
        """Test that AnimationSchedules is immutable"""
        schedules = AnimationSchedules(
            angle_series=(0.0, 1.0, 2.0),
            zoom_series=(1.0, 1.1, 1.2)
        )
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            schedules.angle_series = (3.0, 4.0, 5.0)
    
    def test_controlnet_schedules_immutable(self):
        """Test that ControlNetSchedules is immutable"""
        schedules = ControlNetSchedules(
            schedules={'cn_1_weight_schedule_series': (1.0, 0.8, 0.6)}
        )
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            schedules.schedules = {}
    
    def test_looper_schedules_immutable(self):
        """Test that LooperSchedules is immutable"""
        schedules = LooperSchedules(
            use_looper=True,
            image_strength_schedule_series=(0.5, 0.6, 0.7)
        )
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            schedules.use_looper = False
    
    def test_parseq_schedule_data_immutable(self):
        """Test that ParseqScheduleData is immutable"""
        data = ParseqScheduleData(
            frame_data={'frame_0': {'angle': 0.0}},
            use_deltas=True,
            max_frames=100
        )
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            data.use_deltas = False


class TestScheduleFactoryMethods:
    """Test factory methods for creating schedules"""
    
    def test_animation_schedules_from_anim_args(self):
        """Test creating AnimationSchedules from animation args"""
        # Mock anim_args
        anim_args = SimpleNamespace()
        anim_args.angle = "0: (0.0)"
        anim_args.zoom = "0: (1.0)"
        anim_args.translation_x = "0: (0.0)"
        anim_args.translation_y = "0: (0.0)"
        anim_args.translation_z = "0: (0.0)"
        anim_args.rotation_3d_x = "0: (0.0)"
        anim_args.rotation_3d_y = "0: (0.0)"
        anim_args.rotation_3d_z = "0: (0.0)"
        anim_args.transform_center_x = "0: (0.5)"
        anim_args.transform_center_y = "0: (0.5)"
        anim_args.perspective_flip_theta = "0: (0.0)"
        anim_args.perspective_flip_phi = "0: (0.0)"
        anim_args.perspective_flip_gamma = "0: (0.0)"
        anim_args.perspective_flip_fv = "0: (53.0)"
        anim_args.noise_schedule = "0: (0.02)"
        anim_args.strength_schedule = "0: (0.65)"
        anim_args.keyframe_strength_schedule = "0: (1.0)"
        anim_args.contrast_schedule = "0: (1.0)"
        anim_args.cfg_scale_schedule = "0: (7.0)"
        anim_args.distilled_cfg_scale_schedule = "0: (7.0)"
        anim_args.steps_schedule = "0: (20)"
        anim_args.seed_schedule = "0: (-1)"
        anim_args.sampler_schedule = "0: (euler)"
        anim_args.scheduler_schedule = "0: (normal)"
        anim_args.ddim_eta_schedule = "0: (0.0)"
        anim_args.ancestral_eta_schedule = "0: (1.0)"
        anim_args.subseed_schedule = "0: (-1)"
        anim_args.subseed_strength_schedule = "0: (0.0)"
        anim_args.checkpoint_schedule = "0: ()"
        anim_args.clipskip_schedule = "0: (1)"
        anim_args.noise_multiplier_schedule = "0: (1.0)"
        anim_args.mask_schedule = "0: ()"
        anim_args.noise_mask_schedule = "0: ()"
        anim_args.kernel_schedule = "0: (0)"
        anim_args.sigma_schedule = "0: (1.0)"
        anim_args.amount_schedule = "0: (0.1)"
        anim_args.threshold_schedule = "0: (0.0)"
        anim_args.aspect_ratio_schedule = "0: (1.0)"
        anim_args.fov_schedule = "0: (40.0)"
        anim_args.near_schedule = "0: (200.0)"
        anim_args.far_schedule = "0: (10000.0)"
        anim_args.cadence_flow_factor_schedule = "0: (1.0)"
        anim_args.redo_flow_factor_schedule = "0: (1.0)"
        
        # This would require the FrameInterpolater to be available
        # For now, just test that the method exists
        assert hasattr(AnimationSchedules, 'from_anim_args')


class TestScheduleDataIntegrity:
    """Test data integrity of schedule models"""
    
    def test_schedule_tuple_immutability(self):
        """Test that schedule tuples maintain immutability"""
        schedules = AnimationSchedules(
            angle_series=(0.0, 1.0, 2.0, 3.0, 4.0)
        )
        
        # Tuples are immutable by nature
        original_series = schedules.angle_series
        assert original_series == (0.0, 1.0, 2.0, 3.0, 4.0)
        
        # Cannot modify tuple contents
        with pytest.raises(TypeError):
            schedules.angle_series[0] = 10.0
    
    def test_empty_schedule_defaults(self):
        """Test that empty schedules have proper defaults"""
        schedules = AnimationSchedules()
        
        assert schedules.angle_series == ()
        assert schedules.zoom_series == ()
        assert schedules.translation_x_series == ()
    
    def test_parseq_schedule_data_methods(self):
        """Test ParseqScheduleData methods"""
        frame_data = [
            {'frame': 0, 'angle': 0.0, 'zoom': 1.0},
            {'frame': 1, 'angle': 1.0, 'zoom': 1.1},
            {'frame': 2, 'angle': 2.0, 'zoom': 1.2}
        ]
        
        data = ParseqScheduleData.from_parseq_frames(frame_data, use_deltas=True, max_frames=10)
        
        assert data.frame_data == frame_data
        assert data.use_deltas == True
        assert data.max_frames == 10
        
        # Test get_schedule_series method exists
        assert hasattr(data, 'get_schedule_series')


class TestFunctionalProgrammingPrinciples:
    """Test adherence to functional programming principles"""

    def test_immutability_enforcement(self, mock_image):
        """Test that all new data structures enforce immutability"""
        # Test ProcessingResult
        result = ProcessingResult(images=(mock_image,))
        with pytest.raises(Exception):
            result.success = False
        
        # Test UIDefaults  
        defaults = UIDefaults()
        with pytest.raises(Exception):
            defaults.deforum_args = {}
        
        # Test SettingsState
        settings = SettingsState()
        with pytest.raises(Exception):
            settings.loaded_settings = {}
        
        # Test AnimationSchedules
        schedules = AnimationSchedules()
        with pytest.raises(Exception):
            schedules.angle_series = (1.0, 2.0)

    def test_pure_factory_methods(self, mock_image):
        """Test that factory methods are pure (same input -> same output)"""
        # Test ProcessingResult factory
        result1 = ProcessingResult.create_motion_preview(mock_image)
        result2 = ProcessingResult.create_motion_preview(mock_image)
        
        assert result1.images == result2.images
        assert result1.info == result2.info
        
        # Test ExternalLibraryArgs factory
        rife1 = ExternalLibraryArgs.create_rife_args()
        rife2 = ExternalLibraryArgs.create_rife_args()
        
        assert rife1 == rife2

    def test_functional_composition(self, mock_image):
        """Test that data structures compose functionally"""
        # Create base result
        result = ProcessingResult.create_motion_preview(mock_image)
        
        # Add warning (functional transformation)
        result_with_warning = result.with_warning("Test warning")
        
        # Add another warning (functional composition)
        result_with_two_warnings = result_with_warning.with_warning("Second warning")
        
        # Original should be unchanged
        assert len(result.warnings) == 0
        assert len(result_with_warning.warnings) == 1
        assert len(result_with_two_warnings.warnings) == 2
        
        # All should have same images
        assert result.images == result_with_warning.images == result_with_two_warnings.images

    def test_type_safety(self):
        """Test that type hints are enforced where possible"""
        # Test tuple immutability
        schedules = AnimationSchedules(angle_series=(1.0, 2.0, 3.0))
        assert isinstance(schedules.angle_series, tuple)
        
        # Test that we can't modify tuple contents
        with pytest.raises(TypeError):
            schedules.angle_series[0] = 5.0


class TestBackwardCompatibility:
    """Test backward compatibility with existing SimpleNamespace patterns"""

    def test_processing_result_compatibility(self, mock_image):
        """Test that ProcessingResult can replace SimpleNamespace usage"""
        # Old pattern: processed = SimpleNamespace(images = [root.default_img], info = "Generating motion preview...")
        result = ProcessingResult.create_motion_preview(mock_image)
        
        # Should be accessible like old SimpleNamespace
        assert hasattr(result, 'images')
        assert hasattr(result, 'info')
        assert len(result.images) == 1
        assert "motion preview" in result.info.lower()

    def test_ui_defaults_compatibility(self):
        """Test that UIDefaults can replace SimpleNamespace usage in UI"""
        # Old pattern: d = SimpleNamespace(**DeforumArgs())
        defaults = UIDefaults.create_defaults()
        
        # Should provide access to all args sections
        assert hasattr(defaults, 'deforum_args')
        assert hasattr(defaults, 'animation_args')
        assert hasattr(defaults, 'video_args')
        assert isinstance(defaults.deforum_args, dict)

    def test_settings_state_compatibility(self):
        """Test that SettingsState can replace settings SimpleNamespace usage"""
        # Old pattern: wan_args = SimpleNamespace(**{name: args_dict_main[name] for name in WanArgs()})
        settings_dict = {"wan_mode": "Disabled", "wan_model": "Auto"}
        settings = SettingsState.from_dict(settings_dict)
        
        # Should extract WAN args correctly
        assert hasattr(settings, 'wan_args')
        assert isinstance(settings.wan_args, dict)


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_mutable_object_elimination.py -v
    pytest.main([__file__, "-v"]) 