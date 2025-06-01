"""
Unit tests for immutable data models.

Tests data validation, immutability, enum conversions, and helper functions
for the new dataclass-based argument structures.
"""

import pytest
from dataclasses import FrozenInstanceError
from typing import Dict, Any

from scripts.deforum_helpers.data_models import (
    AnimationArgs, DeforumArgs, VideoArgs, ParseqArgs, WanArgs, RootArgs,
    AnimationMode, ColorCoherence, BorderMode, PaddingMode, SamplingMode, DepthAlgorithm,
    validate_schedule_string, validate_positive_int, validate_non_negative_number, validate_range,
    create_animation_args_from_dict, create_deforum_args_from_dict, create_video_args_from_dict,
    create_parseq_args_from_dict, create_wan_args_from_dict, create_root_args_from_dict
)


class TestEnums:
    """Test enum definitions"""
    
    def test_animation_mode_values(self):
        """Test AnimationMode enum has correct values"""
        assert AnimationMode.TWO_D.value == "2D"
        assert AnimationMode.THREE_D.value == "3D"
        assert AnimationMode.INTERPOLATION.value == "Interpolation"
        assert AnimationMode.WAN_VIDEO.value == "Wan Video"
    
    def test_color_coherence_values(self):
        """Test ColorCoherence enum has correct values"""
        assert ColorCoherence.NONE.value == "None"
        assert ColorCoherence.HSV.value == "HSV"
        assert ColorCoherence.LAB.value == "LAB"
        assert ColorCoherence.RGB.value == "RGB"
        assert ColorCoherence.IMAGE.value == "Image"
    
    def test_border_mode_values(self):
        """Test BorderMode enum has correct values"""
        assert BorderMode.REPLICATE.value == "replicate"
        assert BorderMode.WRAP.value == "wrap"
    
    def test_padding_mode_values(self):
        """Test PaddingMode enum has correct values"""
        assert PaddingMode.BORDER.value == "border"
        assert PaddingMode.REFLECTION.value == "reflection"
        assert PaddingMode.ZEROS.value == "zeros"
    
    def test_sampling_mode_values(self):
        """Test SamplingMode enum has correct values"""
        assert SamplingMode.BICUBIC.value == "bicubic"
        assert SamplingMode.BILINEAR.value == "bilinear"
        assert SamplingMode.NEAREST.value == "nearest"
    
    def test_depth_algorithm_values(self):
        """Test DepthAlgorithm enum has correct values"""
        assert DepthAlgorithm.DEPTH_ANYTHING_V2_SMALL.value == "Depth-Anything-V2-Small"
        assert DepthAlgorithm.MIDAS_3_HYBRID.value == "Midas-3-Hybrid"


class TestValidationFunctions:
    """Test validation utility functions"""
    
    def test_validate_schedule_string_valid(self):
        """Test valid schedule strings pass validation"""
        valid_schedules = [
            "0:(0)",
            "0:(1.0)",
            "0:(0), 30:(10)",
            "0:(0), 15:(5), 30:(10), 45:(5), 60:(0)",
            "0:(sin(3.14*t/120))",
            "0:(1.0+0.002*sin(1.25*3.14*t/120))",
            '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)',  # From actual Deforum usage
            "0:(\"test\")",  # Quoted strings
            "0:()",  # Empty expression (should be allowed)
        ]
        
        for schedule in valid_schedules:
            # Should not raise
            validate_schedule_string(schedule)
    
    def test_validate_schedule_string_invalid(self):
        """Test invalid schedule strings raise ValueError"""
        invalid_schedules = [
            "",
            "invalid",
            "0:0",  # Missing parentheses
            "0:(0,",  # Unclosed parentheses
            "0:(0) 30:(10)",  # Missing comma
            ":(0)",  # Missing frame number
            None,
            123
        ]
        
        for schedule in invalid_schedules:
            with pytest.raises(ValueError, match="must follow format"):
                validate_schedule_string(schedule)
    
    def test_validate_positive_int_valid(self):
        """Test valid positive integers pass validation"""
        valid_values = [1, 5, 100, 1000]
        for value in valid_values:
            validate_positive_int(value)
    
    def test_validate_positive_int_invalid(self):
        """Test invalid positive integers raise ValueError"""
        invalid_values = [0, -1, -10, 0.5, 1.0, "5", None]
        for value in invalid_values:
            with pytest.raises(ValueError, match="must be a positive integer"):
                validate_positive_int(value)
    
    def test_validate_non_negative_number_valid(self):
        """Test valid non-negative numbers pass validation"""
        valid_values = [0, 0.0, 1, 1.5, 100.5]
        for value in valid_values:
            validate_non_negative_number(value)
    
    def test_validate_non_negative_number_invalid(self):
        """Test invalid non-negative numbers raise ValueError"""
        invalid_values = [-1, -0.1, -100, "0", None]
        for value in invalid_values:
            with pytest.raises(ValueError, match="must be non-negative"):
                validate_non_negative_number(value)
    
    def test_validate_range_valid(self):
        """Test valid range values pass validation"""
        validate_range(5, 0, 10)
        validate_range(0.5, 0.0, 1.0)
        validate_range(0, 0, 0)  # Boundary case
    
    def test_validate_range_invalid(self):
        """Test invalid range values raise ValueError"""
        with pytest.raises(ValueError, match="must be between"):
            validate_range(11, 0, 10)
        
        with pytest.raises(ValueError, match="must be between"):
            validate_range(-1, 0, 10)
        
        with pytest.raises(ValueError, match="must be between"):
            validate_range(1.5, 0.0, 1.0)


class TestAnimationArgs:
    """Test AnimationArgs dataclass"""
    
    def test_default_creation(self):
        """Test creating AnimationArgs with defaults"""
        args = AnimationArgs()
        assert args.animation_mode == AnimationMode.THREE_D
        assert args.max_frames == 333
        assert args.border == BorderMode.REPLICATE
        assert args.translation_x == "0: (0)"
        assert args.zoom == "0: (1.0)"
    
    def test_custom_creation(self):
        """Test creating AnimationArgs with custom values"""
        args = AnimationArgs(
            animation_mode=AnimationMode.TWO_D,
            max_frames=120,
            translation_x="0:(0), 30:(10)",
            zoom="0:(1.0), 60:(1.2)"
        )
        assert args.animation_mode == AnimationMode.TWO_D
        assert args.max_frames == 120
        assert args.translation_x == "0:(0), 30:(10)"
        assert args.zoom == "0:(1.0), 60:(1.2)"
    
    def test_immutability(self):
        """Test that AnimationArgs is immutable"""
        args = AnimationArgs()
        with pytest.raises(FrozenInstanceError):
            args.max_frames = 100
        
        with pytest.raises(FrozenInstanceError):
            args.translation_x = "0:(5)"
    
    def test_validation_positive_max_frames(self):
        """Test max_frames validation"""
        # Valid
        AnimationArgs(max_frames=1)
        AnimationArgs(max_frames=1000)
        
        # Invalid
        with pytest.raises(ValueError, match="max_frames must be a positive integer"):
            AnimationArgs(max_frames=0)
        
        with pytest.raises(ValueError, match="max_frames must be a positive integer"):
            AnimationArgs(max_frames=-1)
    
    def test_validation_schedule_strings(self):
        """Test schedule string validation"""
        # Valid
        AnimationArgs(translation_x="0:(0), 30:(10)")
        AnimationArgs(zoom="0:(1.0)")
        
        # Invalid
        with pytest.raises(ValueError, match="translation_x must follow format"):
            AnimationArgs(translation_x="invalid_schedule")
        
        with pytest.raises(ValueError, match="zoom must follow format"):
            AnimationArgs(zoom="0:1.0")  # Missing parentheses
    
    def test_validation_numerical_ranges(self):
        """Test numerical range validation"""
        # Valid midas_weight
        AnimationArgs(midas_weight=0.5)
        AnimationArgs(midas_weight=-1.0)
        AnimationArgs(midas_weight=1.0)
        
        # Invalid midas_weight
        with pytest.raises(ValueError, match="midas_weight must be between"):
            AnimationArgs(midas_weight=1.5)
        
        with pytest.raises(ValueError, match="midas_weight must be between"):
            AnimationArgs(midas_weight=-1.5)
        
        # Valid perlin_persistence
        AnimationArgs(perlin_persistence=0.0)
        AnimationArgs(perlin_persistence=0.5)
        AnimationArgs(perlin_persistence=1.0)
        
        # Invalid perlin_persistence
        with pytest.raises(ValueError, match="perlin_persistence must be between"):
            AnimationArgs(perlin_persistence=1.5)


class TestDeforumArgs:
    """Test DeforumArgs dataclass"""
    
    def test_default_creation(self):
        """Test creating DeforumArgs with defaults"""
        args = DeforumArgs()
        assert args.W == 512
        assert args.H == 512
        assert args.seed == -1
        assert args.sampler == "Euler a"
        assert args.steps == 25
        assert args.scale == 7.0
        assert args.strength == 0.65
        assert args.batch_name == "Deforum"
    
    def test_custom_creation(self):
        """Test creating DeforumArgs with custom values"""
        args = DeforumArgs(
            W=1024,
            H=768,
            seed=12345,
            steps=50,
            scale=10.0,
            strength=0.8
        )
        assert args.W == 1024
        assert args.H == 768
        assert args.seed == 12345
        assert args.steps == 50
        assert args.scale == 10.0
        assert args.strength == 0.8
    
    def test_immutability(self):
        """Test that DeforumArgs is immutable"""
        args = DeforumArgs()
        with pytest.raises(FrozenInstanceError):
            args.W = 1024
        
        with pytest.raises(FrozenInstanceError):
            args.seed = 12345
    
    def test_validation_positive_dimensions(self):
        """Test dimension validation"""
        # Valid
        DeforumArgs(W=512, H=512)
        DeforumArgs(W=1024, H=768)
        
        # Invalid
        with pytest.raises(ValueError, match="W must be a positive integer"):
            DeforumArgs(W=0)
        
        with pytest.raises(ValueError, match="H must be a positive integer"):
            DeforumArgs(H=-1)
    
    def test_validation_ranges(self):
        """Test range validation"""
        # Valid scale
        DeforumArgs(scale=1.0)
        DeforumArgs(scale=15.0)
        DeforumArgs(scale=30.0)
        
        # Invalid scale
        with pytest.raises(ValueError, match="scale must be between"):
            DeforumArgs(scale=0.5)
        
        with pytest.raises(ValueError, match="scale must be between"):
            DeforumArgs(scale=35.0)
        
        # Valid strength
        DeforumArgs(strength=0.0)
        DeforumArgs(strength=0.5)
        DeforumArgs(strength=1.0)
        
        # Invalid strength
        with pytest.raises(ValueError, match="strength must be between"):
            DeforumArgs(strength=-0.1)
        
        with pytest.raises(ValueError, match="strength must be between"):
            DeforumArgs(strength=1.5)


class TestVideoArgs:
    """Test VideoArgs dataclass"""
    
    def test_default_creation(self):
        """Test creating VideoArgs with defaults"""
        args = VideoArgs()
        assert args.fps == 30
        assert args.output_format == "mp4"
        assert args.ffmpeg_crf == 17
        assert args.ffmpeg_preset == "slow"
    
    def test_custom_creation(self):
        """Test creating VideoArgs with custom values"""
        args = VideoArgs(
            fps=60,
            output_format="avi",
            ffmpeg_crf=20,
            ffmpeg_preset="fast"
        )
        assert args.fps == 60
        assert args.output_format == "avi"
        assert args.ffmpeg_crf == 20
        assert args.ffmpeg_preset == "fast"
    
    def test_immutability(self):
        """Test that VideoArgs is immutable"""
        args = VideoArgs()
        with pytest.raises(FrozenInstanceError):
            args.fps = 60
    
    def test_validation_fps(self):
        """Test FPS validation"""
        # Valid
        VideoArgs(fps=1)
        VideoArgs(fps=60)
        
        # Invalid
        with pytest.raises(ValueError, match="fps must be a positive integer"):
            VideoArgs(fps=0)
        
        with pytest.raises(ValueError, match="fps must be a positive integer"):
            VideoArgs(fps=-1)
    
    def test_validation_ffmpeg_crf(self):
        """Test FFmpeg CRF validation"""
        # Valid
        VideoArgs(ffmpeg_crf=0)
        VideoArgs(ffmpeg_crf=17)
        VideoArgs(ffmpeg_crf=51)
        
        # Invalid
        with pytest.raises(ValueError, match="ffmpeg_crf must be between"):
            VideoArgs(ffmpeg_crf=-1)
        
        with pytest.raises(ValueError, match="ffmpeg_crf must be between"):
            VideoArgs(ffmpeg_crf=52)


class TestParseqArgs:
    """Test ParseqArgs dataclass"""
    
    def test_default_creation(self):
        """Test creating ParseqArgs with defaults"""
        args = ParseqArgs()
        assert args.parseq_manifest is None
        assert args.parseq_use_deltas is True
        assert args.parseq_non_schedule_overrides is True
    
    def test_custom_creation(self):
        """Test creating ParseqArgs with custom values"""
        args = ParseqArgs(
            parseq_manifest='{"test": "manifest"}',
            parseq_use_deltas=False,
            parseq_non_schedule_overrides=False
        )
        assert args.parseq_manifest == '{"test": "manifest"}'
        assert args.parseq_use_deltas is False
        assert args.parseq_non_schedule_overrides is False
    
    def test_immutability(self):
        """Test that ParseqArgs is immutable"""
        args = ParseqArgs()
        with pytest.raises(FrozenInstanceError):
            args.parseq_use_deltas = False
    
    def test_validation_manifest_json(self):
        """Test manifest validation for JSON"""
        # Valid JSON
        ParseqArgs(parseq_manifest='{"test": "manifest"}')
        
        # Valid URL
        ParseqArgs(parseq_manifest='https://example.com/manifest.json')
        
        # Valid None/empty
        ParseqArgs(parseq_manifest=None)
        ParseqArgs(parseq_manifest="")
        ParseqArgs(parseq_manifest="   ")
        
        # Invalid
        with pytest.raises(ValueError, match="parseq_manifest must be valid JSON or URL"):
            ParseqArgs(parseq_manifest="invalid_manifest")


class TestWanArgs:
    """Test WanArgs dataclass"""
    
    def test_default_creation(self):
        """Test creating WanArgs with defaults"""
        args = WanArgs()
        assert args.wan_mode == "Disabled"
        assert args.wan_model_path == "models/wan"
        assert args.wan_movement_sensitivity == 1.0
        assert args.wan_guidance_scale == 7.5
        assert args.wan_frame_overlap == 2
    
    def test_custom_creation(self):
        """Test creating WanArgs with custom values"""
        args = WanArgs(
            wan_mode="T2V Only",
            wan_movement_sensitivity=2.0,
            wan_guidance_scale=10.0,
            wan_frame_overlap=5
        )
        assert args.wan_mode == "T2V Only"
        assert args.wan_movement_sensitivity == 2.0
        assert args.wan_guidance_scale == 10.0
        assert args.wan_frame_overlap == 5
    
    def test_immutability(self):
        """Test that WanArgs is immutable"""
        args = WanArgs()
        with pytest.raises(FrozenInstanceError):
            args.wan_mode = "T2V Only"
    
    def test_validation_ranges(self):
        """Test range validation for Wan parameters"""
        # Valid movement sensitivity
        WanArgs(wan_movement_sensitivity=0.1)
        WanArgs(wan_movement_sensitivity=5.0)
        
        # Invalid movement sensitivity
        with pytest.raises(ValueError, match="wan_movement_sensitivity must be between"):
            WanArgs(wan_movement_sensitivity=0.05)
        
        with pytest.raises(ValueError, match="wan_movement_sensitivity must be between"):
            WanArgs(wan_movement_sensitivity=6.0)
        
        # Valid guidance scale
        WanArgs(wan_guidance_scale=1.0)
        WanArgs(wan_guidance_scale=20.0)
        
        # Invalid guidance scale
        with pytest.raises(ValueError, match="wan_guidance_scale must be between"):
            WanArgs(wan_guidance_scale=0.5)
        
        with pytest.raises(ValueError, match="wan_guidance_scale must be between"):
            WanArgs(wan_guidance_scale=25.0)
    
    def test_validation_choices(self):
        """Test choice validation for Wan parameters"""
        # Valid modes
        WanArgs(wan_mode="Disabled")
        WanArgs(wan_mode="T2V Only")
        WanArgs(wan_mode="I2V Chaining")
        
        # Invalid mode
        with pytest.raises(ValueError, match="wan_mode must be one of"):
            WanArgs(wan_mode="Invalid Mode")
        
        # Valid languages
        WanArgs(wan_qwen_language="English")
        WanArgs(wan_qwen_language="Chinese")
        
        # Invalid language
        with pytest.raises(ValueError, match="wan_qwen_language must be one of"):
            WanArgs(wan_qwen_language="Spanish")


class TestRootArgs:
    """Test RootArgs dataclass"""
    
    def test_default_creation(self):
        """Test creating RootArgs with defaults"""
        args = RootArgs()
        assert args.device is None
        assert args.models_path == ""
        assert args.half_precision is True
        assert args.mask_preset_names == ("everywhere", "video_mask")
        assert args.frames_cache == ()
        assert args.animation_prompts == {}
        assert args.prompt_keyframes == []
    
    def test_custom_creation(self):
        """Test creating RootArgs with custom values"""
        args = RootArgs(
            device="cuda",
            models_path="/path/to/models",
            half_precision=False,
            animation_prompts={"0": "test prompt"},
            prompt_keyframes=["0"]
        )
        assert args.device == "cuda"
        assert args.models_path == "/path/to/models"
        assert args.half_precision is False
        assert args.animation_prompts == {"0": "test prompt"}
        assert args.prompt_keyframes == ["0"]
    
    def test_immutability(self):
        """Test that RootArgs is immutable"""
        args = RootArgs()
        with pytest.raises(FrozenInstanceError):
            args.device = "cuda"
        
        # Note: The contained dicts/lists are still mutable, but the reference is immutable
        args.animation_prompts["0"] = "test"  # This is allowed
        with pytest.raises(FrozenInstanceError):
            args.animation_prompts = {"new": "dict"}  # This is not


class TestHelperFunctions:
    """Test helper functions for creating data models from dictionaries"""
    
    def test_create_animation_args_from_dict(self):
        """Test creating AnimationArgs from dictionary"""
        data = {
            "animation_mode": "2D",
            "max_frames": 120,
            "translation_x": "0:(0), 30:(10)",
            "color_coherence": "HSV",
            "border": "wrap",
            "extra_field": "ignored"  # Should be filtered out
        }
        
        args = create_animation_args_from_dict(data)
        assert args.animation_mode == AnimationMode.TWO_D
        assert args.max_frames == 120
        assert args.translation_x == "0:(0), 30:(10)"
        assert args.color_coherence == ColorCoherence.HSV
        assert args.border == BorderMode.WRAP
        # extra_field should be ignored
    
    def test_create_deforum_args_from_dict(self):
        """Test creating DeforumArgs from dictionary"""
        data = {
            "W": 1024,
            "H": 768,
            "seed": 12345,
            "sampler": "Euler",
            "steps": 50,
            "scale": 10.0,
            "extra_field": "ignored"
        }
        
        args = create_deforum_args_from_dict(data)
        assert args.W == 1024
        assert args.H == 768
        assert args.seed == 12345
        assert args.sampler == "Euler"
        assert args.steps == 50
        assert args.scale == 10.0
    
    def test_create_video_args_from_dict(self):
        """Test creating VideoArgs from dictionary"""
        data = {
            "fps": 60,
            "output_format": "avi",
            "ffmpeg_crf": 20,
            "extra_field": "ignored"
        }
        
        args = create_video_args_from_dict(data)
        assert args.fps == 60
        assert args.output_format == "avi"
        assert args.ffmpeg_crf == 20
    
    def test_create_parseq_args_from_dict(self):
        """Test creating ParseqArgs from dictionary"""
        data = {
            "parseq_manifest": '{"test": "manifest"}',
            "parseq_use_deltas": False,
            "parseq_non_schedule_overrides": False,
            "extra_field": "ignored"
        }
        
        args = create_parseq_args_from_dict(data)
        assert args.parseq_manifest == '{"test": "manifest"}'
        assert args.parseq_use_deltas is False
        assert args.parseq_non_schedule_overrides is False
    
    def test_create_wan_args_from_dict(self):
        """Test creating WanArgs from dictionary"""
        data = {
            "wan_mode": "T2V Only",
            "wan_movement_sensitivity": 2.0,
            "wan_guidance_scale": 10.0,
            "extra_field": "ignored"
        }
        
        args = create_wan_args_from_dict(data)
        assert args.wan_mode == "T2V Only"
        assert args.wan_movement_sensitivity == 2.0
        assert args.wan_guidance_scale == 10.0
    
    def test_create_root_args_from_dict(self):
        """Test creating RootArgs from dictionary"""
        data = {
            "device": "cuda",
            "models_path": "/path/to/models",
            "mask_preset_names": ["everywhere", "video_mask"],  # List -> Tuple
            "frames_cache": [1, 2, 3],  # List -> Tuple
            "animation_prompts": {"0": "test"},
            "prompt_keyframes": ["0"],
            "extra_field": "ignored"
        }
        
        args = create_root_args_from_dict(data)
        assert args.device == "cuda"
        assert args.models_path == "/path/to/models"
        assert args.mask_preset_names == ("everywhere", "video_mask")  # Converted to tuple
        assert args.frames_cache == (1, 2, 3)  # Converted to tuple
        assert args.animation_prompts == {"0": "test"}
        assert args.prompt_keyframes == ["0"]
    
    def test_enum_conversion_errors(self):
        """Test that invalid enum values raise errors"""
        with pytest.raises(ValueError):
            create_animation_args_from_dict({"animation_mode": "Invalid Mode"})
        
        with pytest.raises(ValueError):
            create_animation_args_from_dict({"color_coherence": "Invalid Coherence"})
        
        with pytest.raises(ValueError):
            create_animation_args_from_dict({"border": "invalid_border"})


class TestIntegrationWithLegacyCode:
    """Test integration scenarios with legacy code patterns"""
    
    def test_backwards_compatibility_field_names(self):
        """Test that all field names match legacy dictionary keys"""
        # This ensures our dataclasses have the same field names as the original dictionaries
        
        # Test a few key fields from each args type
        args = AnimationArgs()
        assert hasattr(args, 'max_frames')
        assert hasattr(args, 'translation_x')
        assert hasattr(args, 'animation_mode')
        assert hasattr(args, 'color_coherence')
        
        deforum_args = DeforumArgs()
        assert hasattr(deforum_args, 'W')
        assert hasattr(deforum_args, 'H')
        assert hasattr(deforum_args, 'seed')
        assert hasattr(deforum_args, 'sampler')
        
        video_args = VideoArgs()
        assert hasattr(video_args, 'fps')
        assert hasattr(video_args, 'output_format')
        
        wan_args = WanArgs()
        assert hasattr(wan_args, 'wan_mode')
        assert hasattr(wan_args, 'wan_movement_sensitivity')
    
    def test_default_values_match_legacy(self):
        """Test that default values match legacy defaults"""
        args = AnimationArgs()
        assert args.max_frames == 333  # Legacy default
        assert args.translation_x == "0: (0)"  # Legacy default
        assert args.zoom == "0: (1.0)"  # Legacy default
        
        deforum_args = DeforumArgs()
        assert deforum_args.W == 512  # Legacy default
        assert deforum_args.H == 512  # Legacy default
        assert deforum_args.steps == 25  # Legacy default
        assert deforum_args.scale == 7.0  # Legacy default
    
    def test_enum_string_values_match_legacy(self):
        """Test that enum string values match legacy string choices"""
        # Animation mode strings should match UI choices
        assert AnimationMode.TWO_D.value == "2D"
        assert AnimationMode.THREE_D.value == "3D"
        assert AnimationMode.INTERPOLATION.value == "Interpolation"
        assert AnimationMode.WAN_VIDEO.value == "Wan Video"
        
        # Color coherence strings should match UI choices
        assert ColorCoherence.NONE.value == "None"
        assert ColorCoherence.HSV.value == "HSV"
        assert ColorCoherence.LAB.value == "LAB" 