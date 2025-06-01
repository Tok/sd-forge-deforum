"""
Unit tests for the functional configuration system.

Tests the immutable dataclasses, pure conversion functions, and processing logic
while ensuring backward compatibility with the legacy args system.
"""

import pytest
import json
from types import SimpleNamespace
from dataclasses import asdict

# Import with fallback for missing dependencies
try:
    from scripts.deforum_helpers.config import (
        # Models
        DeforumGenerationArgs, DeforumAnimationArgs, DeforumVideoArgs,
        ParseqArgs, WanArgs, RootArgs, ProcessedArguments,
        
        # Conversion functions
        create_deforum_args_from_dict, create_animation_args_from_dict,
        create_video_args_from_dict, create_parseq_args_from_dict,
        create_wan_args_from_dict, create_root_args_from_dict,
        to_legacy_dict, to_legacy_namespace,
        
        # Processing functions
        process_arguments, validate_all_arguments,
        merge_arguments, apply_argument_overrides,
    )

    from scripts.deforum_helpers.config.argument_models import (
        SamplerType, SchedulerType, SeedBehavior, NoiseType, MaskFill
    )

    # Import data models with fallback
    try:
        from scripts.deforum_helpers.data_models import (
            AnimationMode, BorderMode, ColorCoherence
        )
    except ImportError:
        # Create minimal enums for testing if data_models is not available
        from enum import Enum
        
        class AnimationMode(Enum):
            TWO_D = "2D"
            THREE_D = "3D"
            INTERPOLATION = "Interpolation"
            WAN_VIDEO = "Wan Video"
        
        class BorderMode(Enum):
            REPLICATE = "replicate"
            WRAP = "wrap"
        
        class ColorCoherence(Enum):
            NONE = "None"
            HSV = "HSV"
            LAB = "LAB"
            RGB = "RGB"
            IMAGE = "Image"

    CONFIG_AVAILABLE = True
except ImportError as e:
    CONFIG_AVAILABLE = False
    print(f"Config system not available for testing: {e}")


# Skip all tests if config system is not available
pytestmark = pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config system dependencies not available")


class TestDataStructureImmutability:
    """Test that all data structures are properly immutable"""
    
    def test_deforum_args_immutability(self):
        """Test that DeforumGenerationArgs is immutable"""
        args = DeforumGenerationArgs(width=512, height=512)
        
        with pytest.raises(Exception):  # Should raise FrozenInstanceError
            args.width = 1024
        
        assert args.width == 512  # Value should remain unchanged
    
    def test_animation_args_immutability(self):
        """Test that DeforumAnimationArgs is immutable"""
        args = DeforumAnimationArgs(max_frames=100)
        
        with pytest.raises(Exception):
            args.max_frames = 200
        
        assert args.max_frames == 100
    
    def test_processed_arguments_immutability(self):
        """Test that ProcessedArguments is immutable"""
        deforum_args = DeforumGenerationArgs()
        animation_args = DeforumAnimationArgs()
        video_args = DeforumVideoArgs()
        parseq_args = ParseqArgs()
        wan_args = WanArgs()
        root_args = RootArgs()
        
        processed = ProcessedArguments(
            deforum=deforum_args,
            animation=animation_args,
            video=video_args,
            parseq=parseq_args,
            wan=wan_args,
            root=root_args
        )
        
        with pytest.raises(Exception):
            processed.deforum = DeforumGenerationArgs(width=1024)


class TestArgumentValidation:
    """Test validation logic in argument models"""
    
    def test_deforum_args_validation_success(self):
        """Test successful validation of DeforumGenerationArgs"""
        args = DeforumGenerationArgs(
            width=1024,
            height=768,
            steps=20,
            cfg_scale=7.0,
            strength=0.8
        )
        # If we get here without exception, validation passed
        assert args.width == 1024
        assert args.height == 768
    
    def test_deforum_args_validation_failures(self):
        """Test validation failures in DeforumGenerationArgs"""
        # Invalid width (not multiple of 8)
        with pytest.raises(ValueError):
            DeforumGenerationArgs(width=1023)
        
        # Invalid height (not multiple of 8)
        with pytest.raises(ValueError):
            DeforumGenerationArgs(height=767)
        
        # Invalid cfg_scale
        with pytest.raises(ValueError):
            DeforumGenerationArgs(cfg_scale=50.0)
    
    def test_animation_args_validation_success(self):
        """Test successful validation of DeforumAnimationArgs"""
        args = DeforumAnimationArgs(
            max_frames=100,
            shake_intensity=1.5,
            diffusion_cadence=5
        )
        assert args.max_frames == 100
        assert args.shake_intensity == 1.5
    
    def test_animation_args_validation_failures(self):
        """Test validation failures in DeforumAnimationArgs"""
        # Invalid shake_intensity
        with pytest.raises(ValueError):
            DeforumAnimationArgs(shake_intensity=5.0)
        
        # Invalid max_frames
        with pytest.raises(ValueError):
            DeforumAnimationArgs(max_frames=0)


class TestEnumHandling:
    """Test proper enum handling and type safety"""
    
    def test_sampler_enum_handling(self):
        """Test SamplerType enum handling"""
        args = DeforumGenerationArgs(sampler=SamplerType.EULER_A)
        assert args.sampler == SamplerType.EULER_A
        assert args.sampler.value == "Euler a"
    
    def test_animation_mode_enum_handling(self):
        """Test AnimationMode enum handling"""
        args = DeforumAnimationArgs(animation_mode=AnimationMode.TWO_D)
        assert args.animation_mode == AnimationMode.TWO_D
        assert args.animation_mode.value == "2D"
    
    def test_seed_behavior_enum_handling(self):
        """Test SeedBehavior enum handling"""
        args = DeforumGenerationArgs(seed_behavior=SeedBehavior.RANDOM)
        assert args.seed_behavior == SeedBehavior.RANDOM
        assert args.seed_behavior.value == "random"


class TestLegacyConversion:
    """Test conversion between new and legacy formats"""
    
    def test_deforum_args_from_legacy_dict(self):
        """Test conversion from legacy dictionary to DeforumGenerationArgs"""
        legacy_dict = {
            "W": 1024,
            "H": 768,
            "seed": 123456,
            "sampler": "Euler a",
            "scale": 8.0,
            "steps": 25,
            "use_init": True,
            "strength": 0.7
        }
        
        args = create_deforum_args_from_dict(legacy_dict)
        
        assert args.width == 1024
        assert args.height == 768
        assert args.seed == 123456
        assert args.sampler == SamplerType.EULER_A
        assert args.cfg_scale == 8.0
        assert args.steps == 25
        assert args.use_init == True
        assert args.strength == 0.7
    
    def test_animation_args_from_legacy_dict(self):
        """Test conversion from legacy dictionary to DeforumAnimationArgs"""
        legacy_dict = {
            "animation_mode": "3D",
            "max_frames": 150,
            "border": "wrap",
            "angle": "0: (5.0)",
            "zoom": "0: (1.02)",
            "color_coherence": "HSV",
            "noise_type": "uniform"
        }
        
        args = create_animation_args_from_dict(legacy_dict)
        
        assert args.animation_mode == AnimationMode.THREE_D
        assert args.max_frames == 150
        assert args.border == BorderMode.WRAP
        assert args.angle == "0: (5.0)"
        assert args.zoom == "0: (1.02)"
        assert args.color_coherence == ColorCoherence.HSV
        assert args.noise_type == NoiseType.UNIFORM
    
    def test_to_legacy_dict_conversion(self):
        """Test conversion from new format back to legacy dictionary"""
        deforum_args = DeforumGenerationArgs(width=1024, height=768, cfg_scale=8.0)
        animation_args = DeforumAnimationArgs(max_frames=100, animation_mode=AnimationMode.TWO_D)
        video_args = DeforumVideoArgs(fps=30)
        parseq_args = ParseqArgs()
        wan_args = WanArgs()
        root_args = RootArgs()
        
        processed = ProcessedArguments(
            deforum=deforum_args,
            animation=animation_args,
            video=video_args,
            parseq=parseq_args,
            wan=wan_args,
            root=root_args
        )
        
        legacy_dict = to_legacy_dict(processed)
        
        # Check legacy field mappings
        assert legacy_dict["W"] == 1024
        assert legacy_dict["H"] == 768
        assert legacy_dict["scale"] == 8.0
        assert legacy_dict["max_frames"] == 100
        assert legacy_dict["animation_mode"] == "2D"
        assert legacy_dict["fps"] == 30
    
    def test_to_legacy_namespace_conversion(self):
        """Test conversion to legacy SimpleNamespace"""
        deforum_args = DeforumGenerationArgs(width=512, height=512)
        animation_args = DeforumAnimationArgs()
        video_args = DeforumVideoArgs()
        parseq_args = ParseqArgs()
        wan_args = WanArgs()
        root_args = RootArgs()
        
        processed = ProcessedArguments(
            deforum=deforum_args,
            animation=animation_args,
            video=video_args,
            parseq=parseq_args,
            wan=wan_args,
            root=root_args
        )
        
        legacy_ns = to_legacy_namespace(processed)
        
        assert isinstance(legacy_ns, SimpleNamespace)
        assert legacy_ns.W == 512
        assert legacy_ns.H == 512
        assert hasattr(legacy_ns, "max_frames")


class TestArgumentProcessing:
    """Test pure argument processing functions"""
    
    def test_process_arguments_basic(self):
        """Test basic argument processing"""
        args_dict = {
            "W": 1024,
            "H": 768,
            "max_frames": 100,
            "fps": 30
        }
        
        processed = process_arguments(args_dict)
        
        assert isinstance(processed, ProcessedArguments)
        assert processed.deforum.width == 1024
        assert processed.deforum.height == 768
        assert processed.animation.max_frames == 100
        assert processed.video.fps == 30
        assert processed.processing_time > 0
    
    def test_process_arguments_with_overrides(self):
        """Test argument processing with overrides"""
        base_dict = {
            "W": 512,
            "H": 512,
            "max_frames": 100
        }
        
        overrides = {
            "W": 1024,
            "max_frames": 200
        }
        
        processed = process_arguments(base_dict, overrides=overrides)
        
        assert processed.deforum.width == 1024  # Overridden
        assert processed.deforum.height == 512  # Not overridden
        assert processed.animation.max_frames == 200  # Overridden
        assert processed.applied_overrides == overrides
    
    def test_process_arguments_with_timestring(self):
        """Test argument processing with custom timestring"""
        args_dict = {"batch_name": "test_{timestring}"}
        timestring = "20241111123456"
        
        processed = process_arguments(args_dict, timestring=timestring)
        
        assert processed.root.timestring == timestring
        assert timestring in processed.deforum.batch_name


class TestArgumentValidationSystem:
    """Test the argument validation system"""
    
    def test_validate_all_arguments_success(self):
        """Test successful validation of all arguments"""
        processed = ProcessedArguments(
            deforum=DeforumGenerationArgs(),
            animation=DeforumAnimationArgs(),
            video=DeforumVideoArgs(),
            parseq=ParseqArgs(),
            wan=WanArgs(),
            root=RootArgs()
        )
        
        result = validate_all_arguments(processed)
        
        assert result.valid == True
        assert len(result.errors) == 0
        assert result.validated_args is not None
    
    def test_validate_cross_dependencies_failures(self):
        """Test cross-dependency validation failures"""
        # Create args with use_init=True but no init_image
        deforum_args = DeforumGenerationArgs(use_init=True, init_image=None)
        processed = ProcessedArguments(
            deforum=deforum_args,
            animation=DeforumAnimationArgs(),
            video=DeforumVideoArgs(),
            parseq=ParseqArgs(),
            wan=WanArgs(),
            root=RootArgs()
        )
        
        result = validate_all_arguments(processed)
        
        assert result.valid == False
        assert any("use_init is True but no init_image provided" in error for error in result.errors)
    
    def test_validation_warnings_generation(self):
        """Test generation of validation warnings"""
        # Create args that should generate warnings
        deforum_args = DeforumGenerationArgs(
            width=2048, height=2048,  # Large dimensions
            steps=5,  # Low steps
            cfg_scale=25.0  # High CFG
        )
        animation_args = DeforumAnimationArgs(max_frames=2000)  # Many frames
        
        processed = ProcessedArguments(
            deforum=deforum_args,
            animation=animation_args,
            video=DeforumVideoArgs(),
            parseq=ParseqArgs(),
            wan=WanArgs(),
            root=RootArgs()
        )
        
        result = validate_all_arguments(processed)
        
        assert len(result.warnings) > 0
        assert any("memory issues" in warning for warning in result.warnings)
        assert any("poor quality" in warning for warning in result.warnings)


class TestArgumentMerging:
    """Test argument merging functionality"""
    
    def test_merge_arguments_basic(self):
        """Test basic argument merging"""
        base_args = ProcessedArguments(
            deforum=DeforumGenerationArgs(width=512, height=512),
            animation=DeforumAnimationArgs(max_frames=100),
            video=DeforumVideoArgs(fps=30),
            parseq=ParseqArgs(),
            wan=WanArgs(),
            root=RootArgs()
        )
        
        override_args = ProcessedArguments(
            deforum=DeforumGenerationArgs(width=1024),  # Override width
            animation=DeforumAnimationArgs(max_frames=200),  # Override frames
            video=DeforumVideoArgs(),  # Use defaults
            parseq=ParseqArgs(),
            wan=WanArgs(),
            root=RootArgs()
        )
        
        merged = merge_arguments(base_args, override_args)
        
        assert merged.deforum.width == 1024  # Overridden
        assert merged.deforum.height == 512  # From base
        assert merged.animation.max_frames == 200  # Overridden
        assert merged.video.fps == 30  # From base
    
    def test_apply_argument_overrides(self):
        """Test applying specific overrides"""
        processed = ProcessedArguments(
            deforum=DeforumGenerationArgs(width=512, height=512),
            animation=DeforumAnimationArgs(max_frames=100),
            video=DeforumVideoArgs(),
            parseq=ParseqArgs(),
            wan=WanArgs(),
            root=RootArgs()
        )
        
        overrides = {
            "W": 1024,
            "max_frames": 200,
            "fps": 60
        }
        
        updated = apply_argument_overrides(processed, overrides)
        
        assert updated.deforum.width == 1024
        assert updated.animation.max_frames == 200
        assert updated.video.fps == 60


class TestFunctionalProgrammingPrinciples:
    """Test adherence to functional programming principles"""
    
    def test_function_purity_conversion(self):
        """Test that conversion functions are pure (same input -> same output)"""
        test_dict = {
            "W": 1024,
            "H": 768,
            "sampler": "Euler",
            "steps": 20
        }
        
        # Call function multiple times
        result1 = create_deforum_args_from_dict(test_dict)
        result2 = create_deforum_args_from_dict(test_dict)
        
        assert result1 == result2  # Same input should produce same output
        
        # Original dict should not be modified
        assert test_dict == {
            "W": 1024,
            "H": 768,
            "sampler": "Euler",
            "steps": 20
        }
    
    def test_function_purity_processing(self):
        """Test that processing functions are pure"""
        args_dict = {"W": 512, "H": 512, "max_frames": 100}
        
        # Process multiple times
        result1 = process_arguments(args_dict.copy(), timestring="20241111123456")
        result2 = process_arguments(args_dict.copy(), timestring="20241111123456")
        
        # Results should be equivalent (excluding processing_time)
        assert result1.deforum.width == result2.deforum.width
        assert result1.animation.max_frames == result2.animation.max_frames
        assert result1.root.timestring == result2.root.timestring
    
    def test_immutable_composition(self):
        """Test that functions compose properly with immutable data"""
        # Start with basic args
        args_dict = {"W": 512, "H": 512}
        processed1 = process_arguments(args_dict)
        
        # Apply overrides
        overrides = {"W": 1024, "max_frames": 200}
        processed2 = apply_argument_overrides(processed1, overrides)
        
        # Original should be unchanged
        assert processed1.deforum.width == 512
        assert processed1.animation.max_frames == 333  # Default
        
        # New version should have overrides
        assert processed2.deforum.width == 1024
        assert processed2.animation.max_frames == 200


class TestBackwardCompatibility:
    """Test backward compatibility with legacy systems"""
    
    def test_legacy_namespace_compatibility(self):
        """Test that legacy SimpleNamespace format is properly supported"""
        # Create legacy-style args
        legacy_args = SimpleNamespace(
            W=1024,
            H=768,
            seed=12345,
            sampler="Euler a",
            max_frames=150,
            animation_mode="3D"
        )
        
        # Convert to dict and process
        legacy_dict = vars(legacy_args)
        processed = process_arguments(legacy_dict)
        
        # Convert back to legacy format
        new_legacy = to_legacy_namespace(processed)
        
        # Should match original structure
        assert new_legacy.W == legacy_args.W
        assert new_legacy.H == legacy_args.H
        assert new_legacy.seed == legacy_args.seed
        assert new_legacy.max_frames == legacy_args.max_frames
    
    def test_legacy_dict_round_trip(self):
        """Test round-trip conversion legacy dict -> functional -> legacy dict"""
        original_dict = {
            "W": 1024,
            "H": 768,
            "seed": 42,
            "sampler": "Euler",
            "steps": 20,
            "scale": 7.5,
            "max_frames": 100,
            "animation_mode": "3D",
            "fps": 30
        }
        
        # Process through functional system
        processed = process_arguments(original_dict)
        
        # Convert back to legacy
        result_dict = to_legacy_dict(processed)
        
        # Key values should match
        assert result_dict["W"] == original_dict["W"]
        assert result_dict["H"] == original_dict["H"]
        assert result_dict["seed"] == original_dict["seed"]
        assert result_dict["steps"] == original_dict["steps"]
        assert result_dict["scale"] == original_dict["scale"]


class TestComplexScenarios:
    """Test complex real-world scenarios"""
    
    def test_complex_animation_setup(self):
        """Test complex animation configuration"""
        complex_dict = {
            "W": 1024,
            "H": 576,
            "animation_mode": "3D",
            "max_frames": 240,
            "angle": "0: (0), 50: (2.5), 100: (0), 150: (-1.8), 200: (0)",
            "zoom": "0: (1.0), 60: (1.02), 120: (1.0), 180: (0.98), 240: (1.0)",
            "translation_x": "0: (0), 40: (10), 80: (0), 120: (-15), 160: (0), 200: (8), 240: (0)",
            "use_depth_warping": True,
            "depth_algorithm": "Depth-Anything-V2-Small",
            "color_coherence": "LAB",
            "fps": 24,
            "add_soundtrack": "File",
            "wan_mode": "I2V Chaining",
            "wan_enable_movement_analysis": True
        }
        
        processed = process_arguments(complex_dict)
        
        # Validate complex setup
        assert processed.deforum.width == 1024
        assert processed.animation.animation_mode == AnimationMode.THREE_D
        assert processed.animation.max_frames == 240
        assert "50: (2.5)" in processed.animation.angle
        assert processed.animation.use_depth_warping == True
        assert processed.animation.color_coherence == ColorCoherence.LAB
        assert processed.video.fps == 24
        assert processed.wan.wan_mode == "I2V Chaining"
        assert processed.wan.wan_enable_movement_analysis == True
    
    def test_performance_validation(self):
        """Test that processing is performant"""
        large_dict = {
            "W": 1024,
            "H": 1024,
            "max_frames": 1000,
            "angle": "0: (0), " + ", ".join([f"{i}: ({i*0.1})" for i in range(1, 100)]),
            "zoom": "0: (1.0), " + ", ".join([f"{i}: ({1.0 + i*0.001})" for i in range(1, 100)])
        }
        
        processed = process_arguments(large_dict)
        
        # Should complete quickly even with complex schedules
        assert processed.processing_time < 1.0  # Less than 1 second
        assert processed.animation.max_frames == 1000
        assert len(processed.animation.angle) > 100  # Complex schedule preserved


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_config_system.py -v
    pytest.main([__file__, "-v"]) 