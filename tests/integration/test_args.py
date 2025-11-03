"""Unit tests for deforum.config.args module.

Tests argument configuration dictionaries and helper functions.
Most of this file is static configuration data, so we test:
- Dictionary structure validation
- Helper functions (pack_args, get_component_names)
- Simple aspects of process_args (complex logic requires integration tests)
"""

import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace


class TestArgumentDictionaries:
    """Test argument dictionary structure and defaults."""

    def test_root_args_returns_dict(self):
        """RootArgs should return dictionary with required keys."""
        from deforum.config.args import RootArgs

        root_args = RootArgs()

        assert isinstance(root_args, dict)
        assert 'device' in root_args
        assert 'models_path' in root_args
        assert 'frames_cache' in root_args
        assert 'tmp_deforum_run_duplicated_folder' in root_args

    def test_deforum_anim_args_returns_dict(self):
        """DeforumAnimArgs should return dictionary with animation settings."""
        from deforum.config.args import DeforumAnimArgs

        anim_args = DeforumAnimArgs()

        assert isinstance(anim_args, dict)
        # Check for key animation parameters
        assert 'render_mode' in anim_args
        assert 'animation_mode' in anim_args
        assert 'max_frames' in anim_args
        assert 'strength_schedule' in anim_args
        assert 'keyframe_strength_schedule' in anim_args

    def test_deforum_args_returns_dict(self):
        """DeforumArgs should return dictionary with main settings."""
        from deforum.config.args import DeforumArgs

        args = DeforumArgs()

        assert isinstance(args, dict)
        assert 'W' in args
        assert 'H' in args
        assert 'seed' in args
        assert 'sampler' in args
        assert 'steps' in args
        assert 'batch_name' in args

    def test_loop_args_returns_dict(self):
        """LoopArgs should return dictionary with guided image settings."""
        from deforum.config.args import LoopArgs

        loop_args = LoopArgs()

        assert isinstance(loop_args, dict)
        assert 'use_looper' in loop_args
        assert 'init_images' in loop_args

    def test_parseq_args_returns_dict(self):
        """ParseqArgs should return dictionary with Parseq settings."""
        from deforum.config.args import ParseqArgs

        parseq_args = ParseqArgs()

        assert isinstance(parseq_args, dict)
        assert 'parseq_manifest' in parseq_args
        assert 'parseq_use_deltas' in parseq_args

    def test_audio_sync_args_returns_dict(self):
        """AudioSyncArgs should return dictionary with audio sync settings."""
        from deforum.config.args import AudioSyncArgs

        audio_args = AudioSyncArgs()

        assert isinstance(audio_args, dict)
        assert 'enable_audio_sync' in audio_args
        assert 'audio_detection_method' in audio_args
        assert 'audio_sensitivity' in audio_args

    def test_wan_args_returns_dict(self):
        """WanArgs should return dictionary with Wan video settings."""
        from deforum.config.args import WanArgs

        wan_args = WanArgs()

        assert isinstance(wan_args, dict)
        assert 'wan_t2v_model' in wan_args
        assert 'wan_resolution' in wan_args
        assert 'wan_inference_steps' in wan_args

    def test_deforum_output_args_returns_dict(self):
        """DeforumOutputArgs should return dictionary with output settings."""
        from deforum.config.args import DeforumOutputArgs

        output_args = DeforumOutputArgs()

        assert isinstance(output_args, dict)
        assert 'fps' in output_args
        assert 'skip_video_creation' in output_args
        assert 'make_gif' in output_args


class TestComponentNames:
    """Test component name helper functions."""

    def test_get_component_names_includes_all_sections(self):
        """get_component_names should include all argument sections."""
        from deforum.config.args import get_component_names

        names = get_component_names()

        assert isinstance(names, list)
        # Should include animation_prompts
        assert 'animation_prompts' in names
        # Should include DeforumArgs keys
        assert 'W' in names
        assert 'H' in names
        assert 'seed' in names
        # Should include DeforumAnimArgs keys
        assert 'max_frames' in names
        assert 'strength_schedule' in names
        # Should include output args
        assert 'fps' in names
        # Should include Parseq args
        assert 'parseq_manifest' in names
        # Should include AudioSync args
        assert 'enable_audio_sync' in names
        # Should include Wan args
        assert 'wan_t2v_model' in names

    def test_get_settings_component_names_returns_list(self):
        """get_settings_component_names should return list of component names."""
        from deforum.config.args import get_settings_component_names

        names = get_settings_component_names()

        assert isinstance(names, list)
        assert len(names) > 0


class TestPackArgs:
    """Test pack_args helper function."""

    def test_pack_args_basic(self):
        """pack_args should filter dict by keys from function."""
        from deforum.config.args import pack_args

        args_dict = {
            'W': 1280,
            'H': 720,
            'seed': 42,
            'extra_key': 'should_be_ignored'
        }

        def keys_function():
            return ['W', 'H', 'seed']

        result = pack_args(args_dict, keys_function)

        assert result == {'W': 1280, 'H': 720, 'seed': 42}
        assert 'extra_key' not in result

    def test_pack_args_handles_missing_keys(self):
        """pack_args should gracefully handle missing keys."""
        from deforum.config.args import pack_args

        args_dict = {'W': 1280}

        def keys_function():
            return ['W', 'H', 'missing_key']

        result = pack_args(args_dict, keys_function)

        # Should only include keys that exist in args_dict
        assert result == {'W': 1280}
        assert 'H' not in result
        assert 'missing_key' not in result

    def test_pack_args_empty_dict(self):
        """pack_args should handle empty dict."""
        from deforum.config.args import pack_args

        args_dict = {}

        def keys_function():
            return ['W', 'H']

        result = pack_args(args_dict, keys_function)

        assert result == {}


class TestDefaultValues:
    """Test default values in argument dictionaries."""

    def test_render_mode_default(self):
        """Render mode should default to 'Keyframes Only'."""
        from deforum.config.args import DeforumAnimArgs

        anim_args = DeforumAnimArgs()
        assert anim_args['render_mode']['value'] == 'Keyframes Only'

    def test_animation_mode_default(self):
        """Animation mode should default to '3D'."""
        from deforum.config.args import DeforumAnimArgs

        anim_args = DeforumAnimArgs()
        assert anim_args['animation_mode']['value'] == '3D'

    def test_max_frames_default(self):
        """Max frames should default to 333."""
        from deforum.config.args import DeforumAnimArgs

        anim_args = DeforumAnimArgs()
        assert anim_args['max_frames']['value'] == 333

    def test_fps_default(self):
        """FPS should default to 60."""
        from deforum.config.args import DeforumOutputArgs

        output_args = DeforumOutputArgs()
        assert output_args['fps']['value'] == 60

    def test_steps_default(self):
        """Steps should default to 20."""
        from deforum.config.args import DeforumArgs

        args = DeforumArgs()
        assert args['steps']['value'] == 20

    def test_sampler_default(self):
        """Sampler should default to 'Euler'."""
        from deforum.config.args import DeforumArgs

        args = DeforumArgs()
        assert args['sampler']['value'] == 'Euler'

    def test_scheduler_default(self):
        """Scheduler should default to 'Simple'."""
        from deforum.config.args import DeforumArgs

        args = DeforumArgs()
        assert args['scheduler']['value'] == 'Simple'

    def test_resolution_defaults(self):
        """Width and Height should have sensible defaults."""
        from deforum.config.args import DeforumArgs

        args = DeforumArgs()
        assert args['W']['value'] == 1280
        assert args['H']['value'] == 720

    def test_strength_schedule_defaults(self):
        """Strength schedules should have correct defaults."""
        from deforum.config.args import DeforumAnimArgs

        anim_args = DeforumAnimArgs()
        # Normal/tween frames: high strength (more previous frame)
        assert '0.85' in anim_args['strength_schedule']['value']
        # Keyframes: low strength (more diffusion steps)
        assert '0.15' in anim_args['keyframe_strength_schedule']['value']

    def test_reverse_generation_default(self):
        """Reverse generation should default to False."""
        from deforum.config.args import DeforumAnimArgs

        anim_args = DeforumAnimArgs()
        assert anim_args['reverse_generation']['value'] is False

    def test_wan_guidance_scale_default(self):
        """Wan guidance scale should default to 3.5."""
        from deforum.config.args import WanArgs

        wan_args = WanArgs()
        assert wan_args['wan_guidance_scale']['value'] == 3.5

    def test_wan_flf2v_guidance_default(self):
        """Wan FLF2V guidance should default to 3.5."""
        from deforum.config.args import WanArgs

        wan_args = WanArgs()
        assert wan_args['wan_flf2v_guidance_scale']['value'] == 3.5


class TestInfoStrings:
    """Test that info strings are present for key parameters."""

    def test_render_mode_has_info(self):
        """Render mode should have descriptive info string."""
        from deforum.config.args import DeforumAnimArgs

        anim_args = DeforumAnimArgs()
        info = anim_args['render_mode']['info']
        assert len(info) > 0
        assert 'Classic 3D' in info
        assert 'New 3D' in info
        assert 'Keyframes Only' in info

    def test_strength_schedule_has_info(self):
        """Strength schedule should explain its purpose."""
        from deforum.config.args import DeforumAnimArgs

        anim_args = DeforumAnimArgs()
        normal_info = anim_args['strength_schedule']['info']
        keyframe_info = anim_args['keyframe_strength_schedule']['info']

        assert len(normal_info) > 0
        assert len(keyframe_info) > 0
        # Should mention strength values
        assert '0-1' in normal_info or '0.0-1.0' in keyframe_info

    def test_reverse_generation_has_info(self):
        """Reverse generation should explain POV/zoom-in use case."""
        from deforum.config.args import DeforumAnimArgs

        anim_args = DeforumAnimArgs()
        info = anim_args['reverse_generation']['info']

        assert len(info) > 0
        assert 'reverse' in info.lower() or 'forward' in info.lower()

    def test_wan_flf2v_guidance_has_warning(self):
        """FLF2V guidance should warn about 0.0 breaking interpolation."""
        from deforum.config.args import WanArgs

        wan_args = WanArgs()
        info = wan_args['wan_flf2v_guidance_scale']['info']

        assert len(info) > 0
        assert '0.0' in info or 'NEVER' in info


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
