"""Unit tests for wan_button_handler helpers."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to allow direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import directly from the module file to avoid Forge dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "wan_button_handler",
    Path(__file__).parent.parent.parent.parent / "deforum" / "ui" / "handlers" / "wan_button_handler.py"
)
wan_button_handler = importlib.util.module_from_spec(spec)

# Mock the logger before loading the module
class MockLogger:
    def error(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass

# Mock emoji_utils
class MockEmojiUtils:
    @staticmethod
    def maybe_check(): return '✓'
    @staticmethod
    def maybe_cross(): return '✗'
    @staticmethod
    def maybe_warning(): return '⚠'
    @staticmethod
    def download(): return '⬇'
    @staticmethod
    def trash(): return '🗑'
    @staticmethod
    def wrench(): return '🔧'
    @staticmethod
    def bulb(): return '💡'
    @staticmethod
    def signal(): return '📶'
    @staticmethod
    def save(): return '💾'
    @staticmethod
    def refresh_icon(): return '🔄'
    @staticmethod
    def memo(): return '📝'
    @staticmethod
    def movie_camera(): return '🎬'
    @staticmethod
    def target(): return '🎯'
    @staticmethod
    def rocket(): return '🚀'
    @staticmethod
    def chart_increasing(): return '📈'
    @staticmethod
    def palette(): return '🎨'
    @staticmethod
    def hourglass(): return '⏳'
    @staticmethod
    def sleeping(): return '💤'
    @staticmethod
    def fire(): return '🔥'
    @staticmethod
    def get_themed_emoji(emoji_name: str, theme: str = 'classic') -> str:
        """Get emoji based on theme (mock always returns emoji)."""
        if hasattr(MockEmojiUtils, emoji_name):
            return getattr(MockEmojiUtils, emoji_name)()
        return '📦'

# Create mock logging module
mock_logging = type(sys)('deforum.utils.system.logging')
mock_logging.get_logger = lambda: MockLogger()
mock_logging.emoji_if_enabled = lambda emoji_str: emoji_str  # Return emoji as-is
mock_logging.emoji = MockEmojiUtils

sys.modules['deforum.utils.system.logging'] = mock_logging
sys.modules['deforum.utils.system.logging.emoji'] = MockEmojiUtils  # Mock the emoji submodule

spec.loader.exec_module(wan_button_handler)

load_wan_emojis = wan_button_handler.load_wan_emojis
get_wan_auto_download_setting = wan_button_handler.get_wan_auto_download_setting
is_model_valid = wan_button_handler.is_model_valid
validate_discovered_models = wan_button_handler.validate_discovered_models
build_no_models_error_message = wan_button_handler.build_no_models_error_message
extract_animation_prompts_from_args = wan_button_handler.extract_animation_prompts_from_args


class TestLoadWanEmojis:
    """Tests for load_wan_emojis function."""

    def test_returns_dict_with_expected_keys(self):
        """Should return dict with all expected emoji keys."""
        emojis = load_wan_emojis()

        expected_keys = {
            'check', 'cross', 'warning', 'download', 'trash', 'wrench',
            'bulb', 'signal', 'save', 'refresh_icon', 'memo', 'movie_camera',
            'target', 'rocket', 'chart_increasing'
        }

        assert isinstance(emojis, dict)
        assert set(emojis.keys()) == expected_keys

    def test_emoji_values_are_strings(self):
        """All emoji values should be strings."""
        emojis = load_wan_emojis()

        for key, value in emojis.items():
            assert isinstance(value, str), f"Emoji '{key}' should be string, got {type(value)}"


class TestGetWanAutoDownloadSetting:
    """Tests for get_wan_auto_download_setting function."""

    def test_returns_default_when_setting_not_found(self):
        """Should return True (default) when wan_auto_download not in component_names."""
        component_args = (False, "test", 123)
        component_names = ['other_setting', 'another_setting']

        result = get_wan_auto_download_setting(component_args, component_names)

        assert result is True

    def test_extracts_setting_when_present(self):
        """Should extract wan_auto_download value from component_args."""
        component_args = ('value1', False, 'value3')
        component_names = ['setting1', 'wan_auto_download', 'setting3']

        result = get_wan_auto_download_setting(component_args, component_names)

        assert result is False

    def test_handles_index_out_of_range(self):
        """Should return default when index is out of range."""
        component_args = ('value1',)  # Only 1 element
        component_names = ['setting1', 'wan_auto_download', 'setting3']  # Index 1 would be out of range

        result = get_wan_auto_download_setting(component_args, component_names)

        assert result is True  # Default


class TestIsModelValid:
    """Tests for is_model_valid function."""

    def test_valid_ti2v_model_with_model_index(self, tmp_path):
        """TI2V model with model_index.json should be valid."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "model_index.json").touch()

        model = {
            'name': 'Test-TI2V',
            'type': 'TI2V',
            'path': str(model_dir)
        }
        emojis = {'check': '✓'}

        assert is_model_valid(model, emojis) is True

    def test_invalid_ti2v_model_without_model_index(self, tmp_path):
        """TI2V model without model_index.json should be invalid."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        model = {
            'name': 'Test-TI2V',
            'type': 'TI2V',
            'path': str(model_dir)
        }
        emojis = {'check': '✓'}

        assert is_model_valid(model, emojis) is False

    def test_valid_legacy_model_with_transformer(self, tmp_path):
        """Legacy model with transformer directory should be valid."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "transformer").mkdir()

        model = {
            'name': 'Test-Legacy',
            'type': 'Unknown',
            'path': str(model_dir)
        }
        emojis = {'check': '✓'}

        assert is_model_valid(model, emojis) is True

    def test_valid_legacy_model_with_wan_pth(self, tmp_path):
        """Legacy model with wan*.pth files should be valid."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "wan_model.pth").touch()

        model = {
            'name': 'Test-Legacy',
            'type': 'Unknown',
            'path': str(model_dir)
        }
        emojis = {'check': '✓'}

        assert is_model_valid(model, emojis) is True

    def test_invalid_legacy_model_no_structure(self, tmp_path):
        """Legacy model with no recognizable structure should be invalid."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "random_file.txt").touch()

        model = {
            'name': 'Test-Invalid',
            'type': 'Unknown',
            'path': str(model_dir)
        }
        emojis = {'check': '✓'}

        assert is_model_valid(model, emojis) is False


class TestValidateDiscoveredModels:
    """Tests for validate_discovered_models function."""

    def test_separates_valid_and_corrupted_models(self, tmp_path):
        """Should correctly separate valid models from corrupted ones."""
        # Create valid model
        valid_dir = tmp_path / "valid_model"
        valid_dir.mkdir()
        (valid_dir / "model_index.json").touch()

        # Create corrupted model
        corrupted_dir = tmp_path / "corrupted_model"
        corrupted_dir.mkdir()

        models = [
            {'name': 'Valid', 'type': 'TI2V', 'path': str(valid_dir)},
            {'name': 'Corrupted', 'type': 'TI2V', 'path': str(corrupted_dir)},
        ]
        emojis = {'check': '✓'}

        valid, corrupted = validate_discovered_models(models, emojis)

        assert len(valid) == 1
        assert len(corrupted) == 1
        assert valid[0]['name'] == 'Valid'
        assert corrupted[0]['name'] == 'Corrupted'

    def test_returns_empty_lists_for_no_models(self):
        """Should return empty lists when no models provided."""
        models = []
        emojis = {'check': '✓'}

        valid, corrupted = validate_discovered_models(models, emojis)

        assert valid == []
        assert corrupted == []


class TestBuildNoModelsErrorMessage:
    """Tests for build_no_models_error_message function."""

    def test_message_includes_auto_download_help_when_disabled(self):
        """Should include auto-download setup instructions when disabled."""
        emojis = {
            'cross': '✗', 'bulb': '💡', 'wrench': '🔧',
            'check': '✓', 'download': '⬇', 'signal': '📶',
            'save': '💾', 'refresh_icon': '🔄'
        }

        message = build_no_models_error_message(wan_auto_download=False, emojis=emojis)

        assert "AUTO-DOWNLOAD OPTIONS" in message
        assert "Enable \"Auto-Download Models\"" in message
        assert "TI2V-5B" in message
        assert "TI2V-A14B" in message

    def test_message_includes_troubleshooting_when_enabled(self):
        """Should include troubleshooting when auto-download is enabled."""
        emojis = {
            'cross': '✗', 'bulb': '💡', 'wrench': '🔧',
            'check': '✓', 'download': '⬇', 'signal': '📶',
            'save': '💾', 'refresh_icon': '🔄'
        }

        message = build_no_models_error_message(wan_auto_download=True, emojis=emojis)

        assert "TROUBLESHOOTING" in message
        assert "Check internet connection" in message
        assert "disk space" in message

    def test_message_always_includes_model_info(self):
        """Should always include model information regardless of setting."""
        emojis = {
            'cross': '✗', 'bulb': '💡', 'wrench': '🔧',
            'check': '✓', 'download': '⬇', 'signal': '📶',
            'save': '💾', 'refresh_icon': '🔄'
        }

        for auto_download in [True, False]:
            message = build_no_models_error_message(wan_auto_download=auto_download, emojis=emojis)

            assert "No Wan models found" in message
            assert "QUICK SETUP" in message
            assert "24GB VRAM" in message
            assert "32GB+ VRAM" in message


class TestExtractAnimationPromptsFromArgs:
    """Tests for extract_animation_prompts_from_args function."""

    def test_extracts_prompts_when_present(self):
        """Should extract animation_prompts from component_args."""
        prompts_json = '{"0": "test prompt", "60": "another prompt"}'
        component_args = ('value1', prompts_json, 'value3')
        component_names = ['setting1', 'animation_prompts', 'setting3']
        emojis = {'memo': '📝', 'warning': '⚠️'}

        result = extract_animation_prompts_from_args(component_args, component_names, emojis)

        assert result == prompts_json

    def test_returns_default_when_not_found(self):
        """Should return default prompts when animation_prompts not in component_names."""
        component_args = ('value1', 'value2')
        component_names = ['setting1', 'setting2']
        emojis = {'memo': '📝', 'warning': '⚠️'}

        result = extract_animation_prompts_from_args(component_args, component_names, emojis)

        assert result == '{"0": "a beautiful landscape"}'

    def test_handles_index_out_of_range(self):
        """Should return default when index is out of range."""
        component_args = ('value1',)
        component_names = ['setting1', 'animation_prompts']
        emojis = {'memo': '📝', 'warning': '⚠️'}

        result = extract_animation_prompts_from_args(component_args, component_names, emojis)

        assert result == '{"0": "a beautiful landscape"}'


# Integration-style tests for discover_and_prepare_models would require mocking
# WanSimpleIntegration and downloader - skipping for now as those test the orchestration
# rather than pure logic
