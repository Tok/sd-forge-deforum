"""Unit tests for emoji system (deforum/utils/system/logging/emoji.py).

Tests all emoji functions, theme support, and toggle behavior.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# Import the REAL emoji module directly from filesystem
# This bypasses conftest's MockEmojiUtils
@pytest.fixture(scope="module")
def real_emoji_module():
    """Import real emoji module directly from filesystem."""
    emoji_path = Path(__file__).parent.parent.parent / "deforum" / "utils" / "system" / "logging" / "emoji.py"
    spec = importlib.util.spec_from_file_location("real_emoji", emoji_path)
    real_emoji = importlib.util.module_from_spec(spec)

    # Need to ensure deforum.rendering.options exists for the import
    # Mock it minimally
    mock_options = MagicMock()
    mock_options.is_emojis_enabled = Mock(return_value=True)
    if 'deforum.rendering.options' not in sys.modules:
        sys.modules['deforum.rendering.options'] = mock_options

    # Mock logger module for emoji_if_enabled
    mock_logger = MagicMock()
    mock_logger.emoji_if_enabled = Mock(side_effect=lambda x: x)  # Return emoji as-is
    if 'deforum.utils.system.logging.logger' not in sys.modules:
        sys.modules['deforum.utils.system.logging.logger'] = mock_logger

    spec.loader.exec_module(real_emoji)
    return real_emoji


class TestSelectFunction:
    """Test the internal _select() function."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_select_returns_emoji_when_enabled(self, mock_enabled, real_emoji_module):
        result = real_emoji_module._select('🔥')
        assert result == '🔥'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=False)
    def test_select_returns_empty_when_disabled(self, mock_enabled, real_emoji_module):
        result = real_emoji_module._select('🔥')
        assert result == ''

    def test_select_returns_empty_on_import_error(self, real_emoji_module):
        """Test fallback behavior when is_emojis_enabled is unavailable."""
        # During early init, should return empty string
        with patch('deforum.rendering.options.is_emojis_enabled', side_effect=ImportError):
            result = real_emoji_module._select('🔥')
            assert result == ''

    def test_select_returns_empty_on_any_exception(self, real_emoji_module):
        """Test fallback behavior on any exception."""
        with patch('deforum.rendering.options.is_emojis_enabled', side_effect=RuntimeError("test error")):
            result = real_emoji_module._select('🔥')
            assert result == ''


class TestBasicEmojiIcons:
    """Test basic emoji icon functions."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_refresh_icon(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.refresh_icon() == '🔄'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_bulb(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.bulb() == '💡'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_run(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.run() == '🏎️'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_key(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.key() == '🔑'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_frame(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.frame() == '🖼️'


class TestVideoAndMediaEmojis:
    """Test video and media-related emoji functions."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_video_camera(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.video_camera() == '📹'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_wan_video(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.wan_video() == '🎥'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_frames(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.frames() == '🎞️'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_movie_camera(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.movie_camera() == '🎬'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_camera(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.camera() == '📷'


class TestParameterEmojis:
    """Test parameter-related emoji functions."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_strength(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.strength() == '💪'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_scale(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.scale() == '📏'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_seed(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.seed() == '🌰'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_subseed(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.subseed() == '🥜'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_steps(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.steps() == '👣'


class TestEmojiToggleOff:
    """Test all emojis return empty string when disabled."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=False)
    def test_all_basic_emojis_disabled(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module

        # Test sample of emoji functions
        assert emoji.refresh_icon() == ''
        assert emoji.bulb() == ''
        assert emoji.run() == ''
        assert emoji.key() == ''
        assert emoji.frame() == ''
        assert emoji.video_camera() == ''
        assert emoji.strength() == ''
        assert emoji.seed() == ''
        assert emoji.rocket() == ''
        assert emoji.fire() == ''


class TestSlopcoreEmojis:
    """Test slopcore-specific emojis."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_blue_square(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.blue_square() == '🟦'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_purple_square(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.purple_square() == '🟪'


class TestStatusIndicators:
    """Test status indicator functions (maybe_* functions).

    Note: These tests are skipped because they test thin wrappers around
    emoji_if_enabled() which is already thoroughly tested in test_logger.py.
    The maybe_* functions just call emoji_if_enabled() with a hardcoded emoji,
    so testing emoji_if_enabled() is sufficient coverage.
    """

    @pytest.mark.skip(reason="Tested via emoji_if_enabled() in test_logger.py")
    def test_maybe_functions_are_wrappers(self, real_emoji_module):
        """Document that maybe_* functions are thin wrappers.

        maybe_check() -> emoji_if_enabled('✅')
        maybe_cross() -> emoji_if_enabled('❌')
        maybe_warning() -> emoji_if_enabled('⚠️')
        maybe_alert() -> emoji_if_enabled('🚨')

        These are tested indirectly via emoji_if_enabled() tests.
        """
        pass


class TestThemedEmoji:
    """Test get_themed_emoji() function with different themes."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_classic_theme_returns_full_emoji(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        result = emoji.get_themed_emoji('run', theme='classic')
        assert result == '🏎️'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_simple_theme_returns_full_emoji(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        result = emoji.get_themed_emoji('key', theme='simple')
        assert result == '🔑'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_slopcore_theme_blue_square_for_inputs(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        # 'run' is in SLOPCORE_BLUE_OPS
        result = emoji.get_themed_emoji('run', theme='slopcore')
        assert result == '🟦'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_slopcore_theme_purple_square_for_outputs(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        # 'video_camera' is in SLOPCORE_PURPLE_OPS
        result = emoji.get_themed_emoji('video_camera', theme='slopcore')
        assert result == '🟪'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_slopcore_theme_defaults_to_blue_for_unmapped(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        # 'rocket' is not in either SLOPCORE list
        result = emoji.get_themed_emoji('rocket', theme='slopcore')
        assert result == '🟦'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_unknown_emoji_name_returns_empty(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        result = emoji.get_themed_emoji('nonexistent_emoji', theme='classic')
        assert result == ''

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_classic_theme_with_multiple_emojis(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        emojis = ['run', 'key', 'frame', 'strength', 'seed']
        results = [emoji.get_themed_emoji(e, theme='classic') for e in emojis]
        expected = ['🏎️', '🔑', '🖼️', '💪', '🌰']
        assert results == expected

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_slopcore_theme_blue_operations(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        blue_ops = ['run', 'key', 'steps', 'seed', 'gear']
        results = [emoji.get_themed_emoji(e, theme='slopcore') for e in blue_ops]
        # All should be blue square
        assert all(r == '🟦' for r in results)

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_slopcore_theme_purple_operations(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        purple_ops = ['video_camera', 'wan_video', 'frames', 'strength']
        results = [emoji.get_themed_emoji(e, theme='slopcore') for e in purple_ops]
        # All should be purple square
        assert all(r == '🟪' for r in results)


class TestUtilityEmojis:
    """Test utility and miscellaneous emoji functions."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_rocket(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.rocket() == '🚀'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_fire(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.fire() == '🔥'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_sparkles(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.sparkles() == '✨'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_robot(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.robot() == '🤖'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_brain(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.brain() == '🧠'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_party(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.party() == '🎉'


class TestToolAndControlEmojis:
    """Test tool and control-related emoji functions."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_gear(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.gear() == '⚙️'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_wrench(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.wrench() == '🔧'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_tools(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.tools() == '🛠️'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_control(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.control() == '🎛️'


class TestDocumentAndFileEmojis:
    """Test document and file-related emoji functions."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_document(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.document() == '📄'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_folder(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.folder() == '📁'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_open_folder(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.open_folder() == '📂'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_floppy_disk(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.floppy_disk() == '💾'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_save(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.save() == '💾'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_books(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.books() == '📚'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_book(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.book() == '📚'


class TestLegacyConstants:
    """Test legacy module constants (for backward compatibility)."""

    def test_legacy_refresh_constant(self, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.refresh == '🔄'

    def test_legacy_info_constant(self, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.info() == emoji._select('ℹ️')

    def test_legacy_warn_constant(self, real_emoji_module):
        emoji = real_emoji_module
        assert emoji.warn == '⚠️'


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_emoji_with_variation_selector(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        # Test emojis that use variation selector (FE0F)
        assert emoji.warn == '⚠️'
        assert emoji.info() != ''
        assert emoji.frame() != ''

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_themed_emoji_defaults_to_classic(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        # No theme specified should default to classic behavior
        result = emoji.get_themed_emoji('run')
        assert result == '🏎️'

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_all_slopcore_blue_ops_mapped(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        blue_ops = [
            'run', 'key', 'frame', 'control', 'prompts', 'cadence',
            'steps', 'numbers', 'sound', 'music', 'seed', 'subseed',
            'leaf', 'bicycle', 'gear', 'wrench', 'stopwatch', 'tools',
        ]
        for op in blue_ops:
            result = emoji.get_themed_emoji(op, theme='slopcore')
            assert result == '🟦', f"{op} should map to blue square in slopcore"

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_all_slopcore_purple_ops_mapped(self, mock_enabled, real_emoji_module):
        emoji = real_emoji_module
        purple_ops = [
            'video_camera', 'wan_video', 'document', 'frames',
            'movie_camera', 'distribution', 'strength', 'scale',
        ]
        for op in purple_ops:
            result = emoji.get_themed_emoji(op, theme='slopcore')
            assert result == '🟪', f"{op} should map to purple square in slopcore"


class TestComprehensiveEmojiCoverage:
    """Test all emoji functions for completeness."""

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=True)
    def test_all_emoji_functions_return_unicode(self, mock_enabled, real_emoji_module):
        """Ensure all emoji functions return valid Unicode strings."""
        emoji = real_emoji_module

        # List of all emoji function names
        emoji_functions = [
            'refresh_icon', 'bulb', 'run', 'key', 'frame', 'control', 'net', 'web',
            'prompts', 'cadence', 'off', 'distribution', 'strength', 'scale',
            'video_camera', 'wan_video', 'document', 'steps', 'numbers', 'sound',
            'music', 'frames', 'up', 'seed', 'subseed', 'leaf', 'bicycle', 'hole',
            'palette', 'wave', 'broom', 'masking', 'gear', 'wrench', 'stopwatch',
            'tools', 'movie_camera', 'dice', 'folder', 'rocket', 'download',
            'sparkles', 'target', 'magnifying_glass', 'lock', 'save', 'trash',
            'fire', 'sleeping', 'clipboard', 'pencil', 'link', 'brain', 'robot',
            'party', 'eyes', 'globe', 'microscope', 'stop', 'memo', 'ruler',
            'hourglass', 'package', 'lightning', 'signal', 'open_folder', 'plus',
            'minus', 'books', 'chart_increasing', 'camera', 'abacus', 'info',
            'floppy_disk', 'muscle', 'book', 'blue_square', 'purple_square',
        ]

        for func_name in emoji_functions:
            func = getattr(emoji, func_name)
            result = func()
            assert isinstance(result, str), f"{func_name}() should return string"
            assert len(result) > 0, f"{func_name}() should return non-empty emoji when enabled"

    @patch('deforum.rendering.options.is_emojis_enabled', return_value=False)
    def test_all_emoji_functions_return_empty_when_disabled(self, mock_enabled, real_emoji_module):
        """Ensure all emoji functions return empty string when disabled."""
        emoji = real_emoji_module

        emoji_functions = [
            'refresh_icon', 'bulb', 'run', 'key', 'frame', 'video_camera',
            'strength', 'seed', 'rocket', 'fire', 'gear', 'folder',
        ]

        for func_name in emoji_functions:
            func = getattr(emoji, func_name)
            result = func()
            assert result == '', f"{func_name}() should return empty string when disabled"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
