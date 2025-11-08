"""Unit tests for A1111OptionsOverrider context manager."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

# Add parent directory to path to allow direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deforum.utils.system.opts_overrider import A1111OptionsOverrider


@pytest.fixture
def mock_opts():
    """Create mock opts object that mimics modules.shared.opts."""
    opts = SimpleNamespace()
    opts.data = {
        'option1': 'original_value1',
        'option2': 42,
        'option3': True,
    }
    # Also set as attributes
    opts.option1 = 'original_value1'
    opts.option2 = 42
    opts.option3 = True
    return opts


class TestA1111OptionsOverrider:
    """Test suite for A1111OptionsOverrider context manager."""

    def test_init_with_overrides(self):
        """Test initialization with override dictionary."""
        overrides = {'option1': 'new_value'}
        overrider = A1111OptionsOverrider(overrides)

        assert overrider.opts_overrides == overrides
        assert overrider.original_opts is None

    def test_init_without_overrides(self):
        """Test initialization with None defaults to empty dict."""
        overrider = A1111OptionsOverrider()

        assert overrider.opts_overrides == {}
        assert overrider.original_opts is None

    @patch('deforum.utils.system.opts_overrider.opts')
    def test_context_manager_applies_overrides(self, mock_opts_module):
        """Test that overrides are applied when entering context."""
        mock_opts_module.data = {'option1': 'original'}
        mock_opts_module.option1 = 'original'

        overrides = {'option1': 'new_value'}

        with A1111OptionsOverrider(overrides):
            assert mock_opts_module.option1 == 'new_value'
            assert mock_opts_module.data['option1'] == 'new_value'

    @patch('deforum.utils.system.opts_overrider.opts')
    def test_context_manager_restores_originals(self, mock_opts_module):
        """Test that original values are restored after context exit."""
        mock_opts_module.data = {'option1': 'original', 'option2': 100}
        mock_opts_module.option1 = 'original'
        mock_opts_module.option2 = 100

        overrides = {'option1': 'temporary', 'option2': 999}

        with A1111OptionsOverrider(overrides):
            assert mock_opts_module.option1 == 'temporary'
            assert mock_opts_module.option2 == 999

        # After context exit, values should be restored
        assert mock_opts_module.option1 == 'original'
        assert mock_opts_module.option2 == 100

    @patch('deforum.utils.system.opts_overrider.opts')
    def test_empty_overrides_does_nothing(self, mock_opts_module):
        """Test that empty overrides dict doesn't change anything."""
        mock_opts_module.data = {'option1': 'original'}
        mock_opts_module.option1 = 'original'

        with A1111OptionsOverrider({}):
            assert mock_opts_module.option1 == 'original'

        assert mock_opts_module.option1 == 'original'

    @patch('deforum.utils.system.opts_overrider.opts')
    def test_none_overrides_does_nothing(self, mock_opts_module):
        """Test that None overrides defaults to empty dict."""
        mock_opts_module.data = {'option1': 'original'}
        mock_opts_module.option1 = 'original'

        with A1111OptionsOverrider(None):
            assert mock_opts_module.option1 == 'original'

        assert mock_opts_module.option1 == 'original'

    @patch('deforum.utils.system.opts_overrider.opts')
    def test_only_overrides_existing_options(self, mock_opts_module):
        """Test that only existing options are tracked for restoration."""
        mock_opts_module.data = {'option1': 'original'}
        mock_opts_module.option1 = 'original'

        # Try to override non-existent option
        overrides = {'option1': 'new', 'nonexistent': 'value'}

        with A1111OptionsOverrider(overrides):
            # Both should be set
            assert mock_opts_module.option1 == 'new'
            assert mock_opts_module.data['nonexistent'] == 'value'

        # Only option1 should be restored (was in original data)
        assert mock_opts_module.option1 == 'original'

    @patch('deforum.utils.system.opts_overrider.opts')
    @patch('deforum.utils.system.opts_overrider.logger')
    def test_logs_override_actions(self, mock_logger, mock_opts_module):
        """Test that override actions are logged."""
        mock_opts_module.data = {'option1': 'original'}
        mock_opts_module.option1 = 'original'

        overrides = {'option1': 'new_value'}

        with A1111OptionsOverrider(overrides):
            pass

        # Check that logging occurred
        assert mock_logger.debug.called or mock_logger.info.called

    @patch('deforum.utils.system.opts_overrider.opts')
    @patch('deforum.utils.system.opts_overrider.logger')
    def test_logs_restoration(self, mock_logger, mock_opts_module):
        """Test that restoration actions are logged."""
        mock_opts_module.data = {'option1': 'original'}
        mock_opts_module.option1 = 'original'

        overrides = {'option1': 'new_value'}

        with A1111OptionsOverrider(overrides):
            pass

        # Check restoration was logged
        mock_logger.info.assert_any_call("Restoring options: {'option1': 'original'}")

    @patch('deforum.utils.system.opts_overrider.opts')
    @patch('deforum.utils.system.opts_overrider.logger')
    def test_logs_exception_during_context(self, mock_logger, mock_opts_module):
        """Test that exceptions during context are logged."""
        mock_opts_module.data = {'option1': 'original'}
        mock_opts_module.option1 = 'original'

        overrides = {'option1': 'new_value'}

        try:
            with A1111OptionsOverrider(overrides):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Check that exception was logged
        assert mock_logger.warning.called

    @patch('deforum.utils.system.opts_overrider.opts')
    def test_restores_after_exception(self, mock_opts_module):
        """Test that options are restored even if exception occurs."""
        mock_opts_module.data = {'option1': 'original'}
        mock_opts_module.option1 = 'original'

        overrides = {'option1': 'temporary'}

        try:
            with A1111OptionsOverrider(overrides):
                assert mock_opts_module.option1 == 'temporary'
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass

        # Should still restore even after exception
        assert mock_opts_module.option1 == 'original'

    @patch('deforum.utils.system.opts_overrider.opts')
    def test_multiple_overrides(self, mock_opts_module):
        """Test overriding multiple options at once."""
        mock_opts_module.data = {
            'opt1': 'val1',
            'opt2': 'val2',
            'opt3': 'val3',
        }
        mock_opts_module.opt1 = 'val1'
        mock_opts_module.opt2 = 'val2'
        mock_opts_module.opt3 = 'val3'

        overrides = {
            'opt1': 'new1',
            'opt2': 'new2',
            'opt3': 'new3',
        }

        with A1111OptionsOverrider(overrides):
            assert mock_opts_module.opt1 == 'new1'
            assert mock_opts_module.opt2 == 'new2'
            assert mock_opts_module.opt3 == 'new3'

        assert mock_opts_module.opt1 == 'val1'
        assert mock_opts_module.opt2 == 'val2'
        assert mock_opts_module.opt3 == 'val3'

    @patch('deforum.utils.system.opts_overrider.opts')
    def test_nested_context_managers(self, mock_opts_module):
        """Test nested usage of context managers."""
        mock_opts_module.data = {'option1': 'original'}
        mock_opts_module.option1 = 'original'

        with A1111OptionsOverrider({'option1': 'level1'}):
            assert mock_opts_module.option1 == 'level1'

            with A1111OptionsOverrider({'option1': 'level2'}):
                assert mock_opts_module.option1 == 'level2'

            # Should restore to level1 after inner context
            assert mock_opts_module.option1 == 'level1'

        # Should restore to original after outer context
        assert mock_opts_module.option1 == 'original'

    @patch('deforum.utils.system.opts_overrider.opts')
    def test_handles_various_data_types(self, mock_opts_module):
        """Test that various data types can be overridden."""
        mock_opts_module.data = {
            'str_opt': 'string',
            'int_opt': 42,
            'float_opt': 3.14,
            'bool_opt': True,
            'list_opt': [1, 2, 3],
            'dict_opt': {'key': 'value'},
        }
        for key, val in mock_opts_module.data.items():
            setattr(mock_opts_module, key, val)

        overrides = {
            'str_opt': 'new_string',
            'int_opt': 999,
            'float_opt': 2.71,
            'bool_opt': False,
            'list_opt': [4, 5, 6],
            'dict_opt': {'new': 'dict'},
        }

        with A1111OptionsOverrider(overrides):
            assert mock_opts_module.str_opt == 'new_string'
            assert mock_opts_module.int_opt == 999
            assert mock_opts_module.float_opt == 2.71
            assert mock_opts_module.bool_opt is False
            assert mock_opts_module.list_opt == [4, 5, 6]
            assert mock_opts_module.dict_opt == {'new': 'dict'}

        # All should be restored
        assert mock_opts_module.str_opt == 'string'
        assert mock_opts_module.int_opt == 42
        assert mock_opts_module.float_opt == 3.14
        assert mock_opts_module.bool_opt is True
        assert mock_opts_module.list_opt == [1, 2, 3]
        assert mock_opts_module.dict_opt == {'key': 'value'}
