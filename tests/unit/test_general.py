"""Unit tests for deforum.utils.general module.

Tests backward compatibility aliases, version functions, and path handling.
File I/O functions (duplicate_pngs, convert_images, download) are tested in integration/.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import datetime


class TestImportAliases:
    """Test backward compatibility aliases."""

    def test_tickOrCross_alias_exists(self):
        """tickOrCross should be aliased from tick_or_cross."""
        from deforum.utils.general import tickOrCross
        assert callable(tickOrCross)

    def test_clean_folder_name_alias_exists(self):
        """clean_folder_name should be aliased from _clean_folder_name."""
        from deforum.utils.general import clean_folder_name
        assert callable(clean_folder_name)

    def test_checksum_alias_exists(self):
        """checksum should be imported from hashing module."""
        from deforum.utils.general import checksum
        assert callable(checksum)

    def test_get_os_imported(self):
        """get_os should be imported from parsing.strings."""
        from deforum.utils.general import get_os
        assert callable(get_os)

    def test_custom_placeholder_format_imported(self):
        """custom_placeholder_format should be imported from parsing.strings."""
        from deforum.utils.general import custom_placeholder_format
        assert callable(custom_placeholder_format)

    def test_clean_gradio_path_strings_imported(self):
        """clean_gradio_path_strings should be imported from parsing.strings."""
        from deforum.utils.general import clean_gradio_path_strings
        assert callable(clean_gradio_path_strings)


class TestVersionFunctions:
    """Test version and commit date retrieval functions."""

    @patch('deforum.utils.general._get_extension_info')
    def test_get_deforum_version_with_extension(self, mock_get_ext):
        """get_deforum_version should return version from extension info."""
        from deforum.utils.general import get_deforum_version

        mock_ext = MagicMock()
        mock_ext.version = "1.2.3"
        mock_get_ext.return_value = mock_ext

        version = get_deforum_version()
        assert version == "1.2.3"

    @patch('deforum.utils.general._get_extension_info')
    def test_get_deforum_version_without_extension(self, mock_get_ext):
        """get_deforum_version should return 'Unknown' if extension not found."""
        from deforum.utils.general import get_deforum_version

        mock_get_ext.return_value = None

        version = get_deforum_version()
        assert version == "Unknown"

    @patch('deforum.utils.general._get_extension_info')
    def test_get_commit_date_with_extension(self, mock_get_ext):
        """get_commit_date should return formatted timestamp from extension."""
        from deforum.utils.general import get_commit_date

        mock_ext = MagicMock()
        mock_ext.commit_date = 1672531200  # 2023-01-01 00:00:00 UTC
        mock_get_ext.return_value = mock_ext

        commit_date = get_commit_date()
        assert isinstance(commit_date, datetime.datetime)
        assert commit_date.year == 2023
        assert commit_date.month == 1
        assert commit_date.day == 1

    @patch('deforum.utils.general._get_extension_info')
    def test_get_commit_date_without_extension(self, mock_get_ext):
        """get_commit_date should return 'Unknown' if extension not found."""
        from deforum.utils.general import get_commit_date

        mock_get_ext.return_value = None

        commit_date = get_commit_date()
        assert commit_date == "Unknown"


class TestExtensionInfo:
    """Test _get_extension_info internal function."""

    def test_get_extension_info_finds_deforum(self):
        """_get_extension_info should find and return sd-forge-deforum extension."""
        import sys
        from deforum.utils.general import _get_extension_info

        mock_ext = MagicMock()
        mock_ext.name = "sd-forge-deforum"
        mock_ext.enabled = True
        mock_ext.version = "1.0.0"
        mock_ext.read_info_from_repo = MagicMock()

        # Directly set the extensions list on the mock module
        sys.modules['modules.extensions'].extensions = [mock_ext]

        result = _get_extension_info()
        assert result is mock_ext
        mock_ext.read_info_from_repo.assert_called_once()

        # Cleanup: reset to empty list
        sys.modules['modules.extensions'].extensions = []

    def test_get_extension_info_not_found(self):
        """_get_extension_info should return None if extension not found."""
        import sys
        from deforum.utils.general import _get_extension_info

        mock_other_ext = MagicMock()
        mock_other_ext.name = "some-other-extension"
        mock_other_ext.enabled = True

        sys.modules['modules.extensions'].extensions = [mock_other_ext]

        result = _get_extension_info()
        assert result is None

        # Cleanup
        sys.modules['modules.extensions'].extensions = []

    def test_get_extension_info_disabled(self):
        """_get_extension_info should return None if extension is disabled."""
        import sys
        from deforum.utils.general import _get_extension_info

        mock_ext = MagicMock()
        mock_ext.name = "sd-forge-deforum"
        mock_ext.enabled = False

        sys.modules['modules.extensions'].extensions = [mock_ext]

        result = _get_extension_info()
        assert result is None

        # Cleanup
        sys.modules['modules.extensions'].extensions = []

    def test_get_extension_info_handles_exception(self):
        """_get_extension_info should return None and log error on exception."""
        import sys
        from deforum.utils.general import _get_extension_info

        # Create a mock that raises an exception when iterated
        mock_extensions_list = MagicMock()
        mock_extensions_list.__iter__ = MagicMock(side_effect=Exception("Mock error"))
        sys.modules['modules.extensions'].extensions = mock_extensions_list

        result = _get_extension_info()
        assert result is None

        # Cleanup
        sys.modules['modules.extensions'].extensions = []


class TestPathFunctions:
    """Test path handling functions."""

    @patch('deforum.utils.general.test_long_path_support')
    @patch('deforum.utils.general.get_os')
    @patch('deforum.utils.general._get_max_path_length')
    def test_get_max_path_length_windows_with_support(
        self, mock_pure_fn, mock_get_os, mock_test_support
    ):
        """get_max_path_length should test long paths on Windows."""
        from deforum.utils.general import get_max_path_length

        mock_get_os.return_value = 'Windows'
        mock_test_support.return_value = True
        mock_pure_fn.return_value = 32767

        result = get_max_path_length("C:/test")

        mock_test_support.assert_called_once_with("C:/test")
        mock_pure_fn.assert_called_once_with("C:/test", 'Windows', True)
        assert result == 32767

    @patch('deforum.utils.general.test_long_path_support')
    @patch('deforum.utils.general.get_os')
    @patch('deforum.utils.general._get_max_path_length')
    def test_get_max_path_length_linux(self, mock_pure_fn, mock_get_os, mock_test_support):
        """get_max_path_length should not test long paths on Linux."""
        from deforum.utils.general import get_max_path_length

        mock_get_os.return_value = 'Linux'
        mock_pure_fn.return_value = 4096

        result = get_max_path_length("/home/test")

        mock_test_support.assert_not_called()
        mock_pure_fn.assert_called_once_with("/home/test", 'Linux', False)
        assert result == 4096


class TestSubstitutePlaceholders:
    """Test substitute_placeholders function."""

    def test_substitute_placeholders_basic(self):
        """substitute_placeholders should replace valid placeholders."""
        from deforum.utils.general import substitute_placeholders

        # Create mock arg objects
        class MockArgs:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        args = MockArgs(batch_name="test_batch", timestring="20230101-120000")

        with patch('deforum.utils.general.get_max_path_length', return_value=260):
            result = substitute_placeholders(
                "{batch_name}_{timestring}",
                [args],
                "/tmp"
            )

        assert result == "test_batch_20230101-120000"

    def test_substitute_placeholders_handles_invalid(self):
        """substitute_placeholders should replace invalid placeholders with key name."""
        from deforum.utils.general import substitute_placeholders

        class MockArgs:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        args = MockArgs(valid="value")

        with patch('deforum.utils.general.get_max_path_length', return_value=260):
            result = substitute_placeholders(
                "{valid}_{invalid}",
                [args],
                "/tmp"
            )

        # Invalid placeholder is replaced with its key name
        assert result == "value_invalid"

    def test_substitute_placeholders_cleans_invalid_chars(self):
        """substitute_placeholders should replace invalid filesystem characters."""
        from deforum.utils.general import substitute_placeholders

        class MockArgs:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        args = MockArgs(name="test/batch:name<>with|bad*chars")

        with patch('deforum.utils.general.get_max_path_length', return_value=260):
            result = substitute_placeholders(
                "{name}",
                [args],
                "/tmp"
            )

        # All invalid chars should be replaced with underscores
        assert "/" not in result
        assert ":" not in result
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result
        assert "*" not in result

    def test_substitute_placeholders_respects_max_length(self):
        """substitute_placeholders should truncate to max path length."""
        from deforum.utils.general import substitute_placeholders

        class MockArgs:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        args = MockArgs(long_name="A" * 300)

        with patch('deforum.utils.general.get_max_path_length', return_value=50):
            result = substitute_placeholders(
                "{long_name}",
                [args],
                "/tmp"
            )

        assert len(result) <= 50

    def test_substitute_placeholders_resume_timestring(self):
        """substitute_placeholders should update timestring when resuming."""
        from deforum.utils.general import substitute_placeholders

        class MockArgs:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        args1 = MockArgs(
            timestring="20230101-120000",
            resume_from_timestring=True,
            resume_timestring="20230102-140000"
        )
        args2 = MockArgs(batch_name="test")

        with patch('deforum.utils.general.get_max_path_length', return_value=260):
            result = substitute_placeholders(
                "{timestring}_{batch_name}",
                [args1, args2],
                "/tmp"
            )

        # Timestring should be updated to resume_timestring
        assert "20230102-140000" in result
        assert "20230101-120000" not in result

    def test_substitute_placeholders_strips_trailing_underscores(self):
        """substitute_placeholders should remove trailing underscores."""
        from deforum.utils.general import substitute_placeholders

        class MockArgs:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        args = MockArgs(name="test")

        with patch('deforum.utils.general.get_max_path_length', return_value=260):
            result = substitute_placeholders(
                "{name}_{invalid}_",
                [args],
                "/tmp"
            )

        # Should not end with underscores
        assert not result.endswith("_")


class TestFileOperations:
    """Test file operation functions (may require integration tests)."""

    def test_duplicate_pngs_from_folder_signature(self):
        """duplicate_pngs_from_folder should be callable with correct signature."""
        from deforum.utils.general import duplicate_pngs_from_folder
        import inspect

        sig = inspect.signature(duplicate_pngs_from_folder)
        params = list(sig.parameters.keys())

        assert params == ['from_folder', 'to_folder', 'img_batch_id', 'orig_vid_name']

    def test_convert_images_from_list_signature(self):
        """convert_images_from_list should be callable with correct signature."""
        from deforum.utils.general import convert_images_from_list
        import inspect

        sig = inspect.signature(convert_images_from_list)
        params = list(sig.parameters.keys())

        assert params == ['paths', 'output_dir', 'format']

    def test_download_file_with_checksum_signature(self):
        """download_file_with_checksum should be callable with correct signature."""
        from deforum.utils.general import download_file_with_checksum
        import inspect

        sig = inspect.signature(download_file_with_checksum)
        params = list(sig.parameters.keys())

        assert params == ['url', 'expected_checksum', 'dest_folder', 'dest_filename']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
