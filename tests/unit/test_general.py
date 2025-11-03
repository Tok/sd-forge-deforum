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


class TestDebugPrint:
    """Test debug_print function."""

    @patch('deforum.utils.general.opts')
    @patch('deforum.utils.general.logger')
    def test_debug_print_enabled(self, mock_logger, mock_opts):
        """debug_print should log when debug mode is enabled."""
        from deforum.utils.general import debug_print

        mock_opts.data = {'deforum_debug_mode_enabled': True}

        debug_print("Test message")

        mock_logger.debug.assert_called_once_with("Test message")

    @patch('deforum.utils.general.opts')
    @patch('deforum.utils.general.logger')
    def test_debug_print_disabled(self, mock_logger, mock_opts):
        """debug_print should not log when debug mode is disabled."""
        from deforum.utils.general import debug_print

        mock_opts.data = {'deforum_debug_mode_enabled': False}

        debug_print("Test message")

        mock_logger.debug.assert_not_called()

    @patch('deforum.utils.general.opts')
    @patch('deforum.utils.general.logger')
    def test_debug_print_missing_setting(self, mock_logger, mock_opts):
        """debug_print should not log when setting is missing."""
        from deforum.utils.general import debug_print

        mock_opts.data = {}

        debug_print("Test message")

        mock_logger.debug.assert_not_called()


class TestLongPathSupport:
    """Test test_long_path_support function."""

    @patch('deforum.utils.general.shutil.rmtree')
    @patch('deforum.utils.general.os.makedirs')
    def test_long_path_supported(self, mock_makedirs, mock_rmtree):
        """test_long_path_support should return True when long paths work."""
        from deforum.utils.general import test_long_path_support

        mock_makedirs.return_value = None
        mock_rmtree.return_value = None

        result = test_long_path_support("/tmp")

        assert result is True
        mock_makedirs.assert_called_once()
        mock_rmtree.assert_called_once()

    @patch('deforum.utils.general.os.makedirs')
    def test_long_path_not_supported(self, mock_makedirs):
        """test_long_path_support should return False when OSError occurs."""
        from deforum.utils.general import test_long_path_support

        mock_makedirs.side_effect = OSError("Path too long")

        result = test_long_path_support("/tmp")

        assert result is False


class TestDuplicatePngsFromFolder:
    """Test duplicate_pngs_from_folder function."""

    @patch('cv2.imread')
    @patch('cv2.imwrite')
    @patch('deforum.utils.general.shutil.copy')
    @patch('deforum.utils.general.os.listdir')
    @patch('deforum.utils.general.os.makedirs')
    def test_duplicate_pngs_with_video_origin(
        self, mock_makedirs, mock_listdir, mock_copy, mock_imwrite, mock_imread
    ):
        """duplicate_pngs_from_folder should copy files when orig_vid_name is set."""
        from deforum.utils.general import duplicate_pngs_from_folder

        mock_listdir.return_value = ['0001.png', '0002.png', '0003.jpg']

        result = duplicate_pngs_from_folder(
            from_folder="/source",
            to_folder="output",
            img_batch_id=None,
            orig_vid_name="video.mp4"
        )

        assert result == 3
        assert mock_copy.call_count == 3
        mock_imread.assert_not_called()

    @patch('cv2.imread')
    @patch('cv2.imwrite')
    @patch('deforum.utils.general.os.listdir')
    @patch('deforum.utils.general.os.makedirs')
    def test_duplicate_pngs_without_video_origin(
        self, mock_makedirs, mock_listdir, mock_imwrite, mock_imread
    ):
        """duplicate_pngs_from_folder should convert images when orig_vid_name is None."""
        from deforum.utils.general import duplicate_pngs_from_folder

        mock_listdir.return_value = ['0001.png', '0002.png']
        mock_imread.return_value = MagicMock()  # Mock image data

        result = duplicate_pngs_from_folder(
            from_folder="/source",
            to_folder="output",
            img_batch_id=None,
            orig_vid_name=None
        )

        assert result == 2
        assert mock_imread.call_count == 2
        assert mock_imwrite.call_count == 2

    @patch('deforum.utils.general.shutil.copy')
    @patch('deforum.utils.general.os.listdir')
    @patch('deforum.utils.general.os.makedirs')
    def test_duplicate_pngs_filters_depth_files(self, mock_makedirs, mock_listdir, mock_copy):
        """duplicate_pngs_from_folder should skip depth map files."""
        from deforum.utils.general import duplicate_pngs_from_folder

        mock_listdir.return_value = ['0001.png', '0001_depth_0001.png', '0002.png']

        result = duplicate_pngs_from_folder(
            from_folder="/source",
            to_folder="output",
            img_batch_id=None,
            orig_vid_name="video.mp4"
        )

        # Should only process 2 files (excluding depth map)
        assert result == 2

    @patch('deforum.utils.general.shutil.copy')
    @patch('deforum.utils.general.os.listdir')
    @patch('deforum.utils.general.os.makedirs')
    def test_duplicate_pngs_with_batch_id_filter(self, mock_makedirs, mock_listdir, mock_copy):
        """duplicate_pngs_from_folder should filter by batch ID."""
        from deforum.utils.general import duplicate_pngs_from_folder

        mock_listdir.return_value = ['batch1_0001.png', 'batch2_0001.png', 'batch1_0002.png']

        result = duplicate_pngs_from_folder(
            from_folder="/source",
            to_folder="output",
            img_batch_id="batch1",
            orig_vid_name="video.mp4"
        )

        # Should only process files starting with 'batch1'
        assert result == 2


class TestConvertImagesFromList:
    """Test convert_images_from_list function."""

    def test_convert_images_creates_output_dir(self):
        """convert_images_from_list should create output directory."""
        from deforum.utils.general import convert_images_from_list
        from PIL import Image
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            test_img_path = os.path.join(tmpdir, 'test.png')
            Image.new('RGB', (10, 10)).save(test_img_path)

            output_dir = os.path.join(tmpdir, 'output')

            convert_images_from_list([test_img_path], output_dir, 'png')

            # Output directory should be created
            assert os.path.exists(output_dir)

    def test_convert_images_saves_with_format(self):
        """convert_images_from_list should save images in specified format."""
        from deforum.utils.general import convert_images_from_list
        from PIL import Image
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            test_img1 = os.path.join(tmpdir, 'test1.png')
            test_img2 = os.path.join(tmpdir, 'test2.png')
            Image.new('RGB', (10, 10)).save(test_img1)
            Image.new('RGB', (10, 10)).save(test_img2)

            output_dir = os.path.join(tmpdir, 'output')

            convert_images_from_list([test_img1, test_img2], output_dir, 'jpg')

            # Check output files exist
            assert os.path.exists(os.path.join(output_dir, '000000001.jpg'))
            assert os.path.exists(os.path.join(output_dir, '000000002.jpg'))


class TestDownloadFileWithChecksum:
    """Test download_file_with_checksum function."""

    @patch('deforum.utils.general.checksum')
    @patch('deforum.utils.general.download_url_to_file')
    @patch('deforum.utils.general.os.path.exists')
    def test_download_when_file_missing(
        self, mock_exists, mock_download, mock_checksum
    ):
        """download_file_with_checksum should download when file doesn't exist."""
        from deforum.utils.general import download_file_with_checksum

        mock_exists.return_value = False
        mock_checksum.return_value = "abc123"

        download_file_with_checksum(
            url="https://example.com/file.bin",
            expected_checksum="abc123",
            dest_folder="/tmp",
            dest_filename="file.bin"
        )

        mock_download.assert_called_once()
        mock_checksum.assert_called_once()

    @patch('deforum.utils.general.checksum')
    @patch('deforum.utils.general.download_url_to_file')
    @patch('deforum.utils.general.os.path.exists')
    def test_download_raises_on_checksum_mismatch(
        self, mock_exists, mock_download, mock_checksum
    ):
        """download_file_with_checksum should raise exception on checksum mismatch."""
        from deforum.utils.general import download_file_with_checksum

        mock_exists.return_value = False
        mock_checksum.return_value = "wrong_checksum"

        with pytest.raises(Exception, match="Error while downloading"):
            download_file_with_checksum(
                url="https://example.com/file.bin",
                expected_checksum="expected_checksum",
                dest_folder="/tmp",
                dest_filename="file.bin"
            )

    @patch('deforum.utils.general.download_url_to_file')
    @patch('deforum.utils.general.os.path.exists')
    def test_download_skips_when_file_exists(self, mock_exists, mock_download):
        """download_file_with_checksum should skip download when file exists."""
        from deforum.utils.general import download_file_with_checksum

        mock_exists.return_value = True

        download_file_with_checksum(
            url="https://example.com/file.bin",
            expected_checksum="abc123",
            dest_folder="/tmp",
            dest_filename="file.bin"
        )

        mock_download.assert_not_called()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
