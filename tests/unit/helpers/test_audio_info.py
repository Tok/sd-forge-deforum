"""Unit tests for audio info calculation helpers.

Tests the helper functions extracted from get_tab_init().
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add extension root to path
extension_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(extension_root))

# Import directly from module file, bypassing deforum.ui.__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "audio_info",
    extension_root / "deforum" / "ui" / "helpers" / "audio_info.py"
)
audio_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_info)

calculate_audio_info = audio_info.calculate_audio_info
handle_audio_upload = audio_info.handle_audio_upload
auto_load_audio_info = audio_info.auto_load_audio_info


class TestCalculateAudioInfo:
    """Test calculate_audio_info() function."""

    def test_empty_path(self):
        """Should return empty string for empty path."""
        result = calculate_audio_info("", 24)
        assert result == ""

    def test_whitespace_path(self):
        """Should return empty string for whitespace path."""
        result = calculate_audio_info("   ", 24)
        assert result == ""

    def test_none_path(self):
        """Should return empty string for None path."""
        result = calculate_audio_info(None, 24)
        assert result == ""

    @patch('librosa.get_duration')
    @patch('deforum.media.video_audio_utilities.download_audio')
    def test_successful_calculation(self, mock_download, mock_get_duration):
        """Should calculate audio info successfully."""
        # Mock audio file
        mock_download.return_value = "/tmp/test_audio.mp3"
        mock_get_duration.return_value = 10.5

        result = calculate_audio_info("http://example.com/audio.mp3", 30)

        assert "Duration: 10.50s" in result
        assert "Suggested max_frames @ 30 FPS: 315" in result
        mock_download.assert_called_once_with("http://example.com/audio.mp3")
        mock_get_duration.assert_called_once_with(path="/tmp/test_audio.mp3")

    @patch('librosa.get_duration')
    @patch('deforum.media.video_audio_utilities.download_audio')
    def test_default_fps_fallback(self, mock_download, mock_get_duration):
        """Should use default 24 FPS when fps is 0 or None."""
        mock_download.return_value = "/tmp/test_audio.mp3"
        mock_get_duration.return_value = 10.0

        # Test with 0 FPS
        result = calculate_audio_info("/path/audio.mp3", 0)
        assert "@ 24 FPS: 240" in result

        # Test with negative FPS
        result = calculate_audio_info("/path/audio.mp3", -1)
        assert "@ 24 FPS: 240" in result

    @patch('librosa.get_duration')
    @patch('deforum.media.video_audio_utilities.download_audio')
    def test_error_handling(self, mock_download, mock_get_duration):
        """Should return error message when loading fails."""
        mock_download.side_effect = Exception("File not found")

        result = calculate_audio_info("/invalid/path.mp3", 24)

        assert "Could not load audio:" in result
        assert "File not found" in result

    @patch('librosa.get_duration')
    @patch('deforum.media.video_audio_utilities.download_audio')
    def test_local_file_passthrough(self, mock_download, mock_get_duration):
        """Should pass through local file paths."""
        local_path = "/home/user/audio.mp3"
        mock_download.return_value = local_path
        mock_get_duration.return_value = 5.0

        result = calculate_audio_info(local_path, 60)

        assert "Duration: 5.00s" in result
        assert "@ 60 FPS: 300" in result


class TestHandleAudioUpload:
    """Test handle_audio_upload() function."""

    def test_none_filepath(self):
        """Should return None tuple for None filepath."""
        result = handle_audio_upload(None, 24)

        assert result == (None, "File", "")

    @patch('shutil.copy2')
    @patch.object(audio_info, 'calculate_audio_info')
    def test_successful_upload(self, mock_calc, mock_copy):
        """Should successfully handle audio upload."""
        mock_calc.return_value = "Duration: 10.00s | Suggested max_frames @ 30 FPS: 300"

        result = handle_audio_upload("/tmp/gradio/uploaded.mp3", 30)

        abs_path, add_soundtrack, info_text = result
        assert add_soundtrack == "File"
        assert "Duration: 10.00s" in info_text
        assert abs_path.endswith("uploaded.mp3")

        # Verify file was copied
        mock_copy.assert_called_once()

    @patch('shutil.copy2')
    @patch.object(audio_info, 'calculate_audio_info')
    def test_upload_with_spaces_in_filename(self, mock_calc, mock_copy):
        """Should handle filenames with spaces."""
        mock_calc.return_value = "Duration: 5.00s"

        # This should not raise an error
        result = handle_audio_upload("/tmp/my audio file.mp3", 24)

        abs_path, add_soundtrack, info_text = result
        assert add_soundtrack == "File"
        assert "Duration: 5.00s" in info_text


class TestAutoLoadAudioInfo:
    """Test auto_load_audio_info() function."""

    @patch.object(audio_info, 'calculate_audio_info')
    def test_delegates_to_calculate(self, mock_calc):
        """Should delegate to calculate_audio_info."""
        mock_calc.return_value = "Duration: 8.00s | Suggested max_frames @ 24 FPS: 192"

        result = auto_load_audio_info("/path/soundtrack.mp3", 24)

        assert result == "Duration: 8.00s | Suggested max_frames @ 24 FPS: 192"
        mock_calc.assert_called_once_with("/path/soundtrack.mp3", 24)

    @patch.object(audio_info, 'calculate_audio_info')
    def test_empty_path(self, mock_calc):
        """Should handle empty path."""
        mock_calc.return_value = ""

        result = auto_load_audio_info("", 24)

        assert result == ""


class TestIntegration:
    """Integration tests for audio info workflow."""

    @patch('shutil.copy2')
    @patch('librosa.get_duration')
    @patch('deforum.media.video_audio_utilities.download_audio')
    def test_upload_and_auto_load_workflow(self, mock_download, mock_get_duration, mock_copy):
        """Test complete upload and auto-load workflow."""
        # Setup mocks
        mock_download.return_value = "/abs/outputs/audio/test.mp3"
        mock_get_duration.return_value = 12.5

        # Simulate upload
        abs_path, add_soundtrack, info_text = handle_audio_upload("/tmp/test.mp3", 30)

        assert add_soundtrack == "File"
        assert "Duration: 12.50s" in info_text
        assert "@ 30 FPS: 375" in info_text

        # Simulate auto-load on tab open
        auto_info = auto_load_audio_info(abs_path, 30)
        assert auto_info == info_text

    @patch('librosa.get_duration')
    @patch('deforum.media.video_audio_utilities.download_audio')
    def test_fps_change_recalculation(self, mock_download, mock_get_duration):
        """Test recalculating with different FPS."""
        mock_download.return_value = "/path/audio.mp3"
        mock_get_duration.return_value = 10.0

        # Calculate at 24 FPS
        info_24 = calculate_audio_info("/path/audio.mp3", 24)
        assert "@ 24 FPS: 240" in info_24

        # Recalculate at 60 FPS
        info_60 = calculate_audio_info("/path/audio.mp3", 60)
        assert "@ 60 FPS: 600" in info_60

        # Duration should be same
        assert "Duration: 10.00s" in info_24
        assert "Duration: 10.00s" in info_60
