"""Unit tests for Qwen status checking helpers.

Tests the helper functions extracted from check_qwen_models_handler().
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# Add extension root to path
extension_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(extension_root))

# Import directly from module file, bypassing deforum.ui.__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "qwen_status",
    extension_root / "deforum" / "ui" / "handlers" / "qwen_status.py"
)
qwen_status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen_status)

_load_qwen_emojis = qwen_status._load_qwen_emojis
_build_model_selection_status = qwen_status._build_model_selection_status
_build_model_info_status = qwen_status._build_model_info_status
_build_download_status = qwen_status._build_download_status
_build_loading_status = qwen_status._build_loading_status
_build_quick_setup_instructions = qwen_status._build_quick_setup_instructions


class TestLoadQwenEmojis:
    """Test _load_qwen_emojis() function."""

    def test_returns_all_required_emojis(self):
        """Should return dictionary with all required emoji keys."""
        emojis = _load_qwen_emojis()

        expected_keys = {
            'check', 'cross', 'warning', 'palette',
            'hourglass', 'refresh', 'zzz', 'fire'
        }
        assert set(emojis.keys()) == expected_keys

    def test_all_values_are_strings(self):
        """All emoji values should be strings."""
        emojis = _load_qwen_emojis()

        for key, value in emojis.items():
            assert isinstance(value, str), f"Emoji '{key}' should be string"


class TestBuildModelSelectionStatus:
    """Test _build_model_selection_status() function."""

    @pytest.fixture
    def mock_qwen_manager(self):
        """Create mock Qwen manager."""
        manager = Mock()
        manager.auto_select_model.return_value = "qwen2.5-7b-instruct"
        manager.get_model_info.return_value = {
            "name": "qwen2.5-7b-instruct",
            "description": "7B parameter model",
            "vram_gb": 7.0,
            "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        }
        return manager

    @pytest.fixture
    def emojis(self):
        """Mock emoji dictionary."""
        return {
            'check': '✅',
            'cross': '❌',
            'warning': '⚠️',
            'palette': '🎨',
            'hourglass': '⏳',
            'refresh': '🔄',
            'zzz': '💤',
            'fire': '🔥',
        }

    def test_manual_model_selection(self, mock_qwen_manager, emojis):
        """Should return selected model without auto-selection."""
        qwen_model = "qwen2.5-7b-instruct"
        actual_model, model_info, status_parts = _build_model_selection_status(
            qwen_model, mock_qwen_manager, 16.0, emojis
        )

        assert actual_model == "qwen2.5-7b-instruct"
        assert model_info["name"] == "qwen2.5-7b-instruct"
        assert len(status_parts) == 1  # Only selected model line
        assert "Selected Model:" in status_parts[0]
        mock_qwen_manager.auto_select_model.assert_not_called()

    def test_auto_select_model(self, mock_qwen_manager, emojis):
        """Should auto-select model when 'Auto-Select' chosen."""
        qwen_model = "Auto-Select"
        actual_model, model_info, status_parts = _build_model_selection_status(
            qwen_model, mock_qwen_manager, 16.0, emojis
        )

        assert actual_model == "qwen2.5-7b-instruct"  # Auto-selected
        assert len(status_parts) == 3  # Selected + Auto-Selected + Reason
        assert "Auto-Selected:" in status_parts[1]
        assert "Reason:" in status_parts[2]
        assert "16.0GB VRAM" in status_parts[2]
        mock_qwen_manager.auto_select_model.assert_called_once()


class TestBuildModelInfoStatus:
    """Test _build_model_info_status() function."""

    @pytest.fixture
    def emojis(self):
        """Mock emoji dictionary."""
        return {
            'check': '✅',
            'warning': '⚠️',
        }

    def test_none_model_info(self, emojis):
        """Should return empty list for None model info."""
        status_parts = _build_model_info_status(None, 16.0, emojis)
        assert status_parts == []

    def test_model_info_vram_sufficient(self, emojis):
        """Should show success when VRAM requirement met."""
        model_info = {
            "description": "7B parameter model",
            "vram_gb": 7.0,
        }
        status_parts = _build_model_info_status(model_info, 16.0, emojis)

        assert len(status_parts) == 4
        assert "Description:" in status_parts[0]
        assert "VRAM Required:" in status_parts[1]
        assert "7.0GB" in status_parts[1]
        assert "Available VRAM:" in status_parts[2]
        assert "16.0GB" in status_parts[2]
        assert "✅" in status_parts[3]
        assert "VRAM requirement met" in status_parts[3]

    def test_model_info_vram_insufficient(self, emojis):
        """Should show warning when VRAM may be exceeded."""
        model_info = {
            "description": "14B parameter model",
            "vram_gb": 14.0,
        }
        status_parts = _build_model_info_status(model_info, 8.0, emojis)

        assert len(status_parts) == 4
        assert "⚠️" in status_parts[3]
        assert "May exceed available VRAM" in status_parts[3]


class TestBuildDownloadStatus:
    """Test _build_download_status() function."""

    @pytest.fixture
    def emojis(self):
        """Mock emoji dictionary."""
        return {
            'check': '✅',
            'cross': '❌',
        }

    def test_model_downloaded(self, emojis):
        """Should show success when model downloaded."""
        status_parts = _build_download_status(True, None, emojis)

        assert len(status_parts) == 1
        assert "✅" in status_parts[0]
        assert "Model downloaded and available" in status_parts[0]

    def test_model_not_downloaded_no_hf_name(self, emojis):
        """Should show error when model not downloaded."""
        model_info = {"description": "7B model"}  # No hf_name
        status_parts = _build_download_status(False, model_info, emojis)

        assert len(status_parts) == 1
        assert "❌" in status_parts[0]
        assert "Model not downloaded" in status_parts[0]

    def test_model_not_downloaded_with_hf_name(self, emojis):
        """Should show HuggingFace ID when available."""
        model_info = {
            "description": "7B model",
            "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        }
        status_parts = _build_download_status(False, model_info, emojis)

        assert len(status_parts) == 2
        assert "❌" in status_parts[0]
        assert "Model not downloaded" in status_parts[0]
        assert "HuggingFace ID:" in status_parts[1]
        assert "Qwen/Qwen2.5-7B-Instruct" in status_parts[1]


class TestBuildLoadingStatus:
    """Test _build_loading_status() function."""

    @pytest.fixture
    def emojis(self):
        """Mock emoji dictionary."""
        return {
            'zzz': '💤',
            'fire': '🔥',
            'refresh': '🔄',
        }

    def test_no_model_loaded(self, emojis):
        """Should show idle status when no model loaded."""
        status_parts = _build_loading_status(False, None, "qwen2.5-7b-instruct", emojis)

        assert len(status_parts) == 1
        assert "💤" in status_parts[0]
        assert "No model currently loaded" in status_parts[0]

    def test_correct_model_loaded_no_vram(self, emojis):
        """Should show ready status when correct model loaded."""
        loaded_info = {
            "name": "qwen2.5-7b-instruct",
            "vram_usage": 0,
        }
        status_parts = _build_loading_status(
            True, loaded_info, "qwen2.5-7b-instruct", emojis
        )

        assert len(status_parts) == 1
        assert "🔥" in status_parts[0]
        assert "Model currently loaded and ready" in status_parts[0]

    def test_correct_model_loaded_with_vram(self, emojis):
        """Should show VRAM usage when available."""
        loaded_info = {
            "name": "qwen2.5-7b-instruct",
            "vram_usage": 7.2,
        }
        status_parts = _build_loading_status(
            True, loaded_info, "qwen2.5-7b-instruct", emojis
        )

        assert len(status_parts) == 2
        assert "🔥" in status_parts[0]
        assert "Model currently loaded and ready" in status_parts[0]
        assert "Estimated VRAM usage:" in status_parts[1]
        assert "7.2GB" in status_parts[1]

    def test_different_model_loaded(self, emojis):
        """Should show warning when different model loaded."""
        loaded_info = {
            "name": "qwen2.5-3b-instruct",
            "vram_usage": 3.5,
        }
        status_parts = _build_loading_status(
            True, loaded_info, "qwen2.5-7b-instruct", emojis
        )

        assert len(status_parts) == 2
        assert "🔄" in status_parts[0]
        assert "Different model loaded:" in status_parts[0]
        assert "qwen2.5-3b-instruct" in status_parts[0]
        assert "Will switch on next enhancement" in status_parts[1]


class TestBuildQuickSetupInstructions:
    """Test _build_quick_setup_instructions() function."""

    @pytest.fixture
    def emojis(self):
        """Mock emoji dictionary."""
        return {
            'check': '✅',
            'palette': '🎨',
            'hourglass': '⏳',
        }

    def test_model_not_downloaded(self, emojis):
        """Should show download instructions."""
        status_parts = _build_quick_setup_instructions(False, False, emojis)

        assert len(status_parts) == 4
        assert "Quick Setup:" in status_parts[0]
        assert "✅" in status_parts[1]
        assert "Auto-Download" in status_parts[1]
        assert "🎨" in status_parts[2]
        assert "AI Prompt Enhancement" in status_parts[2]
        assert "⏳" in status_parts[3]
        assert "Wait for download" in status_parts[3]

    def test_model_downloaded_not_loaded(self, emojis):
        """Should show ready to use instructions."""
        status_parts = _build_quick_setup_instructions(True, False, emojis)

        assert len(status_parts) == 2
        assert "Ready to Use:" in status_parts[0]
        assert "🎨" in status_parts[1]
        assert "AI Prompt Enhancement" in status_parts[1]

    def test_model_downloaded_and_loaded(self, emojis):
        """Should show ready status."""
        status_parts = _build_quick_setup_instructions(True, True, emojis)

        assert len(status_parts) == 1
        assert "Status:" in status_parts[0]
        assert "Ready for prompt enhancement!" in status_parts[0]


class TestIntegration:
    """Integration tests combining multiple helpers."""

    def test_full_status_build_flow(self):
        """Test building complete status message."""
        # Mock manager
        manager = Mock()
        manager.auto_select_model.return_value = "qwen2.5-7b-instruct"
        manager.get_model_info.return_value = {
            "name": "qwen2.5-7b-instruct",
            "description": "7B parameter model",
            "vram_gb": 7.0,
            "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        }

        # Load emojis
        emojis = _load_qwen_emojis()

        # Build all status sections
        actual_model, model_info, selection_parts = _build_model_selection_status(
            "Auto-Select", manager, 16.0, emojis
        )
        info_parts = _build_model_info_status(model_info, 16.0, emojis)
        download_parts = _build_download_status(True, model_info, emojis)
        loading_parts = _build_loading_status(
            True, {"name": "qwen2.5-7b-instruct", "vram_usage": 7.2}, actual_model, emojis
        )
        setup_parts = _build_quick_setup_instructions(True, True, emojis)

        # Verify all parts exist
        assert len(selection_parts) == 3  # Auto-selection adds 2 extra lines
        assert len(info_parts) == 4
        assert len(download_parts) == 1
        assert len(loading_parts) == 2  # With VRAM info
        assert len(setup_parts) == 1

        # Build final message
        all_parts = selection_parts + info_parts + download_parts + loading_parts + setup_parts
        message = "<br>".join(all_parts)

        # Verify complete message structure
        assert "Auto-Selected:" in message
        assert "VRAM Required:" in message
        assert "Model downloaded and available" in message
        assert "Model currently loaded and ready" in message
        assert "Ready for prompt enhancement!" in message
