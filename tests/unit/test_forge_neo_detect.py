"""Unit tests for Forge Neo detection utilities."""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from types import SimpleNamespace
import sys
import os

from deforum.utils.system.forge_neo_detect import (
    is_forge_neo,
    get_forge_variant,
    is_forge_neo_cached,
    clear_neo_cache,
    _get_webui_root,
    _check_marker_files,
    _check_directory_name,
    _check_sys_path,
    _check_neo_modules,
    NEO_MARKER_FILES,
    NEO_SPECIFIC_MODULES,
)


class TestConstants:
    """Test suite for module constants."""

    def test_neo_marker_files(self):
        """Test Neo marker file constants."""
        assert 'neo.txt' in NEO_MARKER_FILES
        assert 'FORGE_NEO.txt' in NEO_MARKER_FILES
        assert '.forge_neo' in NEO_MARKER_FILES

    def test_neo_specific_modules(self):
        """Test Neo-specific module constants."""
        assert 'modules_neo' in NEO_SPECIFIC_MODULES
        assert 'backend_neo' in NEO_SPECIFIC_MODULES


class TestHelperFunctions:
    """Test suite for internal helper functions."""

    def test_get_webui_root_success(self):
        """Test successfully getting webui root."""
        # Create a mock module hierarchy
        mock_paths_internal = MagicMock()
        mock_paths_internal.script_path = '/path/to/webui'

        mock_modules = MagicMock()
        mock_modules.paths_internal = mock_paths_internal

        # Patch both modules and modules.paths_internal
        with patch.dict('sys.modules', {
            'modules': mock_modules,
            'modules.paths_internal': mock_paths_internal
        }):
            result = _get_webui_root()
            assert result == '/path/to/webui'

    def test_get_webui_root_import_error(self):
        """Test webui root when modules unavailable."""
        # Remove modules from sys.modules temporarily if present
        with patch.dict('sys.modules', {'modules': None, 'modules.paths_internal': None}):
            result = _get_webui_root()
            assert result is None

    def test_check_marker_files_exists(self, tmp_path):
        """Test marker file detection when file exists."""
        # Create a marker file
        marker_path = tmp_path / 'neo.txt'
        marker_path.touch()

        result = _check_marker_files(str(tmp_path))
        assert result is True

    def test_check_marker_files_not_exists(self, tmp_path):
        """Test marker file detection when no files exist."""
        result = _check_marker_files(str(tmp_path))
        assert result is False

    def test_check_marker_files_different_marker(self, tmp_path):
        """Test marker file detection with different marker file."""
        marker_path = tmp_path / 'FORGE_NEO.txt'
        marker_path.touch()

        result = _check_marker_files(str(tmp_path))
        assert result is True

    def test_check_directory_name_with_neo(self):
        """Test directory name check when name contains 'neo'."""
        result = _check_directory_name('/path/to/forge-neo')
        assert result is True

    def test_check_directory_name_with_neo_uppercase(self):
        """Test directory name check with uppercase NEO."""
        result = _check_directory_name('/path/to/FORGE-NEO')
        assert result is True

    def test_check_directory_name_without_neo(self):
        """Test directory name check when name doesn't contain 'neo'."""
        result = _check_directory_name('/path/to/forge')
        assert result is False

    def test_check_sys_path_with_neo(self):
        """Test sys.path check when neo directory present."""
        original_path = sys.path.copy()
        try:
            sys.path.append('/home/user/forge-neo/extensions')
            result = _check_sys_path()
            assert result is True
        finally:
            sys.path = original_path

    def test_check_sys_path_with_neo_underscore(self):
        """Test sys.path check with underscore variant."""
        original_path = sys.path.copy()
        try:
            sys.path.append('/home/user/forge_neo/venv')
            result = _check_sys_path()
            assert result is True
        finally:
            sys.path = original_path

    def test_check_sys_path_without_neo(self):
        """Test sys.path check when no neo directory."""
        original_path = sys.path.copy()
        try:
            # Clear neo paths if any
            sys.path = [p for p in sys.path if 'neo' not in p.lower()]
            result = _check_sys_path()
            assert result is False
        finally:
            sys.path = original_path

    @patch('importlib.util.find_spec')
    def test_check_neo_modules_found(self, mock_find_spec):
        """Test Neo module check when modules exist."""
        mock_find_spec.return_value = MagicMock()  # Non-None = module found
        result = _check_neo_modules()
        assert result is True

    @patch('importlib.util.find_spec')
    def test_check_neo_modules_not_found(self, mock_find_spec):
        """Test Neo module check when modules don't exist."""
        mock_find_spec.return_value = None
        result = _check_neo_modules()
        assert result is False

    @patch('importlib.util.find_spec')
    def test_check_neo_modules_import_error(self, mock_find_spec):
        """Test Neo module check when import fails."""
        mock_find_spec.side_effect = ImportError("Module not found")
        result = _check_neo_modules()
        assert result is False


class TestIsForgeNeo:
    """Test suite for is_forge_neo function."""

    @patch('deforum.utils.system.forge_neo_detect._get_webui_root')
    @patch('deforum.utils.system.forge_neo_detect._check_marker_files')
    def test_detects_via_marker_files(self, mock_check_markers, mock_get_root):
        """Test detection via marker files."""
        mock_get_root.return_value = '/path/to/webui'
        mock_check_markers.return_value = True

        result = is_forge_neo()
        assert result is True

    @patch('deforum.utils.system.forge_neo_detect._get_webui_root')
    @patch('deforum.utils.system.forge_neo_detect._check_marker_files')
    @patch('deforum.utils.system.forge_neo_detect._check_directory_name')
    def test_detects_via_directory_name(
        self, mock_check_dir, mock_check_markers, mock_get_root
    ):
        """Test detection via directory name."""
        mock_get_root.return_value = '/path/to/forge-neo'
        mock_check_markers.return_value = False
        mock_check_dir.return_value = True

        result = is_forge_neo()
        assert result is True

    @patch('deforum.utils.system.forge_neo_detect._get_webui_root')
    @patch('deforum.utils.system.forge_neo_detect._check_marker_files')
    @patch('deforum.utils.system.forge_neo_detect._check_directory_name')
    @patch('deforum.utils.system.forge_neo_detect._check_sys_path')
    def test_detects_via_sys_path(
        self, mock_check_sys, mock_check_dir, mock_check_markers, mock_get_root
    ):
        """Test detection via sys.path."""
        mock_get_root.return_value = '/path/to/webui'
        mock_check_markers.return_value = False
        mock_check_dir.return_value = False
        mock_check_sys.return_value = True

        result = is_forge_neo()
        assert result is True

    @patch('deforum.utils.system.forge_neo_detect._get_webui_root')
    @patch('deforum.utils.system.forge_neo_detect._check_marker_files')
    @patch('deforum.utils.system.forge_neo_detect._check_directory_name')
    @patch('deforum.utils.system.forge_neo_detect._check_sys_path')
    @patch('deforum.utils.system.forge_neo_detect._check_neo_modules')
    def test_detects_via_modules(
        self, mock_check_mods, mock_check_sys, mock_check_dir,
        mock_check_markers, mock_get_root
    ):
        """Test detection via Neo-specific modules."""
        mock_get_root.return_value = '/path/to/webui'
        mock_check_markers.return_value = False
        mock_check_dir.return_value = False
        mock_check_sys.return_value = False
        mock_check_mods.return_value = True

        result = is_forge_neo()
        assert result is True

    @patch('deforum.utils.system.forge_neo_detect._get_webui_root')
    @patch('deforum.utils.system.forge_neo_detect._check_marker_files')
    @patch('deforum.utils.system.forge_neo_detect._check_directory_name')
    @patch('deforum.utils.system.forge_neo_detect._check_sys_path')
    @patch('deforum.utils.system.forge_neo_detect._check_neo_modules')
    def test_not_neo_when_all_checks_fail(
        self, mock_check_mods, mock_check_sys, mock_check_dir,
        mock_check_markers, mock_get_root
    ):
        """Test returns False when all detection methods fail."""
        mock_get_root.return_value = '/path/to/webui'
        mock_check_markers.return_value = False
        mock_check_dir.return_value = False
        mock_check_sys.return_value = False
        mock_check_mods.return_value = False

        result = is_forge_neo()
        assert result is False

    @patch('deforum.utils.system.forge_neo_detect._get_webui_root')
    def test_not_neo_when_no_webui_root(self, mock_get_root):
        """Test returns False when webui root unavailable."""
        mock_get_root.return_value = None

        # All other checks should also fail
        with patch('deforum.utils.system.forge_neo_detect._check_sys_path', return_value=False):
            with patch('deforum.utils.system.forge_neo_detect._check_neo_modules', return_value=False):
                result = is_forge_neo()
                assert result is False

    @patch('deforum.utils.system.forge_neo_detect._get_webui_root')
    @patch('deforum.utils.system.forge_neo_detect.logger')
    def test_handles_exception_gracefully(self, mock_logger, mock_get_root):
        """Test exception handling during detection."""
        mock_get_root.side_effect = RuntimeError("Unexpected error")

        result = is_forge_neo()
        assert result is False
        assert mock_logger.warning.called


class TestGetForgeVariant:
    """Test suite for get_forge_variant function."""

    @patch('deforum.utils.system.forge_neo_detect.is_forge_neo')
    def test_returns_neo_when_detected(self, mock_is_neo):
        """Test returns 'Forge Neo' when Neo detected."""
        mock_is_neo.return_value = True
        result = get_forge_variant()
        assert result == "Forge Neo"

    @patch('deforum.utils.system.forge_neo_detect.is_forge_neo')
    def test_returns_forge_when_not_detected(self, mock_is_neo):
        """Test returns 'Forge' when Neo not detected."""
        mock_is_neo.return_value = False
        result = get_forge_variant()
        assert result == "Forge"


class TestCaching:
    """Test suite for caching functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_neo_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_neo_cache()

    @patch('deforum.utils.system.forge_neo_detect.is_forge_neo')
    def test_cached_version_calls_once(self, mock_is_neo):
        """Test cached version only calls is_forge_neo once."""
        mock_is_neo.return_value = True

        # First call
        result1 = is_forge_neo_cached()
        assert result1 is True

        # Second call should use cache
        result2 = is_forge_neo_cached()
        assert result2 is True

        # Should only be called once
        assert mock_is_neo.call_count == 1

    @patch('deforum.utils.system.forge_neo_detect.is_forge_neo')
    def test_clear_cache_allows_redetection(self, mock_is_neo):
        """Test clearing cache allows re-detection."""
        mock_is_neo.return_value = True

        # First call
        result1 = is_forge_neo_cached()
        assert result1 is True
        assert mock_is_neo.call_count == 1

        # Clear cache and change mock behavior
        clear_neo_cache()
        mock_is_neo.return_value = False

        # Should call is_forge_neo again
        result2 = is_forge_neo_cached()
        assert result2 is False
        assert mock_is_neo.call_count == 2

    def test_clear_cache_idempotent(self):
        """Test clearing cache multiple times is safe."""
        clear_neo_cache()
        clear_neo_cache()
        clear_neo_cache()
        # Should not raise any errors

    @patch('deforum.utils.system.forge_neo_detect.is_forge_neo')
    def test_cache_persists_across_calls(self, mock_is_neo):
        """Test cache persists for multiple calls."""
        mock_is_neo.return_value = True

        # Multiple calls should all use cached value
        for _ in range(5):
            result = is_forge_neo_cached()
            assert result is True

        # is_forge_neo should only be called once
        assert mock_is_neo.call_count == 1
