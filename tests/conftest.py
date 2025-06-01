"""
Pytest configuration and shared fixtures for Deforum tests
"""

import os
import sys
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Add the scripts directory to sys.path for imports
REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Test data directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_DATA_DIR = FIXTURES_DIR / "testdata"


@pytest.fixture(scope="session")
def repo_root():
    """Path to the repository root directory"""
    return REPO_ROOT


@pytest.fixture(scope="session")
def scripts_dir():
    """Path to the scripts directory"""
    return SCRIPTS_DIR


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to the test data directory"""
    return TEST_DATA_DIR


@pytest.fixture
def sample_animation_args():
    """Sample animation arguments for testing"""
    args = SimpleNamespace()
    args.translation_x = "0:(0)"
    args.translation_y = "0:(0)"
    args.translation_z = "0:(0)"
    args.rotation_3d_x = "0:(0)"
    args.rotation_3d_y = "0:(0)"
    args.rotation_3d_z = "0:(0)"
    args.zoom = "0:(1.0)"
    args.angle = "0:(0)"
    args.max_frames = 120
    args.shake_name = "None"
    args.shake_intensity = 1.0
    args.shake_speed = 1.0
    return args


@pytest.fixture
def sample_movement_args():
    """Sample animation arguments with movement for testing"""
    args = SimpleNamespace()
    args.translation_x = "0:(0), 30:(10), 60:(0)"
    args.translation_y = "0:(0), 20:(5), 40:(-5), 60:(0)"
    args.translation_z = "0:(0), 45:(15)"
    args.rotation_3d_x = "0:(0), 30:(2)"
    args.rotation_3d_y = "0:(0), 15:(5), 45:(-3)"
    args.rotation_3d_z = "0:(0)"
    args.zoom = "0:(1.0), 30:(1.2), 60:(1.0)"
    args.angle = "0:(0)"
    args.max_frames = 60
    args.shake_name = "None"
    args.shake_intensity = 1.0
    args.shake_speed = 1.0
    return args


@pytest.fixture
def sample_shakify_args():
    """Sample animation arguments with Camera Shakify enabled"""
    args = SimpleNamespace()
    args.translation_x = "0:(0)"
    args.translation_y = "0:(0)"
    args.translation_z = "0:(0)"
    args.rotation_3d_x = "0:(0)"
    args.rotation_3d_y = "0:(0)"
    args.rotation_3d_z = "0:(0)"
    args.zoom = "0:(1.0)"
    args.angle = "0:(0)"
    args.max_frames = 60
    args.shake_name = "INVESTIGATION"
    args.shake_intensity = 1.0
    args.shake_speed = 1.0
    return args


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing"""
    return {
        "0": "a beautiful landscape",
        "30": "a serene forest",
        "60": "a bustling city"
    }


@pytest.fixture
def sample_enhanced_prompts():
    """Sample enhanced prompts for testing"""
    return {
        "0": "a beautiful landscape with rolling hills and vibrant colors, photorealistic",
        "30": "a serene forest with tall trees and dappled sunlight, cinematic lighting",
        "60": "a bustling city with modern architecture and busy streets, urban photography"
    }


@pytest.fixture
def mock_qwen_manager():
    """Mock Qwen manager for testing prompt enhancement"""
    with patch('deforum_helpers.wan.utils.qwen_manager.qwen_manager') as mock:
        mock.is_model_loaded.return_value = False
        mock.is_model_downloaded.return_value = True
        mock.auto_select_model.return_value = "Qwen2.5-7B-Instruct"
        mock.enhance_prompts.return_value = {
            "0": "enhanced prompt 1",
            "60": "enhanced prompt 2"
        }
        yield mock


@pytest.fixture
def mock_file_system():
    """Mock file system for testing file operations"""
    mock_fs = Mock()
    mock_fs.read_file.return_value = '{"test": "data"}'
    mock_fs.write_file.return_value = None
    mock_fs.exists.return_value = True
    return mock_fs


@pytest.fixture
def temp_settings_file(tmp_path):
    """Create a temporary settings file for testing"""
    settings_content = """
{
    "animation_mode": "3D",
    "max_frames": 120,
    "translation_x": "0:(0), 60:(10)",
    "translation_y": "0:(0)",
    "translation_z": "0:(0)",
    "prompts": {
        "0": "test prompt 1",
        "60": "test prompt 2"
    }
}
"""
    settings_file = tmp_path / "test_settings.txt"
    settings_file.write_text(settings_content.strip())
    return settings_file


@pytest.fixture(autouse=True)
def clean_imports():
    """Clean up imports after each test to avoid state pollution"""
    yield
    # Remove any modules that might have been imported during tests
    modules_to_remove = [
        mod for mod in sys.modules.keys() 
        if mod.startswith('deforum_helpers') or mod.startswith('wan')
    ]
    for mod in modules_to_remove:
        if mod in sys.modules:
            del sys.modules[mod]


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may have controlled dependencies)"
    )
    config.addinivalue_line(
        "markers", "functional: End-to-end functional tests (slow)"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmark tests"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "network: Tests that require network access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location"""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "functional" in str(item.fspath):
            item.add_marker(pytest.mark.functional)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker for tests that might be slow
        if any(keyword in item.name.lower() for keyword in ['slow', 'benchmark', 'performance']):
            item.add_marker(pytest.mark.slow)
        
        # Add GPU marker for tests that require GPU
        if any(keyword in item.name.lower() for keyword in ['gpu', 'cuda', 'torch']):
            item.add_marker(pytest.mark.gpu) 