#!/usr/bin/env python3

import pytest
import sys
from pathlib import Path

# Add the deforum module to path to avoid package-level imports that require WebUI
_test_dir = Path(__file__).parent
_project_root = _test_dir.parent
sys.path.insert(0, str(_project_root))

# Import directly to avoid deforum.__init__ which requires WebUI modules
from deforum.integrations.wan.wan_model_validator import WanModelValidator

def test_wan_validator_discovery():
    """Test WAN model discovery functionality."""
    validator = WanModelValidator()
    models = validator.discover_models()
    
    assert isinstance(models, list)
    # Models list can be empty if no WAN models are installed
    for model in models:
        assert 'name' in model
        assert 'type' in model
        assert 'size_formatted' in model
        assert 'path' in model

def test_wan_validator_integrity():
    """Test WAN model integrity validation."""
    validator = WanModelValidator()
    
    # Test with non-existent path
    result = validator.validate_model_integrity(Path("non_existent_model.pth"))
    assert not result['valid']
    assert 'errors' in result
    assert len(result['errors']) > 0

if __name__ == "__main__":
    pytest.main([__file__]) 