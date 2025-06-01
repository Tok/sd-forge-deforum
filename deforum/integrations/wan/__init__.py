#!/usr/bin/env python3

"""
Wan 2.1 Integration Wrapper for Deforum

This module provides a clean integration layer between Deforum and the original Wan 2.1 repository.

## Architecture:
- **External Repository**: Located at `deforum/integrations/external_repos/wan2.1/`
  - Contains the ORIGINAL, UNMODIFIED Wan 2.1 codebase
  - Should NOT be refactored or changed
  - Maintained 1:1 with upstream repository
  
- **Integration Wrapper**: This module (`deforum/integrations/wan/`)  
  - Contains Deforum-specific integration code
  - Handles argument conversion, UI integration, etc.
  - Can be refactored as part of Deforum modernization

## Usage:
```python
from deforum.integrations.wan import WanIntegration

wan = WanIntegration(model_path="path/to/wan/models")
result = wan.generate_video(prompt="A beautiful scene", frames=60)
```

## Key Components:
- `wan_simple_integration.py`: Main integration interface
- `wan_model_*`: Model management utilities  
- `integration/`: Advanced integration patterns
- `utils/`: Deforum-specific utilities
"""

import sys
import os
from pathlib import Path

# Add the external Wan 2.1 repository to Python path for imports
_current_dir = Path(__file__).parent
_external_wan_path = _current_dir / "external_repos" / "wan2.1"

if _external_wan_path.exists():
    sys.path.insert(0, str(_external_wan_path))

# Public API exports
from .wan_simple_integration import WanSimpleIntegration
from .wan_model_validator import WanModelValidator
from .wan_model_downloader import WanModelDownloader
from .wan_model_discovery import WanModelDiscovery
from .wan_model_cleanup import WanModelCleanup

__all__ = [
    'WanSimpleIntegration',
    'WanModelValidator', 
    'WanModelDownloader',
    'WanModelDiscovery',
    'WanModelCleanup',
]

# Version info
__version__ = "1.0.0"
__wan_version__ = "2.1"  # Version of integrated Wan repository 