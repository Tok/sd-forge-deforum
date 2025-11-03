"""Pytest configuration for Deforum tests.

This file is automatically loaded by pytest before running tests.
It configures the Python path so that the deforum package can be imported.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock Forge backend modules that aren't available in test environment
# This must happen BEFORE any imports that might load backend.loader
# Create proper mock package structure
mock_hf_guess = MagicMock()
mock_hf_guess_utils = MagicMock()
mock_hf_guess_utils.resize_to_batch_size = MagicMock()
mock_hf_guess.utils = mock_hf_guess_utils
mock_hf_guess.model_list = MagicMock()
sys.modules['huggingface_guess'] = mock_hf_guess
sys.modules['huggingface_guess.utils'] = mock_hf_guess_utils

# CRITICAL: Mock sys.argv BEFORE any imports
# modules.shared_cmd_options calls parse_args() at import time,
# which would fail with pytest's arguments
original_argv = sys.argv.copy()
sys.argv = ['webui.py']  # Minimal args that won't cause argparse errors

# Add the extension root directory to Python path
# This allows `from deforum.utils import ...` to work
extension_root = Path(__file__).parent.parent
sys.path.insert(0, str(extension_root))

# Add Forge root directory to Python path
# This allows `import modules` to work (Forge's modules package)
forge_root = extension_root.parent.parent
sys.path.insert(0, str(forge_root))

# Initialize Forge shared state BEFORE any imports that use it
# Many Forge modules access shared.opts, shared.options_templates at import time
try:
    import modules.shared as shared
    from modules.options import OptionInfo

    # Initialize minimal shared state to prevent AttributeError during imports
    if shared.options_templates is None:
        shared.options_templates = {}

    if shared.opts is None:
        # Create a minimal opts object that returns safe defaults for any attribute
        class MinimalOpts:
            def __init__(self):
                self.data = {}

            def get(self, key, default=None):
                return self.data.get(key, default)

            def __getattr__(self, name):
                # Return safe defaults for any attribute access
                # Return empty list for iterables (like hide_samplers),
                # None for other attributes
                # This prevents AttributeError during module imports
                return []

            def __contains__(self, key):
                return key in self.data

        shared.opts = MinimalOpts()

except Exception as e:
    # If initialization fails, tests may still work if they don't need these
    print(f"Warning: Could not initialize Forge shared state: {e}")

# Call deforum_sys_extend() to properly set up all paths
# This is required for Deforum to import properly
try:
    from scripts.deforum_extend_paths import deforum_sys_extend
    deforum_sys_extend()
except ImportError:
    # If we can't import it, the basic path setup above should work
    pass
