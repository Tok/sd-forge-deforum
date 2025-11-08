"""Pytest configuration for Deforum tests.

This file is automatically loaded by pytest before running tests.
It configures the Python path so that the deforum package can be imported.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# CRITICAL: Mock modules package BEFORE any imports that might use it
# This must happen BEFORE any imports, including deforum imports
# Many modules import from 'modules.shared' at the top level

# Mock emoji utilities for consistent test behavior
class MockEmojiUtils:
    """Mock emoji utilities with all required emoji functions."""
    @staticmethod
    def maybe_check(): return '✓'
    @staticmethod
    def maybe_cross(): return '✗'
    @staticmethod
    def maybe_warning(): return '⚠'
    @staticmethod
    def download(): return '⬇'
    @staticmethod
    def trash(): return '🗑'
    @staticmethod
    def wrench(): return '🔧'
    @staticmethod
    def bulb(): return '💡'
    @staticmethod
    def signal(): return '📶'
    @staticmethod
    def save(): return '💾'
    @staticmethod
    def refresh_icon(): return '🔄'
    @staticmethod
    def memo(): return '📝'
    @staticmethod
    def movie_camera(): return '🎬'
    @staticmethod
    def target(): return '🎯'
    @staticmethod
    def rocket(): return '🚀'
    @staticmethod
    def chart_increasing(): return '📈'
    @staticmethod
    def palette(): return '🎨'
    @staticmethod
    def hourglass(): return '⏳'
    @staticmethod
    def sleeping(): return '💤'
    @staticmethod
    def fire(): return '🔥'

    @staticmethod
    def get_themed_emoji(emoji_name: str, theme: str = 'classic') -> str:
        """Get emoji based on theme (mock always returns emoji)."""
        # Try to call the corresponding method if it exists
        if hasattr(MockEmojiUtils, emoji_name):
            return getattr(MockEmojiUtils, emoji_name)()
        return '📦'  # Default emoji for unknown names

# Create minimal opts mock with data dict
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

# Mock modules.shared package
mock_shared = MagicMock()
mock_shared.opts = MinimalOpts()
mock_shared.options_templates = {}
mock_shared.cmd_opts = MagicMock()
mock_shared.state = MagicMock()

# Mock modules.options for OptionInfo
mock_options = MagicMock()
mock_options.OptionInfo = MagicMock

# Mock modules.extensions for extension detection
# Use MagicMock to allow test patching to work correctly
mock_extensions_module = MagicMock()
mock_extensions_module.extensions = []
mock_extensions_module.Extension = MagicMock

# Install mocks into sys.modules BEFORE any imports
mock_modules = MagicMock()
mock_paths = MagicMock()
mock_paths.models_path = "/tmp/models"
mock_paths.script_path = "/tmp/forge"
mock_processing = MagicMock()
mock_processing.get_fixed_seed = lambda x: x if x != -1 else 42
mock_ui = MagicMock()
mock_ui.create_output_panel = MagicMock(return_value=[])
mock_ui.wrap_gradio_call = lambda fn: fn
mock_util = MagicMock()
mock_util.open_folder = MagicMock()
mock_call_queue = MagicMock()
mock_call_queue.wrap_gradio_gpu_call = lambda fn: fn
sys.modules['modules'] = mock_modules
sys.modules['modules.shared'] = mock_shared
sys.modules['modules.options'] = mock_options
sys.modules['modules.extensions'] = mock_extensions_module
sys.modules['modules.shared_cmd_options'] = MagicMock()
sys.modules['modules.paths'] = mock_paths
sys.modules['modules.processing'] = mock_processing
sys.modules['modules.ui'] = mock_ui
sys.modules['modules.util'] = mock_util
sys.modules['modules.call_queue'] = mock_call_queue
sys.modules['modules.scripts'] = MagicMock()
sys.modules['modules.images'] = MagicMock()
sys.modules['modules.sd_models'] = MagicMock()

# CRITICAL: Link module attributes so 'from modules import X' works correctly
# When code does 'from modules import extensions', Python checks sys.modules['modules'].extensions
# So we need to make sure that points to the same object as sys.modules['modules.extensions']
mock_modules.shared = mock_shared
mock_modules.options = mock_options
mock_modules.extensions = mock_extensions_module
mock_modules.paths = mock_paths
mock_modules.processing = mock_processing
mock_modules.ui = mock_ui
mock_modules.util = mock_util

# Mock deforum emoji utilities
sys.modules['deforum.utils.system.logging.emoji'] = MockEmojiUtils

# Mock Forge backend modules that aren't available in test environment
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

# Try to add Forge root if it exists (for local development)
# In CI/GitHub Actions, this won't exist, but mocks above handle it
forge_root = extension_root.parent.parent
if forge_root.exists():
    sys.path.insert(0, str(forge_root))

# Initialize Forge shared state BEFORE any imports that use it
# Many Forge modules access shared.opts, shared.options_templates at import time
try:
    import modules.shared as shared
    from modules.options import OptionInfo

    # Initialize minimal shared state to prevent AttributeError during imports
    if shared.options_templates is None:
        shared.options_templates = {}

    if shared.opts is None or not isinstance(shared.opts, MinimalOpts):
        # Use the MinimalOpts class defined above
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
