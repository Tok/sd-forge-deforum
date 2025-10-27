# 'Deforum' plugin for Automatic1111's Stable Diffusion WebUI.
# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import sys
import os
import subprocess
import importlib.metadata

# Legacy path no longer needed - all code migrated to deforum/ package
# Add extension root to sys.path so we can import deforum package in preload
extension_root = os.path.dirname(os.path.abspath(__file__))
if extension_root not in sys.path:
    sys.path.insert(0, extension_root)


def is_forge_neo_simple() -> bool:
    """
    Simple Neo detection for preload (before full modules available).

    Checks:
    1. Current working directory for 'neo' in path
    2. __file__ path (preload.py location) for 'neo' in parent directories
    3. sys.path for 'neo' in directory names

    Returns:
        True if likely running on Forge Neo, False otherwise
    """
    # Check current working directory
    cwd = os.getcwd()
    if 'forge-neo' in cwd.lower() or 'forge_neo' in cwd.lower():
        return True

    # Check this file's path (extensions/sd-forge-deforum/preload.py)
    # If we're in forge-neo, the path will contain 'forge-neo'
    try:
        preload_path = os.path.abspath(__file__)
        if 'forge-neo' in preload_path.lower() or 'forge_neo' in preload_path.lower():
            return True
    except:
        pass

    # Check sys.path
    for path in sys.path:
        if 'forge-neo' in path.lower() or 'forge_neo' in path.lower():
            return True

    return False


def check_and_fix_huggingface_hub():
    """Check huggingface-hub version and fix compatibility if needed.

    Requires:
    - Gradio 4.40.0 needs HfFolder (removed in 1.0.0+)
    - Forge needs DDUFEntry (added in 0.27.0)
    - diffusers needs >=0.34.0

    Solution: Use 0.36.0 (last version before 1.0 breaking changes)

    Note: Skipped on Forge Neo (has correct versions built-in)
    """
    # Skip all compatibility patches on Forge Neo
    if is_forge_neo_simple():
        print("[Deforum] Running on Forge Neo - skipping compatibility patches")
        return

    try:
        hf_hub_version = importlib.metadata.version("huggingface-hub")
        version_parts = hf_hub_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        # Check if version is incompatible
        needs_fix = major >= 1 or (major == 0 and minor < 27)

        if needs_fix:
            reason = ">= 1.0 (removed HfFolder)" if major >= 1 else f"< 0.27 (missing DDUFEntry)"
            print(f"[Deforum] Detected huggingface-hub {hf_hub_version} ({reason})")
            print("[Deforum] Upgrading to 0.36.0 for compatibility...")

            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "huggingface-hub==0.36.0",
                "--quiet",
            ])

            print("[Deforum] ✓ Installed huggingface-hub 0.36.0")
            print("[Deforum]   Compatible with Gradio, Forge, and diffusers")

    except Exception as e:
        print(f"[Deforum] Warning: Failed to check huggingface-hub version: {e}")


# Fix huggingface-hub compatibility before anything else (skip on Neo)
check_and_fix_huggingface_hub()

try:
    from deforum.utils.system.startup_banner import print_startup_banner
    print_startup_banner()
except Exception as e:
    print(f"[Deforum] Warning: Could not print startup banner: {e}")

def preload(parser):
    parser.add_argument(
        "--deforum-api",
        action="store_true",
        help="Enable the Deforum API",
        default=False,  # Must be False, not None, for proper FlagsModel type inference
    )
    parser.add_argument(
        "--deforum-simple-api",
        action="store_true",
        help="Enable the simplified version of Deforum API",
        default=False,  # Must be False, not None, for proper FlagsModel type inference
    )
    parser.add_argument(
        "--deforum-run-now",
        type=str,
        help="Comma-delimited list of deforum settings files to run immediately on startup",
        default=None,
    )
    parser.add_argument(
        "--deforum-terminate-after-run-now",
        action="store_true",
        help="Whether to shut down the a1111 process immediately after completing the generations passed in to '--deforum-run-now'.",
        default=False,  # Must be False, not None, for proper FlagsModel type inference
    )
    parser.add_argument(
        "--deforum-run-tuning",
        action="store_true",
        help="Enable parameter tuning mode: launches Deforum API and shows Tuning tab for automated quality assessment",
        default=False,
    )