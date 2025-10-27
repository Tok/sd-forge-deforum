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


def check_and_fix_huggingface_hub():
    """Check huggingface-hub version and fix Gradio compatibility if needed.

    Gradio 4.40.0 requires HfFolder class, which was removed in huggingface-hub 1.0.0.
    If 1.0.0+ is installed, downgrade to 0.26.2 (Forge's specified version).
    """
    try:
        hf_hub_version = importlib.metadata.version("huggingface-hub")
        major_version = int(hf_hub_version.split('.')[0])

        if major_version >= 1:
            print(f"[Deforum] Detected huggingface-hub {hf_hub_version} (incompatible with Gradio 4.40.0)")
            print("[Deforum] Downgrading to 0.26.2 for Gradio compatibility...")

            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "huggingface-hub==0.26.2",
                "--quiet",
            ])

            print("[Deforum] ✓ Downgraded to huggingface-hub 0.26.2")
            print("[Deforum]   Note: Wan features will use vendored 0.36.0 internally")

    except Exception as e:
        print(f"[Deforum] Warning: Failed to check huggingface-hub version: {e}")


# Fix huggingface-hub compatibility before anything else
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