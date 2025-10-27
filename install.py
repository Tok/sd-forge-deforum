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

import subprocess
import sys
import launch
import os


def check_and_fix_huggingface_hub():
    """Fix huggingface-hub compatibility BEFORE any other imports.

    This MUST run first to prevent ImportError when webui.py imports gradio.

    Version requirements:
    - Gradio 4.40.0: needs HfFolder (removed in 1.0.0+)
    - Forge: needs DDUFEntry (added in 0.27.0)
    - diffusers: needs >=0.34.0

    Solution: huggingface-hub 0.36.0
    """
    try:
        import importlib.metadata
        hf_hub_version = importlib.metadata.version("huggingface-hub")
        version_parts = hf_hub_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        needs_fix = major >= 1 or (major == 0 and minor < 27)

        if needs_fix:
            reason = ">= 1.0 (removed HfFolder)" if major >= 1 else f"< 0.27 (missing DDUFEntry)"
            print(f"[Deforum] huggingface-hub {hf_hub_version} incompatible ({reason})")
            print("[Deforum] Installing 0.36.0...")

            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "huggingface-hub==0.36.0",
                "--quiet",
            ])

            print("[Deforum] ✓ Installed huggingface-hub 0.36.0")
    except Exception as e:
        print(f"[Deforum] Warning: Could not check huggingface-hub: {e}")


# FIX COMPATIBILITY FIRST - CRITICAL!
check_and_fix_huggingface_hub()

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

print("Deforum: Installing dependencies for Wan 2.2 support...")

# Force upgrade critical dependencies for Wan 2.2 TI2V support
critical_upgrades = {
    'peft': '0.17.1',
    'accelerate': '1.10.1',
}

for package, version in critical_upgrades.items():
    try:
        import importlib.metadata
        current_version = importlib.metadata.version(package)
        if current_version != version:
            print(f"Deforum: Upgrading {package} {current_version} → {version} for Wan 2.2...")
            launch.run_pip(f"install {package}=={version}", f"Deforum Wan 2.2 requirement: {package}=={version}")
    except:
        print(f"Deforum: Installing {package}=={version}...")
        launch.run_pip(f"install {package}=={version}", f"Deforum Wan 2.2 requirement: {package}=={version}")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not lib or lib.startswith('#'):
            continue

        # Force install git diffusers for Wan 2.2 support
        if lib.startswith('git+'):
            print(f"Deforum: Installing diffusers from git for Wan 2.2 support...")
            launch.run_pip(f"install --upgrade {lib}", f"Deforum Wan 2.2 requirement: diffusers (git main)")
            continue

        # Skip version-range packages already handled above
        if any(lib.startswith(pkg) for pkg in ['peft', 'accelerate']):
            continue

        # Install other packages normally
        if not launch.is_installed(lib.split('>=')[0].split('==')[0].split('<')[0]):
            launch.run_pip(f"install {lib}", f"Deforum requirement: {lib}")
