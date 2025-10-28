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


def is_forge_neo_simple() -> bool:
    """
    Simple Neo detection for install (before full modules available).

    Checks:
    1. Current working directory for 'neo' in path
    2. __file__ path (install.py location) for 'neo' in parent directories
    3. sys.path for 'neo' in directory names

    Returns:
        True if likely running on Forge Neo, False otherwise
    """
    # Check current working directory
    cwd = os.getcwd()
    if 'forge-neo' in cwd.lower() or 'forge_neo' in cwd.lower():
        return True

    # Check this file's path (extensions/sd-forge-deforum/install.py)
    # If we're in forge-neo, the path will contain 'forge-neo'
    try:
        install_path = os.path.abspath(__file__)
        if 'forge-neo' in install_path.lower() or 'forge_neo' in install_path.lower():
            return True
    except:
        pass

    # Check sys.path
    for path in sys.path:
        if 'forge-neo' in path.lower() or 'forge_neo' in path.lower():
            return True

    return False


def check_and_fix_huggingface_hub():
    """Fix huggingface-hub compatibility BEFORE any other imports.

    This MUST run first to prevent ImportError when webui.py imports gradio.

    Version requirements:
    - Gradio 4.40.0: needs HfFolder (removed in 1.0.0+)
    - Forge: needs DDUFEntry (added in 0.27.0)
    - diffusers: needs >=0.34.0

    Solution: huggingface-hub 0.36.0

    Note: Skipped on Forge Neo (has correct versions built-in)
    """
    # Skip all compatibility patches on Forge Neo
    if is_forge_neo_simple():
        print("[Deforum] Running on Forge Neo - skipping compatibility patches")
        return

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


# FIX COMPATIBILITY FIRST - CRITICAL! (skip on Neo)
check_and_fix_huggingface_hub()

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

# Skip dependency upgrades on Forge Neo (has correct versions built-in)
if not is_forge_neo_simple():
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
else:
    print("Deforum: Running on Forge Neo - skipping dependency upgrades (using Neo's versions)")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not lib or lib.startswith('#'):
            continue

        # Force install git diffusers for Wan 2.2 support (skip on Neo)
        if lib.startswith('git+'):
            if not is_forge_neo_simple():
                print(f"Deforum: Installing diffusers from git for Wan 2.2 support...")
                launch.run_pip(f"install --upgrade {lib}", f"Deforum Wan 2.2 requirement: diffusers (git main)")
            else:
                print(f"Deforum: Skipping diffusers git install on Forge Neo")
            continue

        # Skip version-range packages already handled above
        if any(lib.startswith(pkg) for pkg in ['peft', 'accelerate']):
            continue

        # Skip stable-audio-tools (optional for Zero-HITL, Python 3.12 incompatible)
        if lib.startswith('stable-audio-tools'):
            try:
                if not launch.is_installed('stable_audio_tools'):
                    launch.run_pip(f"install {lib}", f"Deforum requirement: {lib}")
            except Exception as e:
                print(f"*** Deforum: stable-audio-tools install failed (Python 3.12 issue): {e}")
                print("*** Zero-HITL audio generation will be unavailable, but normal Deforum works fine.")
            continue

        # Install other packages normally
        if not launch.is_installed(lib.split('>=')[0].split('==')[0].split('<')[0]):
            launch.run_pip(f"install {lib}", f"Deforum requirement: {lib}")
