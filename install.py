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

import launch
import os

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not lib or lib.startswith('#'):
            continue
        
        # Special handling for diffusers/transformers/accelerate - only install if not present
        # Don't force upgrades to avoid breaking webui-forge compatibility
        if lib.startswith('diffusers>=') or lib.startswith('transformers>=') or lib.startswith('accelerate>='):
            package_name = lib.split('>=')[0].split('<')[0]  # Handle version ranges like "diffusers>=0.26.0,<0.33.0"
            if not launch.is_installed(package_name):
                print(f"Installing {package_name} for Wan compatibility...")
                launch.run_pip(f"install {lib}", f"Deforum Wan requirement: {lib}")
            else:
                print(f"Skipping {package_name} upgrade to maintain webui-forge compatibility")
        else:
            # Normal installation for other packages
            if not launch.is_installed(lib):
                launch.run_pip(f"install {lib}", f"Deforum requirement: {lib}")
