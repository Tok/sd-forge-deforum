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
import datetime
# Contact the authors: https://deforum.github.io/

import os
import shutil

# Import fork identity constants (no dependencies, safe for early import)
from deforum.constants import FORK_NAME, GITHUB_URL

# noinspection PyUnresolvedReferences
from modules.shared import opts
from torch.hub import download_url_to_file

from deforum.utils.system.logging import log

# Import pure functions from refactored utils module
from deforum.utils.parsing.strings import (
    get_os,
    custom_placeholder_format,
    clean_gradio_path_strings,
    tick_or_cross as tickOrCross,
)
from deforum.utils.filesystem.files import (
    get_max_path_length as _get_max_path_length,
    count_files_in_folder,
)
from deforum.utils.math.interpolation import (
    extract_rife_name,
    clean_folder_name as _clean_folder_name,
    set_interp_out_fps,
    calculate_frames_to_add,
)
from deforum.utils.conversion.hashing import (
    compute_file_checksum_with_factory as checksum,
)

# Backward compatibility aliases
clean_folder_name = _clean_folder_name


def debug_print(message):
    is_debug_mode = opts.data.get("deforum_debug_mode_enabled", False)
    if is_debug_mode:
        log_utils.debug(message)


# checksum imported from deforum.utils.conversion.hashing

# get_os imported from deforum.utils.parsing.strings


# used in src/rife/inference_video.py and more, soon
def duplicate_pngs_from_folder(from_folder, to_folder, img_batch_id, orig_vid_name):
    import cv2
    # TODO: don't copy-paste at all if the input is a video (now it copy-pastes, and if input is deforum run is also converts to make sure no errors rise cuz of 24-32 bit depth differences)
    temp_convert_raw_png_path = os.path.join(from_folder, to_folder)
    os.makedirs(temp_convert_raw_png_path, exist_ok=True)

    frames_handled = 0
    for f in os.listdir(from_folder):
        # Match files that: start with batch ID, OR batch ID is None, OR start with a digit (numeric frame names)
        is_frame = (img_batch_id is not None and f.startswith(img_batch_id)) or \
                   (img_batch_id is None) or \
                   (f[0].isdigit() if f else False)
        if ('png' in f or 'jpg' in f) and '-' not in f and '_depth_' not in f and is_frame:
            frames_handled += 1
            original_img_path = os.path.join(from_folder, f)
            if orig_vid_name is not None:
                shutil.copy(original_img_path, temp_convert_raw_png_path)
            else:
                image = cv2.imread(original_img_path)
                new_path = os.path.join(temp_convert_raw_png_path, f)
                cv2.imwrite(new_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return frames_handled


def convert_images_from_list(paths, output_dir, format):
    import os
    from PIL import Image
    # Ensure that the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all input images
    for i, path in enumerate(paths):
        # Open the image
        with Image.open(path) as img:
            # Generate the output filename
            filename = f"{i + 1:09d}.{format}"
            # Save the image to the output directory
            img.save(os.path.join(output_dir, filename))


def get_deforum_version():
    ext = _get_extension_info()
    return ext.version if ext else "Unknown"


def get_commit_date():
    ext = _get_extension_info()
    formatted = datetime.datetime.fromtimestamp(ext.commit_date)
    return formatted if ext else "Unknown"


def _get_extension_info():
    # noinspection PyUnresolvedReferences
    from modules import extensions as mext
    try:
        for ext in mext.extensions:
            if ext.name in ["sd-forge-deforum"] and ext.enabled:
                ext.read_info_from_repo()  # need this call to get exten info on ui-launch, not to be removed
                return ext
        return None
    except Exception as e:
        log_utils.error(f"Cannot read extension info: {e}.")
        return None


# custom_placeholder_format imported from deforum.utils.parsing.strings


def test_long_path_support(base_folder_path):
    long_folder_name = 'A' * 300
    long_path = os.path.join(base_folder_path, long_folder_name)
    try:
        os.makedirs(long_path)
        shutil.rmtree(long_path)
        return True
    except OSError:
        return False


def get_max_path_length(base_folder_path):
    """Get maximum path length for OS (wrapper with side effects for testing)."""
    os_name = get_os()
    supports_long_paths = test_long_path_support(base_folder_path) if os_name == 'Windows' else False
    return _get_max_path_length(base_folder_path, os_name, supports_long_paths)


def substitute_placeholders(template, arg_list, base_folder_path):
    import re
    # Find and update timestring values if resume_from_timestring is True
    resume_from_timestring = next(
        (arg_obj.resume_from_timestring for arg_obj in arg_list if hasattr(arg_obj, 'resume_from_timestring')), False)
    resume_timestring = next(
        (arg_obj.resume_timestring for arg_obj in arg_list if hasattr(arg_obj, 'resume_timestring')), None)

    if resume_from_timestring and resume_timestring:
        for arg_obj in arg_list:
            if hasattr(arg_obj, 'timestring'):
                arg_obj.timestring = resume_timestring

    max_length = get_max_path_length(base_folder_path)
    values = {attr.lower(): getattr(arg_obj, attr)
              for arg_obj in arg_list
              for attr in dir(arg_obj) if not callable(getattr(arg_obj, attr)) and not attr.startswith('__')}
    
    # FIXED: Properly handle placeholder substitution without leaving stray characters
    # First, substitute valid placeholders
    formatted_string = re.sub(r"{(\w+)}", lambda m: custom_placeholder_format(values, m), template)
    # Then, clean up any remaining invalid placeholders or stray braces
    # Remove any remaining braces entirely instead of replacing with underscores
    formatted_string = re.sub(r'[{}]+', '', formatted_string)  # Remove any remaining braces completely
    formatted_string = re.sub(r'[<>:"/\\|?*\s,]', '_', formatted_string)
    # Clean up any trailing underscores that might result from the cleaning process
    formatted_string = formatted_string.rstrip('_')
    
    return formatted_string[:max_length]


# count_files_in_folder imported from deforum.utils.filesystem.files

# clean_gradio_path_strings imported from deforum.utils.parsing.strings


def download_file_with_checksum(url, expected_checksum, dest_folder, dest_filename):
    expected_full_path = os.path.join(dest_folder, dest_filename)
    if not os.path.exists(expected_full_path) and not os.path.isdir(expected_full_path):
        hash = None
        progress = True
        download_url_to_file(url, str(expected_full_path), hash, progress)
        if checksum(expected_full_path) != expected_checksum:
            raise Exception(f"Error while downloading {dest_filename}.]n" +
                            f"Please manually download from: {url}\nAnd place it in: {dest_folder}")


# tickOrCross imported from deforum.utils.parsing.strings (aliased from tick_or_cross)