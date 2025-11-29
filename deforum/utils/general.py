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

from deforum.utils.system.logging import log, get_logger

# Get logger instance for this module
logger = get_logger()

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


def debug_print(message: str) -> None:
    """Print debug message if debug mode is enabled.

    Args:
        message: Debug message to print
    """
    is_debug_mode = opts.data.get("deforum_debug_mode_enabled", False)
    if is_debug_mode:
        logger.debug(message)


# checksum imported from deforum.utils.conversion.hashing

# get_os imported from deforum.utils.parsing.strings


def _is_valid_frame_file(filename, img_batch_id):
    """Check if filename is a valid frame file for processing.

    Args:
        filename: Name of file to check
        img_batch_id: Batch ID to match, or None to match all

    Returns:
        True if file should be processed as a frame
    """
    if not filename:
        return False

    # Check file extension
    is_image = 'png' in filename or 'jpg' in filename
    if not is_image:
        return False

    # Exclude depth and intermediate files
    if '-' in filename or '_depth_' in filename:
        return False

    # Match batch ID or numeric frame names
    matches_batch = (img_batch_id is not None and filename.startswith(img_batch_id))
    matches_numeric = filename[0].isdigit()
    is_valid_id = img_batch_id is None or matches_batch or matches_numeric

    return is_valid_id


def _copy_frame(original_path, dest_dir):
    """Copy frame file to destination (for video input).

    Args:
        original_path: Source file path
        dest_dir: Destination directory
    """
    shutil.copy(original_path, dest_dir)


def _reencode_frame(original_path, dest_dir, filename):
    """Reencode frame with CV2 to normalize bit depth (for deforum input).

    Args:
        original_path: Source file path
        dest_dir: Destination directory
        filename: Filename for output
    """
    import cv2
    image = cv2.imread(original_path)
    new_path = os.path.join(dest_dir, filename)
    cv2.imwrite(new_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# used in src/rife/inference_video.py and more, soon
def duplicate_pngs_from_folder(from_folder, to_folder, img_batch_id, orig_vid_name):
    """Duplicate PNG/JPG frames from folder, with optional re-encoding.

    For video input, frames are copied directly. For deforum runs, frames are
    re-encoded with CV2 to normalize bit depth (24-32 bit differences).

    Args:
        from_folder: Source directory containing frames
        to_folder: Destination directory name (created inside from_folder)
        img_batch_id: Batch ID prefix to match, or None to match all
        orig_vid_name: Original video name (if from video, enables copy mode)

    Returns:
        Number of frames processed
    """
    # TODO: don't copy-paste at all if the input is a video (now it copy-pastes,
    # and if input is deforum run is also converts to make sure no errors rise
    # cuz of 24-32 bit depth differences)
    temp_convert_raw_png_path = os.path.join(from_folder, to_folder)
    os.makedirs(temp_convert_raw_png_path, exist_ok=True)

    frames_handled = 0
    for f in os.listdir(from_folder):
        if not _is_valid_frame_file(f, img_batch_id):
            continue

        frames_handled += 1
        original_img_path = os.path.join(from_folder, f)

        # Video input: copy directly, Deforum run: re-encode to normalize bit depth
        if orig_vid_name is not None:
            _copy_frame(original_img_path, temp_convert_raw_png_path)
        else:
            _reencode_frame(original_img_path, temp_convert_raw_png_path, f)

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
    if not ext:
        return "Unknown"
    formatted = datetime.datetime.fromtimestamp(ext.commit_date)
    return formatted


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
        logger.error(f"Cannot read extension info: {e}.")
        return None


# custom_placeholder_format imported from deforum.utils.parsing.strings


def test_long_path_support(base_folder_path: str) -> bool:
    """Test if the OS supports long path names (>260 characters).

    Creates a test directory with a 300-character name and removes it.

    Args:
        base_folder_path: Base directory to test in

    Returns:
        True if long paths are supported, False otherwise
    """
    long_folder_name = 'A' * 300
    long_path = os.path.join(base_folder_path, long_folder_name)
    try:
        os.makedirs(long_path)
        shutil.rmtree(long_path)
        return True
    except OSError:
        return False


def get_max_path_length(base_folder_path: str) -> int:
    """Get maximum path length for OS (wrapper with side effects for testing).

    Args:
        base_folder_path: Base directory path to test

    Returns:
        Maximum path length supported by OS
    """
    os_name = get_os()
    supports_long_paths = test_long_path_support(base_folder_path) if os_name == 'Windows' else False
    return _get_max_path_length(base_folder_path, os_name, supports_long_paths)


def _find_resume_timestring(arg_list):
    """Find resume timestring settings from argument list.

    Args:
        arg_list: List of argument objects

    Returns:
        Tuple of (resume_from_timestring bool, resume_timestring str or None)
    """
    resume_from_timestring = next(
        (arg_obj.resume_from_timestring for arg_obj in arg_list
         if hasattr(arg_obj, 'resume_from_timestring')), False)
    resume_timestring = next(
        (arg_obj.resume_timestring for arg_obj in arg_list
         if hasattr(arg_obj, 'resume_timestring')), None)
    return resume_from_timestring, resume_timestring


def _update_timestrings_for_resume(arg_list, resume_timestring):
    """Update timestring attributes for resume mode (SIDE EFFECT).

    Args:
        arg_list: List of argument objects (modified in place)
        resume_timestring: Timestring value to set
    """
    for arg_obj in arg_list:
        if hasattr(arg_obj, 'timestring'):
            arg_obj.timestring = resume_timestring


def _build_values_dict(arg_list):
    """Build dictionary of all non-callable attributes from argument objects.

    Args:
        arg_list: List of argument objects

    Returns:
        Dictionary mapping lowercase attribute names to values
    """
    return {
        attr.lower(): getattr(arg_obj, attr)
        for arg_obj in arg_list
        for attr in dir(arg_obj)
        if not callable(getattr(arg_obj, attr)) and not attr.startswith('__')
    }


def _substitute_and_clean_template(template, values):
    """Substitute placeholders and clean invalid characters from template.

    Args:
        template: Template string with {placeholder} patterns
        values: Dictionary of placeholder values

    Returns:
        Cleaned string with placeholders substituted
    """
    import re
    # Substitute valid placeholders
    formatted = re.sub(
        r"{(\w+)}",
        lambda m: custom_placeholder_format(values, m),
        template
    )
    # Remove any remaining braces
    formatted = re.sub(r'[{}]+', '', formatted)
    # Replace invalid filename characters with underscores
    formatted = re.sub(r'[<>:"/\\|?*\s,]', '_', formatted)
    # Clean up trailing underscores
    return formatted.rstrip('_')


def substitute_placeholders(template, arg_list, base_folder_path):
    """Substitute placeholders in template string with values from arguments.

    Handles resume mode by updating timestrings if needed, builds a dictionary
    of all argument attributes, performs placeholder substitution, cleans
    invalid characters, and truncates to OS max path length.

    Args:
        template: Template string with {placeholder} patterns
        arg_list: List of argument objects containing values
        base_folder_path: Base folder path for max length calculation

    Returns:
        Formatted string with substituted values, cleaned and truncated
    """
    # Handle resume mode
    resume_from_timestring, resume_timestring = _find_resume_timestring(arg_list)
    if resume_from_timestring and resume_timestring:
        _update_timestrings_for_resume(arg_list, resume_timestring)

    # Build values and perform substitution
    values = _build_values_dict(arg_list)
    formatted_string = _substitute_and_clean_template(template, values)

    # Truncate to max path length
    max_length = get_max_path_length(base_folder_path)
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