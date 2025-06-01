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

import json
import os
import sys
import shutil
import time
from datetime import datetime
from types import SimpleNamespace

import modules.shared as sh
from modules.sd_models import FakeInitialModel

from .args import DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, ParseqArgs, get_settings_component_names, \
    pack_args, WanArgs
from .defaults import mask_fill_choices, get_camera_shake_list
from .deforum_controlnet import controlnet_component_names
from .deprecation_utils import handle_deprecated_settings
from .general_utils import get_deforum_version, clean_gradio_path_strings


def get_extension_base_dir():
    """Return the base directory of the extension"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def get_default_settings_path():
    """Return the path to the default settings file in the extension directory"""
    return os.path.join(get_extension_base_dir(), "scripts", "default_settings.txt")

def get_keys_to_exclude():
    # init_image_box is PIL object not string, so ignore.
    return ['init_image_box']

def validate_and_migrate_settings(settings_path, jdata):
    """
    Validate settings file and handle outdated configurations.
    Returns (is_valid, migrated_data, warnings_list)
    """
    warnings = []
    is_outdated = False
    missing_fields = []
    
    # Get current expected fields from all argument functions
    expected_fields = set()
    for args_func in [DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, ParseqArgs, WanArgs]:
        expected_fields.update(args_func().keys())
    
    # Add other expected fields
    expected_fields.update(['prompts', 'animation_prompts_positive', 'animation_prompts_negative'])
    
    # Remove excluded fields
    excluded_fields = set(get_keys_to_exclude())
    expected_fields -= excluded_fields
    
    # Check for missing fields
    current_fields = set(jdata.keys())
    missing_fields = expected_fields - current_fields
    
    # Check for new WanArgs fields specifically (these were added recently)
    wan_fields = set(WanArgs().keys())
    missing_wan_fields = wan_fields - current_fields
    
    if missing_fields:
        is_outdated = True
        warnings.append(f"Settings file is missing {len(missing_fields)} fields")
        
        if missing_wan_fields:
            warnings.append(f"Missing new Wan 2.1 fields: {', '.join(sorted(missing_wan_fields))}")
    
    # Check for very old files (pre-Zirteq fork indicators)
    old_indicators = ['use_zoe_depth', 'histogram_matching', 'depth_adabins', 'depth_leres']
    if any(field in jdata for field in old_indicators):
        is_outdated = True
        warnings.append("Settings file contains deprecated depth algorithms")
    
    # Check git commit to see if it's from original Deforum
    git_commit = jdata.get('deforum_git_commit_id', '')
    if not git_commit or 'forge' not in git_commit.lower():
        is_outdated = True
        warnings.append("Settings file appears to be from original A1111 Deforum")
    
    # Apply migrations and fill missing fields with defaults
    migrated_data = jdata.copy()
    
    # Fill in missing fields with defaults
    for args_func in [DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, ParseqArgs, WanArgs]:
        defaults = args_func()
        for field, config in defaults.items():
            if field not in migrated_data and field not in excluded_fields:
                # Get the default value from the config
                if isinstance(config, dict) and 'value' in config:
                    default_value = config['value']
                else:
                    default_value = config
                migrated_data[field] = default_value
                warnings.append(f"Added missing field '{field}' with default value")
    
    return not is_outdated, migrated_data, warnings

def backup_settings_file(settings_path):
    """Create a backup of the settings file with timestamp"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{settings_path}.backup_{timestamp}"
        shutil.copy2(settings_path, backup_path)
        return backup_path
    except Exception as e:
        print(f"Error creating backup: {e}")
        return None

def handle_outdated_settings_file(settings_path):
    """
    Handle an outdated settings file by offering user options.
    Returns True if settings should be loaded, False if should use defaults.
    """
    print(f"\n{'-'*60}")
    print("🚨 OUTDATED SETTINGS FILE DETECTED 🚨")
    print(f"File: {settings_path}")
    print(f"{'-'*60}")
    print("This settings file appears to be from an older version of Deforum")
    print("and may be missing important new features or contain deprecated settings.")
    print()
    print("Recommended actions:")
    print("1. Backup the old file and use updated defaults (RECOMMENDED)")
    print("2. Try to automatically migrate the settings (MAY HAVE ISSUES)")
    print("3. Use defaults and ignore the outdated file")
    print()
    
    # For automated/headless operation, default to migration
    if hasattr(sh, 'cmd_opts') and getattr(sh.cmd_opts, 'api_only', False):
        print("Running in API mode - automatically migrating settings...")
        return True
    
    # Create backup automatically
    backup_path = backup_settings_file(settings_path)
    if backup_path:
        print(f"✅ Backup created: {backup_path}")
    
    print("Proceeding with automatic migration...")
    print("(The original file will be kept as a backup)")
    print(f"{'-'*60}\n")
    
    return True

def load_args(args_dict_main, args, anim_args, parseq_args, loop_args, controlnet_args, video_args, custom_settings_file, root, run_id):
    custom_settings_file = custom_settings_file[run_id]
    print(f"reading custom settings from {custom_settings_file.name}")
    if not os.path.isfile(custom_settings_file.name):
        print('Custom settings file does not exist. Using in-notebook settings.')
        return True
    
    try:
        with open(custom_settings_file.name, "r") as f:
            jdata = json.loads(f.read())
    except Exception as e:
        print(f"❌ Error loading settings file: {e}")
        return False
    
    # Validate and potentially migrate settings
    is_valid, migrated_data, warnings = validate_and_migrate_settings(custom_settings_file.name, jdata)
    
    if warnings:
        print(f"\n⚠️  Settings validation warnings:")
        for warning in warnings[:10]:  # Limit to first 10 warnings
            print(f"   • {warning}")
        if len(warnings) > 10:
            print(f"   ... and {len(warnings) - 10} more warnings")
        print()
    
    if not is_valid:
        should_migrate = handle_outdated_settings_file(custom_settings_file.name)
        if not should_migrate:
            print("Using default settings instead of outdated file.")
            return True
        
        # Update the data with migrated version
        jdata = migrated_data
        
        # Save the migrated settings back to the file
        try:
            with open(custom_settings_file.name, "w", encoding='utf-8') as f:
                json.dump(jdata, f, ensure_ascii=False, indent=4)
            print(f"✅ Settings file updated with missing fields")
        except Exception as e:
            print(f"⚠️  Could not update settings file: {e}")
    
    # Continue with normal loading process
    handle_deprecated_settings(jdata)
    root.animation_prompts = jdata.get("prompts", root.animation_prompts)
    
    if "animation_prompts_positive" in jdata:
        args_dict_main['animation_prompts_positive'] = jdata["animation_prompts_positive"]
    if "animation_prompts_negative" in jdata:
        args_dict_main['animation_prompts_negative'] = jdata["animation_prompts_negative"]
    
    keys_to_exclude = get_keys_to_exclude()
    
    # Also create wan_args namespace from args_dict_main for loading
    from .args import WanArgs
    wan_args = SimpleNamespace(**{name: args_dict_main[name] for name in WanArgs() if name in args_dict_main})
    
    for args_namespace in [args, anim_args, parseq_args, loop_args, controlnet_args, video_args, wan_args]:
        for k, v in vars(args_namespace).items():
            if k not in keys_to_exclude:
                if k in jdata:
                    setattr(args_namespace, k, jdata[k])
                else:
                    print(f"Key {k} doesn't exist in the custom settings data! Using default value of {v}")
    
    return True

# save settings function that get calls when run_deforum is being called
def save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root, full_out_file_path = None, wan_args = None):
    if full_out_file_path:
        args.__dict__["seed"] = root.raw_seed
        args.__dict__["batch_name"] = root.raw_batch_name
    args.__dict__["prompts"] = root.animation_prompts
    args.__dict__["positive_prompts"] = args.positive_prompts
    args.__dict__["negative_prompts"] = args.negative_prompts
    exclude_keys = get_keys_to_exclude()
    settings_filename = full_out_file_path if full_out_file_path else os.path.join(args.outdir, f"{root.timestring}_settings.txt")
    with open(settings_filename, "w+", encoding="utf-8") as f:
        s = {}
        # Include all argument dictionaries, including wan_args if provided
        dicts_to_merge = [args.__dict__, anim_args.__dict__, parseq_args.__dict__, loop_args.__dict__, controlnet_args.__dict__, video_args.__dict__]
        if wan_args is not None:
            dicts_to_merge.append(wan_args.__dict__)
        
        for d in dicts_to_merge:
            s.update({k: v for k, v in d.items() if k not in exclude_keys})
        s["sd_model_name"] = sh.sd_model.sd_checkpoint_info.name
        s["sd_model_hash"] = sh.sd_model.sd_checkpoint_info.hash
        s["deforum_git_commit_id"] = get_deforum_version()
        json.dump(s, f, ensure_ascii=False, indent=4)

# In gradio gui settings save/ load funcs:
def save_settings(*args, **kwargs):
    settings_path = args[0].strip()
    settings_path = clean_gradio_path_strings(settings_path)
    
    # If path is empty, use a default path in the webui root
    if not settings_path:
        from modules import paths_internal
        settings_path = os.path.join(paths_internal.script_path, "deforum_settings.txt")
        print(f"No settings path provided, using default path in webui root: {settings_path}")
    
    settings_path = os.path.realpath(settings_path)
    
    # Create directory if it doesn't exist
    settings_dir = os.path.dirname(settings_path)
    if not os.path.exists(settings_dir) and settings_dir != '':
        try:
            os.makedirs(settings_dir, exist_ok=True)
            print(f"Created directory: {settings_dir}")
        except Exception as e:
            print(f"Error creating directory {settings_dir}: {str(e)}")
            # If we can't create the directory, save to the webui root as fallback
            from modules import paths_internal
            settings_path = os.path.join(paths_internal.script_path, "deforum_settings.txt")
            print(f"Falling back to saving in webui root: {settings_path}")
    
    settings_component_names = get_settings_component_names()
    data = {settings_component_names[i]: args[i+1] for i in range(0, len(settings_component_names))}
    args_dict = pack_args(data, DeforumArgs)
    anim_args_dict = pack_args(data, DeforumAnimArgs)
    parseq_dict = pack_args(data, ParseqArgs)
    
    # Handle animation prompts with proper error checking
    try:
        args_dict["prompts"] = json.loads(data['animation_prompts'])
    except json.JSONDecodeError as e:
        print(f"Error parsing animation prompts JSON: {str(e)}")
        # Use empty prompts as fallback if JSON is invalid
        args_dict["prompts"] = {}
    
    args_dict["animation_prompts_positive"] = data['animation_prompts_positive']
    args_dict["animation_prompts_negative"] = data['animation_prompts_negative']
    loop_dict = {}
    controlnet_dict = pack_args(data, controlnet_component_names)
    wan_args_dict = pack_args(data, WanArgs)
    video_args_dict = pack_args(data, DeforumOutputArgs)
    combined = {**args_dict, **anim_args_dict, **parseq_dict, **loop_dict, **controlnet_dict, **wan_args_dict, **video_args_dict}
    exclude_keys = get_keys_to_exclude()
    filtered_combined = {k: v for k, v in combined.items() if k not in exclude_keys}
    
    # Add metadata to settings file
    if not isinstance(sh.sd_model, FakeInitialModel):
        filtered_combined["sd_model_name"] = sh.sd_model.sd_checkpoint_info.name
        filtered_combined["sd_model_hash"] = sh.sd_model.sd_checkpoint_info.hash
    filtered_combined["deforum_git_commit_id"] = get_deforum_version()
    
    # Save the file with error handling
    try:
        print(f"Saving settings to {settings_path}")
        with open(settings_path, "w", encoding='utf-8') as f:
            f.write(json.dumps(filtered_combined, ensure_ascii=False, indent=4))
        print(f"Settings saved successfully to {settings_path}")
    except Exception as e:
        print(f"Error saving settings to {settings_path}: {str(e)}")
        # Try to save to webui root as fallback
        try:
            from modules import paths_internal
            fallback_path = os.path.join(paths_internal.script_path, "deforum_settings.txt")
            print(f"Attempting to save to fallback location: {fallback_path}")
            with open(fallback_path, "w", encoding='utf-8') as f:
                f.write(json.dumps(filtered_combined, ensure_ascii=False, indent=4))
            print(f"Settings saved to fallback location: {fallback_path}")
        except Exception as e2:
            print(f"Error saving to fallback location: {str(e2)}")
    
    # Return empty message to clear any previous messages
    return [""]

def update_settings_path(path):
    """Updates the settings path field after loading settings"""
    return path


def load_all_settings(*args, ui_launch=False, update_path=False, **kwargs):
    import gradio as gr
    settings_path = args[0].strip()
    settings_path = clean_gradio_path_strings(settings_path)
    settings_path = os.path.realpath(settings_path)
    settings_component_names = get_settings_component_names()
    data = {settings_component_names[i]: args[i+1] for i in range(len(settings_component_names))}
    
    # First check webui root for deforum_settings.txt if no specific path is provided
    if settings_path == get_default_settings_path() or not os.path.exists(settings_path):
        # Check for a settings file in webui root
        from modules import paths_internal
        webui_root_settings = os.path.join(paths_internal.script_path, "deforum_settings.txt")
        if os.path.isfile(webui_root_settings):
            print(f"Using settings file from webui root: {webui_root_settings}")
            settings_path = webui_root_settings
    
    # Check if the file exists, if not fall back to default settings
    if not os.path.isfile(settings_path):
        default_path = get_default_settings_path()
        print(f"The settings file '{settings_path}' does not exist. Using default settings from {default_path}")
        settings_path = default_path
        # If default file also doesn't exist, return unchanged data
        if not os.path.isfile(settings_path):
            print(f"Default settings file '{default_path}' also not found. The values will be unchanged.")
            if ui_launch:
                return ({key: gr.update(value=value) for key, value in data.items()},)
            else:
                return [settings_path] + list(data.values()) + [""]
    
    print(f"Reading settings from {settings_path}")

    try:
        with open(settings_path, "r", encoding='utf-8') as f:
            jdata = json.load(f)
    except Exception as e:
        print(f"❌ Error loading settings file: {str(e)}")
        # If there's an error loading the file, fall back to default settings
        default_path = get_default_settings_path()
        print(f"Falling back to default settings from {default_path}")
        settings_path = default_path
        if not os.path.isfile(settings_path):
            print(f"Default settings file '{default_path}' also not found. The values will be unchanged.")
            if ui_launch:
                return ({key: gr.update(value=value) for key, value in data.items()},)
            else:
                return [settings_path] + list(data.values()) + [""]
    
    # Validate and potentially migrate settings
    is_valid, migrated_data, warnings = validate_and_migrate_settings(settings_path, jdata)
    
    if warnings:
        print(f"\n⚠️  Settings validation warnings:")
        for warning in warnings[:5]:  # Limit warnings in UI loading
            print(f"   • {warning}")
        if len(warnings) > 5:
            print(f"   ... and {len(warnings) - 5} more warnings")
    
    if not is_valid:
        should_migrate = handle_outdated_settings_file(settings_path)
        if should_migrate:
            jdata = migrated_data
            # Save the migrated settings back to the file
            try:
                with open(settings_path, "w", encoding='utf-8') as f:
                    json.dump(jdata, f, ensure_ascii=False, indent=4)
                print(f"✅ Settings file updated with missing fields")
            except Exception as e:
                print(f"⚠️  Could not update settings file: {e}")
    
    # Continue with normal processing
    handle_deprecated_settings(jdata)
    if 'animation_prompts' in jdata:
        jdata['prompts'] = jdata['animation_prompts']

    result = {}
    for key, default_val in data.items():
        val = jdata.get(key, default_val)
        if key == 'sampler' and isinstance(val, int):
            from modules.sd_samplers import samplers_for_img2img
            val = samplers_for_img2img[val].name
        elif key == 'fill' and isinstance(val, int):
            val = mask_fill_choices[val]
        elif key in {'reroll_blank_frames', 'noise_type'} and key not in jdata:
            default_key_val = (DeforumArgs if key != 'noise_type' else DeforumAnimArgs)[key]
            print(f"{key} not found in load file, using default value: {default_key_val}")
            val = default_key_val
        elif key in {'animation_prompts_positive', 'animation_prompts_negative'}:
            val = jdata.get(key, default_val)
        elif key == 'animation_prompts':
            val = json.dumps(jdata['prompts'], ensure_ascii=False, indent=4)
        # Special handling for camera shake
        elif key == 'shake_name':
            # Check if the value is a key in the camera shake list
            camera_shake_list = get_camera_shake_list()
            if val in camera_shake_list.keys():
                # If it's a key, convert it to the display name
                print(f"Converting camera shake key '{val}' to display name '{camera_shake_list[val]}'")
                val = camera_shake_list[val]
            # Make sure the value exists in the list of display names
            elif val not in camera_shake_list.values():
                print(f"Warning: Unknown camera shake value '{val}'. Using default 'Investigation'.")
                val = 'Investigation'

        result[key] = val

    # Include the settings path in the results
    if ui_launch:
        updates = {key: gr.update(value=value) for key, value in result.items()}
        # Add the settings path update
        updates['settings_path'] = gr.update(value=settings_path)
        return (updates,)
    else:
        # Return values for all components
        return list(result.values())


def load_video_settings(*args, **kwargs):
    video_settings_path = args[0].strip()
    video_settings_path = clean_gradio_path_strings(video_settings_path)
    video_settings_path = os.path.realpath(video_settings_path)
    vid_args_names = list(DeforumOutputArgs().keys())
    data = {vid_args_names[i]: args[i+1] for i in range(0, len(vid_args_names))}
    
    # First check webui root for deforum_settings.txt if no specific path is provided
    if video_settings_path == get_default_settings_path() or not os.path.exists(video_settings_path):
        # Check for a settings file in webui root
        from modules import paths_internal
        webui_root_settings = os.path.join(paths_internal.script_path, "deforum_settings.txt")
        if os.path.isfile(webui_root_settings):
            print(f"Using video settings from webui root file: {webui_root_settings}")
            video_settings_path = webui_root_settings
    
    # Check if the file exists, if not fall back to default settings
    if not os.path.isfile(video_settings_path):
        default_path = get_default_settings_path()
        print(f"The video settings file '{video_settings_path}' does not exist. Using default settings from {default_path}")
        video_settings_path = default_path
        # If default file also doesn't exist, return unchanged data
        if not os.path.isfile(video_settings_path):
            print(f"Default settings file '{default_path}' also not found. The values will be unchanged.")
            return [data[name] for name in vid_args_names]
    
    print(f"Reading video settings from {video_settings_path}")
    
    try:
        with open(video_settings_path, "r", encoding='utf-8') as f:
            jdata = json.load(f)
            handle_deprecated_settings(jdata)
    except Exception as e:
        print(f"Error loading video settings file: {str(e)}")
        # If there's an error loading the file, fall back to default settings
        default_path = get_default_settings_path()
        print(f"Falling back to default settings from {default_path}")
        video_settings_path = default_path
        if not os.path.isfile(video_settings_path):
            print(f"Default settings file '{default_path}' also not found. The values will be unchanged.")
            return [data[name] for name in vid_args_names]
        with open(video_settings_path, "r", encoding='utf-8') as f:
            jdata = json.load(f)
            handle_deprecated_settings(jdata)
        
    ret = []

    for key in data:
        if key == 'add_soundtrack':
            # Handle the add_soundtrack property with error checking
            if key in jdata:
                add_soundtrack_val = jdata[key]
                if type(add_soundtrack_val) == bool:
                    ret.append('File' if add_soundtrack_val else 'None')
                else:
                    ret.append(add_soundtrack_val)
            else:
                # Default to None if not specified
                ret.append('None')
        elif key in jdata:
            ret.append(jdata[key])
        else:
            ret.append(data[key])
    
    return ret