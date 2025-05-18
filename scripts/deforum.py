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

import os

import modules.paths as ph
from modules import script_callbacks
from modules.shared import cmd_opts
from scripts.deforum_extend_paths import deforum_sys_extend


def setup_controlnet_compatibility():
    """
    Set up compatibility with ControlNet by patching its processing methods
    
    This ensures that ControlNet works correctly with the Deforum extension,
    especially fixing the resize_mode attribute requirement.
    """
    try:
        from modules import shared, scripts, processing
        from deforum_helpers.deforum_controlnet import debug_print
        
        # Create placeholder classes to test patching
        class DummyProcessing:
            def __init__(self):
                self.scripts = scripts.Scripts()
                self.scripts.alwayson_scripts = []
                
        # Create placeholder script
        class DummyControlNetScript:
            def __init__(self):
                pass
                
            def title(self):
                return "ControlNet"
                
            def process(self, p, *args, **kwargs):
                pass
        
        # Add dummy script to test patching logic
        dummy_p = DummyProcessing()
        dummy_script = DummyControlNetScript()
        dummy_p.scripts.alwayson_scripts.append(dummy_script)
        
        # Try to patch the dummy object
        from deforum_helpers.deforum_controlnet import patch_controlnet_process
        success = patch_controlnet_process(dummy_p)
        
        if success:
            debug_print("Successfully set up ControlNet compatibility during initialization")
        else:
            debug_print("ControlNet compatibility setup deferred to runtime")
            
        return success
        
    except Exception as e:
        print(f"Note: Could not set up ControlNet compatibility during initialization: {e}")
        print("This is not critical - compatibility will be applied during generation")
        return False


def init_deforum():
    # use sys.path.extend to make sure all of our files are available for importation
    deforum_sys_extend()

    # create the Models/Deforum folder, where many of the deforum related models/ packages will be downloaded
    os.makedirs(ph.models_path + '/Deforum', exist_ok=True)
    
    # Import our new preload module with comprehensive ControlNet patches
    try:
        from deforum_helpers import preload
        print("Applied preload patches for Deforum")
    except Exception as e:
        print(f"Note: Error applying preload patches: {e}")
        
        # Import our ControlNet patcher as fallback
        try:
            from deforum_helpers import controlnet_patcher
            print("Loaded ControlNet patcher for Deforum")
        except Exception as e2:
            print(f"Note: Could not load ControlNet patcher: {e2}")
    
    # Import and preload ControlNet models for improved FLUX compatibility
    try:
        from deforum_helpers.deforum_controlnet import preload_controlnet_models, register_controlnet_models
        
        # Attempt to preload and register ControlNet models
        if os.path.exists(os.path.join(ph.models_path, "ControlNet")) or os.path.exists(os.path.join(ph.models_path, "controlnet")):
            print("Preloading ControlNet models for Deforum...")
            preload_controlnet_models()
            register_controlnet_models()
            
            # Set up ControlNet compatibility
            setup_controlnet_compatibility()
    except Exception as e:
        print(f"Note: ControlNet preloading for Deforum not available. This is normal if ControlNet is not installed: {e}")

    # import our on_ui_tabs and on_ui_settings functions from the respected files
    from deforum_helpers.ui_right import on_ui_tabs
    from deforum_helpers.ui_settings import on_ui_settings

    # trigger webui's extensions mechanism using our imported main functions -
    # first to create the actual deforum gui, then to make the deforum tab in webui's settings section
    script_callbacks.on_ui_tabs(on_ui_tabs)
    script_callbacks.on_ui_settings(on_ui_settings)

init_deforum()
