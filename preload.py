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

import os
import sys
import importlib.util

# Global flag to indicate if we've applied the patches
__controlnet_patched = False

def preload_controlnet():
    """Apply ControlNet patches early in the startup process"""
    global __controlnet_patched
    
    # Only apply once
    if __controlnet_patched:
        return
    
    try:
        print("[Deforum] Preloading ControlNet patch for WebUI Forge compatibility")
        
        # Try to load the forge-specific patcher first
        current_dir = os.path.dirname(os.path.abspath(__file__))
        forge_patcher_path = os.path.join(current_dir, "scripts", "deforum_helpers", "forge_controlnet_patcher.py")
        
        if os.path.exists(forge_patcher_path):
            print(f"[Deforum] Found WebUI Forge ControlNet patcher at {forge_patcher_path}")
            
            # Load and execute the forge-specific patcher
            try:
                spec = importlib.util.spec_from_file_location("forge_controlnet_patcher", forge_patcher_path)
                patcher_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(patcher_module)
                
                # First try to use the new initialize_all_patches function
                if hasattr(patcher_module, "initialize_all_patches"):
                    print("[Deforum] Using enhanced early initialization for WebUI Forge compatibility")
                    success = patcher_module.initialize_all_patches()
                    print(f"[Deforum] WebUI Forge early patches applied successfully: {success}")
                    __controlnet_patched = True
                # Fall back to the old method if initialize_all_patches doesn't exist
                else:
                    # Create dummy processing object for initial patching
                    # Handle the case where creating a processing object might fail
                    dummy_p = None
                    try:
                        from modules import processing, scripts
                        if hasattr(processing, 'StableDiffusionProcessingTxt2Img') and hasattr(scripts, 'scripts_txt2img'):
                            dummy_p = processing.StableDiffusionProcessingTxt2Img()
                            dummy_p.scripts = scripts.scripts_txt2img
                    except:
                        print("[Deforum] Could not create dummy processing object for patching")
                    
                    # Call the apply_all_patches function
                    if hasattr(patcher_module, "apply_all_patches"):
                        success = patcher_module.apply_all_patches(dummy_p)
                        print(f"[Deforum] WebUI Forge ControlNet patches applied successfully: {success}")
                        __controlnet_patched = True
                    else:
                        print("[Deforum] No apply_all_patches function found in forge patcher module")
            except Exception as e:
                print(f"[Deforum] Error loading WebUI Forge ControlNet patcher: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Fall back to regular patcher if forge-specific isn't available
            patcher_path = os.path.join(current_dir, "scripts", "deforum_helpers", "controlnet_patcher.py")
            
            if os.path.exists(patcher_path):
                print(f"[Deforum] Found standard ControlNet patcher at {patcher_path}")
                
                # Load and execute the standard module
                try:
                    spec = importlib.util.spec_from_file_location("controlnet_patcher", patcher_path)
                    patcher_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(patcher_module)
                    
                    # Call the apply_all_patches function
                    if hasattr(patcher_module, "apply_all_patches"):
                        success = patcher_module.apply_all_patches()
                        print(f"[Deforum] Standard ControlNet patches applied successfully: {success}")
                        __controlnet_patched = True
                    else:
                        print("[Deforum] No apply_all_patches function found in standard patcher module")
                except Exception as e:
                    print(f"[Deforum] Error loading standard ControlNet patcher: {e}")
            else:
                print(f"[Deforum] No ControlNet patcher found. This may affect ControlNet compatibility.")
    except Exception as e:
        print(f"[Deforum] Error in preload_controlnet: {e}")
        import traceback
        traceback.print_exc()

def preload(parser):
    parser.add_argument(
        "--deforum-api",
        action="store_true",
        help="Enable the Deforum API",
        default=None,
    )
    parser.add_argument(
        "--deforum-simple-api",
        action="store_true",
        help="Enable the simplified version of Deforum API",
        default=None,
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
        default=None,
    )
    
    # Apply ControlNet patches during the preload phase
    preload_controlnet()