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


def init_deforum():
    # use sys.path.extend to make sure all of our files are available for importation
    deforum_sys_extend()

    # Apply compatibility patches for diffusers git main + Forge
    try:
        from deforum.integrations.flux_controlnet.diffusers_compat import apply_all_patches
        apply_all_patches()
    except Exception as e:
        print(f"⚠️ Deforum: Failed to apply diffusers compatibility patches: {e}")

    # Flux 2 compatibility patch: Add vec_in_dim fallback for GGUF models
    try:
        from deforum.integrations.flux2.compat_patch import ensure_flux2_compatibility
        ensure_flux2_compatibility()
    except Exception as e:
        print(f"⚠️ Deforum: Failed to apply Flux 2 compatibility patch: {e}")
        import traceback
        traceback.print_exc()

    # Fractional img2img patches: RE-ENABLED with debug logging
    try:
        from deforum.pipeline.fractional_img2img_patch import apply_fractional_img2img_patch
        from deforum.pipeline.fractional_sigma_slicer_patch import patch_kdiffusion_sampler_class
        apply_fractional_img2img_patch()
        patch_kdiffusion_sampler_class()
    except Exception as e:
        print(f"⚠️ Deforum: Failed to apply fractional img2img patches: {e}")
        import traceback
        traceback.print_exc()

    # create the Models/Deforum folder, where many of the deforum related models/ packages will be downloaded
    os.makedirs(ph.models_path + '/Deforum', exist_ok=True)

    # Auto-download Flux model and VAE if not present
    try:
        from deforum.utils.system.flux_model_downloader import auto_download_flux_if_needed
        from deforum.utils.system.flux_check import is_flux_available

        if not is_flux_available():
            print("[Deforum] Flux model not detected - starting auto-download...")
            print("[Deforum] This will download ~15GB across 4 files:")
            print("[Deforum]   - flux1-dev-bnb-nf4-v2.safetensors (~5GB)")
            print("[Deforum]   - clip_l.safetensors (~250MB)")
            print("[Deforum]   - t5xxl_fp16.safetensors (~9.8GB)")
            print("[Deforum]   - ae.safetensors (~300MB)")
            print("[Deforum] Please wait...")
            auto_download_flux_if_needed()
    except Exception as e:
        print(f"⚠️ Deforum: Failed to auto-download Flux: {e}")
        print("[Deforum] You can manually download Flux from the Wan Models tab")

    # Auto-download Wan 2.1 FLF2V model if not present
    try:
        from deforum.utils.system.wan_model_downloader import auto_download_wan_flf2v_if_needed
        auto_download_wan_flf2v_if_needed()
    except Exception as e:
        print(f"⚠️ Deforum: Failed to auto-download Wan FLF2V: {e}")
        print("[Deforum] You can manually download Wan models from the Wan Models tab")

    # import our on_ui_tabs and on_ui_settings functions from the respected files
    from deforum.ui.ui_right import on_ui_tabs
    from deforum.ui.ui_settings import on_ui_settings

    # trigger webui's extensions mechanism using our imported main functions -
    # first to create the actual deforum gui, then to make the deforum tab in webui's settings section
    script_callbacks.on_ui_tabs(on_ui_tabs)
    script_callbacks.on_ui_settings(on_ui_settings)

    # Register tuning tab if --deforum-run-tuning flag is set
    if getattr(cmd_opts, 'deforum_run_tuning', False):
        from deforum.ui.ui_tuning import create_tuning_tab
        print("[Deforum] Tuning mode enabled - registering Tuning tab")
        print("[Deforum] Note: Deforum API will be auto-enabled (see api.py)")

        def on_tuning_tab():
            return [create_tuning_tab()]

        script_callbacks.on_ui_tabs(on_tuning_tab)

init_deforum()

