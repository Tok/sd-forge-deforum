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

from decimal import Decimal, getcontext

param_dict = {
    "angle": {"backend": "angle_series", "user": "Angle", "print": "Angle"},
    "transform_center_x": {"backend": "transform_center_x_series", "user": "Trans Center X", "print": "Tr.C.X"},
    "transform_center_y": {"backend": "transform_center_y_series", "user": "Trans Center Y", "print": "Tr.C.Y"},
    "zoom": {"backend": "zoom_series", "user": "Zoom", "print": "Zoom"},
    "translation_x": {"backend": "translation_x_series", "user": "Trans X", "print": "TrX"},
    "translation_y": {"backend": "translation_y_series", "user": "Trans Y", "print": "TrY"},
    "translation_z": {"backend": "translation_z_series", "user": "Trans Z", "print": "TrZ"},
    "rotation_3d_x": {"backend": "rotation_3d_x_series", "user": "Rot 3D X", "print": "RotX"},
    "rotation_3d_y": {"backend": "rotation_3d_y_series", "user": "Rot 3D Y", "print": "RotY"},
    "rotation_3d_z": {"backend": "rotation_3d_z_series", "user": "Rot 3D Z", "print": "RotZ"},
    "perspective_flip_theta": {"backend": "perspective_flip_theta_series", "user": "Per Fl Theta", "print": "PerFlT"},
    "perspective_flip_phi": {"backend": "perspective_flip_phi_series", "user": "Per Fl Phi", "print": "PerFlP"},
    "perspective_flip_gamma": {"backend": "perspective_flip_gamma_series", "user": "Per Fl Gamma", "print": "PerFlG"},
    "perspective_flip_fv": {"backend": "perspective_flip_fv_series", "user": "Per Fl FV", "print": "PerFlFV"},
    "noise_schedule": {"backend": "noise_schedule_series", "user": "Noise Sch", "print": "Noise"},
    "strength_schedule": {"backend": "strength_schedule_series", "user": "Str Sch", "print": "StrSch"},
    "keyframe_strength_schedule": {"backend": "keyframe_strength_schedule_series", "user": "Kfr Str Sch", "print": "KfrStrSch"},
    "contrast_schedule": {"backend": "contrast_schedule_series", "user": "Contrast Sch", "print": "CtrstSch"},
    "cfg_scale_schedule": {"backend": "cfg_scale_schedule_series", "user": "CFG Sch", "print": "CFGSch"},
    "distilled_cfg_scale_schedule": {"backend": "distilled_cfg_scale_schedule_series", "user": "Dist. CFG Sch", "print": "DistCFGSch"},
    "subseed_schedule": {"backend": "subseed_schedule_series", "user": "Subseed Sch", "print": "SubSSch"},
    "subseed_strength_schedule": {"backend": "subseed_strength_schedule_series", "user": "Subseed Str Sch", "print": "SubSStrSch"},
    "checkpoint_schedule": {"backend": "checkpoint_schedule_series", "user": "Ckpt Sch", "print": "CkptSch"},
    "steps_schedule": {"backend": "steps_schedule_series", "user": "Steps Sch", "print": "StepsSch"},
    "seed_schedule": {"backend": "seed_schedule_series", "user": "Seed Sch", "print": "SeedSch"},
    "sampler_schedule": {"backend": "sampler_schedule_series", "user": "Sampler Sch", "print": "SamplerSchedule"},
    "scheduler_schedule": {"backend": "scheduler_schedule_series", "user": "Scheduler Sch", "print": "SchedulerSchedule"},
    "clipskip_schedule": {"backend": "clipskip_schedule_series", "user": "Clipskip Sch", "print": "ClipskipSchedule"},
    "noise_multiplier_schedule": {"backend": "noise_multiplier_schedule_series", "user": "Noise Multp Sch", "print": "NoiseMultiplierSchedule"},
    "mask_schedule": {"backend": "mask_schedule_series", "user": "Mask Sch", "print": "MaskSchedule"},
    "noise_mask_schedule": {"backend": "noise_mask_schedule_series", "user": "Noise Mask Sch", "print": "NoiseMaskSchedule"},
    "amount_schedule": {"backend": "amount_schedule_series", "user": "Ant.Blr Amount Sch", "print": "AmountSchedule"},
    "kernel_schedule": {"backend": "kernel_schedule_series", "user": "Ant.Blr Kernel Sch", "print": "KernelSchedule"},
    "sigma_schedule": {"backend": "sigma_schedule_series", "user": "Ant.Blr Sigma Sch", "print": "SigmaSchedule"},
    "threshold_schedule": {"backend": "threshold_schedule_series", "user": "Ant.Blr Threshold Sch", "print": "ThresholdSchedule"},
    "aspect_ratio_schedule": {"backend": "aspect_ratio_series", "user": "Aspect Ratio Sch", "print": "AspectRatioSchedule"},
    "fov_schedule": {"backend": "fov_series", "user": "FOV Sch", "print": "FieldOfViewSchedule"},
    "near_schedule": {"backend": "near_series", "user": "Near Sch", "print": "NearSchedule"},
    "cadence_flow_factor_schedule": {"backend": "cadence_flow_factor_schedule_series", "user": "Cadence Flow Factor Sch", "print": "CadenceFlowFactorSchedule"},
    "redo_flow_factor_schedule": {"backend": "redo_flow_factor_schedule_series", "user": "Redo Flow Factor Sch", "print": "RedoFlowFactorSchedule"},
    "far_schedule": {"backend": "far_series", "user": "Far Sch", "print": "FarSchedule"},
    "hybrid_comp_alpha_schedule": {"backend": "hybrid_comp_alpha_schedule_series", "user": "Hyb Comp Alpha Sch", "print": "HybridCompAlphaSchedule"},
    "hybrid_comp_mask_blend_alpha_schedule": {"backend": "hybrid_comp_mask_blend_alpha_schedule_series", "user": "Hyb Comp Mask Blend Alpha Sch", "print": "HybridCompMaskBlendAlphaSchedule"},
    "hybrid_comp_mask_contrast_schedule": {"backend": "hybrid_comp_mask_contrast_schedule_series", "user": "Hyb Comp Mask Ctrst Sch", "print": "HybridCompMaskContrastSchedule"},
    "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": {"backend": "hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series", "user": "Hyb Comp Mask Auto Contrast Cutoff High Sch", "print": "HybridCompMaskAutoContrastCutoffHighSchedule"},
    "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": {"backend": "hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series", "user": "Hyb Comp Mask Auto Ctrst Cut Low Sch", "print": "HybridCompMaskAutoContrastCutoffLowSchedule"},
    "hybrid_flow_factor_schedule": {"backend": "hybrid_flow_factor_schedule_series", "user": "Hybrid Flow Factor Sch", "print": "HybridFlowFactorSchedule"},
}


def time_to_srt_format(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{int(milliseconds * 1000):03}"


def init_srt_file(filename, fps, precision=20):
    with open(filename, "w") as f:
        pass
    return calculate_frame_duration(fps)


def calculate_frame_duration(fps, precision=20):
    getcontext().prec = precision
    frame_duration = Decimal(1) / Decimal(fps)
    return frame_duration


def write_frame_subtitle(filename, frame_number, frame_duration, text):
    # Used by stable core only. Meant to be used when subtitles are intended to change with every frame.
    frames_per_subtitle = 1
    # For low FPS animations and for debugging this is fine, but at higher FPS the file may be too bloated and not fit
    # for YouTube upload, in which case "write_subtitle_from_to" may be used directly with a longer duration instead.
    start_time = frame_time(frame_number, frame_duration)
    end_time = (Decimal(frame_number) + Decimal(frames_per_subtitle)) * frame_duration
    write_subtitle_from_to(filename, frame_number, start_time, end_time, text)


def frame_time(frame_number, frame_duration):
    return Decimal(frame_number) * frame_duration


def write_subtitle_from_to(filename, frame_number, start_time_s, end_time_s, text):
    # start_time should be the same as end_time of the last call
    with open(filename, "a", encoding="utf-8") as f:
        # see https://en.wikipedia.org/wiki/SubRip#Format
        f.write(f"{frame_number + 1}\n")
        f.write(f"{time_to_srt_format(start_time_s)} --> {time_to_srt_format(end_time_s)}\n")
        f.write(f"{text}\n\n")


def format_animation_params(keys, prompt_series, frame_idx, params_to_print):
    params_string = ""
    for key, value in param_dict.items():
        if value['user'] in params_to_print:
            backend_key = value['backend']
            print_key = value['print']
            param_value = getattr(keys, backend_key)[frame_idx]
            formatted_value = _format_value(param_value)
            params_string += f"{print_key}: {formatted_value}; "

    if "Prompt" in params_to_print:
        params_string += f"Prompt: {prompt_series[frame_idx]}; "

    params_string = params_string.rstrip("; ")  # Remove trailing semicolon and whitespace
    return params_string


def _format_value(param_value):
    """
    Format the input value as a string.

    If the input is a float that represents an integer (e.g., 3.0), it converts it to an integer string (e.g., "3").
    If it's a float with a decimal component (e.g., 3.14), it formats it to three decimal places (e.g., "3.140").
    For all other types (including integers), it simply converts the value to a string.
    """
    is_float = isinstance(param_value, float)
    if is_float and param_value == int(param_value):
        return str(int(param_value))
    elif is_float and not param_value.is_integer():  # TODO? un-flaw logic by combining check in reverse (or something).
        return f"{param_value:.3f}"
    else:
        return f"{param_value}"


def get_user_values():
    items = [v["user"] for v in param_dict.values()]
    items.append("Prompt")
    return items
