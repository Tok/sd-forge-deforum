"""
Core Arguments
Contains basic generation and core Deforum arguments
"""

import modules.paths as ph
import modules.shared as sh
from .defaults import get_samplers_list, get_schedulers_list


def DeforumArgs():
    """Core Deforum arguments for basic generation settings."""
    return {
        "W": {
            "label": "Width",
            "type": "slider",
            "value": 1024,
            "minimum": 128,
            "maximum": 2048,
            "step": 64,
            "info": "width in pixels. Use bigger sizes for Flux. Recommended = 1024"
        },
        "H": {
            "label": "Height", 
            "type": "slider",
            "value": 1024,
            "minimum": 128,
            "maximum": 2048,
            "step": 64,
            "info": "height in pixels. Use bigger sizes for Flux. Recommended = 1024"
        },
        "show_info_on_ui": {
            "type": "checkbox",
            "value": False,
            "label": "Show more info",
            "info": "adds additional explanatory text to various parameters"
        },
        "show_controlnet_tab": {
            "type": "checkbox",
            "value": False,
            "label": "Show ControlNet tab",
            "info": "Shows the ControlNet tab (experimental - not fully working yet)"
        },
        "tiling": {
            "label": "Tiling",
            "type": "checkbox",
            "value": False,
            "info": "enable for seamless-tiling of each generated image. Experimental"
        },
        "restore_faces": {
            "label": "Restore faces",
            "type": "checkbox",
            "value": False,
            "info": "enable to trigger webui's face restoration on each frame during the generation"
        },
        "seed_resize_from_w": {
            "label": "Resize seed from width",
            "type": "slider",
            "minimum": 0,
            "maximum": 2048,
            "step": 64,
            "value": 0,
        },
        "seed_resize_from_h": {
            "label": "Resize seed from height",
            "type": "slider",
            "minimum": 0,
            "maximum": 2048,
            "step": 64,
            "value": 0,
        },
        "seed": {
            "label": "Seed",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "Starting seed for the animation. -1 for random"
        },
        "sampler": {
            "label": "Sampler",
            "type": "dropdown",
            "choices": get_samplers_list().values(),
            "value": "Euler",
        },
        "scheduler": {
            "label": "Scheduler",
            "type": "dropdown",
            "choices": get_schedulers_list().values(),
            "value": "Simple",
        },
        "steps": {
            "label": "Steps",
            "type": "slider",
            "minimum": 1,
            "maximum": 200,
            "step": 1,
            "value": 20,
        },
        "batch_name": {
            "label": "Batch name",
            "type": "textbox",
            "value": "Deforum_{timestring}",
            "info": "output images will be placed in a folder with this name ({timestring} token will be replaced) inside the img2img output folder. Supports params placeholders. e.g {seed}, {w}, {h}, {prompts}"
        },
        "seed_behavior": {
            "label": "Seed behavior",
            "type": "radio",
            "choices": ['iter', 'fixed', 'random', 'ladder', 'alternate', 'schedule'],
            "value": "iter",
            "info": "controls the seed behavior that is used for animation. Hover on the options to see more info"
        },
        "seed_iter_N": {
            "label": "Seed iter N",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "for how many frames the same seed should stick before iterating to the next one"
        },
        "use_init": {
            "label": "Use init",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "strength": {
            "label": "Strength",
            "type": "slider",
            "minimum": 0,
            "maximum": 1,
            "step": 0.01,
            "value": 0.85,
            "info": "the inverse of denoise; lower values alter the init image more (high denoise); higher values alter it less (low denoise)"
        },
        "strength_0_no_init": {
            "label": "Strength 0 no init",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "init_image": {
            "label": "Init image URL",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/I1.png",
            "info": "Use web address or local path. Note: if the image box below is used then this field is ignored."
        },
        "init_image_box": {
            "label": "Init image box",
            "type": "image",
            "type_param": "pil",
            "source": "upload",
            "interactive": True,
            "info": ""
        },
        "use_mask": {
            "label": "Use mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "use_alpha_as_mask": {
            "label": "Use alpha as mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_file": {
            "label": "Mask file",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/M1.jpg",
            "info": ""
        },
        "invert_mask": {
            "label": "Invert mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_contrast_adjust": {
            "label": "Mask contrast adjust",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": ""
        },
        "mask_brightness_adjust": {
            "label": "Mask brightness adjust",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": ""
        },
        "overlay_mask": {
            "label": "Overlay mask",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "mask_overlay_blur": {
            "label": "Mask overlay blur",
            "type": "slider",
            "minimum": 0,
            "maximum": 64,
            "step": 1,
            "value": 4,
        },
        "fill": {
            "label": "Fill mode",
            "type": "radio",
            "choices": ['stretch', 'fit', 'crop'],
            "value": "stretch",
            "info": ""
        },
        "full_res_mask": {
            "label": "Full resolution mask",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "full_res_mask_padding": {
            "label": "Full resolution mask padding",
            "type": "slider",
            "minimum": 0,
            "maximum": 512,
            "step": 4,
            "value": 1,
        },
        "reroll_blank_frames": {
            "label": "Reroll blank frames",
            "type": "radio",
            "choices": ['ignore', 'reroll', 'interrupt'],
            "value": "ignore",
            "info": "what to do with blank frames (they may result from glitches or the NSFW filter being turned on)"
        },
        "reroll_patience": {
            "label": "Reroll patience",
            "type": "slider",
            "minimum": 1.0,
            "maximum": 10.0,
            "step": 0.1,
            "value": 2.0,
            "info": "how many rerolls to do before giving up"
        },
        "motion_preview_mode": {
            "label": "Motion preview mode",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "prompts_path": {
            "label": "Prompts path",
            "type": "textbox",
            "value": "",
            "info": ""
        },
        "negative_prompts_path": {
            "label": "Negative prompts path",
            "type": "textbox",
            "value": "",
            "info": ""
        },
        "cn_1_overwrite_frames": {
            "label": "ControlNet 1 overwrite frames",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "cn_1_vid_path": {
            "label": "ControlNet 1 video path",
            "type": "textbox",
            "value": "",
            "info": ""
        },
        "cn_1_mask_vid_path": {
            "label": "ControlNet 1 mask video path",
            "type": "textbox",
            "value": "",
            "info": ""
        },
        "cn_1_enabled": {
            "label": "ControlNet 1 enabled",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "cn_1_use_vid_as_input": {
            "label": "ControlNet 1 use video as input",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "cn_1_low_vram": {
            "label": "ControlNet 1 low VRAM",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "cn_1_weight": {
            "label": "ControlNet 1 weight",
            "type": "slider",
            "minimum": 0,
            "maximum": 2,
            "step": 0.01,
            "value": 1,
        },
        "cn_1_guidance_start": {
            "label": "ControlNet 1 guidance start",
            "type": "slider",
            "minimum": 0,
            "maximum": 1,
            "step": 0.01,
            "value": 0,
        },
        "cn_1_guidance_end": {
            "label": "ControlNet 1 guidance end",
            "type": "slider",
            "minimum": 0,
            "maximum": 1,
            "step": 0.01,
            "value": 1,
        },
        "cn_1_threshold_a": {
            "label": "ControlNet 1 threshold A",
            "type": "slider",
            "minimum": 64,
            "maximum": 1024,
            "step": 1,
            "value": 64,
        },
        "cn_1_threshold_b": {
            "label": "ControlNet 1 threshold B",
            "type": "slider",
            "minimum": 64,
            "maximum": 1024,
            "step": 1,
            "value": 64,
        },
        "cn_1_resize_mode": {
            "label": "ControlNet 1 resize mode",
            "type": "radio",
            "choices": ["Envelope (Outer Fit)", "Scale to Fit (Inner Fit)", "Just Resize"],
            "value": "Scale to Fit (Inner Fit)",
            "info": ""
        },
        "cn_1_control_mode": {
            "label": "ControlNet 1 control mode",
            "type": "radio",
            "choices": ["Balanced", "My prompt is more important", "ControlNet is more important"],
            "value": "Balanced",
            "info": ""
        },
        "cn_1_loopback_mode": {
            "label": "ControlNet 1 loopback mode",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
    } 