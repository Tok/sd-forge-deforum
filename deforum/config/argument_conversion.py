"""
Pure conversion functions for backward compatibility with legacy args system.

This module provides functions to convert between the new immutable dataclasses
and the old mutable dictionary/SimpleNamespace system, ensuring zero breaking changes.
"""

import json
import os
import tempfile
import time
from types import SimpleNamespace
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from .argument_models import (
    DeforumGenerationArgs, DeforumAnimationArgs, DeforumVideoArgs,
    ParseqArgs, WanArgs, RootArgs, ProcessedArguments,
    SamplerType, SchedulerType, SeedBehavior, NoiseType, RerollBehavior, MaskFill
)

from ..data_models import (
    AnimationMode, BorderMode, ColorCoherence, DepthAlgorithm,
    PaddingMode, SamplingMode
)


# Pure conversion functions from dictionaries to immutable models
def create_deforum_args_from_dict(data: Dict[str, Any]) -> DeforumGenerationArgs:
    """Pure function: legacy dict -> immutable DeforumGenerationArgs"""
    # Handle enum conversions
    sampler = data.get("sampler", "Euler")
    try:
        sampler_enum = SamplerType(sampler)
    except ValueError:
        sampler_enum = SamplerType.EULER
    
    scheduler = data.get("scheduler", "Simple")
    try:
        scheduler_enum = SchedulerType(scheduler)
    except ValueError:
        scheduler_enum = SchedulerType.SIMPLE
    
    seed_behavior = data.get("seed_behavior", "iter")
    try:
        seed_behavior_enum = SeedBehavior(seed_behavior)
    except ValueError:
        seed_behavior_enum = SeedBehavior.ITER
    
    fill = data.get("fill", "original")
    try:
        fill_enum = MaskFill(fill)
    except ValueError:
        fill_enum = MaskFill.ORIGINAL
    
    reroll_blank_frames = data.get("reroll_blank_frames", "ignore")
    try:
        reroll_enum = RerollBehavior(reroll_blank_frames)
    except ValueError:
        reroll_enum = RerollBehavior.IGNORE
    
    return DeforumGenerationArgs(
        # Map dictionary keys to dataclass fields
        width=data.get("W", 1024),
        height=data.get("H", 1024),
        seed=data.get("seed", -1),
        sampler=sampler_enum,
        scheduler=scheduler_enum,
        steps=data.get("steps", 20),
        cfg_scale=data.get("scale", 7.0),  # Legacy "scale" -> "cfg_scale"
        
        tiling=data.get("tiling", False),
        restore_faces=data.get("restore_faces", False),
        seed_resize_from_w=data.get("seed_resize_from_w", 0),
        seed_resize_from_h=data.get("seed_resize_from_h", 0),
        
        seed_behavior=seed_behavior_enum,
        seed_iter_n=data.get("seed_iter_N", 1),
        
        use_init=data.get("use_init", False),
        strength=data.get("strength", 0.85),
        strength_0_no_init=data.get("strength_0_no_init", True),
        init_image=data.get("init_image"),
        init_image_box=data.get("init_image_box"),
        
        use_mask=data.get("use_mask", False),
        use_alpha_as_mask=data.get("use_alpha_as_mask", False),
        mask_file=data.get("mask_file", ""),
        invert_mask=data.get("invert_mask", False),
        mask_contrast_adjust=data.get("mask_contrast_adjust", 1.0),
        mask_brightness_adjust=data.get("mask_brightness_adjust", 1.0),
        overlay_mask=data.get("overlay_mask", True),
        mask_overlay_blur=data.get("mask_overlay_blur", 4),
        fill=fill_enum,
        full_res_mask=data.get("full_res_mask", True),
        full_res_mask_padding=data.get("full_res_mask_padding", 4),
        
        batch_name=data.get("batch_name", "Deforum_{timestring}"),
        
        reroll_blank_frames=reroll_enum,
        reroll_patience=data.get("reroll_patience", 10),
        
        motion_preview_mode=data.get("motion_preview_mode", False),
        
        show_info_on_ui=data.get("show_info_on_ui", False),
        show_controlnet_tab=data.get("show_controlnet_tab", False),
    )


def create_animation_args_from_dict(data: Dict[str, Any]) -> DeforumAnimationArgs:
    """Pure function: legacy dict -> immutable DeforumAnimationArgs"""
    # Handle enum conversions
    animation_mode = data.get("animation_mode", "3D")
    try:
        if animation_mode == "2D":
            animation_mode_enum = AnimationMode.TWO_D
        elif animation_mode == "3D":
            animation_mode_enum = AnimationMode.THREE_D
        elif animation_mode == "Interpolation":
            animation_mode_enum = AnimationMode.INTERPOLATION
        elif animation_mode == "Wan Video":
            animation_mode_enum = AnimationMode.WAN_VIDEO
        else:
            animation_mode_enum = AnimationMode.THREE_D
    except:
        animation_mode_enum = AnimationMode.THREE_D
    
    border = data.get("border", "replicate")
    try:
        border_enum = BorderMode.REPLICATE if border == "replicate" else BorderMode.WRAP
    except:
        border_enum = BorderMode.REPLICATE
    
    color_coherence = data.get("color_coherence", "None")
    try:
        if color_coherence == "HSV":
            color_coherence_enum = ColorCoherence.HSV
        elif color_coherence == "LAB":
            color_coherence_enum = ColorCoherence.LAB
        elif color_coherence == "RGB":
            color_coherence_enum = ColorCoherence.RGB
        elif color_coherence == "Image":
            color_coherence_enum = ColorCoherence.IMAGE
        else:
            color_coherence_enum = ColorCoherence.NONE
    except:
        color_coherence_enum = ColorCoherence.NONE
    
    noise_type = data.get("noise_type", "perlin")
    try:
        noise_type_enum = NoiseType.PERLIN if noise_type == "perlin" else NoiseType.UNIFORM
    except:
        noise_type_enum = NoiseType.PERLIN
    
    depth_algorithm = data.get("depth_algorithm", "Depth-Anything-V2-Small")
    try:
        if "Depth-Anything" in depth_algorithm:
            depth_algorithm_enum = DepthAlgorithm.DEPTH_ANYTHING_V2_SMALL
        else:
            depth_algorithm_enum = DepthAlgorithm.MIDAS_3_HYBRID
    except:
        depth_algorithm_enum = DepthAlgorithm.DEPTH_ANYTHING_V2_SMALL
    
    padding_mode = data.get("padding_mode", "border")
    try:
        if padding_mode == "reflection":
            padding_mode_enum = PaddingMode.REFLECTION
        elif padding_mode == "zeros":
            padding_mode_enum = PaddingMode.ZEROS
        else:
            padding_mode_enum = PaddingMode.BORDER
    except:
        padding_mode_enum = PaddingMode.BORDER
    
    sampling_mode = data.get("sampling_mode", "bicubic")
    try:
        if sampling_mode == "bilinear":
            sampling_mode_enum = SamplingMode.BILINEAR
        elif sampling_mode == "nearest":
            sampling_mode_enum = SamplingMode.NEAREST
        else:
            sampling_mode_enum = SamplingMode.BICUBIC
    except:
        sampling_mode_enum = SamplingMode.BICUBIC
    
    return DeforumAnimationArgs(
        animation_mode=animation_mode_enum,
        max_frames=data.get("max_frames", 333),
        border=border_enum,
        
        # Movement schedules
        angle=data.get("angle", "0: (0)"),
        zoom=data.get("zoom", "0: (1.0)"),
        translation_x=data.get("translation_x", "0: (0)"),
        translation_y=data.get("translation_y", "0: (0)"),
        translation_z=data.get("translation_z", "0: (0)"),
        transform_center_x=data.get("transform_center_x", "0: (0.5)"),
        transform_center_y=data.get("transform_center_y", "0: (0.5)"),
        rotation_3d_x=data.get("rotation_3d_x", "0: (0)"),
        rotation_3d_y=data.get("rotation_3d_y", "0: (0)"),
        rotation_3d_z=data.get("rotation_3d_z", "0: (0)"),
        
        # Camera shake
        shake_name=data.get("shake_name", "INVESTIGATION"),
        shake_intensity=data.get("shake_intensity", 1.0),
        shake_speed=data.get("shake_speed", 1.0),
        
        # Perspective flip
        enable_perspective_flip=data.get("enable_perspective_flip", False),
        perspective_flip_theta=data.get("perspective_flip_theta", "0: (0)"),
        perspective_flip_phi=data.get("perspective_flip_phi", "0: (0)"),
        perspective_flip_gamma=data.get("perspective_flip_gamma", "0: (0)"),
        perspective_flip_fv=data.get("perspective_flip_fv", "0: (53)"),
        
        # All the schedules
        noise_schedule=data.get("noise_schedule", "0: (0.065)"),
        strength_schedule=data.get("strength_schedule", "0: (0.85)"),
        keyframe_strength_schedule=data.get("keyframe_strength_schedule", "0: (0.50)"),
        contrast_schedule=data.get("contrast_schedule", "0: (1.0)"),
        cfg_scale_schedule=data.get("cfg_scale_schedule", "0: (1.0)"),
        distilled_cfg_scale_schedule=data.get("distilled_cfg_scale_schedule", "0: (3.5)"),
        
        enable_steps_scheduling=data.get("enable_steps_scheduling", False),
        steps_schedule=data.get("steps_schedule", "0: (20)"),
        
        fov_schedule=data.get("fov_schedule", "0: (70)"),
        aspect_ratio_schedule=data.get("aspect_ratio_schedule", "0: (1.0)"),
        aspect_ratio_use_old_formula=data.get("aspect_ratio_use_old_formula", False),
        near_schedule=data.get("near_schedule", "0: (200)"),
        far_schedule=data.get("far_schedule", "0: (10000)"),
        
        seed_schedule=data.get("seed_schedule", '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)'),
        enable_subseed_scheduling=data.get("enable_subseed_scheduling", False),
        subseed_schedule=data.get("subseed_schedule", "0: (1)"),
        subseed_strength_schedule=data.get("subseed_strength_schedule", "0: (0)"),
        
        enable_sampler_scheduling=data.get("enable_sampler_scheduling", False),
        sampler_schedule=data.get("sampler_schedule", '0: ("Euler")'),
        
        enable_scheduler_scheduling=data.get("enable_scheduler_scheduling", False),
        scheduler_schedule=data.get("scheduler_schedule", '0: ("Simple")'),
        
        use_noise_mask=data.get("use_noise_mask", False),
        mask_schedule=data.get("mask_schedule", '0: ("{video_mask}")'),
        noise_mask_schedule=data.get("noise_mask_schedule", '0: ("{video_mask}")'),
        
        enable_checkpoint_scheduling=data.get("enable_checkpoint_scheduling", False),
        checkpoint_schedule=data.get("checkpoint_schedule", '0: ("model1.ckpt"), 100: ("model2.safetensors")'),
        
        enable_clipskip_scheduling=data.get("enable_clipskip_scheduling", False),
        clipskip_schedule=data.get("clipskip_schedule", "0: (2)"),
        
        enable_noise_multiplier_scheduling=data.get("enable_noise_multiplier_scheduling", True),
        noise_multiplier_schedule=data.get("noise_multiplier_schedule", "0: (1.05)"),
        
        resume_from_timestring=data.get("resume_from_timestring", False),
        resume_timestring=data.get("resume_timestring", "20241111111111"),
        
        enable_ddim_eta_scheduling=data.get("enable_ddim_eta_scheduling", False),
        ddim_eta_schedule=data.get("ddim_eta_schedule", "0: (0)"),
        enable_ancestral_eta_scheduling=data.get("enable_ancestral_eta_scheduling", False),
        ancestral_eta_schedule=data.get("ancestral_eta_schedule", "0: (1)"),
        
        amount_schedule=data.get("amount_schedule", "0: (0.1)"),
        kernel_schedule=data.get("kernel_schedule", "0: (5)"),
        sigma_schedule=data.get("sigma_schedule", "0: (1)"),
        threshold_schedule=data.get("threshold_schedule", "0: (0)"),
        
        color_coherence=color_coherence_enum,
        color_coherence_image_path=data.get("color_coherence_image_path", ""),
        color_coherence_video_every_n_frames=data.get("color_coherence_video_every_N_frames", 1),
        color_force_grayscale=data.get("color_force_grayscale", False),
        legacy_colormatch=data.get("legacy_colormatch", False),
        
        keyframe_distribution=data.get("keyframe_distribution", "Redistributed"),
        diffusion_cadence=data.get("diffusion_cadence", 10),
        optical_flow_cadence=data.get("optical_flow_cadence", "None"),
        cadence_flow_factor_schedule=data.get("cadence_flow_factor_schedule", "0: (1)"),
        optical_flow_redo_generation=data.get("optical_flow_redo_generation", "None"),
        redo_flow_factor_schedule=data.get("redo_flow_factor_schedule", "0: (1)"),
        diffusion_redo=data.get("diffusion_redo", "0"),
        
        noise_type=noise_type_enum,
        perlin_w=data.get("perlin_w", 8.0),
        perlin_h=data.get("perlin_h", 8.0),
        perlin_octaves=data.get("perlin_octaves", 4),
        perlin_persistence=data.get("perlin_persistence", 0.5),
        
        use_depth_warping=data.get("use_depth_warping", True),
        depth_algorithm=depth_algorithm_enum,
        midas_weight=data.get("midas_weight", 0.2),
        padding_mode=padding_mode_enum,
        sampling_mode=sampling_mode_enum,
        save_depth_maps=data.get("save_depth_maps", False),
        
        video_init_path=data.get("video_init_path", 'https://deforum.github.io/a1/V1.mp4'),
        extract_nth_frame=data.get("extract_nth_frame", 1),
        extract_from_frame=data.get("extract_from_frame", 0),
        extract_to_frame=data.get("extract_to_frame", -1),
        overwrite_extracted_frames=data.get("overwrite_extracted_frames", False),
        
        # Video mask
        use_mask_video=data.get("use_mask_video", False),
        video_mask_path=data.get("video_mask_path", 'https://deforum.github.io/a1/VM1.mp4'),
        
        # Note: hybrid video functionality removed - all hybrid arguments eliminated
    )


def create_video_args_from_dict(data: Dict[str, Any]) -> DeforumVideoArgs:
    """Pure function: legacy dict -> immutable DeforumVideoArgs"""
    return DeforumVideoArgs(
        skip_video_creation=data.get("skip_video_creation", False),
        fps=data.get("fps", 60),
        make_gif=data.get("make_gif", False),
        
        delete_imgs=data.get("delete_imgs", False),
        delete_input_frames=data.get("delete_input_frames", False),
        
        image_path=data.get("image_path", "C:/SD/20241111111111_%09d.png"),
        
        add_soundtrack=data.get("add_soundtrack", "File"),
        soundtrack_path=data.get("soundtrack_path", "https://ia801303.us.archive.org/26/items/amen-breaks/cw_amen13_173.mp3"),
        
        r_upscale_video=data.get("r_upscale_video", False),
        r_upscale_factor=data.get("r_upscale_factor", "x2"),
        r_upscale_model=data.get("r_upscale_model", 'realesr-animevideov3'),
        r_upscale_keep_imgs=data.get("r_upscale_keep_imgs", True),
        
        store_frames_in_ram=data.get("store_frames_in_ram", False),
        
        frame_interpolation_engine=data.get("frame_interpolation_engine", "None"),
        frame_interpolation_x_amount=data.get("frame_interpolation_x_amount", 2),
        frame_interpolation_slow_mo_enabled=data.get("frame_interpolation_slow_mo_enabled", False),
        frame_interpolation_slow_mo_amount=data.get("frame_interpolation_slow_mo_amount", 2),
        frame_interpolation_keep_imgs=data.get("frame_interpolation_keep_imgs", False),
        frame_interpolation_use_upscaled=data.get("frame_interpolation_use_upscaled", False),
    )


def create_parseq_args_from_dict(data: Dict[str, Any]) -> ParseqArgs:
    """Pure function: legacy dict -> immutable ParseqArgs"""
    return ParseqArgs(
        parseq_manifest=data.get("parseq_manifest"),
        parseq_use_deltas=data.get("parseq_use_deltas", True),
        parseq_non_schedule_overrides=data.get("parseq_non_schedule_overrides", True),
    )


def create_wan_args_from_dict(data: Dict[str, Any]) -> WanArgs:
    """Pure function: legacy dict -> immutable WanArgs"""
    return WanArgs(
        wan_mode=data.get("wan_mode", "Disabled"),
        wan_model_path=data.get("wan_model_path", "models/wan"),
        wan_model_name=data.get("wan_model_name", "Auto-Select"),
        wan_enable_prompt_enhancement=data.get("wan_enable_prompt_enhancement", False),
        wan_qwen_model=data.get("wan_qwen_model", "Auto-Select"),
        wan_qwen_language=data.get("wan_qwen_language", "English"),
        wan_auto_download=data.get("wan_auto_download", True),
        wan_preferred_size=data.get("wan_preferred_size", "1.3B VACE (Recommended)"),
        
        wan_enable_movement_analysis=data.get("wan_enable_movement_analysis", True),
        wan_movement_sensitivity=data.get("wan_movement_sensitivity", 1.0),
        
        wan_style_prompt=data.get("wan_style_prompt", ""),
        wan_style_strength=data.get("wan_style_strength", 0.5),
        wan_i2v_strength=data.get("wan_i2v_strength", 0.8),
        
        wan_t2v_model=data.get("wan_t2v_model", "1.3B VACE"),
        wan_i2v_model=data.get("wan_i2v_model", "Use Primary Model"),
        wan_resolution=data.get("wan_resolution", "864x480 (Landscape)"),
        wan_seed=data.get("wan_seed", -1),
        wan_guidance_scale=data.get("wan_guidance_scale", 7.5),
        
        wan_strength_override=data.get("wan_strength_override", True),
        wan_fixed_strength=data.get("wan_fixed_strength", 1.0),
        wan_guidance_override=data.get("wan_guidance_override", True),
        wan_motion_strength_override=data.get("wan_motion_strength_override", False),
        wan_motion_strength=data.get("wan_motion_strength", 1.0),
        wan_frame_overlap=data.get("wan_frame_overlap", 2),
        wan_enable_interpolation=data.get("wan_enable_interpolation", True),
        wan_interpolation_strength=data.get("wan_interpolation_strength", 0.5),
        wan_flash_attention_mode=data.get("wan_flash_attention_mode", "Auto (Recommended)"),
        wan_qwen_auto_download=data.get("wan_qwen_auto_download", True),
    )


def create_root_args_from_dict(data: Dict[str, Any]) -> RootArgs:
    """Pure function: legacy dict -> immutable RootArgs"""
    return RootArgs(
        device=data.get("device"),
        models_path=data.get("models_path", ""),
        half_precision=data.get("half_precision", True),
        clipseg_model=data.get("clipseg_model"),
        mask_preset_names=tuple(data.get("mask_preset_names", ["everywhere", "video_mask"])),
        frames_cache=tuple(data.get("frames_cache", [])),
        raw_batch_name=data.get("raw_batch_name"),
        raw_seed=data.get("raw_seed"),
        timestring=data.get("timestring", ""),
        subseed=data.get("subseed", -1),
        subseed_strength=data.get("subseed_strength", 0.0),
        seed_internal=data.get("seed_internal", 0),
        init_sample=data.get("init_sample"),
        noise_mask=data.get("noise_mask"),
        initial_info=data.get("initial_info"),
        first_frame=data.get("first_frame"),
        default_img=data.get("default_img"),
        animation_prompts=dict(data.get("animation_prompts", {})),
        prompt_keyframes=list(data.get("prompt_keyframes", [])),
        current_user_os=data.get("current_user_os", ""),
        tmp_deforum_run_duplicated_folder=data.get("tmp_deforum_run_duplicated_folder", ""),
        job_id=data.get("job_id"),
    )


# Pure conversion functions to legacy format
def to_legacy_dict(args: ProcessedArguments) -> Dict[str, Any]:
    """Pure function: immutable args -> legacy dictionary format"""
    # Convert all dataclasses to dictionaries
    deforum_dict = asdict(args.deforum)
    animation_dict = asdict(args.animation)
    video_dict = asdict(args.video)
    parseq_dict = asdict(args.parseq)
    wan_dict = asdict(args.wan)
    root_dict = asdict(args.root)
    
    # Apply legacy field name mappings
    deforum_dict["W"] = deforum_dict.pop("width")
    deforum_dict["H"] = deforum_dict.pop("height")
    deforum_dict["scale"] = deforum_dict.pop("cfg_scale")
    deforum_dict["seed_iter_N"] = deforum_dict.pop("seed_iter_n")
    
    # Convert enums back to strings for deforum args
    deforum_dict["sampler"] = deforum_dict["sampler"].value if hasattr(deforum_dict["sampler"], 'value') else str(deforum_dict["sampler"])
    deforum_dict["scheduler"] = deforum_dict["scheduler"].value if hasattr(deforum_dict["scheduler"], 'value') else str(deforum_dict["scheduler"])
    deforum_dict["seed_behavior"] = deforum_dict["seed_behavior"].value if hasattr(deforum_dict["seed_behavior"], 'value') else str(deforum_dict["seed_behavior"])
    deforum_dict["fill"] = deforum_dict["fill"].value if hasattr(deforum_dict["fill"], 'value') else str(deforum_dict["fill"])
    deforum_dict["reroll_blank_frames"] = deforum_dict["reroll_blank_frames"].value if hasattr(deforum_dict["reroll_blank_frames"], 'value') else str(deforum_dict["reroll_blank_frames"])
    
    # Convert enums back to strings for animation args
    animation_dict["animation_mode"] = animation_dict["animation_mode"].value if hasattr(animation_dict["animation_mode"], 'value') else str(animation_dict["animation_mode"])
    animation_dict["border"] = animation_dict["border"].value if hasattr(animation_dict["border"], 'value') else str(animation_dict["border"])
    animation_dict["color_coherence"] = animation_dict["color_coherence"].value if hasattr(animation_dict["color_coherence"], 'value') else str(animation_dict["color_coherence"])
    animation_dict["noise_type"] = animation_dict["noise_type"].value if hasattr(animation_dict["noise_type"], 'value') else str(animation_dict["noise_type"])
    animation_dict["depth_algorithm"] = animation_dict["depth_algorithm"].value if hasattr(animation_dict["depth_algorithm"], 'value') else str(animation_dict["depth_algorithm"])
    animation_dict["padding_mode"] = animation_dict["padding_mode"].value if hasattr(animation_dict["padding_mode"], 'value') else str(animation_dict["padding_mode"])
    animation_dict["sampling_mode"] = animation_dict["sampling_mode"].value if hasattr(animation_dict["sampling_mode"], 'value') else str(animation_dict["sampling_mode"])
    
    # Fix field name mapping for legacy compatibility
    animation_dict["color_coherence_video_every_N_frames"] = animation_dict.pop("color_coherence_video_every_n_frames")
    
    # Combine all dictionaries
    combined = {}
    combined.update(deforum_dict)
    combined.update(animation_dict)
    combined.update(video_dict)
    combined.update(parseq_dict)
    combined.update(wan_dict)
    combined.update(root_dict)
    
    return combined


def to_legacy_namespace(args: ProcessedArguments) -> SimpleNamespace:
    """Pure function: immutable args -> legacy SimpleNamespace format"""
    legacy_dict = to_legacy_dict(args)
    return SimpleNamespace(**legacy_dict)


def create_separate_legacy_namespaces(args: ProcessedArguments) -> tuple:
    """Pure function: immutable args -> separate legacy SimpleNamespace objects"""
    deforum_dict = asdict(args.deforum)
    animation_dict = asdict(args.animation)
    video_dict = asdict(args.video)
    parseq_dict = asdict(args.parseq)
    wan_dict = asdict(args.wan)
    root_dict = asdict(args.root)
    
    # Apply legacy field mappings
    deforum_dict["W"] = deforum_dict.pop("width")
    deforum_dict["H"] = deforum_dict.pop("height")
    deforum_dict["scale"] = deforum_dict.pop("cfg_scale")
    deforum_dict["seed_iter_N"] = deforum_dict.pop("seed_iter_n")
    
    # Fix animation field mapping
    animation_dict["color_coherence_video_every_N_frames"] = animation_dict.pop("color_coherence_video_every_n_frames")
    
    # Convert enums to strings for all dictionaries
    for enum_field in ["sampler", "scheduler", "seed_behavior", "fill", "reroll_blank_frames"]:
        if enum_field in deforum_dict:
            deforum_dict[enum_field] = str(deforum_dict[enum_field])
    
    for enum_field in ["animation_mode", "border", "color_coherence", "noise_type", "depth_algorithm", "padding_mode", "sampling_mode"]:
        if enum_field in animation_dict:
            animation_dict[enum_field] = str(animation_dict[enum_field])
    
    return (
        SimpleNamespace(**deforum_dict),
        SimpleNamespace(**animation_dict),
        SimpleNamespace(**video_dict),
        SimpleNamespace(**parseq_dict),
        SimpleNamespace(**wan_dict),
        SimpleNamespace(**root_dict),
    ) 