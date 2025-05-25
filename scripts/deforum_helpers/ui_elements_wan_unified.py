def wan_generate_video(*component_args):
    """
    Unified function to handle WAN video generation from the WAN tab
    Uses the consolidated WAN integration with proper fallback handling
    """
    try:
        # Import unified modules
        from .args import get_component_names, process_args
        from .wan_integration_unified import WanVideoGenerator, WanPromptScheduler, validate_wan_settings
        from .render_wan_unified import handle_frame_overlap, save_clip_frames, create_generation_summary
        from modules.processing import StableDiffusionProcessing
        import uuid
        import modules.shared as shared
        import os
        from datetime import datetime
        
        print("🎬 WAN video generation triggered from WAN tab")
        print("🔒 Using unified WAN generation pipeline")
        
        # Generate a unique ID for this run
        job_id = str(uuid.uuid4())[:8]
        
        # Get component names to understand the argument structure
        component_names = get_component_names()
        
        # Create the arguments dict for processing
        args_dict = {}
        for i, name in enumerate(component_names):
            if i < len(component_args):
                args_dict[name] = component_args[i]
            else:
                args_dict[name] = None
        
        # Add required fields for process_args
        args_dict['override_settings_with_file'] = False
        args_dict['custom_settings_file'] = ""
        args_dict['animation_prompts'] = args_dict.get('animation_prompts', '{"0": "a beautiful landscape"}')
        args_dict['animation_prompts_positive'] = args_dict.get('animation_prompts_positive', "")
        args_dict['animation_prompts_negative'] = args_dict.get('animation_prompts_negative', "")
        
        # Force animation mode to WAN Video
        args_dict['animation_mode'] = 'Wan Video'
        
        # Create proper output directory for WAN images
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        batch_name = args_dict.get('batch_name', 'Deforum')
        
        # Use webui-forge's output directory structure with wan-images folder
        webui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        deforum_output_dir = os.path.join(webui_root, 'outputs', 'wan-images', f"{batch_name}_{timestamp}")
        os.makedirs(deforum_output_dir, exist_ok=True)
        
        class MockProcessing:
            def __init__(self):
                self.outpath_samples = deforum_output_dir
        
        args_dict['p'] = MockProcessing()
        
        print(f"📊 Processing {len(component_args)} component arguments...")
        
        # Process arguments using Deforum's argument processing
        args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, wan_args = process_args(args_dict, job_id)
        
        if not args_loaded_ok:
            return "❌ Failed to load arguments for WAN generation"
        
        # Validate WAN settings
        try:
            validate_wan_settings(wan_args)
        except ValueError as e:
            error_msg = f"❌ WAN validation failed: {e}"
            print(error_msg)
            return error_msg
        
        if not wan_args.wan_enabled:
            return "❌ WAN Video mode selected but WAN is not enabled. Please enable WAN in the WAN Video tab."
        
        print(f"✅ Arguments processed successfully")
        print(f"📁 Output directory: {args.outdir}")
        print(f"🎯 Model path: {wan_args.wan_model_path}")
        print(f"📐 Resolution: {wan_args.wan_resolution}")
        print(f"🎬 FPS: {wan_args.wan_fps}")
        print(f"⏱️ Clip Duration: {wan_args.wan_clip_duration}s")
        
        # Initialize unified WAN generator
        print("🔧 Initializing unified WAN generator...")
        try:
            wan_generator = WanVideoGenerator(wan_args.wan_model_path, shared.device)
            
            # Load the model (auto-detects Open-Sora vs SD fallback)
            print("🔄 Loading WAN model...")
            wan_generator.load_model()
            
            # Show what mode we're using
            if wan_generator.use_opensora:
                print("✅ Using Open-Sora video generation pipeline")
            else:
                print("✅ Using Stable Diffusion video generation pipeline")
            
        except Exception as e:
            # Handle any errors during initialization
            error_msg = f"""❌ WAN Generation Error

Failed to initialize unified WAN generator: {e}

This could indicate:
- Model path issues: {wan_args.wan_model_path}
- Missing dependencies  
- Repository setup problems
- File system access issues

The system will attempt to use Stable Diffusion fallback mode.
Check the console output above for more details.
"""
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
        
        try:
            # Parse prompts and calculate timing
            print("📋 Parsing animation prompts...")
            prompt_scheduler = WanPromptScheduler(root.animation_prompts, wan_args, video_args)
            prompt_schedule = prompt_scheduler.parse_prompts_and_timing()
            
            print(f"Found {len(prompt_schedule)} clips to generate:")
            for i, (prompt, start_time, duration) in enumerate(prompt_schedule):
                frame_count = int(duration * wan_args.wan_fps)
                print(f"  Clip {i+1}: {frame_count} frames ({duration:.1f}s) - '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            
            # Generate video clips using unified pipeline
            all_frames = []
            total_frames_generated = 0
            previous_frame = None
            
            for clip_index, (prompt, start_time, duration) in enumerate(prompt_schedule):
                print(f"\n🎬 Generating Clip {clip_index + 1}/{len(prompt_schedule)}")
                print(f"Prompt: {prompt}")
                print(f"Duration: {duration}s")
                
                try:
                    # Prepare generation parameters
                    generation_kwargs = {
                        'prompt': prompt,
                        'duration': duration,
                        'fps': wan_args.wan_fps,
                        'resolution': wan_args.wan_resolution,
                        'steps': wan_args.wan_inference_steps,
                        'guidance_scale': wan_args.wan_guidance_scale,
                        'seed': args.seed if hasattr(args, 'seed') else -1,
                        'motion_strength': wan_args.wan_motion_strength
                    }
                    
                    # Add SD-specific parameters if using SD fallback
                    if not wan_generator.use_opensora:
                        from .parseq_adapter import ParseqAdapter
                        from .animation_key_frames import DeformAnimKeys
                        
                        parseq_adapter = ParseqAdapter(parseq_args, anim_args, video_args, controlnet_args, loop_args, freeu_args, kohya_hrfix_args)
                        keys = DeformAnimKeys(anim_args, args.seed) if not parseq_adapter.use_parseq else parseq_adapter.anim_keys
                        
                        generation_kwargs.update({
                            'args': args,
                            'keys': keys,
                            'anim_args': anim_args,
                            'loop_args': loop_args,
                            'controlnet_args': controlnet_args,
                            'freeu_args': freeu_args,
                            'kohya_hrfix_args': kohya_hrfix_args,
                            'root': root,
                            'parseq_adapter': parseq_adapter
                        })
                    
                    # Determine if this is a continuation (image2video) or new generation (text2video)
                    is_continuation = (clip_index > 0 and previous_frame is not None and wan_args.wan_frame_overlap > 0)
                    
                    if is_continuation:
                        print(f"🔗 Generating image-to-video continuation from previous frame")
                        clip_frames = wan_generator.generate_img2video(
                            init_image=previous_frame,
                            **generation_kwargs
                        )
                    else:
                        print(f"🎨 Generating text-to-video from prompt")
                        clip_frames = wan_generator.generate_txt2video(**generation_kwargs)
                    
                    if not clip_frames:
                        raise RuntimeError(f"No frames generated for clip {clip_index + 1}")
                    
                    print(f"✅ Generated {len(clip_frames)} frames for clip {clip_index + 1}")
                    
                    # Handle frame overlapping between clips
                    if clip_index > 0 and wan_args.wan_frame_overlap > 0:
                        processed_frames = handle_frame_overlap(
                            frames=clip_frames,
                            previous_frame=previous_frame,
                            overlap_count=wan_args.wan_frame_overlap,
                            is_continuation=True
                        )
                    else:
                        processed_frames = clip_frames
                    
                    # Save frames to disk
                    frames_saved = save_clip_frames(
                        frames=processed_frames,
                        outdir=args.outdir,
                        timestring=root.timestring,
                        clip_index=clip_index,
                        start_frame_number=total_frames_generated,
                        args=args,
                        video_args=video_args,
                        root=root
                    )
                    
                    all_frames.extend(processed_frames)
                    total_frames_generated += frames_saved
                    
                    # Store last frame for potential continuation
                    if processed_frames:
                        previous_frame = wan_generator.extract_last_frame(processed_frames)
                    
                    print(f"✅ Clip {clip_index + 1} completed: {frames_saved} frames saved")
                    
                except Exception as e:
                    error_msg = f"❌ Error generating clip {clip_index + 1}: {e}"
                    print(error_msg)
                    return error_msg
            
            # Create generation summary
            summary = create_generation_summary(prompt_schedule, wan_args, total_frames_generated)
            print(f"\n{summary}")
            
            success_msg = f"""✅ Unified WAN Video Generation Completed Successfully!

Total frames generated: {total_frames_generated}
Total clips: {len(prompt_schedule)}
Output directory: {args.outdir}
Generation method: {"Open-Sora" if wan_generator.use_opensora else "Stable Diffusion"}

{summary}

You can now create a video from the generated frames using Deforum's video creation tools or external software.
"""
            
            print(success_msg)
            return success_msg
            
        except Exception as e:
            error_msg = f"❌ Error during unified WAN video generation: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
            
        finally:
            # Always attempt cleanup
            try:
                wan_generator.unload_model()
                print("🧹 Unified WAN model cleanup completed")
            except Exception as e:
                print(f"Warning: Error during model cleanup: {e}")
            
    except Exception as e:
        error_msg = f"❌ CRITICAL ERROR during unified WAN video generation setup: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg
