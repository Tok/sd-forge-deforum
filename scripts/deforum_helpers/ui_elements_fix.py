def wan_generate_video(*component_args):
    """
    Function to handle Wan video generation from the Wan tab
    This bypasses run_deforum to avoid diffusers corruption during model loading
    """
    try:
        # Import here to avoid circular imports - only what we need for direct Wan generation
        from .args import get_component_names, process_args
        from .wan_integration import WanVideoGenerator, WanPromptScheduler, validate_wan_settings
        from .render_wan import handle_frame_overlap, save_clip_frames, create_generation_summary
        from modules.processing import StableDiffusionProcessing
        import uuid
        import modules.shared as shared
        import os
        from datetime import datetime
        
        print("🎬 Wan video generation triggered from Wan tab")
        print("🔒 Using isolated Wan generation path (bypassing run_deforum)")
        
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
        
        # Force animation mode to Wan Video
        args_dict['animation_mode'] = 'Wan Video'
        
        # Create proper output directory for Wan images
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
            return "❌ Failed to load arguments for Wan generation"
        
        # Validate Wan settings
        validation_errors = validate_wan_settings(wan_args)
        if validation_errors:
            error_msg = "❌ Wan validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors)
            print(error_msg)
            return error_msg
        
        if not wan_args.wan_enabled:
            return "❌ Wan Video mode selected but Wan is not enabled. Please enable Wan in the Wan Video tab."
        
        print(f"✅ Arguments processed successfully")
        print(f"📁 Output directory: {args.outdir}")
        print(f"🎯 Model path: {wan_args.wan_model_path}")
        print(f"📐 Resolution: {wan_args.wan_resolution}")
        print(f"🎬 FPS: {wan_args.wan_fps}")
        print(f"⏱️ Clip Duration: {wan_args.wan_clip_duration}s")
        
        # Initialize Wan generator with full isolation
        print("🔧 Initializing Wan generator with full diffusers isolation...")
        wan_generator = WanVideoGenerator(wan_args.wan_model_path, shared.device)
        
        try:
            # This will fail fast if Wan is not available - NO diffusers corruption
            wan_generator.load_model()
            
            # Parse animation prompts and calculate timing
            prompt_scheduler = WanPromptScheduler(root.animation_prompts, wan_args, video_args)
            prompt_schedule = prompt_scheduler.parse_prompts_and_timing()
            
            print(f"📝 Generated {len(prompt_schedule)} video clips:")
            for i, (prompt, start_time, duration) in enumerate(prompt_schedule):
                print(f"  Clip {i+1}: '{prompt[:50]}...' (start: {start_time:.1f}s, duration: {duration:.1f}s)")
            
            # Generate video clips with full isolation
            all_frames = []
            previous_frame = None
            total_frames_generated = 0
            
            for i, (prompt, start_time, duration) in enumerate(prompt_schedule):
                if shared.state.interrupted:
                    print("⏹️ Generation interrupted by user")
                    break
                    
                print(f"\n🎬 Generating clip {i+1}/{len(prompt_schedule)}: {prompt[:50]}...")
                
                # Calculate seed for this clip
                clip_seed = wan_args.wan_seed if wan_args.wan_seed != -1 else -1
                
                try:
                    if i == 0:
                        # First clip: text-to-video
                        print(f"🎭 Mode: Text-to-Video")
                        frames = wan_generator.generate_txt2video(
                            prompt=prompt,
                            duration=duration,
                            fps=wan_args.wan_fps,
                            resolution=wan_args.wan_resolution,
                            steps=wan_args.wan_inference_steps,
                            guidance_scale=wan_args.wan_guidance_scale,
                            seed=clip_seed,
                            motion_strength=wan_args.wan_motion_strength
                        )
                    else:
                        # Subsequent clips: image-to-video
                        print(f"🖼️ Mode: Image-to-Video (using previous frame as init)")
                        frames = wan_generator.generate_img2video(
                            init_image=previous_frame,
                            prompt=prompt,
                            duration=duration,
                            fps=wan_args.wan_fps,
                            resolution=wan_args.wan_resolution,
                            steps=wan_args.wan_inference_steps,
                            guidance_scale=wan_args.wan_guidance_scale,
                            seed=clip_seed,
                            motion_strength=wan_args.wan_motion_strength
                        )
                    
                    if not frames:
                        print(f"❌ ERROR: No frames generated for clip {i+1}")
                        continue
                    
                    print(f"✅ Generated {len(frames)} frames for clip {i+1}")
                    
                    # Handle frame overlap and save frames
                    processed_frames = handle_frame_overlap(frames, previous_frame, wan_args.wan_frame_overlap, i > 0)
                    
                    # Save frames to disk
                    saved_frame_count = save_clip_frames(processed_frames, args.outdir, root.timestring, i, total_frames_generated)
                    total_frames_generated += saved_frame_count
                    
                    # Add frames to the all_frames list
                    all_frames.extend(processed_frames)
                    
                    # Extract last frame for next clip initialization
                    previous_frame = wan_generator.extract_last_frame(frames)
                    
                    # Update progress
                    shared.state.job = f"Wan clip {i+1}/{len(prompt_schedule)}"
                    shared.state.job_no = i + 1
                    shared.state.job_count = len(prompt_schedule)
                    
                    print(f"✅ Clip {i+1} completed successfully")
                    
                except Exception as e:
                    print(f"❌ ERROR generating clip {i+1}: {e}")
                    # Continue with next clip instead of failing completely
                    continue
            
            print(f"\n🎉 Wan Video Generation Complete!")
            print(f"📊 Generated {len(prompt_schedule)} clips with {total_frames_generated} total frames")
            print(f"📁 Output directory: {args.outdir}")
            
            return f"✅ Wan video generation completed successfully!\n📊 Generated {total_frames_generated} frames\n📁 Output: {args.outdir}"
            
        except Exception as e:
            error_msg = f"❌ FATAL ERROR in Wan video generation: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
            
        finally:
            # Always unload the model to free GPU memory
            wan_generator.unload_model()
            print("🧹 Wan model unloaded, GPU memory freed")
            
    except Exception as e:
        error_msg = f"❌ Error during Wan video generation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg