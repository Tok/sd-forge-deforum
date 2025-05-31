def analyze_movement_handler(wan_enhanced_prompts):
    """Analyze Deforum movement schedules and add movement descriptions to wan prompts"""
    
    try:
        import gradio as gr
        from ...args import DeforumAnimArgs
        from .movement_analyzer import analyze_deforum_movement
        
        print(f"🎬 Starting movement analysis...")
        
        # Get current Deforum settings using the same method as normal renders
        deforum_args = DeforumAnimArgs()
        
        # Create a simple namespace with required movement parameters
        from types import SimpleNamespace
        anim_args = SimpleNamespace()
        
        # Get movement schedules from UI components (these contain actual schedule strings)
        try:
            # Translation schedules
            anim_args.translation_x = str(deforum_args['translation_x'])
            anim_args.translation_y = str(deforum_args['translation_y'])
            anim_args.translation_z = str(deforum_args['translation_z'])
            
            # Rotation schedules  
            anim_args.rotation_3d_x = str(deforum_args['rotation_3d_x'])
            anim_args.rotation_3d_y = str(deforum_args['rotation_3d_y'])
            anim_args.rotation_3d_z = str(deforum_args['rotation_3d_z'])
            
            # Zoom and angle
            anim_args.zoom = str(deforum_args['zoom'])
            anim_args.angle = str(deforum_args.get('angle', '0:(0)'))
            
            # Camera Shakify parameters (enhanced feature)
            anim_args.shake_name = str(deforum_args.get('shake_name', 'None'))
            anim_args.shake_intensity = float(deforum_args.get('shake_intensity', 0.0))
            anim_args.shake_speed = float(deforum_args.get('shake_speed', 1.0))
            
        except Exception as e:
            print(f"⚠️ Error accessing movement schedules: {e}")
            # Fallback to static values
            anim_args.translation_x = "0:(0)"
            anim_args.translation_y = "0:(0)"
            anim_args.translation_z = "0:(0)"
            anim_args.rotation_3d_x = "0:(0)"
            anim_args.rotation_3d_y = "0:(0)"
            anim_args.rotation_3d_z = "0:(0)"
            anim_args.zoom = "0:(1.0)"
            anim_args.angle = "0:(0)"
            anim_args.shake_name = "None"
            anim_args.shake_intensity = 0.0
            anim_args.shake_speed = 1.0
        
        print(f"📋 Movement Analysis Parameters:")
        print(f"   🎯 Translation X: {anim_args.translation_x}")
        print(f"   🎯 Translation Y: {anim_args.translation_y}")
        print(f"   🎯 Translation Z: {anim_args.translation_z}")
        print(f"   🔄 Rotation X: {anim_args.rotation_3d_x}")
        print(f"   🔄 Rotation Y: {anim_args.rotation_3d_y}")
        print(f"   🔄 Rotation Z: {anim_args.rotation_3d_z}")
        print(f"   🔍 Zoom: {anim_args.zoom}")
        print(f"   📐 Angle: {anim_args.angle}")
        print(f"   🎬 Shake Name: {anim_args.shake_name}")
        print(f"   💪 Shake Intensity: {anim_args.shake_intensity}")
        print(f"   ⚡ Shake Speed: {anim_args.shake_speed}")
        
        # Analyze movement with enhanced detection
        movement_desc, movement_strength = analyze_deforum_movement(
            anim_args, 
            sensitivity=1.0, 
            max_frames=120
        )
        
        print(f"🎯 Movement Analysis Result:")
        print(f"   📝 Description: {movement_desc}")
        print(f"   💪 Strength: {movement_strength:.3f}")
        
        # Parse existing prompts
        try:
            if not wan_enhanced_prompts or wan_enhanced_prompts.strip() == "":
                current_prompts = {}
            else:
                current_prompts = json.loads(wan_enhanced_prompts)
        except json.JSONDecodeError as e:
            print(f"⚠️ Invalid JSON in wan prompts: {e}")
            current_prompts = {}
        
        # Add movement descriptions to prompts
        enhanced_prompts = {}
        for frame_key, prompt in current_prompts.items():
            # Add movement description to each prompt
            if movement_desc and movement_desc != "static camera position":
                enhanced_prompt = f"{prompt}, {movement_desc}"
            else:
                enhanced_prompt = prompt
            
            enhanced_prompts[frame_key] = enhanced_prompt
        
        # If no prompts exist, create a basic one with movement
        if not enhanced_prompts and movement_desc != "static camera position":
            enhanced_prompts["0"] = f"cinematic scene with {movement_desc}"
        
        # Convert back to JSON
        result_json = json.dumps(enhanced_prompts, indent=2)
        
        print(f"✅ Movement analysis complete. Updated {len(enhanced_prompts)} prompts with movement descriptions.")
        
        return result_json
        
    except Exception as e:
        print(f"❌ Movement analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return wan_enhanced_prompts  # Return original if analysis fails 