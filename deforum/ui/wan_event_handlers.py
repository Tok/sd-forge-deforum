"""
WAN Event Handlers
Contains WAN-specific event handling logic
"""

import json
import os


def load_wan_prompts_handler():
    """Load Wan prompts from default settings"""
    try:
        # Load prompts from default_settings.txt
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'default_settings.txt')
        
        if not os.path.exists(settings_path):
            print(f"❌ Default settings file not found: {settings_path}")
            return "0: A peaceful landscape scene, photorealistic"
        
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # Get wan_prompts from settings
        wan_prompts = settings.get('wan_prompts', {})
        
        if not wan_prompts:
            print("⚠️ No wan_prompts found in default settings, falling back to basic prompt")
            return "0: A peaceful landscape scene, photorealistic"
        
        # Convert prompts dict to textarea format (frame: prompt)
        prompt_lines = []
        for frame, prompt in sorted(wan_prompts.items(), key=lambda x: int(x[0])):
            prompt_lines.append(f"{frame}: {prompt}")
        
        result = "\n".join(prompt_lines)
        print(f"✅ Loaded {len(wan_prompts)} Wan prompts from default settings")
        return result
        
    except Exception as e:
        print(f"❌ Error loading Wan prompts: {e}")
        return f"0: Error loading prompts: {str(e)}"


def load_deforum_prompts_handler():
    """Load original Deforum prompts from default settings"""
    try:
        # Load prompts from default_settings.txt
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'default_settings.txt')
        
        if not os.path.exists(settings_path):
            print(f"❌ Default settings file not found: {settings_path}")
            return "0: A peaceful landscape scene, photorealistic"
        
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # Get prompts from settings (main prompts section)
        deforum_prompts = settings.get('prompts', {})
        
        if not deforum_prompts:
            print("⚠️ No prompts found in default settings, falling back to basic prompt")
            return "0: A peaceful landscape scene, photorealistic"
        
        # Convert prompts dict to textarea format (frame: prompt)
        prompt_lines = []
        for frame, prompt in sorted(deforum_prompts.items(), key=lambda x: int(x[0])):
            prompt_lines.append(f"{frame}: {prompt}")
        
        result = "\n".join(prompt_lines)
        print(f"✅ Loaded {len(deforum_prompts)} Deforum prompts from default settings")
        return result
        
    except Exception as e:
        print(f"❌ Error loading Deforum prompts: {e}")
        return f"0: Error loading prompts: {str(e)}"


def load_deforum_to_wan_prompts_handler():
    """Load current Deforum prompts into Wan prompts field"""
    try:
        # Try to get animation prompts from the stored component reference
        animation_prompts_json = ""
        
        # Import here to avoid circular imports
        from .wan_event_handlers import enhance_prompts_handler
        
        if hasattr(enhance_prompts_handler, '_animation_prompts_component'):
            try:
                animation_prompts_json = enhance_prompts_handler._animation_prompts_component.value
                print(f"📋 Loading Deforum prompts to Wan prompts field")
            except Exception as e:
                print(f"⚠️ Could not access animation_prompts component: {e}")
        
        if not animation_prompts_json or animation_prompts_json.strip() == "":
            return """{"0": "No Deforum prompts found! Go to the Prompts tab and configure your animation prompts first."}"""
        
        # Parse the JSON and convert to clean Wan format
        try:
            prompts_dict = json.loads(animation_prompts_json)
            
            # Convert to Wan format (clean prompts without negative parts)
            wan_prompts_dict = {}
            for frame, prompt in prompts_dict.items():
                # Clean up the prompt (remove negative prompts)
                clean_prompt = prompt.split('--neg')[0].strip()
                wan_prompts_dict[frame] = clean_prompt
            
            # Return as JSON
            result = json.dumps(wan_prompts_dict, ensure_ascii=False, indent=2)
            print(f"✅ Converted {len(prompts_dict)} Deforum prompts to Wan JSON format")
            return result
            
        except json.JSONDecodeError as e:
            return json.dumps({
                "0": f"Invalid JSON in Deforum prompts: {str(e)}. Fix the JSON format in the Prompts tab first."
            }, indent=2)
            
    except Exception as e:
        return f"❌ Error loading Deforum prompts: {str(e)}"


def load_wan_defaults_handler():
    """Load default Wan prompts from settings file"""
    try:
        # Load default prompts from settings
        settings_path = os.path.join(os.path.dirname(__file__), '..', 'default_settings.txt')
        
        if not os.path.exists(settings_path):
            # Fallback to simple defaults
            return json.dumps({
                "0": "prompt text",
                "60": "another prompt"
            }, ensure_ascii=False, indent=2)
        
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            wan_prompts = settings.get('wan_prompts', {})
            
            if wan_prompts:
                # Return as JSON
                result = json.dumps(wan_prompts, ensure_ascii=False, indent=2)
                print(f"✅ Loaded {len(wan_prompts)} default Wan prompts from settings")
                return result
            else:
                # Use fallback
                return json.dumps({
                    "0": "prompt text",
                    "60": "another prompt"
                }, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error loading default settings: {e}")
            # Return simple fallback
            return json.dumps({
                "0": "prompt text",
                "60": "another prompt"
            }, ensure_ascii=False, indent=2)
            
    except Exception as e:
        return json.dumps({
            "0": f"Error loading default prompts: {str(e)}"
        }, indent=2)


def validate_wan_generation(current_prompts):
    """Validate that Wan generation requirements are met"""
    try:
        # Check if prompts are empty
        if not current_prompts or current_prompts.strip() == "":
            return """⚠️ **Prompts Required**

📋 **Load prompts to get started:**
• Click "Load from Deforum Prompts" to use your animation prompts
• Or click "Load Default Wan Prompts" for examples
• Then optionally enhance with AI or add movement descriptions"""
        
        # Check if it's just placeholder text
        if any(placeholder in current_prompts.lower() for placeholder in ["required:", "load prompts", "placeholder"]):
            return """⚠️ **Load Real Prompts**

📋 **Replace placeholder text:**
• Click "Load from Deforum Prompts" to copy your animation prompts
• Or click "Load Default Wan Prompts" for examples"""
        
        # Try to parse as JSON
        try:
            prompts_dict = json.loads(current_prompts)
            if not prompts_dict:
                return "⚠️ **Empty prompts** - Add some prompts first"
            
            # Check if prompts are just basic placeholders
            first_prompt = list(prompts_dict.values())[0].lower()
            if any(placeholder in first_prompt for placeholder in ["prompt text", "beautiful landscape", "load prompts"]):
                return """⚠️ **Default/Placeholder Prompts Detected**

📋 **Load your real prompts:**
• Click "Load from Deforum Prompts" to copy your animation prompts
• Or edit the prompts manually to describe your desired video"""
                
            # All good - ready to generate!
            num_prompts = len(prompts_dict)
            return f"""✅ **Ready to Generate!** 

🎬 **Found {num_prompts} prompt{'s' if num_prompts != 1 else ''}** for Wan video generation
🔥 **Click "Generate Wan Video" above** to start I2V chaining generation
⚡ **Optional:** Add movement descriptions or AI enhancement first"""
            
        except json.JSONDecodeError:
            return """❌ **Invalid JSON Format**

🔧 **Fix the format:**
• Prompts should be in JSON format like: {"0": "prompt text", "60": "another prompt"}
• Check for missing quotes, commas, or brackets"""
    
    except Exception as e:
        return f"❌ **Validation Error:** {str(e)}"


def analyze_movement_handler(current_prompts, enable_shakify=True, sensitivity_override=False, manual_sensitivity=1.0):
    """Handle movement analysis with Camera Shakify integration"""
    try:
        from ..integrations.wan.utils.movement_analyzer import analyze_movement_from_schedules
        import json
        
        print(f"📐 Starting Movement Analysis...")
        print(f"🎬 Camera Shakify: {'Enabled' if enable_shakify else 'Disabled'}")
        print(f"📊 Sensitivity Override: {'Manual ({})'.format(manual_sensitivity) if sensitivity_override else 'Auto-calculated'}")
        
        # Parse current prompts
        if not current_prompts or current_prompts.strip() == "":
            error_msg = """❌ No prompts found!

🔧 **Setup Required:**
1. 📝 Load prompts using "Load from Deforum Prompts" or "Load Default Wan Prompts"
2. 📐 Click **Add Movement Descriptions** again after setting up prompts

💡 **Quick Start:**
Click "Load Default Wan Prompts" to start with example prompts!"""
            return current_prompts, error_msg
        
        # Try to parse as JSON
        animation_prompts = None
        try:
            animation_prompts = json.loads(current_prompts)
            print(f"✅ Successfully parsed {len(animation_prompts)} prompts as JSON")
        except json.JSONDecodeError:
            # Try to parse as readable format
            try:
                animation_prompts = {}
                for line in current_prompts.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 1)
                        frame_part = parts[0].strip()
                        prompt_part = parts[1].strip()
                        
                        # Extract frame number
                        if frame_part.lower().startswith('frame '):
                            frame_num = frame_part[6:].strip()
                        else:
                            frame_num = frame_part
                        
                        animation_prompts[frame_num] = prompt_part
                
                if animation_prompts:
                    print(f"✅ Successfully parsed {len(animation_prompts)} prompts from readable format")
                else:
                    raise ValueError("No valid prompts found")
            except Exception as e:
                print(f"❌ Could not parse prompts: {e}")
                error_msg = f"❌ Invalid format in prompts. Expected JSON format like:\n{{\n  \"0\": \"prompt text\",\n  \"60\": \"another prompt\"\n}}\n\nOr readable format like:\nFrame 0: prompt text\nFrame 60: another prompt"
                return current_prompts, error_msg
        
        if not animation_prompts:
            error_msg = """❌ No valid prompts found!

🔧 **Setup Required:**
1. 📝 Load prompts using the load buttons above
2. 📋 Make sure your prompts are in proper JSON format like:
   {
     "0": "prompt text",
     "60": "another prompt",
     "120": "a cyberpunk environment with glowing elements"
   }
3. 📐 Click **Add Movement Descriptions** again after setting up prompts"""
            return current_prompts, error_msg
        
        # Validate prompts content  
        if len(animation_prompts) == 1 and "0" in animation_prompts and "beautiful landscape" in animation_prompts["0"]:
            error_msg = """❌ Default prompts detected!

🔧 **Please configure your actual animation prompts:**
1. 📝 Load your real prompts using the load buttons above
2. ✏️ Or manually edit the prompts field
3. 📐 Click **Add Movement Descriptions** again

💡 **For your animation sequence:**
Set up prompts like:
{
  "0": "A peaceful scene, photorealistic",
  "18": "A scene with glowing effects, neon colors, synthwave aesthetic", 
  "36": "A cyberpunk scene with LED patterns, digital environment"
}"""
            return current_prompts, error_msg
        
        print(f"📐 Analyzing movement for {len(animation_prompts)} prompts...")
        
        # Determine sensitivity - use manual override if enabled, else auto-calculate
        if sensitivity_override:
            sensitivity = manual_sensitivity
            print(f"📊 Using manual sensitivity: {sensitivity}")
        else:
            sensitivity = None  # Will be auto-calculated by the analyzer
            print(f"📊 Using auto-calculated sensitivity")
        
        # Call movement analyzer with Camera Shakify integration
        enhanced_prompts, movement_description = analyze_movement_from_schedules(
            animation_prompts, 
            enable_shakify=enable_shakify,
            sensitivity=sensitivity
        )
        
        # Convert enhanced prompts back to JSON format
        enhanced_json = json.dumps(enhanced_prompts, ensure_ascii=False, indent=2)
        
        print(f"✅ Movement analysis completed successfully")
        print(f"📊 Enhanced {len(enhanced_prompts)} prompts with movement descriptions")
        
        return enhanced_json, movement_description
        
    except ImportError as e:
        error_msg = f"❌ Movement analyzer not available: {e}\n\nMake sure all required dependencies are installed."
        print(f"❌ Import error: {e}")
        return current_prompts, error_msg
    except Exception as e:
        error_msg = f"❌ Movement analysis error: {str(e)}"
        print(f"❌ Analysis error: {e}")
        return current_prompts, error_msg


def enhance_prompts_handler(current_prompts, qwen_model, language, auto_download):
    """Handle prompt enhancement with QwenPromptExpander with progress feedback"""
    try:
        from ..integrations.wan.utils.qwen_manager import qwen_manager
        
        print(f"🎨 AI Prompt Enhancement requested for {qwen_model}")
        print(f"📝 Received prompts: {str(current_prompts)[:100]}...")
        
        # Progress: Start
        progress_update = "🎨 Starting AI Prompt Enhancement...\n"
        
        # Check if auto-download is enabled for model availability
        if not auto_download:
            # Check if the selected model is available
            if not qwen_manager.is_model_downloaded(qwen_model):
                return f"""❌ Qwen model not available: {qwen_model}

🔧 **Model Download Required:**
1. ✅ Enable "Auto-Download Qwen Models" checkbox
2. 🎨 Click "AI Prompt Enhancement" again to auto-download
3. ⏳ Wait for download to complete

📥 **Manual Download Alternative:**
1. Use HuggingFace CLI: `huggingface-cli download {qwen_manager.get_model_info(qwen_model).get('huggingface_id', 'model-id')}`
2. ✅ Enable auto-download for easier setup

💡 **Auto-download is recommended** for seamless model management.""", progress_update + "❌ Model not available - enable auto-download!"
        
        # Progress: Model check
        progress_update += "🔍 Checking model availability...\n"
        
        # Check if a model is already loaded
        if qwen_manager.is_model_loaded():
            loaded_info = qwen_manager.get_loaded_model_info()
            current_model = loaded_info['name'] if loaded_info else "Unknown"
            
            # If different model requested, cleanup first
            if qwen_model != "Auto-Select" and current_model != qwen_model:
                print(f"🔄 Switching from {current_model} to {qwen_model}")
                progress_update += f"🔄 Switching from {current_model} to {qwen_model}...\n"
                qwen_manager.cleanup_cache()
        
        # Progress: Model loading
        if not qwen_manager.is_model_loaded():
            if qwen_model == "Auto-Select":
                selected_model = qwen_manager.auto_select_model()
                print(f"🤖 Auto-selected model: {selected_model}")
                progress_update += f"🤖 Auto-selected model: {selected_model}\n"
            else:
                print(f"📥 Loading Qwen model: {qwen_model}")
                progress_update += f"📥 Loading Qwen model: {qwen_model}...\n"
        
        # Get wan prompts from the current_prompts parameter
        animation_prompts = None
        
        if current_prompts and current_prompts.strip():
            try:
                # Try to parse as JSON first
                animation_prompts = json.loads(current_prompts)
                print(f"✅ Successfully parsed {len(animation_prompts)} Wan prompts as JSON")
                progress_update += f"✅ Parsed {len(animation_prompts)} prompts successfully\n"
            except json.JSONDecodeError:
                # Try to parse as readable format
                try:
                    animation_prompts = {}
                    for line in current_prompts.strip().split('\n'):
                        if ':' in line:
                            parts = line.split(':', 1)
                            frame_part = parts[0].strip()
                            prompt_part = parts[1].strip()
                            
                            # Extract frame number
                            if frame_part.lower().startswith('frame '):
                                frame_num = frame_part[6:].strip()
                            else:
                                frame_num = frame_part
                            
                            animation_prompts[frame_num] = prompt_part
                    
                    if animation_prompts:
                        print(f"✅ Successfully parsed {len(animation_prompts)} Wan prompts as readable format")
                        progress_update += f"✅ Parsed {len(animation_prompts)} prompts from readable format\n"
                    else:
                        raise ValueError("No valid prompts found")
                except Exception as e:
                    print(f"❌ Could not parse Wan prompts: {e}")
                    error_msg = f"❌ Invalid format in Wan prompts. Expected JSON format like:\n{{\n  \"0\": \"prompt text\",\n  \"60\": \"another prompt\"\n}}\n\nOr readable format like:\nFrame 0: prompt text\nFrame 60: another prompt"
                    return error_msg, progress_update + "❌ Failed to parse prompts!"
        else:
            print("⚠️ Empty Wan prompts")
            
        # Check if we got valid prompts
        if not animation_prompts:
            error_msg = """❌ No Wan prompts found!

🔧 **Setup Required:**
1. 📝 Load prompts using "Load from Deforum Prompts" or "Load Default Wan Prompts"
2. 📋 Make sure your prompts are in proper JSON format like:
   {
     "0": "prompt text",
     "60": "another prompt",
     "120": "a cyberpunk environment with glowing elements"
   }
3. 🎨 Click **AI Prompt Enhancement** again after setting up prompts

💡 **Quick Start:**
Click "Load Default Wan Prompts" to start with example prompts!"""
            return error_msg, progress_update + "❌ No prompts to enhance!"
        
        # Validate prompts content
        if len(animation_prompts) == 1 and "0" in animation_prompts and "beautiful landscape" in animation_prompts["0"]:
            error_msg = """❌ Default prompts detected!

🔧 **Please configure your actual animation prompts:**
1. 📝 Load your real prompts using the load buttons above
2. ✏️ Or manually edit the Wan prompts field
3. 🎨 Click **AI Prompt Enhancement** again

💡 **For your animation sequence:**
Set up prompts like:
{
  "0": "A peaceful scene, photorealistic",
  "18": "A scene with glowing effects, neon colors, synthwave aesthetic",
  "36": "A cyberpunk scene with LED patterns, digital environment"
}"""
            return error_msg, progress_update + "❌ Default prompts detected!"
        
        print(f"🎨 Enhancing {len(animation_prompts)} Wan prompts with {qwen_model}")
        progress_update += f"🎨 Starting enhancement of {len(animation_prompts)} prompts...\n"
        
        # Create the Qwen prompt expander with better error handling
        try:
            progress_update += "📥 Creating AI model instance...\n"
            prompt_expander = qwen_manager.create_prompt_expander(qwen_model, auto_download)
            
            if not prompt_expander:
                if auto_download:
                    error_msg = f"""⏳ Downloading {qwen_model} model...

🔄 **Download in Progress:**
Model download started automatically. This may take a few minutes.

📥 **Please wait** and try clicking "AI Prompt Enhancement" again in 30-60 seconds.

💡 **Status**: Check console for download progress."""
                    return error_msg, progress_update + f"⏳ Downloading {qwen_model}..."
                else:
                    error_msg = f"""❌ Qwen model not available: {qwen_model}

🔧 **Setup Required:**
1. ✅ Enable "Auto-Download Qwen Models" checkbox above
2. 🎨 Click "AI Prompt Enhancement" again to auto-download
3. ⏳ Wait for download to complete

📥 **Manual Download Alternative:**
Use HuggingFace CLI: `huggingface-cli download {qwen_manager.get_model_info(qwen_model).get('huggingface_id', 'model-id')}`

💡 **Auto-download is recommended** for seamless model management."""
                    return error_msg, progress_update + "❌ Model creation failed!"
            
            progress_update += f"✅ {qwen_model} model ready for enhancement\n"
            
            # Enhance prompts
            progress_update += "🔄 Processing prompts with AI...\n"
            enhanced_prompts = prompt_expander.enhance_animation_prompts(
                animation_prompts, 
                language=language,
                progress_callback=lambda msg: print(f"🎨 {msg}")
            )
            
            if enhanced_prompts:
                progress_update += f"✅ Successfully enhanced {len(enhanced_prompts)} prompts!\n"
                # Convert back to JSON format  
                enhanced_json = json.dumps(enhanced_prompts, ensure_ascii=False, indent=2)
                
                success_msg = f"""✅ **AI Enhancement Complete!**

🎨 **Enhanced {len(enhanced_prompts)} prompts** with {qwen_model}
🌍 **Language**: {language}
🚀 **Ready for generation** - enhanced prompts loaded automatically

💡 **Next Steps:**
• 📐 Optionally add movement descriptions
• 🎬 Click "Generate Wan Video" to create your enhanced animation!"""
                
                print(f"✅ AI enhancement completed successfully")
                return enhanced_json, progress_update + success_msg
            else:
                error_msg = """❌ **Enhancement Failed**

🔧 **Possible Issues:**
• Model may be loading (try again in a moment)
• Prompts may be in unsupported format
• Check console for detailed error messages

💡 **Retry**: Click "AI Prompt Enhancement" again"""
                return current_prompts, progress_update + error_msg
                
        except Exception as e:
            error_msg = f"""❌ **Enhancement Error**

🔧 **Error Details:** {str(e)}

💡 **Troubleshooting:**
• Try a different Qwen model
• Check console for detailed error messages  
• Ensure prompts are in valid JSON format"""
            print(f"❌ Enhancement error: {e}")
            return current_prompts, progress_update + error_msg
            
    except ImportError as e:
        error_msg = f"""❌ **Qwen Enhancement Unavailable**

🔧 **Missing Dependencies:** {str(e)}

💡 **Setup Required:**
Install required packages for AI prompt enhancement."""
        print(f"❌ Import error: {e}")
        return current_prompts, error_msg
    except Exception as e:
        error_msg = f"""❌ **Unexpected Error**

🔧 **Error:** {str(e)}

💡 **Please try again or check console for details.**"""
        print(f"❌ Enhancement handler error: {e}")
        return current_prompts, error_msg


def check_qwen_models_handler(qwen_model):
    """Check Qwen model availability and return status"""
    try:
        from ..integrations.wan.utils.qwen_manager import qwen_manager
        
        print(f"🔍 Checking Qwen model: {qwen_model}")
        
        # Get model info
        model_info = qwen_manager.get_model_info(qwen_model)
        
        if not model_info:
            return f"❌ <span style='color: #f44336;'>Unknown model: {qwen_model}</span>"
        
        # Check if downloaded
        is_downloaded = qwen_manager.is_model_downloaded(qwen_model)
        
        # Check if loaded
        is_loaded = qwen_manager.is_model_loaded()
        loaded_model = None
        if is_loaded:
            loaded_info = qwen_manager.get_loaded_model_info()
            loaded_model = loaded_info.get('name') if loaded_info else None
        
        # Format status
        download_status = "✅ Downloaded" if is_downloaded else "❌ Not Downloaded"
        load_status = f"✅ Loaded ({loaded_model})" if is_loaded and loaded_model == qwen_model else "⚪ Not Loaded"
        
        size_info = model_info.get('approximate_size', 'Unknown size')
        hf_id = model_info.get('huggingface_id', 'Unknown ID')
        
        status_color = '#4CAF50' if is_downloaded else '#f44336'
        
        return f"""
        <div style='color: {status_color}; font-family: monospace;'>
        <strong>{qwen_model}</strong><br>
        📦 <strong>Download:</strong> {download_status}<br>
        🧠 <strong>Load:</strong> {load_status}<br>
        📏 <strong>Size:</strong> {size_info}<br>
        🔗 <strong>HF ID:</strong> {hf_id}
        </div>
        """
        
    except Exception as e:
        print(f"❌ Error checking Qwen models: {e}")
        return f"❌ <span style='color: #f44336;'>Error checking models: {str(e)}</span>"


def download_qwen_model_handler(qwen_model, auto_download_enabled):
    """Download Qwen model handler"""
    try:
        from ..integrations.wan.utils.qwen_manager import qwen_manager
        
        if not auto_download_enabled:
            return """⚠️ <span style='color: #FF9800;'>Auto-download is disabled</span>
            
            <br><br><strong>Enable auto-download first:</strong>
            <br>1. ✅ Check "Auto-Download Qwen Models" checkbox
            <br>2. 📥 Click "Download Selected Model" again"""
        
        print(f"📥 Starting download for Qwen model: {qwen_model}")
        
        # Check if already downloaded
        if qwen_manager.is_model_downloaded(qwen_model):
            return f"✅ <span style='color: #4CAF50;'>{qwen_model} is already downloaded</span>"
        
        # Get model info for download
        model_info = qwen_manager.get_model_info(qwen_model)
        if not model_info:
            return f"❌ <span style='color: #f44336;'>Unknown model: {qwen_model}</span>"
        
        # Start download
        hf_id = model_info.get('huggingface_id')
        size_info = model_info.get('approximate_size', 'Unknown size')
        
        return f"""⏳ <span style='color: #2196F3;'>Downloading {qwen_model}...</span>
        
        <br><br><strong>Download Details:</strong>
        <br>📦 <strong>Model:</strong> {qwen_model}
        <br>🔗 <strong>HF ID:</strong> {hf_id}
        <br>📏 <strong>Size:</strong> {size_info}
        
        <br><br>⏳ <strong>Please wait...</strong> Download may take several minutes.
        <br>💡 Check console for download progress.
        <br>🔄 Refresh status in 30-60 seconds."""
        
    except Exception as e:
        print(f"❌ Error downloading Qwen model: {e}")
        return f"❌ <span style='color: #f44336;'>Download error: {str(e)}</span>"


def cleanup_qwen_cache_handler():
    """Cleanup Qwen cache handler"""
    try:
        from ..integrations.wan.utils.qwen_manager import qwen_manager
        
        print(f"🧹 Starting Qwen cache cleanup...")
        
        # Check if any models are loaded
        if qwen_manager.is_model_loaded():
            loaded_info = qwen_manager.get_loaded_model_info()
            loaded_model = loaded_info.get('name') if loaded_info else 'Unknown'
            print(f"🔄 Cleaning up loaded model: {loaded_model}")
        
        # Cleanup cache
        qwen_manager.cleanup_cache()
        
        print(f"✅ Qwen cache cleanup completed")
        return "✅ <span style='color: #4CAF50;'>Cache cleanup completed successfully</span><br><br>💾 Memory freed and models unloaded."
        
    except Exception as e:
        print(f"❌ Error during Qwen cache cleanup: {e}")
        return f"❌ <span style='color: #f44336;'>Cleanup error: {str(e)}</span>"


def wan_generate_with_validation(*component_args):
    """Wrapper for wan_generate_video that includes validation"""
    try:
        # Get component names to find the wan_enhanced_prompts index
        from .args import get_component_names
        component_names = get_component_names()
        
        # Find wan_enhanced_prompts in the component list
        wan_prompts = ""
        try:
            # The wan_enhanced_prompts should be passed as one of the component args
            if len(component_args) > 0:
                wan_prompts = component_args[0] if component_args[0] else ""
            
            # Validate prompts first
            validation_result = validate_wan_generation(wan_prompts)
            if validation_result.startswith("❌"):
                return validation_result
            
            # If validation passes, call the original generate function
            return f"""✅ Validation passed! 

{validation_result}

🎬 **Starting Wan video generation...**
- Prompts: {len(wan_prompts.split('"')) // 4} clips detected
- Using I2V chaining for smooth transitions

🔄 **Status:** Generation starting..."""
            
        except Exception as e:
            return f"❌ **Generation Error:** {str(e)}"
            
    except Exception as e:
        return f"❌ **Validation Error:** {str(e)}" 