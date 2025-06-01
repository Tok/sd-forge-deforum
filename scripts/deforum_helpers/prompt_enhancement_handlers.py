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

"""
Enhanced Prompt Handlers for AI enhancement with style and theme controls.
Supports both Deforum and Wan prompts with consistency across video sequences.
"""

import json
import random


def random_style_handler():
    """Return a random style choice"""
    styles = [
        "Photorealistic", "Cinematic", "Anime/Manga", "Oil Painting", "Watercolor",
        "Digital Art", "3D Render", "Sketch/Drawing", "Vintage Film", "Studio Photography",
        "Street Photography", "Fine Art", "Impressionist", "Pop Art", "Minimalist"
    ]
    return random.choice(styles)


def random_theme_handler():
    """Return a random theme choice"""
    themes = [
        "Cyberpunk", "Synthwave/Vaporwave", "Frutiger Aero", "Steampunk", "Post-Apocalyptic",
        "Nature/Organic", "Urban/Metropolitan", "Retro-Futuristic", "Noir/Moody", "Ethereal/Dreamy",
        "Industrial/Brutalist", "Art Deco", "Bauhaus/Minimal", "Cosmic/Space", "Medieval Fantasy",
        "Tropical Paradise", "Winter Wonderland", "Desert Mystique", "Underwater World"
    ]
    return random.choice(themes)


def random_both_handler():
    """Return random style and theme"""
    return random_style_handler(), random_theme_handler()


def reset_to_photorealistic_handler():
    """Reset to photorealistic style and no theme"""
    return "Photorealistic", "None", "", ""  # style, theme, custom_style, custom_theme


def cycle_creative_themes_handler():
    """Cycle through creative themes while keeping current style"""
    creative_themes = [
        "Cyberpunk", "Synthwave/Vaporwave", "Frutiger Aero", "Steampunk", 
        "Retro-Futuristic", "Ethereal/Dreamy", "Art Deco", "Cosmic/Space"
    ]
    return random.choice(creative_themes)


def build_style_theme_string(style_dropdown, theme_dropdown, custom_style, custom_theme):
    """Build style and theme string for prompt enhancement"""
    # Use custom inputs if provided, otherwise use dropdown values
    final_style = custom_style.strip() if custom_style.strip() else style_dropdown
    final_theme = custom_theme.strip() if custom_theme.strip() else theme_dropdown
    
    # Build style string
    style_parts = []
    if final_style and final_style != "None":
        style_parts.append(f"in {final_style.lower()} style")
    
    if final_theme and final_theme != "None":
        style_parts.append(f"with {final_theme.lower()} aesthetic")
    
    return ", ".join(style_parts)


def enhance_deforum_prompts_handler(animation_prompts, style_dropdown, theme_dropdown, custom_style, custom_theme, qwen_model, language, auto_download):
    """Enhance Deforum animation prompts with AI and apply style/theme"""
    try:
        from .wan.utils.qwen_manager import qwen_manager
        
        print(f"🎨 Enhancing Deforum prompts with style: {style_dropdown}, theme: {theme_dropdown}")
        
        # Build style/theme string
        style_theme_str = build_style_theme_string(style_dropdown, theme_dropdown, custom_style, custom_theme)
        print(f"🎭 Style/theme string: {style_theme_str}")
        
        # Parse animation_prompts
        if not animation_prompts or not animation_prompts.strip():
            return animation_prompts, "❌ No Deforum prompts found! Please add prompts in the Main Prompts tab first.", "❌ No prompts to enhance"
        
        try:
            prompts_dict = json.loads(animation_prompts)
        except json.JSONDecodeError:
            return animation_prompts, "❌ Invalid JSON format in Deforum prompts! Please check the Main Prompts tab.", "❌ JSON parse error"
        
        if not prompts_dict:
            return animation_prompts, "❌ Empty prompts dictionary in Deforum prompts!", "❌ Empty prompts"
        
        # Enhance each prompt with AI
        enhanced_prompts = {}
        progress_messages = []
        
        for frame_key, prompt in prompts_dict.items():
            # Add style/theme to prompt
            enhanced_prompt = prompt
            if style_theme_str:
                enhanced_prompt = f"{prompt}, {style_theme_str}"
            
            # Use AI enhancement if model is available
            try:
                if qwen_manager.is_model_downloaded(qwen_model) or auto_download:
                    ai_enhanced = qwen_manager.enhance_prompts(
                        prompts={frame_key: enhanced_prompt},
                        model_name=qwen_model,
                        language=language,
                        auto_download=auto_download
                    )
                    enhanced_prompts[frame_key] = ai_enhanced.get(frame_key, enhanced_prompt)
                    progress_messages.append(f"✅ Enhanced frame {frame_key}")
                else:
                    # Just apply style/theme without AI enhancement
                    enhanced_prompts[frame_key] = enhanced_prompt
                    progress_messages.append(f"🎨 Applied style to frame {frame_key}")
            except Exception as e:
                print(f"⚠️ AI enhancement failed for frame {frame_key}: {e}")
                enhanced_prompts[frame_key] = enhanced_prompt
                progress_messages.append(f"⚠️ Style only for frame {frame_key}")
        
        # Format as JSON
        enhanced_json = json.dumps(enhanced_prompts, ensure_ascii=False, indent=2)
        
        status_message = f"""✅ Enhanced {len(enhanced_prompts)} Deforum prompts!

🎨 Style: {style_dropdown}
🌍 Theme: {theme_dropdown}
{"🎭 Custom Style: " + custom_style if custom_style.strip() else ""}
{"🏛️ Custom Theme: " + custom_theme if custom_theme.strip() else ""}

The enhanced prompts have been applied to your Deforum animation prompts.
You can now generate your animation with improved, stylistically consistent descriptions!"""
        
        progress_message = "\n".join(progress_messages)
        
        return enhanced_json, status_message, progress_message
        
    except Exception as e:
        error_msg = f"❌ Error enhancing Deforum prompts: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return animation_prompts, error_msg, f"❌ Error: {str(e)}"


def enhance_wan_prompts_handler_with_style(wan_prompts, style_dropdown, theme_dropdown, custom_style, custom_theme, qwen_model, language, auto_download):
    """Enhance Wan prompts with AI and apply style/theme"""
    try:
        from .ui_elements import enhance_prompts_handler
        
        # This uses the existing enhance_prompts_handler but with added style/theme
        style_theme_str = build_style_theme_string(style_dropdown, theme_dropdown, custom_style, custom_theme)
        
        # First apply style/theme to wan prompts
        if style_theme_str and wan_prompts:
            try:
                prompts_dict = json.loads(wan_prompts)
                styled_prompts = {}
                for frame_key, prompt in prompts_dict.items():
                    styled_prompts[frame_key] = f"{prompt}, {style_theme_str}"
                styled_wan_prompts = json.dumps(styled_prompts, ensure_ascii=False, indent=2)
            except:
                styled_wan_prompts = wan_prompts
        else:
            styled_wan_prompts = wan_prompts
        
        # Use the existing enhancement handler
        enhanced_prompts, progress = enhance_prompts_handler(
            styled_wan_prompts, qwen_model, language, auto_download
        )
        
        # Add style/theme info to status
        style_info = f"""🎨 Style: {style_dropdown}
🌍 Theme: {theme_dropdown}
{"🎭 Custom Style: " + custom_style if custom_style.strip() else ""}
{"🏛️ Custom Theme: " + custom_theme if custom_theme.strip() else ""}

"""
        
        enhanced_status = style_info + enhanced_prompts
        
        return enhanced_prompts, enhanced_status, progress
        
    except Exception as e:
        error_msg = f"❌ Error enhancing Wan prompts: {str(e)}"
        print(error_msg)
        return wan_prompts, error_msg, f"❌ Error: {str(e)}"


def apply_style_only_handler(prompts, style_dropdown, theme_dropdown, custom_style, custom_theme, prompt_type="Deforum"):
    """Apply only style/theme to prompts without AI enhancement"""
    try:
        style_theme_str = build_style_theme_string(style_dropdown, theme_dropdown, custom_style, custom_theme)
        
        if not prompts or not prompts.strip():
            return prompts, f"❌ No {prompt_type} prompts found!"
        
        if not style_theme_str:
            return prompts, f"ℹ️ No style or theme selected - prompts unchanged"
        
        try:
            prompts_dict = json.loads(prompts)
        except json.JSONDecodeError:
            return prompts, f"❌ Invalid JSON format in {prompt_type} prompts!"
        
        styled_prompts = {}
        for frame_key, prompt in prompts_dict.items():
            styled_prompts[frame_key] = f"{prompt}, {style_theme_str}"
        
        styled_json = json.dumps(styled_prompts, ensure_ascii=False, indent=2)
        
        status_message = f"""✅ Applied style to {len(styled_prompts)} {prompt_type} prompts!

🎨 Style: {style_dropdown}
🌍 Theme: {theme_dropdown}
{"🎭 Custom Style: " + custom_style if custom_style.strip() else ""}
{"🏛️ Custom Theme: " + custom_theme if custom_theme.strip() else ""}

Style and theme have been added to all prompts for consistent visual appearance."""
        
        return styled_json, status_message
        
    except Exception as e:
        error_msg = f"❌ Error applying style to {prompt_type} prompts: {str(e)}"
        print(error_msg)
        return prompts, error_msg 