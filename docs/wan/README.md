# Wan 2.1 Video Generation Integration

## Overview

This integration provides seamless Wan 2.1 video generation within Deforum, featuring:

- **🔍 Auto-Discovery**: Automatically finds and loads Wan models from common locations
- **📥 Auto-Download**: Downloads missing models from HuggingFace automatically  
- **🔗 Deforum Integration**: Uses Deforum's prompt scheduling, FPS, and seed systems
- **🎬 I2V Chaining**: Seamless transitions between clips using Image-to-Video
- **⚡ Flash Attention Compatibility**: Automatic fallback when Flash Attention unavailable
- **💪 Strength Scheduling**: Control continuity vs creativity with Deforum schedules

## Quick Start

1. **Configure Prompts** (Prompts tab):
   ```json
   {
     "0": "a serene beach at sunset",
     "90": "a misty forest in the morning", 
     "180": "a bustling city street at night"
   }
   ```

2. **Set FPS** (Output tab): Choose your desired FPS (e.g., 30 or 60)

3. **Choose Model** (Wan Video tab): Select model size preference

4. **Generate**: Click "Generate Wan Video"

## Model Management

### Auto-Discovery
Models are automatically discovered from:
- `models/wan/`
- `models/wan/` 
- `models/Wan/`
- HuggingFace cache

### Auto-Download
Missing models are downloaded automatically when enabled:

```bash
# 1.3B model (recommended) - ~17GB
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan

# 14B model (high quality) - ~75GB  
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir models/wan
```

### Model Selection Options
- **Auto-Detect**: Finds best available model automatically
- **1.3B T2V/I2V**: Faster, lower VRAM usage
- **14B T2V/I2V**: Higher quality, more VRAM required
- **Use T2V Model (No Continuity)**: ⚠️ Uses T2V for I2V - equivalent to Deforum strength 0.0
- **Custom Path**: Use your own model directory

**I2V Model Impact:**
- **Continuity Options (Auto-Detect/I2V models)**: Uses last frame from previous clip as input, respects Deforum strength schedules, creates seamless transitions
- **No Continuity Option (T2V model)**: Generates each clip independently from text only, maximum creative freedom but abrupt transitions between clips

**Note**: The "Use T2V Model (No Continuity)" option gives the model complete creative freedom to interpret each prompt independently, similar to setting Deforum strength to 0.0. This results in more creative interpretations but breaks visual continuity between clips.

## Deforum Integration

### Settings Sources
Wan uses these settings from other Deforum tabs:

- **📝 Prompts**: From Prompts tab
- **🎬 FPS**: From Output tab (no separate Wan FPS needed)
- **🎲 Seed**: From Keyframes → Seed & SubSeed tab  
- **💪 Strength**: From Keyframes → Strength tab (for I2V chaining)
- **⏱️ Duration**: Auto-calculated from prompt timing

### Prompt Schedule Integration
- Each prompt with a frame number becomes a video clip
- Duration calculated from frame differences
- Example: `{"0": "beach", "120": "forest"}` creates two clips
- Frame 0→120 at 30fps = 4 second clips

### Strength Schedule Integration  
Control I2V chaining continuity:
- **Higher values (0.7-0.9)**: Strong continuity, smoother transitions
- **Lower values (0.3-0.6)**: More creative freedom, less continuity
- **Override option**: Use fixed strength for maximum continuity

### Seed Schedule Integration
- Uses Deforum's seed schedule from Keyframes → Seed & SubSeed
- Set **Seed behavior** to 'schedule' for custom seed scheduling
- Example: `0:(12345), 60:(67890)` uses different seeds per clip

## Advanced Features

### I2V Chaining
- Uses last frame of previous clip as input for next clip
- Ensures seamless transitions between prompt changes
- Supports Deforum strength scheduling for continuity control
- Enhanced parameters for maximum continuity

### Frame Management
- **Wan 4n+1 Requirement**: Automatically calculates proper frame counts
- **Frame Discarding**: Removes excess frames to match exact timing
- **PNG Output**: High-quality frame sequences

### Flash Attention Compatibility
- Automatic detection of Flash Attention availability
- Seamless fallback to PyTorch native attention
- No manual installation required

## Settings Reference

### Essential Settings
- **T2V Model**: Text-to-Video model selection
- **I2V Model**: Image-to-Video model for chaining (separate from T2V for continuity)
- **Auto-Download**: Automatic model downloading
- **Preferred Size**: 1.3B (recommended) or 14B (high quality)
- **Resolution**: 1280x720, 720x1280, 854x480, 480x854
- **Inference Steps**: 5-100 (5-15 for testing, 20-50 for quality)
- **Guidance Scale**: 1.0-20.0 (prompt adherence)

### Advanced Settings
- **Frame Overlap**: Overlapping frames between clips (0-10)
- **Motion Strength**: Strength of motion in videos (0.0-2.0)
- **Enable Interpolation**: Frame interpolation between clips
- **Interpolation Strength**: Strength of interpolation (0.0-1.0)

### Continuity Control
- **Strength Override**: Override Deforum schedules with fixed value
- **Fixed Strength**: 1.0 = maximum continuity, 0.0 = maximum creativity

## Troubleshooting

### Common Issues

**No models found:**
```bash
# Download recommended model
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan
```

**Flash Attention errors:**
- Automatic fallback is applied
- No manual installation required
- Check console for compatibility messages

**Generation fails:**
1. Verify prompts in Prompts tab
2. Check FPS setting in Output tab
3. Ensure models are downloaded
4. Check console for detailed errors

### Performance Tips
- Use 1.3B models for faster generation
- Lower inference steps (5-15) for testing
- Higher steps (30-50) for final quality
- Monitor VRAM usage with 14B models

## Technical Details

### Model Requirements
- **1.3B models**: ~8GB VRAM minimum
- **14B models**: ~24GB VRAM minimum
- **Storage**: 17GB (1.3B) or 75GB (14B) per model

### Frame Calculation
- Duration = (frame_difference / fps) seconds
- Automatic 4n+1 frame count calculation
- Frame discarding for exact timing preservation

### Integration Architecture
- Direct pipeline integration with Deforum
- Separate T2V and I2V model loading
- Enhanced I2V parameters for continuity
- Automatic compatibility patching 