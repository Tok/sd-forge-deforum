# Wan 2.1 Video Generation

Wan 2.1 is Alibaba's state-of-the-art text-to-video generation model, fully integrated with Deforum's scheduling system for precision video creation.

## Quick Start

### 1. Install Models

Choose one model based on your system capabilities:

```bash
# Recommended: 1.3B model (8GB+ VRAM)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan

# High Quality: 14B model (16GB+ VRAM) 
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan
```

### 2. Configure Prompts

Set up your prompt schedule in the **Prompts tab**:

```json
{
  "0": "A serene mountain landscape at dawn",
  "30": "Morning mist rising from the valleys", 
  "60": "Golden sunlight breaking through clouds",
  "90": "Full daylight illuminating the peaks"
}
```

### 3. Set FPS

Configure your desired frame rate in the **Output tab** (e.g., 30 FPS).

### 4. Generate

Go to the **Wan Video tab** and click **"Generate Wan Video"**.

## Features

### 🎯 Deforum Integration

- **Prompt Scheduling**: Uses Deforum's prompt system for precise timing
- **FPS Integration**: Single FPS setting controls everything
- **Seed Scheduling**: Optional seed control for consistency
- **Strength Scheduling**: I2V chaining with continuity control

### 🔍 Auto-Discovery

Models are automatically discovered from:
- `models/wan/`
- `models/WAN/`
- HuggingFace cache
- Downloads folder

### 🚀 Advanced Features

- **I2V Chaining**: Seamless transitions between clips
- **4n+1 Frame Calculation**: Automatic Wan frame requirement handling
- **PNG Frame Output**: Individual frames before video compilation
- **Flash Attention Fallback**: Works with or without flash-attn

## Model Comparison

| Model | Size | VRAM | Speed | Quality | Best For |
|-------|------|------|--------|---------|----------|
| **T2V-1.3B** | ~17GB | 8GB+ | Fast | Good | Most Users ⭐ |
| **T2V-14B** | ~75GB | 16GB+ | Slow | Excellent | High-end Systems |

## Settings Reference

### Essential Settings

- **Model Size**: Choose between 1.3B (recommended) or 14B (high quality)
- **Resolution**: Output video resolution (e.g., 1024x576, 1280x720)
- **Inference Steps**: 5-15 (quick test), 20-50 (quality), 50+ (high quality)
- **Guidance Scale**: 7.5 (default), higher for stronger prompt adherence

### Advanced Settings

- **Frame Overlap**: Overlapping frames between clips (default: 2)
- **Motion Strength**: Strength of motion in videos (default: 1.0)
- **Enable Interpolation**: Frame interpolation between clips
- **Interpolation Strength**: Strength of interpolation (default: 0.5)

## Scheduling Integration

### Prompt Schedule

Each prompt with a frame number becomes a video clip:

```json
{
  "0": "beach sunset",     // Clip 1: frames 0-59
  "60": "forest morning",  // Clip 2: frames 60-119  
  "120": "city night"      // Clip 3: frames 120+
}
```

Duration = (frame_difference / fps) seconds per clip.

### Seed Schedule

Control seeds for consistency (set Seed behavior to 'schedule'):

```
0:(12345), 60:(67890), 120:(54321)
```

### Strength Schedule

Control I2V continuity (Keyframes → Strength tab):

```
0:(0.85), 60:(0.7), 120:(0.5)
```

- Higher values (0.7-0.9): Strong continuity, smoother transitions
- Lower values (0.3-0.6): More creative freedom, less continuity

## Frame Calculation

Wan requires frame counts in 4n+1 format (5, 9, 13, 17, 21, etc.):

- **Requested**: 15 frames → **Generated**: 17 frames → **Discarded**: 2 frames
- **Requested**: 20 frames → **Generated**: 21 frames → **Discarded**: 1 frame
- **Requested**: 21 frames → **Generated**: 21 frames → **Discarded**: 0 frames

The system automatically calculates the nearest 4n+1 value and discards extra frames from the middle.

## Troubleshooting

### Common Issues

**No models found**
- Download models using the commands above
- Check that models are in `models/wan/` directory
- Restart WebUI after downloading

**Generation fails**
- Try the 1.3B model if using 14B
- Check VRAM usage (8GB+ required for 1.3B, 16GB+ for 14B)
- Verify prompts are configured in Prompts tab

**Flash attention errors**
- Compatibility layer should handle this automatically
- No manual flash-attn installation required

**Button not working**
- Restart WebUI completely
- Check console for error messages
- Verify prompts are configured

### Performance Tips

- Start with 1.3B model for testing
- Use lower inference steps (5-15) for quick tests
- Reduce resolution for faster generation
- Monitor VRAM usage during generation

## Technical Details

### I2V Chaining Process

1. **First Clip**: Generated using Text-to-Video (T2V)
2. **Subsequent Clips**: Use last frame of previous clip as starting image (I2V)
3. **Strength Control**: Deforum strength schedule controls influence
4. **Seamless Transitions**: Better continuity than pure T2V

### Compatibility

- **Flash Attention**: Optional, automatic fallback to PyTorch native
- **Original Wan Repo**: Uses compatibility layer without modifications
- **Error Handling**: Comprehensive error messages and guidance
- **Memory Management**: Efficient handling of large generations

### Output Structure

```
output_directory/
├── clip_0_frames/          # PNG frames for clip 1
├── clip_1_frames/          # PNG frames for clip 2  
├── clip_2_frames/          # PNG frames for clip 3
└── final_video.mp4         # Stitched final video
```

## Examples

### Music Video (60 FPS)

```json
{
  "0": "drummer on stage, dramatic lighting",
  "60": "guitar solo, crowd cheering",
  "120": "bass drop, strobing lights",
  "180": "finale, confetti falling"
}
```

Each clip = 1 second at 60 FPS.

### Nature Documentary (30 FPS)

```json
{
  "0": "sunrise over savanna, golden hour",
  "90": "elephants walking to watering hole", 
  "180": "lions resting under acacia tree",
  "270": "sunset, birds flying home"
}
```

Each clip = 3 seconds at 30 FPS.

### Abstract Art (24 FPS)

```json
{
  "0": "swirling colors, fluid motion",
  "48": "geometric patterns emerging",
  "96": "crystalline structures forming",
  "144": "dissolution into particles"
}
```

Each clip = 2 seconds at 24 FPS. 