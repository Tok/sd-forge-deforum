# Wan 2.1 Video Generation Integration

## Overview

This integration provides seamless Wan 2.1 video generation within Deforum, featuring:

- **🔍 Auto-Discovery**: Automatically finds and loads Wan models from common locations
- **📥 Auto-Download**: Downloads missing models from HuggingFace automatically  
- **🔗 Deforum Integration**: Uses Deforum's prompt scheduling, FPS, and seed systems
- **🎬 I2V Chaining**: Seamless transitions between clips using Image-to-Video
- **⚡ Flash Attention Compatibility**: Automatic fallback when Flash Attention unavailable
- **💪 Strength Scheduling**: Control continuity vs creativity with Deforum schedules
- **🎯 VACE Support**: All-in-one T2V+I2V models for perfect consistency

## VACE Models - Recommended Architecture ✨

**VACE (Video Adaptive Conditional Enhancement)** represents Wan's latest unified architecture that handles both Text-to-Video and Image-to-Video generation with a single model:

### 🔄 **Unified Architecture Benefits**
- **Single Model**: One model handles both T2V and I2V generation
- **Perfect Consistency**: Same weights ensure visual continuity between clips
- **Memory Efficient**: No need to load separate T2V and I2V models
- **Enhanced Quality**: Latest architecture with improved video generation

### 🎯 **VACE T2V Mode**
VACE models can generate pure text-to-video content by using blank frame transformation:
- Creates dummy blank frames as input
- Transforms them based on text prompts
- Produces high-quality T2V output with I2V architecture benefits
- Maintains consistency for later I2V chaining

### 📊 **VACE vs T2V Model Comparison**
| Feature | VACE Models | T2V Models | I2V Models |
|---------|-------------|------------|------------|
| **T2V Generation** | ✅ Via blank frame transformation | ✅ Native | ❌ Not supported |
| **I2V Generation** | ✅ Native | ❌ Not supported | ✅ Native |
| **I2V Chaining** | ✅ Perfect continuity | ❌ No continuity | ✅ Good continuity |
| **Memory Usage** | ⚡ Single model load | 🔄 T2V model only | 🔄 I2V model only |
| **Consistency** | 🎯 Perfect (same weights) | ⚠️ Variable (different models) | ⚠️ I2V only (no T2V) |
| **Status** | 🆕 Latest | 🔄 Legacy | 🔄 Legacy |

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

3. **Choose Model** (Wan Video tab): Select VACE model for best results

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
# VACE 1.3B model (recommended) - ~17GB - All-in-one T2V+I2V
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/wan

# VACE 14B model (high quality) - ~75GB - All-in-one T2V+I2V
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan

# T2V 1.3B model (T2V only) - ~17GB - No I2V chaining
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan

# T2V 14B model (T2V only) - ~75GB - No I2V chaining  
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir models/wan

# I2V 1.3B model (I2V only) - ~17GB - Legacy I2V chaining
huggingface-cli download Wan-AI/Wan2.1-I2V-1.3B --local-dir models/wan

# I2V 14B model (I2V only) - ~75GB - Legacy I2V chaining
huggingface-cli download Wan-AI/Wan2.1-I2V-14B --local-dir models/wan
```

### Model Selection Options
- **Auto-Detect**: Finds best available model automatically (prefers VACE)
- **VACE 1.3B**: All-in-one T2V+I2V, faster, lower VRAM usage ⭐
- **VACE 14B**: All-in-one T2V+I2V, higher quality, more VRAM required
- **T2V 1.3B**: T2V only, no I2V chaining capability
- **T2V 14B**: T2V only, no I2V chaining capability
- **I2V 1.3B**: I2V only, legacy model for I2V chaining
- **I2V 14B**: I2V only, legacy model for I2V chaining
- **Custom Path**: Use your own model directory

**VACE Model Impact:**
- **Perfect Continuity**: Uses same model for both T2V and I2V, ensuring visual consistency
- **Seamless Transitions**: Last frame from previous clip becomes input for next clip
- **Strength Scheduling**: Respects Deforum strength schedules for continuity control
- **T2V Mode**: Can generate pure T2V content using blank frame transformation

**T2V Model Impact:**
- **Independent Generation**: Each clip generated independently from text only
- **No Continuity**: Maximum creative freedom but abrupt transitions between clips
- **Memory Efficient**: Smaller model footprint for T2V-only workflows

**I2V Model Impact (Legacy):**
- **I2V Only**: Can only perform image-to-video generation, no T2V capability
- **Good Continuity**: Designed for I2V chaining but requires separate T2V model for initial frame
- **Legacy Compatibility**: Useful for existing workflows that rely on separate T2V+I2V model pairs
- **Two-Stage Process**: Requires T2V model to generate first frame, then I2V for chaining

**💡 Recommendation**: Use VACE models for I2V chaining workflows, T2V models only for independent clip generation.

## Deforum Integration

### **Legacy T2V + I2V Workflow (Pre-VACE)**

Before VACE models, Wan 2.1 used separate models for text-to-video and image-to-video generation. This approach is still supported for compatibility:

#### **When to Use Legacy Models:**
- **Existing Workflows**: You have working setups with T2V + I2V model pairs
- **Specialized Use Cases**: Need pure T2V or pure I2V generation only
- **Resource Constraints**: Want to load only T2V or I2V models to save VRAM
- **Compatibility**: Working with older scripts that expect separate models

#### **Legacy Workflow Process:**
1. **T2V Model**: Generates the first frame/clip from text prompt
2. **I2V Model**: Takes the last frame from T2V output and generates the next clip
3. **Chaining**: Repeats I2V process for each subsequent prompt
4. **Result**: Video with I2V transitions, but requires two separate model loads

#### **Legacy vs VACE Comparison:**
| Aspect | Legacy (T2V + I2V) | VACE |
|--------|-------------------|------|
| **Model Count** | 2 separate models | 1 unified model |
| **Memory Usage** | Need both T2V + I2V loaded | Single model load |
| **Setup Complexity** | More complex (two models) | Simple (one model) |
| **Consistency** | Different model weights | Same weights throughout |
| **Compatibility** | Older workflows | Latest approach |

**💡 Migration Tip**: If you're using legacy T2V + I2V workflows, consider switching to VACE models for better consistency and simpler setup.

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

### Frame-Specific Movement Analysis ✨ NEW
- **Varied Descriptions**: Each prompt gets unique movement analysis based on its frame position
- **Directional Specificity**: Specific descriptions like "panning left", "tilting down", "dolly forward"
- **Camera Shakify Integration**: Analyzes actual shake patterns at each frame offset
- **Intensity & Duration**: Describes movement as "subtle/gentle/moderate" and "brief/extended/sustained"
- **No More Generic Text**: Eliminates identical "investigative handheld camera movement" across all prompts

**Example Results**:
- Frame 0: "camera movement with subtle panning left (sustained) and gentle moving down (extended)"
- Frame 43: "camera movement with moderate panning right (brief) and subtle rotating left (sustained)"
- Frame 106: "camera movement with gentle dolly forward (extended) and subtle rolling clockwise (brief)"

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