# SD-Forge Deforum Extension

A comprehensive Deforum extension for Stable Diffusion WebUI Forge, featuring advanced animation capabilities and WAN 2.1 video generation with precise audio synchronization.

## Features

- **Traditional Deforum Animation**: Advanced keyframe-based image animation with motion, depth warping, and coherence controls
- **WAN 2.1 Video Generation**: State-of-the-art AI video generation using Alibaba WAN models with **precise frame timing for audio sync**
- **Prompt Scheduling**: Complex prompt interpolation and transitions with exact frame-based timing
- **Motion Control**: Advanced camera and motion effects including 3D transformations
- **Multiple Output Formats**: Support for various video formats and resolutions
- **Audio Synchronization**: Frame-perfect timing for music video generation and audio-visual projects

## WAN 2.1 Video Generation ✨

### 🎯 **NEW: Precision Frame Timing**
- **Audio Sync Perfect**: Frame counts calculated exactly from prompt schedule (e.g., prompt at frame 0 with next at frame 12 = exactly 12 frames)
- **No Duration Guessing**: Uses actual frame differences between prompts for precise timing
- **Music Video Ready**: Perfect for synchronized audio-visual content

### 🚀 **Auto-Discovery System**
- **Zero Configuration**: Automatically finds WAN models in common locations
- **Smart Detection**: Scans `models/wan/`, `models/WAN/`, HuggingFace cache, Downloads folder
- **Model Validation**: Ensures all required files are present before generation

### 📊 **Model Options**
| Model | Size | VRAM | Speed | Quality | Best For |
|-------|------|------|--------|---------|----------|
| **T2V-1.3B** | ~17GB | 8GB+ | Fast | Good | Testing, Most Users ⭐ |
| **T2V-14B** | ~75GB | 16GB+ | Slow | Excellent | High-end Systems |

## Installation

1. Clone this repository into your `webui/extensions` directory:
```bash
cd webui/extensions
git clone https://github.com/deforum-art/sd-forge-deforum
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Restart WebUI Forge

## WAN Setup (Super Easy!)

### 🎯 Quick Start (Recommended)
```bash
# Install HuggingFace CLI (if not installed)
pip install huggingface_hub

# Download 1.3B model (recommended for testing)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan
```

### 🏆 High Quality Option
```bash
# Download 14B model (requires 16GB+ VRAM)
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan
```

### ✅ That's It!
The extension will automatically discover your model. No manual path configuration needed!

## Usage

### Traditional Deforum Animation
1. Go to the **Deforum** tab in WebUI Forge
2. Set up your prompts in the **Prompts** tab with frame timing
3. Configure motion parameters in **Keyframes** tab
4. Click **Generate** to create your animation

### WAN Video Generation
1. Go to the **WAN Video** tab
2. Enable WAN and select your preferred model size
3. Set up **frame-based prompts** in the Prompts tab:
   ```json
   {
     "0": "A serene landscape at dawn",
     "30": "The same landscape with morning sun",
     "60": "Golden hour lighting on the landscape",
     "90": "Sunset colors filling the sky"
   }
   ```
4. Click **Generate WAN Video** for precise frame-timed generation

### 🎵 **Audio Sync Example**
For a 60 FPS video with music:
```json
{
  "0": "Beat drop - explosive energy",
  "240": "Verse starts - calm flowing motion", 
  "480": "Chorus hits - dynamic camera movement",
  "720": "Bridge section - intimate close-up"
}
```
This creates exactly 240 frames (4 seconds), 240 frames, 240 frames, ensuring perfect sync with your audio track.

## Key Features

### 🎯 **Precision Timing**
- **Frame-Perfect**: Uses exact frame differences from prompt schedule
- **Audio Sync**: No approximations - each clip has precisely calculated frame count
- **Music Video Ready**: Perfect for beat-matched content

### 🔍 **Smart Discovery** 
- **Zero Config**: Models found automatically in common locations
- **Multi-Location**: Scans multiple directories and caches
- **Validation**: Ensures all required model files present

### 🚀 **Performance Optimized**
- **Memory Management**: Fixed imageio memory limits for large videos
- **Efficient Generation**: Smart frame calculation prevents excessive clip count
- **Error Recovery**: Robust fallback systems and clear error messages

### 🛠️ **Developer Friendly**
- **Organized Codebase**: All WAN code in `scripts/deforum_helpers/wan/` structure
- **Clean APIs**: Unified integration with clear interfaces
- **Extensible**: Easy to add new pipeline types and model formats

## System Requirements

### Minimum (1.3B Model)
- **GPU**: 8GB+ VRAM
- **Storage**: 20GB free space  
- **RAM**: 16GB+ system RAM

### Recommended (14B Model)
- **GPU**: 16GB+ VRAM
- **Storage**: 80GB free space
- **RAM**: 32GB+ system RAM

## Troubleshooting

### WAN Issues
- **No models found**: Run download commands above
- **Generation fails**: Try 1.3B model if using 14B
- **Out of memory**: Reduce resolution or inference steps
- **Audio sync off**: Check frame numbers in prompt schedule

### General Issues
- **Import errors**: Restart WebUI after installation
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Performance issues**: Check VRAM usage and reduce settings

## File Structure

```
scripts/deforum_helpers/wan/
├── configs/          # Model configurations
├── models/           # Model components  
├── utils/            # Utility functions
├── pipelines/        # Pipeline implementations
└── integration/      # Integration layer
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

- **Deforum Team**: Original animation framework
- **Alibaba WAN Team**: WAN 2.1 video generation models
- **WebUI Forge**: Advanced Stable Diffusion interface
- **Community Contributors**: Continuous improvements and bug fixes

## Support

- **Documentation**: Check the built-in help in each tab
- **Issues**: Report bugs via GitHub Issues
- **Community**: Join discussions in WebUI communities
- **Updates**: Watch repository for latest improvements

---

**🎬 Ready to create frame-perfect AI videos with audio sync? Download a model and start generating!**