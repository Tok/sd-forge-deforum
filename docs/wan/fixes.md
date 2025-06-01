# WAN Fixes & Updates

This document tracks bug fixes, updates, and resolutions for the WAN (Wan Video) system in Deforum.

## 🔧 Recent Fixes

### v2.1.3 - Movement Analysis Improvements
**Date**: Current Release
**Status**: ✅ Resolved

#### Issues Fixed:
- **Generic Movement Descriptions**: Fixed issue where all prompts received identical "investigative handheld camera movement" descriptions
- **Frame-Specific Analysis**: Implemented unique movement analysis for each frame position
- **Movement Pattern Recognition**: Enhanced detection of pan, tilt, dolly, and roll movements

#### Changes Made:
```python
# Before: Generic descriptions for all frames
def analyze_movement_generic(args):
    return "investigative handheld camera movement"

# After: Frame-specific analysis with detailed patterns
def analyze_movement_at_frame(animation_schedules, frame_offset, sensitivity):
    movements = detect_movement_patterns(animation_schedules, frame_offset)
    return generate_specific_descriptions(movements, sensitivity)
```

#### Results:
- **Unique Descriptions**: Each prompt now gets contextually relevant movement descriptions
- **Directional Accuracy**: Specific movements like "panning left", "tilting down", "dolly forward"
- **Intensity Classification**: Movements classified as "subtle", "gentle", or "moderate"
- **Duration Analysis**: Movements described as "brief", "extended", or "sustained"

### v2.1.2 - Memory Management Fixes
**Date**: Previous Release
**Status**: ✅ Resolved

#### Issues Fixed:
- **VRAM Leaks**: Fixed memory leaks during model switching
- **Flash Attention Fallback**: Improved fallback when Flash Attention unavailable
- **Model Loading Timeouts**: Fixed occasional model loading timeouts

#### Changes Made:
- Implemented proper cleanup in model switching
- Added graceful Flash Attention fallback with performance monitoring
- Increased model loading timeout and added retry mechanism

### v2.1.1 - I2V Chaining Stability
**Date**: Previous Release  
**Status**: ✅ Resolved

#### Issues Fixed:
- **Frame Transition Artifacts**: Fixed visual artifacts between I2V chained clips
- **Strength Schedule Integration**: Fixed issues with Deforum strength schedule parsing
- **VACE Model Support**: Enhanced support for VACE unified models

#### Changes Made:
- Improved frame overlap handling for smoother transitions
- Fixed strength schedule interpolation edge cases
- Added specialized VACE model loading with proper T2V/I2V mode switching

## 🐛 Known Issues

### High Priority Issues

#### Issue #001: Large Model Loading Times
**Status**: 🟡 In Progress
**Description**: 14B models take significant time to load on systems with limited VRAM
**Workaround**: Use 1.3B models for faster loading, or pre-load models
**ETA**: Next minor release

#### Issue #002: Resolution Scaling Edge Cases
**Status**: 🟡 In Progress  
**Description**: Some custom resolutions may not work optimally with certain models
**Workaround**: Use standard resolutions (864x480, 720x1280)
**ETA**: Next patch release

### Medium Priority Issues

#### Issue #003: Prompt Enhancement Language Detection
**Status**: 🔵 Planned
**Description**: Auto-detection of prompt language could be more accurate
**Workaround**: Manually specify language in WAN settings
**ETA**: Future release

#### Issue #004: Multi-GPU Support
**Status**: 🔵 Planned
**Description**: WAN currently uses single GPU, multi-GPU support planned
**Workaround**: Use single powerful GPU for now
**ETA**: Major release (v2.2)

## 🔄 Troubleshooting Guide

### Common Issues & Solutions

#### "Out of Memory" Errors
**Symptoms**: CUDA out of memory during generation
**Solutions**:
1. Switch to smaller model (1.3B instead of 14B)
2. Reduce frame overlap in advanced settings
3. Lower resolution setting
4. Close other GPU-intensive applications

```python
# Automatic memory optimization
wan_config = WanConfig(
    model_size="1.3B",  # Smaller model
    frame_overlap=1,    # Reduced overlap
    resolution="480x854"  # Lower resolution
)
```

#### "Model Not Found" Errors  
**Symptoms**: WAN can't locate installed models
**Solutions**:
1. Check model installation in `models/wan/` directory
2. Enable auto-download in WAN settings
3. Manually specify model path
4. Verify model directory structure

#### "Flash Attention" Warnings
**Symptoms**: Console warnings about Flash Attention
**Solutions**:
1. This is normal - automatic fallback is working
2. For better performance, install Flash Attention: `pip install flash-attn`
3. No action required - generation will continue with PyTorch native attention

#### Poor Generation Quality
**Symptoms**: Generated videos don't match prompts well
**Solutions**:
1. Enable prompt enhancement in WAN settings
2. Increase guidance scale (7.5 → 10.0)
3. Use more descriptive prompts
4. Enable movement analysis for better context

#### Slow Generation Speed
**Symptoms**: Generation takes much longer than expected
**Solutions**:
1. Reduce inference steps for testing (50 → 20)
2. Use 1.3B model instead of 14B
3. Enable Flash Attention if available
4. Check system VRAM usage

## 🔍 Debugging Tips

### Enable Debug Mode
```python
# Add to your configuration
debug_mode = True
verbose_logging = True

# Check console output for detailed information
```

### Monitor System Resources
- **VRAM Usage**: Monitor GPU memory during generation
- **System RAM**: Ensure sufficient system memory
- **Storage Space**: Verify adequate disk space for output
- **CPU Usage**: Check CPU utilization during model loading

### Log Analysis
Key log messages to watch for:
- `🔧 WAN Model Discovery: Found X models`
- `⚡ Flash Attention: Available/Not Available`
- `🎬 Generation Progress: X/Y frames complete`
- `💾 Memory Usage: X GB VRAM used`

### Performance Monitoring
```python
# Monitor generation performance
def monitor_wan_performance():
    start_time = time.time()
    memory_start = torch.cuda.memory_allocated()
    
    # ... generation code ...
    
    generation_time = time.time() - start_time
    memory_peak = torch.cuda.max_memory_allocated()
    
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Peak VRAM: {memory_peak / 1e9:.2f}GB")
```

## 📊 Performance Optimization

### Model Selection Guidelines
- **1.3B VACE**: Best balance of quality and speed (8GB+ VRAM)
- **14B VACE**: Highest quality (16GB+ VRAM)
- **Legacy Models**: Only if needed for specific workflows

### Resolution Recommendations
- **High Quality**: 1280x720 or 720x1280
- **Balanced**: 864x480 or 480x854  
- **Fast Testing**: 640x360 or 360x640

### Generation Settings
```python
# Optimized settings for different use cases

# Fast Testing
fast_settings = {
    'inference_steps': 15,
    'guidance_scale': 7.5,
    'frame_overlap': 1
}

# Balanced Quality
balanced_settings = {
    'inference_steps': 25,
    'guidance_scale': 8.0,
    'frame_overlap': 2
}

# High Quality
high_quality_settings = {
    'inference_steps': 50,
    'guidance_scale': 9.0,
    'frame_overlap': 3
}
```

## 🚀 Future Improvements

### Planned Fixes
- **Enhanced Memory Management**: Better VRAM optimization
- **Faster Model Loading**: Optimized model initialization
- **Improved Error Messages**: More helpful error descriptions
- **Better Progress Tracking**: Detailed generation progress

### Feature Enhancements
- **Real-time Preview**: Live generation preview
- **Batch Processing**: Multiple prompt batches
- **Custom Model Support**: User-trained model integration
- **Advanced Interpolation**: Neural frame interpolation

## 📝 Reporting Issues

### How to Report Bugs
1. **Check Known Issues**: Review this document first
2. **Gather Information**: Include system specs, model used, error messages
3. **Provide Steps**: Clear reproduction steps
4. **Include Logs**: Console output and error messages

### Information to Include
- **System**: OS, GPU, VRAM amount
- **Models**: Which WAN models installed/used
- **Settings**: WAN configuration used
- **Error Messages**: Complete error text
- **Steps**: What you were trying to do

### Where to Report
- **GitHub Issues**: Technical bugs and feature requests
- **Community Forums**: Usage questions and discussions
- **Documentation Issues**: Corrections and improvements

---

*This fixes documentation is actively maintained. Check the [enhancements](enhancements.md) for upcoming features and improvements.* 