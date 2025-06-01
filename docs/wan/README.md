# WAN (Wan Video) - Advanced Video Generation

WAN (Wan Video) is an advanced video generation system integrated into Deforum that provides sophisticated AI-powered video creation capabilities with intelligent prompt enhancement and movement analysis.

## 🎯 Overview

WAN extends Deforum's animation capabilities with:

- **Advanced AI Models**: Support for multiple T2V and I2V models
- **Intelligent Prompt Enhancement**: AI-powered prompt refinement using Qwen models
- **Movement Analysis**: Automatic detection and description of camera movements
- **Seamless Integration**: Native integration with Deforum's animation pipeline

## 🚀 Key Features

### Text-to-Video (T2V) Generation
- Multiple model sizes (1.3B VACE, 600M, etc.)
- Optimized for different hardware configurations
- Advanced prompt processing and enhancement

### Image-to-Video (I2V) Chaining
- Seamless conversion from static images to video
- Maintains temporal consistency across frames
- Support for complex animation sequences

### Prompt Enhancement
- **Qwen Integration**: Advanced language model for prompt refinement
- **Multi-language Support**: English and Chinese language processing
- **Style Prompting**: Integrated style enhancement capabilities
- **Movement Descriptions**: Automatic camera movement descriptions

### Movement Analysis
- **Automatic Detection**: Analyzes animation parameters for movement patterns
- **Intelligent Descriptions**: Generates natural language descriptions of camera motion
- **Sensitivity Control**: Adjustable analysis sensitivity for different animation styles
- **Pattern Recognition**: Identifies complex movement combinations

## 🛠 Configuration

### Basic Setup
```python
wan_mode: str = "I2V Chaining"  # "Disabled", "T2V Only", "I2V Chaining"
wan_model_name: str = "Auto-Select"
wan_enable_prompt_enhancement: bool = True
wan_enable_movement_analysis: bool = True
```

### Advanced Settings
```python
wan_movement_sensitivity: float = 1.0     # Movement detection sensitivity
wan_style_strength: float = 0.5           # Style prompt influence
wan_guidance_scale: float = 7.5           # Generation guidance strength
wan_frame_overlap: int = 2                # Frame overlap for consistency
```

## 📋 Supported Models

### T2V Models
- **1.3B VACE (Recommended)**: Best quality, requires more VRAM
- **600M**: Balanced performance and resource usage
- **300M**: Lightweight option for limited hardware

### Prompt Enhancement Models
- **Qwen-2.5-0.5B**: Compact model for basic enhancement
- **Qwen-2.5-1.5B**: Balanced model (recommended)
- **Qwen-2.5-3B**: High-quality enhancement for powerful systems

## 🎮 Usage Examples

### Basic T2V Generation
```python
animation_config = {
    'wan_mode': 'T2V Only',
    'wan_model_name': '1.3B VACE',
    'wan_enable_prompt_enhancement': True,
    'prompts': {
        '0': 'A serene mountain landscape at sunset'
    }
}
```

### Advanced I2V Chaining
```python
animation_config = {
    'wan_mode': 'I2V Chaining',
    'wan_i2v_strength': 0.8,
    'wan_enable_movement_analysis': True,
    'wan_movement_sensitivity': 1.2,
    'animation_mode': '3D',
    'translation_z': '0: (0), 30: (-5), 60: (0)'
}
```

## 🔧 Integration with Deforum

WAN integrates seamlessly with Deforum's existing animation system:

1. **Animation Parameters**: Automatically analyzes Deforum's keyframe schedules
2. **Movement Detection**: Identifies camera movements from 3D parameters
3. **Prompt Enhancement**: Enhances prompts based on detected movement patterns
4. **Video Generation**: Produces high-quality video output with temporal consistency

## 📊 Performance Considerations

### VRAM Requirements
- **1.3B VACE**: 8GB+ VRAM recommended
- **600M**: 6GB+ VRAM 
- **300M**: 4GB+ VRAM

### Generation Speed
- **T2V**: ~2-5 seconds per frame (depending on model)
- **I2V**: ~1-3 seconds per frame
- **Prompt Enhancement**: ~0.1-0.5 seconds per prompt

## 🐛 Troubleshooting

### Common Issues
1. **Out of Memory**: Try smaller models or reduce frame overlap
2. **Slow Generation**: Check model selection and hardware optimization
3. **Poor Quality**: Adjust guidance scale or enable prompt enhancement
4. **Movement Detection Issues**: Tune movement sensitivity

### Performance Tips
- Use "Auto-Select" for model choice on first run
- Enable flash attention for compatible hardware
- Adjust frame overlap based on scene complexity
- Use prompt enhancement for better results

## 📈 Development Status

- **🟢 Stable**: Basic T2V and I2V generation
- **🟡 Active Development**: Advanced movement analysis
- **🔵 Experimental**: Multi-language prompt enhancement
- **🟠 Planned**: Custom model training support

## 🔗 Related Documentation

- [Technical Details](technical.md) - Implementation details and architecture
- [Fixes & Updates](fixes.md) - Bug fixes and resolution notes  
- [Enhancements](enhancements.md) - Feature improvements and roadmap
- [Prompt Enhancement](../user-guide/prompt-enhancement.md) - User guide for prompt features

---

*WAN is actively developed and improved. Check the [enhancements](enhancements.md) for the latest features and improvements.* 