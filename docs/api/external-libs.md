# External Libraries Integration

Documentation for external library integrations in Deforum.

## Overview

Deforum integrates with several external libraries to provide advanced functionality:
- **RIFE** - Real-time Intermediate Flow Estimation for frame interpolation
- **FILM** - Frame Interpolation for Large Motion for advanced interpolation
- **MiDaS** - Monocular depth estimation
- **ControlNet** - Image conditioning and control

## RIFE Integration

### Purpose
RIFE provides high-quality frame interpolation for smooth video generation.

### Configuration
```python
from deforum_helpers.data_models import ExternalLibraryArgs

# Create RIFE configuration
rife_config = ExternalLibraryArgs(
    modelDir="path/to/rife/model",
    fp16=False,
    UHD=False,
    scale=1.0,
    fps=30,
    interp_x_amount=2
)
```

### Usage
```python
from deforum_helpers.external_libs.rife.inference_video import run_rife_new_video_infer

# Run RIFE interpolation
result = run_rife_new_video_infer(
    output="output_path",
    model="rife_model_path",
    fps=30,
    interp_x_amount=2
)
```

### Features
- Real-time frame interpolation
- GPU acceleration support
- Multiple interpolation factors
- High-quality output

## FILM Integration

### Purpose
FILM provides advanced frame interpolation for large motion scenarios.

### Configuration
```python
# Create FILM configuration
film_config = ExternalLibraryArgs(
    model_path="path/to/film/model",
    input_folder="input_frames",
    save_folder="output_frames",
    inter_frames=4
)
```

### Usage
```python
from deforum_helpers.external_libs.film_interpolation.film_inference import run_film_interp_infer

# Run FILM interpolation
run_film_interp_infer(
    model_path="film_model.pkl",
    input_folder="input_frames/",
    save_folder="output_frames/",
    inter_frames=4
)
```

### Features
- Large motion handling
- High-quality interpolation
- Batch processing
- Memory efficient

## MiDaS Integration

### Purpose
MiDaS provides monocular depth estimation for 3D effects and depth-aware processing.

### Configuration
```python
# MiDaS is configured through the main animation args
depth_algorithm = "Depth-Anything-V2-Small"  # or "Midas-3-Hybrid"
```

### Features
- Multiple depth algorithms
- Real-time depth estimation
- Integration with 3D animation
- Depth map generation

## ControlNet Integration

### Purpose
ControlNet provides precise control over image generation using various conditioning inputs.

### Configuration
```python
# ControlNet configuration through UI or args
controlnet_args = {
    "controlnet_enabled": True,
    "controlnet_model": "control_v11p_sd15_canny",
    "controlnet_weight": 1.0,
    "controlnet_guidance_start": 0.0,
    "controlnet_guidance_end": 1.0
}
```

### Features
- Multiple control types (Canny, Depth, OpenPose, etc.)
- Weight and guidance control
- Multi-model support
- Real-time conditioning

## Integration Patterns

### Immutable Configuration
All external libraries use immutable configuration objects:

```python
@dataclass(frozen=True)
class ExternalLibraryArgs:
    """Immutable configuration for external tools"""
    # Common fields
    model_path: str = ""
    input_params: Dict[str, Any] = field(default_factory=dict)
    
    # Library-specific fields
    # ... (see data_models.py for complete definition)
```

### Error Handling
```python
try:
    result = run_external_library(config)
except ExternalLibraryError as e:
    # Handle library-specific errors
    logger.error(f"External library error: {e}")
    # Fallback or user notification
```

### Performance Optimization
- **Memory management**: Automatic cleanup after processing
- **GPU utilization**: Efficient GPU memory usage
- **Batch processing**: Process multiple frames efficiently
- **Caching**: Cache models and intermediate results

## Installation Requirements

### RIFE
- PyTorch with CUDA support
- OpenCV
- NumPy

### FILM
- TensorFlow or PyTorch
- OpenCV
- NumPy

### MiDaS
- PyTorch
- Transformers library
- OpenCV

### ControlNet
- Diffusers library
- Transformers
- PyTorch

## Troubleshooting

### Common Issues

#### Memory Errors
- Reduce batch size
- Use FP16 precision
- Clear GPU cache between operations

#### Model Loading Failures
- Check model file paths
- Verify model compatibility
- Ensure sufficient disk space

#### Performance Issues
- Use appropriate hardware acceleration
- Optimize batch sizes
- Monitor GPU memory usage

### Debug Mode
Enable debug logging for detailed information:
```python
import logging
logging.getLogger('deforum_helpers.external_libs').setLevel(logging.DEBUG)
```

## Best Practices

### Configuration Management
- Use immutable configuration objects
- Validate parameters before processing
- Provide sensible defaults

### Resource Management
- Clean up GPU memory after use
- Monitor system resources
- Use appropriate batch sizes

### Error Recovery
- Implement graceful fallbacks
- Provide clear error messages
- Log detailed error information

### Performance
- Profile before optimizing
- Use appropriate precision settings
- Cache expensive operations 