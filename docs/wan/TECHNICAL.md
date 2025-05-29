# Wan Technical Reference

Technical documentation for developers and advanced users working with the Wan integration.

## Architecture

### Integration Layer

The Wan integration uses a compatibility layer approach:

- **No Wan Repo Modification**: Original Wan repository remains untouched
- **Compatibility Wrapper**: `wan_simple_integration.py` provides interface
- **Auto-Discovery**: Dynamic model detection and validation
- **Error Handling**: Comprehensive fallback mechanisms

### Key Components

```
scripts/deforum_helpers/
├── wan_simple_integration.py    # Main integration logic
├── ui_elements.py              # UI components (Wan tab)
├── args.py                     # Wan arguments definition
└── run_deforum.py             # Main execution pipeline
```

## Model Discovery

### Discovery Locations

Models are searched in order:

1. `models/wan/` (primary)
2. `models/WAN/` (alternative)
3. HuggingFace cache (`~/.cache/huggingface/`)
4. Downloads folder (`~/Downloads/`)

### Model Validation

```python
def discover_models():
    """Auto-discover Wan models with validation"""
    models = []
    for path in search_paths:
        if validate_model_structure(path):
            models.append({
                'name': extract_model_name(path),
                'path': path,
                'size': determine_model_size(path),
                'type': 'T2V' or 'I2V'
            })
    return models
```

### Model Structure

Expected directory structure:

```
models/wan/
├── model_index.json           # Model configuration
├── scheduler/                 # Scheduler components
├── text_encoder/             # Text encoder
├── tokenizer/                # Tokenizer
├── transformer/              # Main transformer model
├── vae/                      # VAE components
└── feature_extractor/        # Feature extraction
```

## Frame Calculation

### 4n+1 Requirement

Wan models require frame counts in 4n+1 format:

```python
def calculate_wan_frames(requested_frames):
    """Calculate nearest valid Wan frame count"""
    if (requested_frames - 1) % 4 == 0:
        return requested_frames  # Already valid
    
    # Find nearest 4n+1 values
    lower = ((requested_frames - 1) // 4) * 4 + 1
    upper = lower + 4
    
    # Choose closest
    if abs(requested_frames - lower) <= abs(requested_frames - upper):
        return lower
    else:
        return upper
```

### Frame Discarding

Extra frames are discarded from the middle to preserve timing:

```python
def discard_middle_frames(frames, target_count):
    """Discard frames from middle to match target count"""
    if len(frames) <= target_count:
        return frames
    
    excess = len(frames) - target_count
    start_discard = len(frames) // 2 - excess // 2
    end_discard = start_discard + excess
    
    return frames[:start_discard] + frames[end_discard:]
```

## I2V Chaining

### Process Flow

1. **T2V Generation**: First clip uses Text-to-Video
2. **Frame Extraction**: Extract last frame as starting image
3. **I2V Generation**: Subsequent clips use Image-to-Video
4. **Strength Control**: Apply Deforum strength schedule

### Strength Scheduling

```python
def parse_strength_schedule(schedule_str, frame_number):
    """Parse Deforum strength schedule for I2V"""
    # Parse: "0:(0.85), 120:(0.6)"
    schedule = parse_deforum_schedule(schedule_str)
    
    # Find applicable strength value
    strength = interpolate_schedule_value(schedule, frame_number)
    
    return max(0.0, min(1.0, strength))
```

## Memory Management

### VRAM Optimization

```python
def optimize_memory_usage():
    """Optimize VRAM usage during generation"""
    # Clear cache before generation
    torch.cuda.empty_cache()
    
    # Use gradient checkpointing
    model.enable_gradient_checkpointing()
    
    # Offload to CPU when not in use
    if not actively_generating:
        model.to('cpu')
```

### Batch Processing

```python
def process_clips_sequentially(clips):
    """Process clips one at a time to manage memory"""
    results = []
    
    for clip in clips:
        # Load model to GPU
        model.to('cuda')
        
        # Generate clip
        result = generate_clip(clip)
        results.append(result)
        
        # Offload model
        model.to('cpu')
        torch.cuda.empty_cache()
    
    return results
```

## Error Handling

### Flash Attention Fallback

```python
def setup_attention_mechanism():
    """Setup attention with automatic fallback"""
    try:
        import flash_attn
        return 'flash_attention'
    except ImportError:
        print("Flash attention not available, using PyTorch native")
        return 'pytorch_native'
```

### Model Loading Errors

```python
def safe_model_loading(model_path):
    """Safely load model with comprehensive error handling"""
    try:
        model = load_wan_model(model_path)
        return model, None
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        
        # Provide specific guidance
        if "out of memory" in str(e).lower():
            error_msg += "\n💡 Try the 1.3B model or reduce batch size"
        elif "file not found" in str(e).lower():
            error_msg += "\n💡 Check model path and file permissions"
        
        return None, error_msg
```

## Performance Optimization

### Model Selection

```python
def select_optimal_model(available_models, vram_gb):
    """Select best model based on available VRAM"""
    if vram_gb >= 16:
        return find_model_by_size(available_models, '14B')
    elif vram_gb >= 8:
        return find_model_by_size(available_models, '1.3B')
    else:
        raise ValueError("Insufficient VRAM (8GB+ required)")
```

### Generation Parameters

```python
def optimize_generation_params(model_size, target_quality):
    """Optimize parameters based on model and quality target"""
    if model_size == '1.3B':
        return {
            'inference_steps': 20 if target_quality == 'fast' else 50,
            'guidance_scale': 7.5,
            'batch_size': 1
        }
    elif model_size == '14B':
        return {
            'inference_steps': 30 if target_quality == 'fast' else 75,
            'guidance_scale': 8.0,
            'batch_size': 1
        }
```

## Integration Points

### Deforum Pipeline

```python
def integrate_with_deforum(args, anim_args, video_args):
    """Integration point with main Deforum pipeline"""
    if anim_args.animation_mode == 'Wan Video':
        return generate_wan_video(
            args=args,
            anim_args=anim_args,
            video_args=video_args,
            animation_prompts=args.animation_prompts
        )
    else:
        return standard_deforum_generation(args, anim_args, video_args)
```

### UI Components

```python
def create_wan_ui_components():
    """Create Wan-specific UI components"""
    components = {}
    
    # Essential settings
    components['model_size'] = gr.Dropdown(
        choices=['1.3B (Recommended)', '14B (High Quality)'],
        value='1.3B (Recommended)'
    )
    
    # Advanced settings
    components['inference_steps'] = gr.Slider(
        minimum=5, maximum=100, step=5, value=50
    )
    
    return components
```

## Testing

### Unit Tests

```python
def test_frame_calculation():
    """Test 4n+1 frame calculation"""
    assert calculate_wan_frames(15) == 17
    assert calculate_wan_frames(20) == 21
    assert calculate_wan_frames(21) == 21

def test_model_discovery():
    """Test model auto-discovery"""
    models = discover_models()
    assert len(models) > 0
    assert all('path' in model for model in models)
```

### Integration Tests

```python
def test_end_to_end_generation():
    """Test complete generation pipeline"""
    prompts = {"0": "test prompt", "30": "second prompt"}
    result = generate_wan_video(
        prompts=prompts,
        fps=30,
        model_size='1.3B'
    )
    assert result is not None
    assert os.path.exists(result['output_path'])
```

## Debugging

### Debug Output

Enable debug mode for detailed logging:

```python
# In ui_elements.py
print("🔧 DEBUG: Creating Wan inference steps slider")
print(f"🔧 DEBUG: Slider properties - min: {slider.minimum}")
```

### Common Debug Points

1. **Model Discovery**: Check console for discovered models
2. **Frame Calculation**: Verify 4n+1 calculations
3. **Memory Usage**: Monitor VRAM during generation
4. **Pipeline Integration**: Verify Deforum argument passing

### Log Analysis

```python
def analyze_generation_logs():
    """Analyze generation logs for issues"""
    logs = read_generation_logs()
    
    # Check for common issues
    if "CUDA out of memory" in logs:
        return "Reduce model size or batch size"
    elif "Model not found" in logs:
        return "Check model installation"
    elif "Flash attention" in logs:
        return "Flash attention fallback working correctly"
```

## API Reference

### Main Functions

- `discover_models()`: Auto-discover available Wan models
- `generate_wan_video()`: Main video generation function
- `calculate_wan_frames()`: Calculate valid frame counts
- `parse_strength_schedule()`: Parse Deforum strength schedules

### Configuration

- `WanArgs()`: Wan-specific arguments definition
- `wan_generate_video()`: UI callback function
- `get_tab_wan()`: UI tab creation function

### Utilities

- `validate_model_structure()`: Validate model directory
- `optimize_memory_usage()`: VRAM optimization
- `safe_model_loading()`: Error-safe model loading 