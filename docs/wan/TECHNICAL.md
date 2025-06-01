# WAN Technical Documentation

This document provides technical implementation details for the WAN (Wan Video) integration in Deforum.

## 🏗️ Architecture Overview

WAN is implemented as a modular system that integrates with Deforum's functional programming architecture:

```
WAN System Architecture:
├── Core Generation Engine
│   ├── T2V Pipeline (Text-to-Video)
│   ├── I2V Pipeline (Image-to-Video)
│   └── VACE Unified Pipeline
├── Prompt Enhancement System
│   ├── Qwen Model Integration
│   ├── Movement Analysis
│   └── Style Enhancement
├── Movement Analysis Engine
│   ├── Animation Parameter Analysis
│   ├── Pattern Recognition
│   └── Natural Language Generation
└── Integration Layer
    ├── Deforum Schedule Integration
    ├── UI Component Integration
    └── State Management
```

## 🔧 Core Components

### 1. Generation Pipelines

#### T2V Pipeline
- **Model Loading**: Dynamic model discovery and loading
- **Prompt Processing**: Advanced prompt enhancement and scheduling
- **Frame Generation**: High-quality video frame synthesis
- **Memory Management**: Efficient VRAM usage and cleanup

#### I2V Pipeline  
- **Frame Chaining**: Seamless frame-to-frame transitions
- **Continuity Control**: Strength-based temporal consistency
- **Overlap Management**: Frame overlap for smooth transitions
- **Quality Enhancement**: Advanced interpolation techniques

#### VACE Unified Pipeline
- **Dual Mode Operation**: Single model for both T2V and I2V
- **Blank Frame Transformation**: T2V via I2V architecture
- **Consistency Optimization**: Same weights for perfect continuity
- **Memory Efficiency**: Single model load for both operations

### 2. Prompt Enhancement Engine

```python
@dataclass(frozen=True)
class PromptEnhancementConfig:
    """Immutable configuration for prompt enhancement"""
    qwen_model: str = "Qwen-2.5-1.5B"
    language: str = "English"
    enable_style_prompting: bool = True
    enable_movement_descriptions: bool = True
    style_strength: float = 0.5
    
    def enhance_prompt(self, base_prompt: str, movement_context: str) -> str:
        """Pure function to enhance prompts with movement context"""
        # Implementation details...
```

#### Qwen Model Integration
- **Multi-language Support**: English and Chinese processing
- **Context-aware Enhancement**: Movement-based prompt refinement
- **Style Integration**: Seamless style prompt integration
- **Performance Optimization**: Cached model loading and inference

### 3. Movement Analysis System

```python
@dataclass(frozen=True)
class MovementAnalysisResult:
    """Immutable result from movement analysis"""
    movements: Tuple[str, ...] = field(default_factory=tuple)
    descriptions: Tuple[str, ...] = field(default_factory=tuple)
    intensity: str = "subtle"
    duration: str = "brief"
    
    @classmethod
    def analyze_animation_schedules(cls, schedules: AnimationSchedules, 
                                  frame_offset: int = 0,
                                  sensitivity: float = 1.0) -> 'MovementAnalysisResult':
        """Pure function to analyze movement from animation schedules"""
        # Implementation details...
```

#### Pattern Recognition
- **Directional Analysis**: Identifies pan, tilt, dolly, roll movements
- **Intensity Classification**: Subtle, gentle, moderate movement levels
- **Duration Analysis**: Brief, extended, sustained movement patterns
- **Frame-specific Context**: Unique analysis for each frame position

### 4. Integration with Deforum

#### Schedule System Integration
```python
def integrate_wan_with_schedules(animation_schedules: AnimationSchedules,
                               wan_config: WanConfig) -> WanGenerationPlan:
    """Pure function to create WAN generation plan from Deforum schedules"""
    # Extract keyframes from schedules
    keyframes = extract_keyframes_from_schedules(animation_schedules)
    
    # Analyze movement patterns
    movement_analysis = analyze_movement_patterns(keyframes, wan_config.sensitivity)
    
    # Generate enhanced prompts
    enhanced_prompts = enhance_prompts_with_movement(
        keyframes.prompts, movement_analysis, wan_config
    )
    
    return WanGenerationPlan(
        keyframes=keyframes,
        enhanced_prompts=enhanced_prompts,
        movement_descriptions=movement_analysis.descriptions
    )
```

#### UI State Management
```python
@dataclass(frozen=True)
class WanUIState:
    """Immutable WAN UI state"""
    model_status: str = "Not Loaded"
    available_models: Tuple[str, ...] = field(default_factory=tuple)
    generation_progress: float = 0.0
    current_operation: str = "Idle"
    
    def with_model_loaded(self, model_name: str) -> 'WanUIState':
        """Return new state with model loaded"""
        return replace(self, model_status=f"Loaded: {model_name}")
```

## 🚀 Performance Optimizations

### Memory Management
- **Lazy Loading**: Models loaded only when needed
- **Memory Cleanup**: Automatic cleanup after generation
- **VRAM Monitoring**: Real-time VRAM usage tracking
- **Model Caching**: Intelligent model caching strategies

### Generation Optimization
- **Flash Attention**: Automatic detection and usage
- **Batch Processing**: Efficient batch generation
- **Frame Overlap**: Optimized overlap for quality vs performance
- **Resolution Scaling**: Dynamic resolution adjustments

### Prompt Processing
- **Caching**: Enhanced prompt caching
- **Batch Enhancement**: Multiple prompt enhancement
- **Context Reuse**: Movement context reuse across frames
- **Language Detection**: Automatic language detection

## 🔬 Testing Framework

### Unit Tests
```python
class TestWanIntegration:
    """Comprehensive WAN integration tests"""
    
    def test_prompt_enhancement_immutability(self):
        """Test prompt enhancement creates immutable results"""
        
    def test_movement_analysis_pure_functions(self):
        """Test movement analysis uses pure functions"""
        
    def test_generation_pipeline_isolation(self):
        """Test generation pipeline isolates side effects"""
        
    def test_schedule_integration_functional(self):
        """Test schedule integration follows functional principles"""
```

### Integration Tests
- **End-to-end Generation**: Full pipeline testing
- **Memory Usage**: VRAM and RAM usage validation
- **Quality Metrics**: Generated video quality assessment
- **Performance Benchmarks**: Generation speed and efficiency

### Mock Testing
- **Model Mocking**: Test without actual model loading
- **UI State Testing**: Pure UI state transformations
- **Schedule Integration**: Mock schedule integration testing
- **Error Handling**: Comprehensive error scenario testing

## 🔄 Data Flow

### Generation Pipeline Flow
```
1. Deforum Schedules → Movement Analysis
2. Movement Analysis → Prompt Enhancement  
3. Enhanced Prompts → T2V/I2V Generation
4. Generated Frames → Quality Processing
5. Processed Frames → Video Assembly
6. Final Video → Output Pipeline
```

### State Management Flow
```
1. User Input → UI State Update (Immutable)
2. UI State → Configuration Creation (Pure Function)
3. Configuration → Generation Plan (Pure Function)
4. Generation Plan → Pipeline Execution (Side Effects Isolated)
5. Pipeline Results → State Update (Immutable)
```

## 🛠️ Configuration Management

### Model Configuration
```python
@dataclass(frozen=True)
class WanModelConfig:
    """Immutable WAN model configuration"""
    t2v_model_path: str = ""
    i2v_model_path: str = ""
    qwen_model_path: str = ""
    auto_download: bool = True
    preferred_size: str = "1.3B"
    flash_attention: bool = True
```

### Generation Configuration
```python
@dataclass(frozen=True)
class WanGenerationConfig:
    """Immutable generation configuration"""
    resolution: str = "864x480"
    inference_steps: int = 25
    guidance_scale: float = 7.5
    motion_strength: float = 1.0
    frame_overlap: int = 2
    enable_interpolation: bool = True
```

## 📊 Monitoring and Metrics

### Performance Metrics
- **Generation Speed**: Frames per second
- **Memory Usage**: Peak VRAM and RAM usage
- **Model Loading Time**: Time to load models
- **Prompt Enhancement Speed**: Enhancement operations per second

### Quality Metrics
- **Temporal Consistency**: Frame-to-frame similarity
- **Prompt Adherence**: Generated content vs prompt alignment
- **Movement Accuracy**: Movement description accuracy
- **Enhancement Quality**: Prompt enhancement effectiveness

## 🔒 Error Handling

### Graceful Degradation
- **Model Loading Failures**: Fallback to alternative models
- **Memory Issues**: Automatic resolution adjustment
- **Enhancement Failures**: Fallback to original prompts
- **Generation Errors**: Partial result recovery

### Error Recovery
```python
def handle_wan_generation_error(error: WanError) -> WanRecoveryPlan:
    """Pure function to create error recovery plan"""
    if isinstance(error, OutOfMemoryError):
        return WanRecoveryPlan(
            reduce_resolution=True,
            reduce_batch_size=True,
            clear_cache=True
        )
    elif isinstance(error, ModelLoadError):
        return WanRecoveryPlan(
            try_alternative_model=True,
            enable_auto_download=True
        )
    # ... other error types
```

## 🔮 Future Enhancements

### Planned Features
- **Custom Model Training**: Support for user-trained models
- **Multi-GPU Support**: Distributed generation across multiple GPUs
- **Advanced Interpolation**: Neural frame interpolation
- **Real-time Preview**: Live generation preview

### Performance Improvements
- **Model Quantization**: 8-bit and 4-bit model support
- **Optimized Attention**: Custom attention mechanisms
- **Streaming Generation**: Frame streaming for long videos
- **Adaptive Quality**: Dynamic quality adjustment

---

*This technical documentation is actively maintained as WAN evolves. Check the [enhancements](enhancements.md) for the latest development updates.* 