# API Reference

Technical documentation for Deforum's APIs and data structures.

## Core APIs

### [Data Models](data-models.md)
Immutable data structures used throughout Deforum:
- `ProcessingResult` - Generation results
- `UIDefaults` - UI configuration defaults
- `SettingsState` - Application settings
- `ExternalLibraryArgs` - External library configuration
- `TestFixtureArgs` - Testing utilities

### [Schedule System](schedules.md)
Animation scheduling and keyframe management:
- `AnimationSchedules` - Core animation parameters
- `ControlNetSchedules` - ControlNet integration
- `ParseqScheduleData` - Parseq integration
- Schedule interpolation and validation

### [Core Functions](core.md)
Main Deforum functionality:
- Generation pipeline
- Image processing
- Video creation
- Configuration management

## External Integrations

### [External Libraries](external-libs.md)
Integration with external tools:
- **RIFE** - Frame interpolation
- **FILM** - Advanced interpolation
- **MiDaS** - Depth estimation
- **ControlNet** - Image conditioning

### [WAN AI Integration](../wan/technical.md)
Advanced AI video generation:
- Text-to-video generation
- Image-to-video conversion
- Prompt enhancement
- Model management

## UI Components

### [UI System](ui-components.md)
User interface components and state management:
- Component architecture
- State management patterns
- Event handling
- Form validation

## Data Validation

### Type Safety
All APIs use comprehensive type hints:
```python
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class ExampleConfig:
    width: int
    height: int
    seed: Optional[int] = None
```

### Validation Functions
Built-in validation for all data structures:
- Range validation
- Type checking
- Format validation
- Cross-field validation

## Error Handling

### Exception Types
- `ValidationError` - Data validation failures
- `ConfigurationError` - Invalid configuration
- `ProcessingError` - Generation failures
- `ExternalLibraryError` - External tool failures

### Error Recovery
- Graceful degradation
- Fallback configurations
- User-friendly error messages
- Detailed logging

## Performance Considerations

### Memory Management
- Immutable data structures prevent memory leaks
- Efficient tuple usage for large datasets
- Lazy evaluation where appropriate

### Optimization
- Pure functions enable caching
- Functional composition reduces overhead
- Type hints enable compiler optimizations

## Examples

### Basic Usage
```python
from deforum_helpers.data_models import ProcessingResult

# Create immutable result
result = ProcessingResult.create_generation_result(
    images=[generated_image],
    info="Generation completed successfully"
)
```

### Advanced Configuration
```python
from deforum_helpers.schedules_models import AnimationSchedules

# Create animation schedule
schedule = AnimationSchedules.from_anim_args(
    anim_args=animation_config,
    max_frames=100,
    seed=42
)
``` 