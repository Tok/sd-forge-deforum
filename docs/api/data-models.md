# Data Models API Reference

This document provides comprehensive API reference for Deforum's immutable data models that replace SimpleNamespace usage throughout the codebase.

## 🏗️ Overview

Deforum's data models are built using frozen dataclasses to ensure immutability, type safety, and functional programming principles. All models provide validation, factory methods, and pure functional transformations.

### Core Principles
- **Immutability**: All data structures are frozen and cannot be modified after creation
- **Type Safety**: Comprehensive type hints and runtime validation
- **Pure Functions**: All methods are side-effect free
- **Validation**: Built-in validation with clear error messages
- **Factory Methods**: Clean creation patterns for common use cases

## 📦 Module: data_models.py

### ProcessingResult

Immutable processing result to replace SimpleNamespace objects in generation pipeline.

```python
@dataclass(frozen=True)
class ProcessingResult:
    """Immutable processing result with comprehensive metadata"""
    images: Tuple[Image.Image, ...] = field(default_factory=tuple)
    info: str = ""
    success: bool = True
    processing_time: float = 0.0
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    seeds: Tuple[int, ...] = field(default_factory=tuple)
    prompts: Tuple[str, ...] = field(default_factory=tuple)
```

#### Factory Methods

##### `create_motion_preview(image: Image.Image) -> ProcessingResult`
Creates a processing result for motion preview generation.

```python
# Example usage
preview_image = Image.new('RGB', (512, 512))
result = ProcessingResult.create_motion_preview(preview_image)
print(result.info)  # "Generating motion preview..."
```

##### `create_generation_result(images: List[Image.Image], info: str) -> ProcessingResult`
Creates a processing result for actual generation.

```python
# Example usage
generated_images = [img1, img2, img3]
result = ProcessingResult.create_generation_result(
    generated_images, 
    "Generated 3 frames successfully"
)
print(len(result.images))  # 3
```

#### Instance Methods

##### `with_warning(warning: str) -> ProcessingResult`
Returns new ProcessingResult with added warning (functional transformation).

```python
# Example usage
result = ProcessingResult.create_motion_preview(image)
result_with_warning = result.with_warning("Low VRAM detected")
# Original result unchanged, new result has warning
```

### UIDefaults

Immutable UI defaults to replace mutable SimpleNamespace objects in UI system.

```python
@dataclass(frozen=True)
class UIDefaults:
    """Immutable UI default configurations"""
    deforum_args: Dict[str, Any] = field(default_factory=dict)
    animation_args: Dict[str, Any] = field(default_factory=dict)
    video_args: Dict[str, Any] = field(default_factory=dict)
    parseq_args: Dict[str, Any] = field(default_factory=dict)
    wan_args: Dict[str, Any] = field(default_factory=dict)
    root_args: Dict[str, Any] = field(default_factory=dict)
```

#### Factory Methods

##### `create_defaults() -> UIDefaults`
Creates UI defaults using functional args system.

```python
# Example usage
defaults = UIDefaults.create_defaults()
# Auto-populates with current default configurations
```

### SettingsState

Immutable settings state to replace mutable objects in settings system.

```python
@dataclass(frozen=True)
class SettingsState:
    """Immutable settings state with validation support"""
    loaded_settings: Dict[str, Any] = field(default_factory=dict)
    validation_errors: Tuple[str, ...] = field(default_factory=tuple)
    last_modified: float = 0.0
    file_path: Optional[str] = None
    wan_args: Dict[str, Any] = field(default_factory=dict)
```

#### Factory Methods

##### `from_dict(settings_dict: Dict[str, Any], file_path: Optional[str] = None) -> SettingsState`
Creates settings state from dictionary with automatic WAN args extraction.

```python
# Example usage
settings_data = {
    "W": 1024,
    "H": 768,
    "wan_mode": "I2V Chaining",
    "max_frames": 100
}
settings = SettingsState.from_dict(settings_data, "/path/to/settings.json")
```

#### Instance Methods

##### `with_validation_error(error: str) -> SettingsState`
Returns new SettingsState with added validation error.

```python
# Example usage
settings = SettingsState.from_dict(config_dict)
settings_with_error = settings.with_validation_error("Invalid resolution")
```

### ExternalLibraryArgs

Immutable args for external libraries (RIFE, FILM) to replace SimpleNamespace objects.

```python
@dataclass(frozen=True)
class ExternalLibraryArgs:
    """Immutable configuration for external video processing libraries"""
    multi: float = 1.0
    video: str = ""
    output: str = ""
    img: str = ""
    exp: int = 1
    ratio: float = 0.0
    rthreshold: float = 0.02
    rmaxcycles: int = 8
    UHD: bool = False
    scale: float = 1.0
    fps: Optional[float] = None
    png: bool = False
    ext: str = "mp4"
```

#### Factory Methods

##### `create_rife_args() -> ExternalLibraryArgs`
Creates RIFE-specific configuration.

```python
# Example usage
rife_config = ExternalLibraryArgs.create_rife_args()
# Pre-configured for RIFE interpolation
```

##### `create_film_args() -> ExternalLibraryArgs`
Creates FILM-specific configuration.

```python
# Example usage  
film_config = ExternalLibraryArgs.create_film_args()
# Pre-configured for FILM interpolation
```

### TestFixtureArgs

Immutable args for test fixtures to replace SimpleNamespace objects in tests.

```python
@dataclass(frozen=True)
class TestFixtureArgs:
    """Immutable test fixture configuration"""
    W: int = 512
    H: int = 512
    seed: int = 42
    sampler: str = "Euler"
    steps: int = 20
    cfg_scale: float = 7.0
    strength: float = 0.85
    use_init: bool = False
    animation_mode: str = "3D"
    max_frames: int = 100
    fps: int = 30
    batch_name: str = "test_batch"
```

#### Factory Methods

##### `create_minimal_args() -> TestFixtureArgs`
Creates minimal test configuration.

##### `create_animation_args() -> TestFixtureArgs`
Creates animation-specific test configuration.

##### `create_video_args() -> TestFixtureArgs`
Creates video-specific test configuration.

```python
# Example usage
minimal_config = TestFixtureArgs.create_minimal_args()
animation_config = TestFixtureArgs.create_animation_args()
video_config = TestFixtureArgs.create_video_args()
```

## 🎛️ Configuration Data Classes

### AnimationArgs

Immutable animation arguments with comprehensive validation.

```python
@dataclass(frozen=True)
class AnimationArgs:
    """Immutable animation configuration with validation"""
    animation_mode: AnimationMode = AnimationMode.THREE_D
    max_frames: int = 333
    border: BorderMode = BorderMode.REPLICATE
    
    # Movement schedules
    angle: str = "0: (0)"
    zoom: str = "0: (1.0)"
    translation_x: str = "0: (0)"
    translation_y: str = "0: (0)"
    translation_z: str = "0: (0)"
    # ... many more schedule fields
```

#### Key Features
- **Schedule Validation**: All schedule strings validated for correct format
- **Enum Types**: Type-safe enums for animation modes, color coherence, etc.
- **Range Validation**: Numeric values validated within acceptable ranges
- **Post-init Validation**: Comprehensive validation after object creation

### DeforumArgs

Immutable Deforum generation arguments.

```python
@dataclass(frozen=True)
class DeforumArgs:
    """Immutable generation configuration"""
    W: int = 512
    H: int = 512
    seed: int = -1
    sampler: str = "Euler a"
    steps: int = 25
    scale: float = 7.0
    # ... comprehensive generation settings
```

### VideoArgs

Immutable video output configuration.

```python
@dataclass(frozen=True)
class VideoArgs:
    """Immutable video output configuration"""
    fps: int = 30
    output_format: str = "mp4"
    ffmpeg_location: str = "ffmpeg"
    ffmpeg_crf: int = 17
    ffmpeg_preset: str = "slow"
    # ... video encoding settings
```

## 🔧 Utility Functions

### Validation Functions

#### `validate_schedule_string(schedule_str: str, parameter_name: str = "schedule") -> None`
Validates schedule string format with detailed error messages.

```python
# Example usage
try:
    validate_schedule_string("0: (1.0), 30: (2.0)", "zoom_schedule")
except ValueError as e:
    print(f"Validation error: {e}")
```

#### `validate_positive_int(value: int, parameter_name: str = "value") -> None`
Validates positive integer values.

#### `validate_non_negative_number(value: Union[int, float], parameter_name: str = "value") -> None`
Validates non-negative numeric values.

#### `validate_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float], parameter_name: str = "value") -> None`
Validates values within specified range.

### Factory Functions

#### `create_animation_args_from_dict(data: Dict[str, Any]) -> AnimationArgs`
Creates AnimationArgs from legacy dictionary data with enum conversion.

#### `create_deforum_args_from_dict(data: Dict[str, Any]) -> DeforumArgs`
Creates DeforumArgs from legacy dictionary data.

```python
# Example usage
legacy_data = {
    "animation_mode": "3D",
    "max_frames": 100,
    "zoom": "0: (1.0), 50: (1.5)"
}
animation_config = create_animation_args_from_dict(legacy_data)
```

### Validation Functions

#### `validate_processing_result(result: ProcessingResult) -> bool`
Validates ProcessingResult data integrity.

#### `validate_ui_defaults(defaults: UIDefaults) -> bool`
Validates UIDefaults data integrity.

```python
# Example usage
result = ProcessingResult.create_motion_preview(image)
is_valid = validate_processing_result(result)  # True
```

## 🔍 Type Aliases

### Common Type Patterns
```python
ImageTuple = Tuple[Image.Image, ...]
StringTuple = Tuple[str, ...]
FloatTuple = Tuple[float, ...]
```

### Usage Examples
```python
def process_images(images: ImageTuple) -> ProcessingResult:
    """Process multiple images with type safety"""
    return ProcessingResult(images=images, info="Processed batch")

def create_prompt_series(prompts: StringTuple) -> Dict[int, str]:
    """Create frame-indexed prompt series"""
    return {i: prompt for i, prompt in enumerate(prompts)}
```

## 🚀 Usage Patterns

### Creating Immutable Configurations
```python
# Instead of mutable SimpleNamespace:
# args = SimpleNamespace(W=1024, H=768, strength=0.8)

# Use immutable dataclass:
config = DeforumArgs(W=1024, H=768, strength=0.8)
```

### Functional Transformations
```python
# Instead of mutation:
# result.warnings.append("New warning")  # Error - tuple is immutable

# Use functional transformation:
result_with_warning = result.with_warning("New warning")
```

### Factory Method Patterns
```python
# Create specialized configurations:
rife_config = ExternalLibraryArgs.create_rife_args()
motion_preview = ProcessingResult.create_motion_preview(image)
ui_defaults = UIDefaults.create_defaults()
```

### Validation Integration
```python
# Automatic validation on creation:
try:
    config = AnimationArgs(max_frames=-1)  # Raises ValueError
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

## 🔬 Testing Support

All data models include comprehensive test coverage:

- **Creation Tests**: Verify proper initialization
- **Immutability Tests**: Ensure frozen behavior
- **Factory Method Tests**: Validate factory patterns
- **Validation Tests**: Test validation logic
- **Functional Transformation Tests**: Verify pure functions

```python
# Example test usage
def test_processing_result_immutability():
    result = ProcessingResult.create_motion_preview(mock_image)
    with pytest.raises(Exception):
        result.success = False  # Should raise FrozenInstanceError
```

---

*This API documentation is automatically generated from type hints and docstrings. For implementation details, see the source code in `scripts/deforum_helpers/data_models.py`.* 