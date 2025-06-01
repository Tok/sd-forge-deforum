# Naming Changes Migration Guide

This document provides a comprehensive list of all renamed files, functions, variables, and classes during the functional refactoring of Deforum. Use this guide to update any custom code or references.

## 📁 File and Directory Renaming

### Core Directory Structure
| Old Path | New Path | Reason |
|----------|----------|---------|
| `scripts/deforum_helpers/` | `scripts/deforum_core/` | Better indicates core functionality |
| `scripts/deforum_helpers/ui_elements.py` | `scripts/deforum_core/ui/interface.py` | Clearer purpose, modular structure |
| `scripts/deforum_helpers/ui_left.py` | `scripts/deforum_core/ui/components.py` | More descriptive, modular structure |
| `scripts/deforum_helpers/animation_key_frames.py` | `scripts/deforum_core/animation/schedules.py` | Matches new immutable schedules |
| `scripts/deforum_helpers/parseq_adapter.py` | `scripts/deforum_core/integration/parseq_integration.py` | Better describes role |
| `scripts/deforum_helpers/video_audio_utilities.py` | `scripts/deforum_core/media/processing.py` | Shorter, clearer |

### New Modular Structure
```
scripts/deforum_core/
├── __init__.py
├── animation/                 # Animation system
│   ├── schedules.py          # Renamed from animation_key_frames.py
│   ├── interpolation.py      # Renamed from FrameInterpolater
│   └── movement_analysis.py  # Movement detection
├── ui/                       # User interface
│   ├── interface.py          # Renamed from ui_elements.py
│   ├── components.py         # Renamed from ui_left.py
│   ├── state/                # UI state management
│   └── builders/             # UI builders
├── generation/               # Core generation
│   ├── pipeline.py           # Renamed from generate.py parts
│   ├── rendering.py          # Renamed from render.py parts
│   └── processing.py         # Processing utilities
├── integration/              # External integrations
│   ├── parseq_integration.py # Renamed from parseq_adapter.py
│   ├── wan_integration.py    # WAN integration
│   └── controlnet.py         # ControlNet integration
├── media/                    # Media processing
│   ├── processing.py         # Renamed from video_audio_utilities.py
│   ├── encoding.py           # Video encoding
│   └── audio.py              # Audio processing
├── config/                   # Configuration system
│   ├── models.py             # Immutable config models
│   ├── validation.py         # Validation functions
│   └── legacy_adapter.py     # Legacy compatibility
└── utils/                    # Utilities
    ├── math_utils.py         # Mathematical utilities
    ├── file_utils.py         # File operations
    └── logging_utils.py      # Logging utilities
```

## 🏗️ Class and Object Renaming

### Core Data Structures
| Old Name | New Name | Reason |
|----------|----------|---------|
| `DeformAnimKeys` | `AnimationScheduleBuilder` | Clearer intent |
| `ControlNetKeys` | `ControlNetScheduleBuilder` | Consistent naming |
| `FreeUAnimKeys` | `FreeUScheduleBuilder` | Clear purpose |
| `KohyaHRFixAnimKeys` | `KohyaScheduleBuilder` | Simplified naming |
| `LooperAnimKeys` | `LooperScheduleBuilder` | Consistent pattern |
| `SimpleNamespace` instances | Descriptive immutable dataclasses | Type safety, immutability |

### Arguments and Configuration
| Old Name | New Name | Reason |
|----------|----------|---------|
| `DeforumArgs()` | `GenerationConfig()` | Better domain naming |
| `DeforumAnimArgs()` | `AnimationConfig()` | Clearer purpose |
| `DeforumOutputArgs()` | `VideoOutputConfig()` | More specific |
| `ParseqArgs()` | `ParseqIntegrationConfig()` | Clearer role |
| `WanArgs()` | `WanConfig()` | Consistent naming |
| `RootArgs()` | `RuntimeState()` | Better describes content |

### UI Components
| Old Name | New Name | Reason |
|----------|----------|---------|
| `UIDefaults` | `UIApplicationDefaults` | More specific |
| `UIState` | `UIApplicationState` | Clearer scope |
| `TabBuilder` | `UITabBuilder` | Consistent naming |
| `ComponentBuilder` | `UIComponentBuilder` | Clear purpose |

## 🔧 Function Renaming

### Animation and Scheduling
| Old Name | New Name | Reason |
|----------|----------|---------|
| `FrameInterpolater` | `FrameInterpolator` | Correct spelling |
| `parse_inbetweens` | `interpolate_keyframes` | More descriptive |
| `get_inbetweens` | `build_interpolated_series` | Clearer purpose |
| `parseq_to_series` | `extract_parseq_schedule_series` | More specific |
| `create_anim_keys` | `create_animation_schedules` | Clear intent |

### Generation and Processing
| Old Name | New Name | Reason |
|----------|----------|---------|
| `generate` | `generate_animation_sequence` | More specific |
| `render` | `render_animation_frame` | Clear scope |
| `process_video` | `process_video_sequence` | More descriptive |
| `create_video` | `encode_video_from_frames` | Specific action |
| `save_video` | `export_video_file` | Clear intent |

### UI Functions
| Old Name | New Name | Reason |
|----------|----------|---------|
| `create_ui` | `build_application_interface` | More descriptive |
| `get_tab_*` | `build_*_tab_interface` | Consistent pattern |
| `update_ui` | `update_interface_state` | Clearer purpose |
| `handle_*` | `process_*_event` | Consistent event naming |

### Utility Functions
| Old Name | New Name | Reason |
|----------|----------|---------|
| `get_*` | `extract_*` or `calculate_*` | More specific verbs |
| `set_*` | `configure_*` or `update_*` | Clearer intent |
| `check_*` | `validate_*` or `verify_*` | More specific |
| `do_*` | `execute_*` or `perform_*` | Professional naming |

## 📝 Variable Renaming

### Animation Configuration
| Old Name | New Name | Reason |
|----------|----------|---------|
| `anim_args` | `animation_config` | More professional |
| `deforum_args` | `generation_config` | Clearer purpose |
| `root_args` | `runtime_state` | Better describes content |
| `video_args` | `video_output_config` | More specific |
| `parseq_args` | `parseq_integration_config` | Clearer role |

### Cryptic Variable Elimination
| Old Name | New Name | Reason |
|----------|----------|---------|
| `da` | `animation_defaults` | Eliminate cryptic names |
| `d` | `generation_defaults` | Clear purpose |
| `dv` | `video_defaults` | Descriptive |
| `fi` | `interpolator` | No abbreviations |
| `processed` | `generation_result` | More specific |
| `args` | `config` or specific type | Context-dependent |

### UI Variables
| Old Name | New Name | Reason |
|----------|----------|---------|
| `ui` | `interface_state` | More descriptive |
| `tab` | `tab_interface` | Clear scope |
| `component` | `ui_component` | Specific type |
| `state` | `application_state` | Clear scope |
| `data` | `component_data` | More specific |

### Processing Variables
| Old Name | New Name | Reason |
|----------|----------|---------|
| `img` | `image` | No abbreviations |
| `vid` | `video` | Full words |
| `seq` | `sequence` | Complete word |
| `res` | `result` | Full word |
| `cfg` | `config` | Complete word |

## 🔄 Method and Property Renaming

### Immutable Data Structure Methods
| Old Pattern | New Pattern | Reason |
|-------------|-------------|---------|
| `.get_*()` | `.extract_*()` | More specific action |
| `.set_*()` | `.with_*()` | Immutable pattern |
| `.update_*()` | `.with_updated_*()` | Clear immutability |
| `.create_*()` | `.build_*()` | Consistent factory pattern |

### Example Transformations:
```python
# Old mutable pattern:
def set_strength(self, value):
    self.strength = value

# New immutable pattern:
def with_strength(self, value: float) -> 'AnimationConfig':
    return replace(self, strength=value)

# Old generic getter:
def get_schedule(self, name):
    return self.schedules[name]

# New specific extractor:
def extract_schedule_series(self, schedule_name: str) -> Tuple[float, ...]:
    return self.schedules.get(schedule_name, tuple())
```

## 📦 Import Statement Updates

### Old Import Patterns
```python
# OLD - Will break after refactoring
from scripts.deforum_helpers.ui_elements import *
from scripts.deforum_helpers.args import DeforumArgs
from scripts.deforum_helpers.animation_key_frames import FrameInterpolater
```

### New Import Patterns
```python
# NEW - Functional imports with clear naming
from scripts.deforum_core.ui.interface import build_application_interface
from scripts.deforum_core.config.models import GenerationConfig
from scripts.deforum_core.animation.interpolation import FrameInterpolator
from scripts.deforum_core.animation.schedules import AnimationScheduleBuilder

# Immutable data structures
from scripts.deforum_core.data_models import (
    ProcessingResult, UIApplicationDefaults, SettingsState
)
from scripts.deforum_core.schedules_models import (
    AnimationSchedules, ControlNetSchedules, ParseqScheduleData
)
```

## 🔧 Configuration Updates

### Legacy Configuration Format
```python
# OLD configuration style
args = SimpleNamespace(**{
    'W': 512,
    'H': 512,
    'strength': 0.75
})

# OLD variable names
anim_args = da
deforum_args = d
video_args = dv
```

### New Configuration Format
```python
# NEW immutable configuration
generation_config = GenerationConfig(
    width=512,
    height=512,
    strength=0.75
)

# NEW clear variable names
animation_config = animation_defaults
generation_config = generation_defaults
video_output_config = video_defaults
```

## 🚀 Migration Automation

### Automated Migration Script
```python
def migrate_legacy_code(file_path: str) -> None:
    """Automated migration helper for common renaming patterns"""
    
    # File path updates
    path_mappings = {
        'deforum_helpers': 'deforum_core',
        'ui_elements.py': 'ui/interface.py',
        'ui_left.py': 'ui/components.py',
        'animation_key_frames.py': 'animation/schedules.py',
        'parseq_adapter.py': 'integration/parseq_integration.py'
    }
    
    # Variable name updates
    variable_mappings = {
        r'\banim_args\b': 'animation_config',
        r'\bdeforum_args\b': 'generation_config',
        r'\broot_args\b': 'runtime_state',
        r'\bda\b': 'animation_defaults',
        r'\bd\b': 'generation_defaults',
        r'\bdv\b': 'video_defaults',
        r'\bfi\b': 'interpolator'
    }
    
    # Class name updates
    class_mappings = {
        'DeforumArgs': 'GenerationConfig',
        'DeforumAnimArgs': 'AnimationConfig',
        'FrameInterpolater': 'FrameInterpolator',
        'DeformAnimKeys': 'AnimationScheduleBuilder'
    }
    
    # Apply all mappings...
```

## 📋 Checklist for Manual Updates

### For Custom Extensions
- [ ] Update all import statements to new module structure
- [ ] Replace old class names with new equivalents
- [ ] Update variable names to follow new conventions
- [ ] Replace mutable SimpleNamespace with immutable dataclasses
- [ ] Update function calls to use new naming patterns

### For Configuration Files
- [ ] Replace old argument class names
- [ ] Update variable names in config dictionaries
- [ ] Use new immutable data structure constructors
- [ ] Update any hardcoded file paths

### For UI Customizations
- [ ] Update UI component imports
- [ ] Replace old UI builder function names
- [ ] Update event handler function names
- [ ] Use new state management patterns

## 🔍 Breaking Changes Summary

### High Impact Changes
1. **Module Structure**: Complete reorganization of deforum_helpers → deforum_core
2. **Immutable Data**: All SimpleNamespace → frozen dataclasses
3. **Class Names**: All *Args → *Config classes
4. **Function Names**: Generic get/set → specific extract/with patterns

### Medium Impact Changes
1. **Variable Names**: Cryptic abbreviations → descriptive names
2. **File Names**: Generic names → purpose-specific names
3. **Method Names**: Mutable patterns → immutable patterns

### Low Impact Changes
1. **Import Organization**: Better import grouping and ordering
2. **Documentation**: Updated docstrings and type hints
3. **Code Style**: Consistent naming conventions

## 📚 Additional Resources

### Migration Tools
- **Automated Scanner**: Detects old patterns in your code
- **Batch Replacer**: Applies common renaming patterns
- **Compatibility Checker**: Verifies migration completeness
- **Documentation Generator**: Updates documentation after migration

### Support Resources
- **Migration FAQ**: Common migration questions and answers
- **Video Tutorials**: Step-by-step migration walkthroughs  
- **Community Forum**: Get help with complex migration scenarios
- **Code Examples**: Before/after examples for common patterns

---

*This naming changes guide is comprehensive but may not cover every edge case. When in doubt, follow the functional programming principles: use descriptive names, avoid abbreviations, prefer immutable patterns, and maintain clear separation of concerns.* 