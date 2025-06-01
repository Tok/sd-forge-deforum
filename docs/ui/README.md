# UI Documentation

User interface components and design patterns used in Deforum.

## Overview

Deforum's UI is built using Gradio with a component-based architecture that emphasizes:
- **Immutable state management** - UI state is managed through immutable data structures
- **Event-driven updates** - Components respond to user actions through pure functions
- **Modular design** - Each tab and component has clear responsibilities

## Tab Structure

### [Main Tabs](tab-structure.md)
The UI is organized into workflow-oriented tabs:

1. **Setup** - Core generation settings (resolution, steps, etc.)
2. **Animation** - Movement and timing parameters
3. **Prompts** - Text prompts and scheduling
4. **WAN AI** - Advanced AI video generation
5. **Init** - Input sources (images, videos, masks)
6. **Advanced** - Fine-tuning parameters
7. **Output** - Video rendering settings
8. **Post-Process** - Enhancement and effects

## Component Architecture

### State Management
```python
@dataclass(frozen=True)
class UIDefaults:
    """Immutable UI configuration defaults"""
    deforum_args: Dict[str, Any]
    animation_args: Dict[str, Any]
    video_args: Dict[str, Any]
    # ... other configuration
```

### Component Builders
- **Tab builders** - Create complete tab interfaces
- **Component builders** - Create individual UI elements
- **Layout builders** - Manage component arrangement

### Event Handling
- Pure functions for all event handlers
- Immutable state updates
- Clear separation of UI and business logic

## Key Components

### Animation Controls
- Movement parameter sliders
- Schedule input fields
- Preview and validation

### Prompt Management
- Multi-frame prompt editing
- AI enhancement integration
- Template loading and saving

### Video Output
- Format selection
- Quality settings
- Export progress tracking

### WAN AI Integration
- Model selection
- Generation parameters
- Real-time feedback

## Design Patterns

### Functional UI Updates
```python
def update_animation_tab(state: UIState, changes: Dict[str, Any]) -> UIState:
    """Pure function to update UI state"""
    return state.with_updates(changes)
```

### Component Composition
```python
def build_animation_tab(defaults: UIDefaults) -> gr.Tab:
    """Compose animation tab from smaller components"""
    with gr.Tab("Animation"):
        movement_controls = build_movement_controls(defaults.animation_args)
        timing_controls = build_timing_controls(defaults.animation_args)
        return gr.Column([movement_controls, timing_controls])
```

### Event Binding
```python
def setup_event_handlers(components: Dict[str, gr.Component]) -> None:
    """Bind event handlers to UI components"""
    components['generate_btn'].click(
        fn=handle_generation_request,
        inputs=get_generation_inputs(components),
        outputs=[components['output_gallery']]
    )
```

## Styling and Themes

### CSS Classes
- Consistent styling across components
- Responsive design principles
- Accessibility considerations

### Color Scheme
- Dark/light theme support
- High contrast options
- Color-blind friendly palette

## Accessibility

### Keyboard Navigation
- Tab order optimization
- Keyboard shortcuts
- Screen reader support

### Visual Design
- Clear visual hierarchy
- Sufficient color contrast
- Readable typography

## Performance

### Lazy Loading
- Components loaded on demand
- Efficient state updates
- Minimal re-rendering

### Memory Management
- Immutable state prevents memory leaks
- Efficient component lifecycle
- Garbage collection optimization

## Testing

### Component Testing
- Unit tests for individual components
- Integration tests for tab functionality
- User interaction simulation

### Visual Testing
- Screenshot comparison
- Layout validation
- Cross-browser compatibility

## Best Practices

### Component Design
- Keep components small and focused
- Use immutable props
- Implement clear interfaces
- Provide comprehensive documentation

### State Management
- Use immutable data structures
- Minimize state complexity
- Implement clear update patterns
- Validate all state changes

### Event Handling
- Use pure functions for handlers
- Implement proper error handling
- Provide user feedback
- Log important events 