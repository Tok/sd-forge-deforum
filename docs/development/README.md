# Development Guide

Guide for developers working on Deforum's codebase.

## Architecture Overview

### Functional Programming Approach
Deforum uses modern functional programming principles:
- **Immutable data structures** - All configuration and state objects are immutable
- **Pure functions** - Business logic is side-effect free and predictable  
- **Type safety** - Comprehensive type hints and validation
- **Modular design** - Clear separation of concerns

### Project Structure
```
scripts/deforum_helpers/
├── external_libs/          # External library integrations (RIFE, FILM, etc.)
├── data_models.py         # Immutable data structures
├── schedules_models.py    # Animation scheduling system
├── ui_left.py            # Main UI components
├── generate.py           # Core generation logic
├── args.py               # Argument processing
└── wan/                  # WAN AI integration
```

## Getting Started

### Development Setup
1. Clone the repository
2. Install development dependencies
3. Run the test suite
4. Set up your IDE with type checking

### Code Standards
- **Function size**: Maximum 50 lines per function
- **File size**: Maximum 500 lines per module
- **Type hints**: Required on all public interfaces
- **Documentation**: Complete docstrings for public functions
- **Testing**: 85%+ coverage for new code

## Core Systems

### [Data Models](../api/data-models.md)
- Immutable dataclasses for all configuration
- Validation and type safety
- Factory methods for common patterns

### [Schedule System](../api/schedules.md)
- Animation parameter scheduling
- Keyframe interpolation
- Pure functional approach

### [UI System](../ui/README.md)
- Component-based architecture
- Immutable state management
- Event-driven updates

## Testing

### [Testing Guide](testing.md)
- Unit testing with pytest
- Coverage reporting
- Test data management
- Mocking external dependencies

### Running Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=scripts/deforum_helpers

# Run specific test file
python -m pytest tests/unit/test_data_models.py
```

## Contributing

### [Contributing Guidelines](contributing.md)
- Code review process
- Pull request guidelines
- Issue reporting
- Documentation standards

### Development Workflow
1. Create feature branch
2. Implement changes with tests
3. Ensure all tests pass
4. Submit pull request
5. Address review feedback

## API Reference

### [Core APIs](../api/core.md)
- Main Deforum interfaces
- Generation pipeline
- Configuration management

### [External Libraries](../api/external-libs.md)
- RIFE integration
- FILM integration
- MiDaS depth estimation

## Best Practices

### Code Organization
- Keep functions small and focused
- Use immutable data structures
- Isolate side effects
- Write comprehensive tests

### Performance
- Profile before optimizing
- Use appropriate data structures
- Minimize memory allocations
- Cache expensive computations

### Debugging
- Use type checking tools
- Write descriptive error messages
- Add logging for complex operations
- Test edge cases thoroughly 