# Deforum Coding Guide

**Purpose:** This guide establishes coding standards and best practices for the sd-forge-deforum extension.

**Status:** ✅ **Phase 2 Refactoring COMPLETE** (October 2025)
- All code migrated from `scripts/deforum_helpers/` to `deforum/` package
- Clean package structure established
- Functional programming patterns applied throughout

**Scope:** These standards apply to all new code and refactoring work. They prioritize:
- **Functional programming patterns** - Pure functions, immutability, composition
- **Code quality** - Type hints, documentation, complexity limits
- **Testability** - Small, testable units with comprehensive coverage

See also: [CONTRIBUTING.md](CONTRIBUTING.md) for contribution workflow and [CLAUDE.md](CLAUDE.md) for AI assistant instructions.

## Architecture and Package Structure

### Directory Structure

Following WebUI extension conventions and modern Python packaging standards:

```
extensions/sd-forge-deforum/
├── deforum/                    # ✅ Clean library package (modern Python standard)
│   ├── __init__.py
│   ├── api/                    # ✅ REST API module
│   │   ├── api.py              # API endpoints
│   │   ├── models.py           # API data models
│   │   └── __init__.py
│   ├── config/                 # ✅ Configuration and settings
│   │   ├── args.py             # Argument definitions
│   │   ├── defaults.py         # Default values
│   │   ├── settings.py         # Settings management
│   │   └── default_settings.txt  # Default configuration template
│   ├── utils/                  # ✅ Pure utility functions (organized)
│   │   ├── generation/         # Seeds, noise, prompts
│   │   ├── image/              # Processing, geometry, transforms, color
│   │   ├── media/              # Depth, optical flow, subtitles
│   │   ├── parsing/            # Schedules, expressions, strings
│   │   ├── filesystem/         # Paths, files, URLs
│   │   ├── math/               # Core math, interpolation, resolution
│   │   ├── system/             # Logging, HTTP, opts_overrider
│   │   ├── camera/             # Camera analysis
│   │   ├── conversion/         # Type conversion, formats, hashing
│   │   ├── ui/                 # Progress bars, console
│   │   ├── validation/         # Validators, deprecation
│   │   ├── functional.py       # Functional programming helpers
│   │   └── general.py          # General utilities
│   ├── core/                   # ✅ Core business logic
│   │   ├── keyframes.py        # Keyframe scheduling
│   │   ├── prompts.py          # Prompt scheduling
│   │   ├── seeds.py            # Seed iteration
│   │   └── masking/            # Masking subsystem
│   ├── rendering/              # ✅ Rendering pipeline
│   │   ├── core.py             # Main render loop (THE render core)
│   │   ├── img_2_img_tubes.py  # Functional img2img pipeline
│   │   ├── wan_flux.py         # Flux/Wan hybrid mode
│   │   ├── noise.py            # Noise application
│   │   ├── options.py          # Rendering options
│   │   ├── helpers/            # Helper modules (organized)
│   │   └── data/               # Render data structures
│   ├── integrations/           # ✅ External integrations
│   │   ├── wan/                # Wan AI video integration
│   │   ├── parseq/             # Parseq keyframe integration
│   │   ├── flux_controlnet/    # Flux ControlNet integration
│   │   └── external_repos/     # Third-party libraries
│   │       ├── clipseg/        # CLIPSeg (word masking)
│   │       ├── rife/           # RIFE (frame interpolation)
│   │       ├── film_interpolation/  # FILM (frame interpolation)
│   │       └── wan2.1/         # Wan 2.1 model repository
│   ├── media/                  # ✅ Media I/O
│   ├── animation/              # ✅ Animation logic
│   ├── depth/                  # ✅ Depth estimation
│   ├── orchestration/          # ✅ High-level orchestration
│   └── ui/                     # ✅ UI components
├── scripts/                    # ✅ Forge integration layer ONLY
│   ├── deforum.py              # Main WebUI script (Forge entry point)
│   └── deforum_extend_paths.py  # sys.path setup
├── tests/                      # ✅ Test suite
├── preload.py                  # ✅ Extension preload hook
├── CODING_GUIDE.md             # This document
└── README.md                   # User documentation
```

### Design Rationale

**Why `deforum/` at extension root?**
1. **WebUI Convention**: Scripts in `scripts/` are discovered by WebUI
2. **Modern Python Standard**: `src/` layout or package-at-root per Python Packaging Guide
3. **Forge Pattern**: Matches ControlNet's `lib_controlnet/` pattern
4. **Clean Separation**: Distinguishes new clean code from legacy `scripts/deforum_helpers/`

**Migration Status: ✅ COMPLETE**
- All code migrated from `scripts/deforum_helpers/` to `deforum/` package
- Utils organized into logical subdirectories
- Rendering helpers consolidated
- "Experimental" terminology removed

### Package Organization

**`deforum/utils/`** - Pure utility functions (organized into subdirectories)
- **generation/**: Seeds, noise, prompts
- **image/**: Image processing, geometry, transforms, color
- **media/**: Depth, optical flow, subtitles
- **parsing/**: Schedules, expressions, strings
- **filesystem/**: Paths, files, URLs
- **math/**: Core math, interpolation, resolution
- **system/**: Logging, HTTP, opts_overrider, startup banner
- **camera/**: Camera analysis
- **conversion/**: Type conversion, formats, hashing
- **ui/**: Progress bars, console output
- **validation/**: Validators, deprecation warnings

**`deforum/core/`** - Core business logic
- Keyframe scheduling, prompt handling, seed iteration
- Masking subsystem for selective generation

**`deforum/rendering/`** - Rendering pipeline
- **core.py**: Main render loop (frame iteration, transformations, video output)
- **img_2_img_tubes.py**: Functional img2img pipeline
- **wan_flux.py**: Flux + Wan hybrid mode
- **helpers/**: Helper modules (depth, filename, flux_controlnet, image, memory, subtitle, turbo, webui)
- **data/**: Data structures (RenderData, DiffusionFrame, etc.)

**`scripts/`** - Forge integration layer ONLY
- **deforum.py**: Main WebUI script (Forge entry point)
- **deforum_extend_paths.py**: sys.path setup

## Functional Programming Principles (Python-Adapted)

### 1. Prefer Expressions Over Statements
- Use ternary operators, logical operators, and comprehensions instead of verbose if/else blocks
- Use list/dict/set comprehensions over imperative loops where readable

**Bad:**
```python
if condition:
    result = value_a
else:
    result = value_b
```

**Good:**
```python
result = value_a if condition else value_b
```

### 2. Small Pure Functions
- **Maximum 20 lines per function**
- **Single responsibility** - one function does one thing well
- **Side-effect free** where possible - same input always gives same output
- Return values, don't modify arguments

**Bad:**
```python
def process_data(data):
    # 50 lines of mixed logic
    data['modified'] = True  # Side effect!
    return data
```

**Good:**
```python
def calculate_result(data: dict) -> int:
    """Pure calculation - no side effects."""
    return data['a'] + data['b']

def create_modified_data(data: dict) -> dict:
    """Returns new dict, doesn't modify input."""
    return {**data, 'modified': True}
```

### 3. Static Context for Pure Functions
- Group related pure functions in utility modules
- Use module-level functions, not classes, for stateless utilities
- Organize in: `deforum/utils/` package (e.g., `deforum/utils/seed_utils.py`, `deforum/utils/transform_utils.py`)

### 4. Avoid Void Functions
- Functions should return values
- Separate side effects (I/O, logging, state changes) from pure logic
- Use verbs for side-effect functions: `save_image()`, `log_error()`
- Use nouns for pure functions: `calculate_aspect_ratio()`, `get_next_frame()`

### 5. Use Functional List Operations
- Prefer `map()`, `filter()`, `reduce()` and comprehensions over imperative loops
- Use `itertools` and `functools` for advanced functional patterns

**Bad:**
```python
results = []
for item in items:
    if item > 0:
        results.append(item * 2)
```

**Good:**
```python
results = [item * 2 for item in items if item > 0]
# Or for complex transforms:
results = list(map(lambda x: x * 2, filter(lambda x: x > 0, items)))
```

### 6. Minimize and Group State
- Keep mutable state isolated and well-contained
- Use dataclasses/named tuples for immutable data structures
- Make state explicit in function signatures

### 7. No Magic Numbers
- Extract all magic numbers to named constants
- Use configuration objects or dataclasses for grouped constants
- Put constants at module top or in dedicated `constants.py`

**Bad:**
```python
if width > 1920:
    scale = 0.5
```

**Good:**
```python
MAX_WIDTH = 1920
LARGE_IMAGE_SCALE_FACTOR = 0.5

if width > MAX_WIDTH:
    scale = LARGE_IMAGE_SCALE_FACTOR
```

### 8. Immutable by Default
- Use tuples over lists when data won't change
- Use `dataclass(frozen=True)` for immutable data structures
- Return new objects instead of modifying existing ones

### 9. Function Composition
- Design small functions that can be composed together
- Use `functools.partial` and `functools.reduce` for composition
- Think in pipelines: data → transform1 → transform2 → result

## Python-Specific Standards

### 1. Complexity Limit (STRICT REQUIREMENT)
**All functions MUST have McCabe complexity ≤ 10**

- Use helper methods to break down complex logic
- Extract nested loops and conditionals into separate functions
- Aim for single responsibility per function
- Run `radon cc` to measure complexity

**Tools:**
```bash
pip install radon
radon cc scripts/deforum_helpers/ -a -nc
```

### 2. Type Hints (STRICT REQUIREMENT)
**Complete type annotations required on ALL functions**

```python
from typing import List, Dict, Optional, Tuple, Union

def process_frame(
    frame_number: int,
    width: int,
    height: int,
    settings: Dict[str, any]
) -> Tuple[np.ndarray, bool]:
    """Process a single frame with given settings.

    Args:
        frame_number: Frame index (0-based)
        width: Frame width in pixels
        height: Frame height in pixels
        settings: Configuration dictionary

    Returns:
        Tuple of (processed_frame, success_flag)
    """
    ...
```

**Use modern type hints (Python 3.10+):**
```python
# Prefer these over typing module when possible
def get_frames(count: int) -> list[dict]:  # Not List[Dict]
    return [{"id": i} for i in range(count)]

def optional_value(x: int | None) -> str:  # Not Optional[int]
    return str(x) if x is not None else "none"
```

### 3. Error Handling (STRICT REQUIREMENT)
**Comprehensive try-catch blocks with graceful fallbacks**

```python
def load_image(path: str) -> Image | None:
    """Load image with error handling.

    Returns None on failure rather than crashing.
    """
    try:
        return Image.open(path)
    except FileNotFoundError:
        logger.error(f"Image not found: {path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load image {path}: {e}")
        return None
```

- Catch specific exceptions where possible
- Log errors with context
- Provide user-friendly error messages
- Never use bare `except:` - use `except Exception:` at minimum

### 4. Documentation (PRAGMATIC APPROACH)
**Keep it minimal and meaningful**

**Three levels of documentation:**

**Level 1: Simple/Internal Functions** - Type hints + one-liner or nothing
```python
def calculate_tween_weight(prev_keyframe: int, next_keyframe: int, current_frame: int) -> float:
    """Linear interpolation weight between keyframes."""
    if not prev_keyframe <= current_frame <= next_keyframe:
        raise ValueError(f"Frame {current_frame} not between {prev_keyframe} and {next_keyframe}")
    return (current_frame - prev_keyframe) / (next_keyframe - prev_keyframe)
```

**Level 2: Public API / Complex Logic** - Full docstring when needed
```python
def process_frame_distribution(
    data: RenderData,
    distribution: KeyFrameDistribution,
    start_index: int
) -> list[DiffusionFrame]:
    """Calculate frame indices using keyframe distribution algorithm.

    Args:
        data: Central rendering state
        distribution: Distribution mode (OFF, KEYFRAMES_ONLY, ADDITIVE, REDISTRIBUTED)
        start_index: Starting frame index

    Returns:
        List of DiffusionFrame objects with calculated indices
    """
    # Complex logic here
    pass
```

**Level 3: REST API / External Interfaces** - Full documentation
```python
def create_animation(
    animation_mode: str,
    max_frames: int,
    prompts: dict[int, str],
    **kwargs
) -> dict:
    """Create a Deforum animation with given parameters.

    Args:
        animation_mode: Animation mode ('3D', 'Flux/Wan', 'Interpolation')
        max_frames: Total number of frames to generate
        prompts: Dict mapping frame numbers to prompt strings
        **kwargs: Additional animation parameters

    Returns:
        Dict containing:
            - status: "success" or "error"
            - output_path: Path to generated video
            - frame_count: Number of frames generated

    Raises:
        ValueError: If animation_mode is invalid
        RuntimeError: If generation fails

    Example:
        >>> create_animation("3D", 120, {0: "mountain", 60: "valley"})
        {"status": "success", "output_path": "...", "frame_count": 120}
    """
    pass
```

**When to skip docstrings entirely:**
- Function name + type hints make it obvious: `def lerp(a: float, b: float, t: float) -> float`
- Getter/setter with clear names: `def get_frame_count(data: RenderData) -> int`
- Simple wrappers: `def is_3d_mode(mode: str) -> bool`

### 5. Code Style (STRICT REQUIREMENT)

**Copyright headers:**
- **DO NOT include copyright headers in refactored files**
- Use the central `LICENSE` file at repository root (AGPL-3.0)
- **Reason:** Copyright headers waste LLM context (~15 lines × 150 files = 2250 lines)
- Only the LICENSE file needs to declare copyright and terms

**Black formatting, flake8 linting (must pass)**

```bash
# Format code
black deforum/ scripts/deforum_helpers/ tests/ --line-length 100

# Check formatting (don't modify)
black deforum/ scripts/deforum_helpers/ tests/ --line-length 100 --check

# Lint code
flake8 deforum/ scripts/deforum_helpers/ tests/ --max-line-length 100

# Type checking
mypy deforum/ scripts/deforum_helpers/ --strict
```

**Configuration (`pyproject.toml`):**
```toml
[tool.black]
line-length = 100
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

## Code Quality Standards

### 0. Logging Best Practices

**Use the Centralized Logger:**
All logging must use the centralized logger system from `deforum.utils.system.logging`:

```python
from deforum.utils.system.logging import get_logger

# Safe at module level - lazy proxy pattern
logger = get_logger()

# Use appropriate log levels
logger.debug("Detailed diagnostic information")
logger.info("Normal operation progress")
logger.warning("Warning about potential issues")
logger.error("Error occurred but recoverable")
logger.critical("Critical error, cannot continue")
```

**Never use `print()` directly except:**
- ✅ Ctrl+C interrupt handling (immediate console output required)
- ✅ ASCII art banners (startup_banner.py, dashboard.py)
- ✅ Logger implementation itself (utils/system/logging/)
- ❌ **All other cases must use logger**

**Lazy Logger Pattern:**
Module-level `logger = get_logger()` is safe - it returns a lazy proxy that defers initialization until first method call. This solves `opts` timing issues during imports.

**Theme-Aware Output:**
The logger respects the user's console theme setting (slopcore, classic, simple) and emoji preferences. Use the logger for all user-facing output.

### 1. No Duplicate Code (DRY Principle)
- Extract common functionality into shared utilities
- If you copy-paste, you must refactor into a shared function
- Maximum 3 lines of similar code before extraction required

### 2. No Dead Code
- Remove unused functions, variables, and imports
- Run `vulture` to find dead code: `vulture scripts/deforum_helpers/`
- Comment out during development, delete before commit

### 3. Pure Functions First
- Separate data transformation from side effects
- Pure functions go in `utils/` modules
- Side effects go in orchestration/runner modules

### 4. Immutable Data Patterns
```python
# Bad - mutates input
def add_field(data: dict) -> dict:
    data['new_field'] = 'value'
    return data

# Good - returns new dict
def add_field(data: dict) -> dict:
    return {**data, 'new_field': 'value'}

# Best - use dataclasses
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class FrameData:
    number: int
    width: int
    height: int

def add_frame_number(data: FrameData, number: int) -> FrameData:
    return replace(data, number=number)
```

### 5. Explicit Dependencies
Function parameters should clearly show what data is needed:

```python
# Bad - unclear what's needed
def process(config):
    width = config['video']['resolution']['width']
    # ...

# Good - explicit parameters
def process(width: int, height: int, fps: int):
    # ...
```

### 6. Descriptive Naming
- **Functions:** `calculate_aspect_ratio()`, `get_next_keyframe()`, `save_frame()`
- **Variables:** `frame_count`, `is_keyframe`, `tween_weight`
- **Constants:** `MAX_RESOLUTION`, `DEFAULT_FPS`
- **No abbreviations** unless widely known (fps, api, url are OK)

### 7. Low Cyclomatic Complexity
**Maximum complexity: 10** (enforced by radon)

- Use guard clauses to reduce nesting
- Extract complex conditions into named variables
- Break complex functions into smaller helpers

```python
# Bad - high complexity
def process(value):
    if value:
        if value > 0:
            if value < 100:
                return value * 2
            else:
                return 100
        else:
            return 0
    else:
        return None

# Good - guard clauses
def process(value: int | None) -> int | None:
    if value is None:
        return None
    if value <= 0:
        return 0
    if value >= 100:
        return 100
    return value * 2
```

### 8. Single Responsibility
Each function should do one thing well:

```python
# Bad - multiple responsibilities
def load_and_process_and_save_image(path):
    img = load(path)
    processed = process(img)
    save(processed)

# Good - separate responsibilities
def load_image(path: str) -> Image: ...
def process_image(img: Image) -> Image: ...
def save_image(img: Image, path: str) -> None: ...

# Compose in orchestrator
def run_image_pipeline(input_path: str, output_path: str) -> None:
    img = load_image(input_path)
    processed = process_image(img)
    save_image(processed, output_path)
```

### 9. Early Returns (Guard Clauses)
Use early returns to reduce nesting:

```python
# Bad - nested
def get_value(data):
    if data:
        if 'key' in data:
            if data['key']:
                return data['key']
            else:
                return 'default'
        else:
            return 'default'
    else:
        return 'default'

# Good - guard clauses
def get_value(data: dict | None) -> str:
    if not data:
        return 'default'
    if 'key' not in data:
        return 'default'
    return data['key'] or 'default'
```

## Refactoring Priorities (Execution Order)

When refactoring a module, follow this order:

1. **Add type hints** to all functions (enables better tooling)
2. **Add minimal docstrings** (only for public APIs or non-obvious logic)
3. **Eliminate code duplication** through utility extraction
4. **Extract magic numbers** to named constants
5. **Separate pure calculations** from side effects
6. **Convert imperative loops** to functional operations (where readable)
7. **Group related pure functions** into utility modules
8. **Break down complex functions** (complexity > 10)
9. **Minimize mutable state** and make it explicit
10. **Add error handling** and input validation
11. **Remove dead code** and unused imports
12. **Format with Black** and **lint with flake8**
13. **Write unit tests** for extracted pure functions

**Documentation priority:**
- REST API endpoints: Full docstrings (Args, Returns, Raises, Examples)
- Public interfaces: Full docstrings when behavior is complex
- Internal functions: Type hints only, or one-liner if needed
- Obvious functions: No docstring (name + types are documentation)

## Tools and Commands

### Code Quality Checks
```bash
# Format code
black scripts/deforum_helpers/ tests/ --line-length 100

# Lint
flake8 scripts/deforum_helpers/ tests/ --max-line-length 100

# Find dead code
vulture scripts/deforum_helpers/

# Check complexity
radon cc scripts/deforum_helpers/ -a -nc  # Show only complex functions
radon cc scripts/deforum_helpers/ -a      # Show all

# Type checking
mypy scripts/deforum_helpers/ --strict
```

### Testing

**Test Organization:**
- `tests/unit/` - Fast, isolated tests (no GPU/Forge backend)
- `tests/integration/` - Full-stack tests (requires Forge + GPU)

**Run tests:**
```bash
# Unit tests only (fast, CI-friendly)
pytest tests/unit/ -v

# Unit tests with coverage
pytest tests/unit/ --cov=deforum --cov=scripts --cov-report=html

# Integration tests (requires server)
pytest tests/integration/ --start-server

# Require minimum coverage threshold
pytest tests/unit/ --cov-fail-under=70
```

**Testing Best Practices:**
- Write unit tests for all pure functions in `deforum/utils/`
- Integration tests for full rendering pipelines
- Separate unit/integration to keep CI fast
- Use `@pytest.mark.integration` for integration tests
- Mock external dependencies in unit tests (Forge, GPU, filesystem)
- Target 70%+ coverage for utility modules
- See [tests/README.md](tests/README.md) for comprehensive testing guide

**Recent Coverage Achievements:**
- `deforum/utils/general.py`: 5% → 55%
- `deforum/utils/system/logging/log.py`: 39% → 71%
- Lazy logger pattern comprehensively tested

## Examples of Good Refactoring

### Before: Imperative, Complex, No Types
```python
def process_frames(frames, settings):
    results = []
    for i in range(len(frames)):
        frame = frames[i]
        if frame is not None:
            if settings['mode'] == '3D':
                if frame['width'] > 1920:
                    scale = 0.5
                else:
                    scale = 1.0
                processed = {'data': frame['data'] * scale, 'id': i}
                results.append(processed)
    return results
```

### After: Functional, Typed, Minimal Docs
```python
from dataclasses import dataclass

# Constants
MAX_WIDTH_BEFORE_SCALING = 1920
LARGE_FRAME_SCALE_FACTOR = 0.5
NORMAL_FRAME_SCALE_FACTOR = 1.0

@dataclass(frozen=True)
class Frame:
    data: np.ndarray
    width: int
    height: int

@dataclass(frozen=True)
class ProcessedFrame:
    data: np.ndarray
    id: int

def calculate_scale_factor(width: int) -> float:
    # Large frames (>1920px) scaled down to save memory
    return (
        LARGE_FRAME_SCALE_FACTOR
        if width > MAX_WIDTH_BEFORE_SCALING
        else NORMAL_FRAME_SCALE_FACTOR
    )

def process_single_frame(frame: Frame, frame_id: int) -> ProcessedFrame:
    scale = calculate_scale_factor(frame.width)
    return ProcessedFrame(data=frame.data * scale, id=frame_id)

def process_frames(frames: list[Frame | None], mode: str) -> list[ProcessedFrame]:
    """Process frames in 3D mode with scaling (skips None frames)."""
    if mode != '3D':
        return []
    return [
        process_single_frame(frame, idx)
        for idx, frame in enumerate(frames)
        if frame is not None
    ]
```

## Commit Message Format

When committing refactored code:

```
refactor: <module_name> - apply functional patterns

Changes:
- Extract pure functions: calculate_X(), transform_Y()
- Add type hints to all functions
- Reduce complexity from 15 to 8 (calculate_frame)
- Move constants to module top
- Add unit tests for pure functions

Complexity: 15 → 8
Coverage: 0% → 75%
```

## Review Checklist

Before committing refactored code, verify:

- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] No function has complexity > 10 (run `radon cc`)
- [ ] No magic numbers (all extracted to constants)
- [ ] No code duplication
- [ ] Black formatting applied
- [ ] Flake8 passes with no errors
- [ ] Unit tests written for pure functions
- [ ] Coverage report shows improvement
- [ ] No dead code or unused imports

## Anti-Patterns to Avoid

### ❌ Don't Do This:
```python
# Mixing side effects with logic
def calculate_and_save_result(data):
    result = complex_calculation(data)
    with open('output.txt', 'w') as f:  # Side effect!
        f.write(str(result))
    return result

# Modifying inputs
def process(items: list):
    items.append('new')  # Mutates input!
    return items

# Unclear function purpose
def do_stuff(x, y, z):  # What does this do?
    ...

# Deep nesting
def process(x):
    if x:
        if x > 0:
            if x < 100:
                if x % 2 == 0:
                    return x
```

### ✅ Do This Instead:
```python
# Separate calculation from I/O
def calculate_result(data: dict) -> int:
    """Pure calculation."""
    return complex_calculation(data)

def save_result(result: int, path: str) -> None:
    """I/O side effect."""
    with open(path, 'w') as f:
        f.write(str(result))

# Return new list
def process(items: list) -> list:
    """Returns new list, doesn't modify input."""
    return [*items, 'new']

# Clear function names
def calculate_aspect_ratio(width: int, height: int) -> float:
    """Computes width/height ratio."""
    return width / height

# Guard clauses
def process(x: int | None) -> int | None:
    if x is None:
        return None
    if x <= 0:
        return None
    if x >= 100:
        return None
    if x % 2 != 0:
        return None
    return x
```
