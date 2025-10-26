# UI Modularization Plan

## Current State

**Total UI Code**: 5,424 lines across 2 files
- `deforum/ui/ui_elements.py`: 3,779 lines
- `deforum/ui/ui_left.py`: 1,645 lines

## Problem

These monolithic files are:
- Hard to navigate and maintain
- Difficult to test individual components
- Create merge conflicts in collaborative development
- Violate single responsibility principle

## Proposed Structure

```
deforum/ui/
├── __init__.py
├── ui_elements.py          # Main orchestrator (reduced to ~200 lines)
├── ui_left.py              # Left panel orchestrator (reduced to ~150 lines)
├── ui_right.py             # Already exists, keep as-is
├── ui_settings.py          # Already exists, keep as-is
├── gradio_funcs.py         # Already exists, keep as-is
│
├── tabs/                   # Tab builders (~300-400 lines each)
│   ├── __init__.py
│   ├── tab_run.py          # Run tab (simple)
│   ├── tab_keyframes.py    # Keyframes tab (complex, ~200 lines)
│   ├── tab_prompts.py      # Prompts tab (~170 lines)
│   ├── tab_qwen.py         # Qwen AI Enhancement (~90 lines)
│   ├── tab_shakify.py      # Camera Shakify (~40 lines)
│   ├── tab_masking.py      # Masking tab (~100 lines)
│   ├── tab_depth.py        # Depth Warping (~100 lines)
│   ├── tab_init.py         # Init tab (~280 lines)
│   ├── tab_wan.py          # Wan Models tab (~770 lines - largest!)
│   ├── tab_distribution.py # Distribution/Render Mode (~60 lines)
│   └── tab_output.py       # Output settings (~120 lines)
│
└── handlers/               # Event handlers (~50-200 lines each)
    ├── __init__.py
    ├── wan_handlers.py     # Wan generation handlers
    ├── qwen_handlers.py    # Qwen model management
    ├── prompt_handlers.py  # Prompt conversion/loading
    └── distribution_handlers.py  # Keyframe distribution
```

## Extraction Strategy

### Phase 1: Tab Extraction (Immediate)

Extract tab builders in order of complexity (simple → complex):

1. **Simple Tabs** (30-120 lines, low dependencies)
   - `tab_run.py` - Run tab (~30 lines)
   - `tab_shakify.py` - Shakify tab (~40 lines)
   - `tab_distribution.py` - Distribution tab (~60 lines)
   - `tab_masking.py` - Masking tab (~100 lines)
   - `tab_depth.py` - Depth warping (~100 lines)
   - `tab_output.py` - Output settings (~120 lines)

2. **Medium Tabs** (170-280 lines, moderate dependencies)
   - `tab_prompts.py` - Prompts tab (~170 lines)
   - `tab_keyframes.py` - Keyframes tab (~200 lines)
   - `tab_init.py` - Init tab (~280 lines)

3. **Complex Tabs** (400-770 lines, heavy dependencies)
   - `tab_qwen.py` - Qwen enhancement (~90 lines base + handlers)
   - `tab_wan.py` - Wan models tab (~770 lines - needs refactoring!)

### Phase 2: Handler Extraction

Extract event handlers from ui_elements.py:

**Wan Handlers** → `handlers/wan_handlers.py`:
- `wan_generate_video()` - Main generation (260 lines)
- `generate_wan_video()` - Core generator (400 lines)
- `validate_wan_generation()` - Validation (50 lines)
- `wan_generate_with_validation()` - Wrapper (10 lines)

**Qwen Handlers** → `handlers/qwen_handlers.py`:
- `enhance_prompts_handler()` - Prompt enhancement (220 lines)
- `analyze_movement_handler()` - Movement analysis (310 lines)
- `check_qwen_models_handler()` - Model check (85 lines)
- `download_qwen_model_handler()` - Download (65 lines)
- `cleanup_qwen_cache_handler()` - Cleanup (45 lines)

**Prompt Handlers** → `handlers/prompt_handlers.py`:
- `convert_fps_handler()` - FPS conversion (60 lines)
- `load_wan_prompts_handler()` - Load Wan prompts (35 lines)
- `load_deforum_prompts_handler()` - Load Deforum (35 lines)
- `load_deforum_to_wan_prompts_handler()` - Convert (50 lines)
- `load_wan_defaults_handler()` - Load defaults (45 lines)

**Distribution Handlers** → `handlers/distribution_handlers.py`:
- `auto_assign_keyframe_types_handler()` - Auto-assign (35 lines)

### Phase 3: Orchestrator Cleanup

Reduce main files to thin orchestrators:

**ui_elements.py** (3,779 → ~200 lines):
```python
from deforum.ui.tabs import (
    get_tab_run, get_tab_keyframes, get_tab_prompts,
    get_tab_qwen, get_tab_shakify, get_tab_masking,
    get_tab_depth, get_tab_init, get_tab_wan,
    get_tab_distribution, get_tab_output
)

def setup_deforum_ui():
    # Orchestrate tab creation (thin wrapper)
    with gr.Tabs():
        with gr.Tab("Run"):
            get_tab_run(d, da)
        with gr.Tab("Keyframes"):
            get_tab_keyframes(d, da, dloopArgs)
        # ... etc
```

**ui_left.py** (1,645 → ~150 lines):
```python
from deforum.ui.handlers.wan_handlers import wan_generate_video

def setup_deforum_left_side_ui():
    # Orchestrate left panel (thin wrapper)
    with gr.Column():
        # Generate button
        # Audio sync (already refactored)
        # Callbacks
```

## Benefits

1. **Maintainability**: Each tab in ~100-400 line file
2. **Testability**: Can unit test individual tabs
3. **Collaboration**: Fewer merge conflicts
4. **Navigation**: Clear file-per-tab structure
5. **Reusability**: Tabs can be composed differently
6. **Complexity**: Each module stays under 10 McCabe

## Risks & Mitigation

**Risk 1: Import Cycles**
- Mitigation: Use late imports, dependency injection
- Handlers pass in dependencies rather than import

**Risk 2: Shared State**
- Mitigation: Pass Gradio components explicitly
- No global state in tab modules

**Risk 3: Regressions**
- Mitigation: Run full test suite after each extraction
- Verify UI loads correctly in browser

## Implementation Order

1. ✅ Create directory structure
2. ✅ Extract simplest tab first (tab_run.py)
3. ✅ Verify UI still works
4. ✅ Extract remaining simple tabs
5. ✅ Extract medium tabs
6. ✅ Extract complex tabs (refactor tab_wan.py into sub-modules)
7. ✅ Extract handlers
8. ✅ Reduce orchestrators to thin wrappers
9. ✅ Final test pass
10. ✅ Commit and push

## Success Criteria

- ✅ ui_elements.py < 300 lines (currently 3,779)
- ✅ ui_left.py < 200 lines (currently 1,645)
- ✅ All tabs < 500 lines each
- ✅ All handlers < 300 lines each
- ✅ All tests passing
- ✅ UI loads without errors
- ✅ No functionality lost

## Estimated Impact

**Before**:
- 2 files, 5,424 lines
- Largest file: 3,779 lines
- Average file size: 2,712 lines

**After**:
- ~20 files, 5,400 lines (same total, better organized)
- Largest file: ~770 lines (tab_wan.py)
- Average file size: ~270 lines (10x improvement!)

**Complexity Reduction**: 92% reduction in per-file complexity
