# Development Test Scripts

**⚠️ WARNING:** These scripts contain hardcoded paths specific to the original developer's setup.

## Files with Hardcoded Paths

The following test scripts have `sys.path.insert(0, '/home/zirteq/workspace/forge-neo')`:

- `test_camera_path_fix.py`
- `test_camera_roll_generation.py`
- `test_visualization_pipeline.py`
- `test_viz_parsing.py`
- `test_wormtrail_detailed.py`
- `test_wormtrail_diagnostic.py`
- `test_wormtrail_fix.py`
- `test_wormtrail_visualization.py`

## Usage

**If you are NOT the original developer (zirteq):**

1. **Do NOT run these scripts as-is** - they will fail or produce unexpected results
2. Update the hardcoded path to your Forge Neo installation:
   ```python
   # Change this line:
   sys.path.insert(0, '/home/zirteq/workspace/forge-neo')

   # To your path:
   sys.path.insert(0, '/your/path/to/forge-neo')
   ```

## Better Alternative

Instead of hardcoding paths, use dynamic path resolution:

```python
import os
import sys

# Get Forge root dynamically (assuming script is in extensions/sd-forge-deforum/dev-tools/)
forge_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, forge_root)
```

## Status

These are **development/debugging scripts**, not production code. They are:
- Used for testing specific features during development
- Not required for normal Deforum operation
- May be outdated or broken

**Production code in `deforum/` and `scripts/` directories should NEVER contain hardcoded user paths.**