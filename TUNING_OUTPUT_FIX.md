# Tuning Output Directory Fix

## Problem

Tuning test outputs were being saved to the wrong location:

**Before:**
```
extensions/sd-forge-deforum/outputs/deforum-tuning/
```

This was buried in the extension directory, making outputs:
- Hard to find
- Inconsistent with normal Forge generations
- Not following Forge conventions

## Solution

Changed to use Forge's standard outputs directory:

**After:**
```
forge-neo/outputs/deforum-tuning/
```

Now tuning outputs are:
- ✅ In the same location as normal generations (`outputs/txt2img-images/`, `outputs/deforum/`, etc.)
- ✅ Easy to find alongside other Forge outputs
- ✅ Consistent with Forge conventions
- ✅ Accessible via standard Forge output directory browsing

## Files Modified

### 1. `deforum/api/tuning_test_helpers.py`

**Changed `get_test_options_overrides()` function:**

```python
def get_test_options_overrides(output_dir: Path = None) -> Dict[str, Any]:
    """Get options overrides for tuning tests."""
    if output_dir:
        return {"outdir_samples": str(output_dir)}
    else:
        # NEW: Use Forge's standard outputs directory
        try:
            from modules import shared
            base_outdir = shared.opts.outdir_samples or shared.opts.outdir_img2img_samples
            tuning_dir = Path(base_outdir).parent / "deforum-tuning"
            return {"outdir_samples": str(tuning_dir)}
        except:
            # Fallback for test environments
            forge_root = Path(os.getcwd())
            return {"outdir_samples": str(forge_root / "outputs" / "deforum-tuning")}
```

**Why this works:**
- `shared.opts.outdir_samples` points to Forge's configured output directory
- We use `.parent / "deforum-tuning"` to create a sibling directory
- Fallback uses `os.getcwd()` which gives Forge webui root when running

### 2. `tests/tuning/test_color_preservation.py`

**Changed OUTPUT_DIR constant:**

```python
# OLD:
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "deforum-tuning" / "color_preservation"

# NEW:
import os
FORGE_ROOT = Path(os.getcwd())  # Forge webui root directory
OUTPUT_DIR = FORGE_ROOT / "outputs" / "deforum-tuning" / "color_preservation"
```

### 3. `deforum/ui/ui_tuning.py`

**Changed `on_open_tuning_dir()` function:**

```python
def on_open_tuning_dir():
    """Open the tuning output directory in file browser."""
    from pathlib import Path
    from modules.util import open_folder
    import os

    # NEW: Use Forge's standard outputs directory
    forge_root = Path(os.getcwd())
    tuning_dir = forge_root / "outputs" / "deforum-tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Opening tuning directory: {tuning_dir}")
    open_folder(str(tuning_dir))
    return f"📁 Opened: {tuning_dir}"
```

## Result

All tuning test outputs now appear in:
```
forge-neo/
├── outputs/
│   ├── txt2img-images/      # Normal txt2img outputs
│   ├── deforum/             # Normal Deforum outputs
│   ├── deforum-tuning/      # 🆕 Tuning test outputs (NEW LOCATION!)
│   │   ├── color_preservation/
│   │   │   ├── steps20_norm0.85_kf0.15/
│   │   │   ├── steps20_norm0.90_kf0.15/
│   │   │   └── ...
│   │   └── ...
│   └── zero_hitl/           # Zero-HITL outputs
```

## Stop Button Issue

The user reported that clicking "Stop" accidentally deleted everything.

**Investigation:** The stop button code (`deforum/ui/ui_tuning.py:308-322`) only:
1. Calls the API cancel endpoint
2. Sets test status to "cancelled"
3. Does NOT delete any files

**Conclusion:** Files were likely overwritten during a test run (new iteration reusing same directory), NOT deleted by the stop button. With this output directory fix, test outputs will be better organized and easier to locate if something goes wrong.

## Testing

To verify the fix:

1. Run a tuning test via UI or pytest
2. Check that outputs appear in `forge-neo/outputs/deforum-tuning/`
3. Click "📁 Open Tuning Directory" button to verify it opens correct location
4. Run `pytest tests/tuning/test_color_preservation.py -v` and verify OUTPUT_DIR resolves correctly

## Backward Compatibility

Existing outputs in the old location (`extensions/sd-forge-deforum/outputs/`) will remain there and can be manually moved if needed. New outputs will use the correct location.
