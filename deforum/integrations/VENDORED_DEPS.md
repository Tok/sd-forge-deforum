# Vendored Dependencies

## Problem

Forge specifies `huggingface-hub==0.26.2` in `requirements_versions.txt`, but our extension requires `diffusers` (for Wan video support) which needs `huggingface-hub>=0.34.0,<2.0`. This creates a dependency conflict.

## Quick Fix (If Forge Won't Start)

If you see `ImportError: cannot import name 'HfFolder'` or `'DDUFEntry' from 'huggingface_hub'` when starting Forge:

```bash
cd /path/to/stable-diffusion-webui-forge
./venv/bin/pip install 'huggingface-hub==0.36.0'
```

Then restart Forge normally. This fixes compatibility between:
- **Gradio 4.40.0** - Needs HfFolder (removed in 1.0.0+)
- **Forge** - Needs DDUFEntry (added in 0.27.0)
- **diffusers** - Needs >=0.34.0

## Solution

We vendor a compatible version of `huggingface-hub` (0.36.0) in a local `.vendored/` directory that only gets used when loading Wan diffusers pipelines. This approach:

1. ✅ **Doesn't modify Forge's venv** - No "works-on-my-machine" issues
2. ✅ **Isolated to Wan features** - Only affects diffusers imports, not other extensions
3. ✅ **Auto-installs on first use** - No manual setup required
4. ✅ **Compatible with Gradio 4.40.0** - Uses 0.36.0 which works with both diffusers and Gradio

## Implementation

### vendored_hf_hub.py

Located at `deforum/integrations/vendored_hf_hub.py`, this module provides:

```python
from deforum.integrations.vendored_hf_hub import use_vendored_hf_hub

# Call before importing diffusers
use_vendored_hf_hub()
import diffusers
```

**How it works:**
1. Installs `huggingface-hub==0.36.0` to `.vendored/` using pip `--target` and `--no-deps`
2. Prepends `.vendored/` to `sys.path` so imports use the vendored version
3. Creates a marker file to avoid reinstalling on every launch

### Integration Point

Used in `deforum/integrations/wan/wan_simple_integration.py:346`:

```python
# Use vendored huggingface-hub for diffusers compatibility
from deforum.integrations.vendored_hf_hub import use_vendored_hf_hub
use_vendored_hf_hub()

# Now safe to import diffusers
from diffusers import WanPipeline, AutoencoderKLWan
```

## Why 0.36.0?

- **Lower bound:** diffusers requires `>=0.34.0`
- **Upper bound:** Gradio 4.40.0 (Forge's version) needs `<1.0` (1.0.0+ removed `HfFolder` class)
- **Choice:** 0.36.0 is the last stable version before 1.0 breaking changes

## Version Matrix

| Component | Required Version | Actual Version | Status |
|-----------|------------------|----------------|--------|
| Forge huggingface-hub | `==0.26.2` | 0.26.2 | ✅ Unchanged |
| diffusers requirement | `>=0.34.0,<2.0` | 0.36.0 (vendored) | ✅ Satisfied |
| Gradio 4.40.0 requirement | `<1.0` | 0.36.0 (vendored) | ✅ Compatible |

## Directory Structure

```
extensions/sd-forge-deforum/
├── .vendored/                        # Git-ignored
│   ├── huggingface_hub/              # Vendored package
│   │   ├── __init__.py
│   │   ├── ._installed_0.36.0        # Marker file
│   │   └── ...
│   └── (other potential future vendored packages)
├── deforum/
│   └── integrations/
│       ├── vendored_hf_hub.py        # Vendoring logic
│       └── wan/
│           └── wan_simple_integration.py  # Uses vendored version
└── .gitignore                        # Excludes .vendored/
```

## Alternatives Considered

### 1. Update Forge's huggingface-hub ❌
**Problem:** Creates "works-on-my-machine" issue, breaks other users' installations

### 2. Downgrade diffusers to 0.26.3 ❌
**Problem:** Wan pipelines (`WanImageToVideoPipeline`) only exist in diffusers main branch

### 3. Fork diffusers and patch version requirements ❌
**Problem:** Maintenance nightmare, breaks automatic updates

### 4. Use environment variables to override pip requirements ❌
**Problem:** Fragile, affects entire venv, hard to document

### 5. Vendor entire diffusers package ❌
**Problem:** Huge (~200MB), complex dependencies, duplicates torch/transformers

### 6. Vendor huggingface-hub (CHOSEN) ✅
**Advantages:**
- Small package (~10MB)
- No dependencies (uses Forge's existing requests/filelock/etc)
- Surgical fix (only affects Wan diffusers imports)
- Auto-installs on first use

## Testing

Verify the vendored installation works:

```bash
cd /path/to/forge
./venv/bin/python -c "
import sys
sys.path.insert(0, 'extensions/sd-forge-deforum')
from deforum.integrations.vendored_hf_hub import use_vendored_hf_hub
use_vendored_hf_hub()
import diffusers
import huggingface_hub
print(f'diffusers: {diffusers.__version__}')
print(f'huggingface_hub: {huggingface_hub.__version__}')
"
```

Expected output:
```
[Deforum] Using vendored huggingface-hub from .../extensions/sd-forge-deforum/.vendored
diffusers: 0.36.0.dev0
huggingface_hub: 0.36.0
```

## Maintenance

### Updating Vendored Version

If a newer huggingface-hub is needed:

1. Update `VENDORED_HF_HUB_VERSION` in `vendored_hf_hub.py`
2. Delete `.vendored/` directory
3. Restart Forge - will auto-install new version

### Monitoring Upstream

Watch for:
- **Forge updates:** If Forge upgrades to `huggingface-hub>=0.34.0`, this vendoring can be removed
- **diffusers updates:** If diffusers version requirement changes, adjust `VENDORED_HF_HUB_VERSION`
- **Gradio updates:** If Forge upgrades Gradio to 5.x, may need different huggingface-hub version
