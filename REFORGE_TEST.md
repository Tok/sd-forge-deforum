# reForge Compatibility Testing

This branch (`test/reforge`) is for testing compatibility with [stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge).

## Why reForge?

- **Original Forge status:** Hibernating/dead (last active development months ago)
- **reForge benefits:**
  - Actively maintained community fork
  - Better resource management (SDXL @ 4GB VRAM, SD1.5 @ 2GB)
  - No manual flags needed (`medvram`, `lowvram`, etc. removed)
  - Ongoing development and bug fixes

## Test Plan

### Phase 1: Installation
1. Clone reForge in parallel directory:
   ```bash
   cd ~/workspace
   git clone https://github.com/Panchovix/stable-diffusion-webui-reForge.git
   cd stable-diffusion-webui-reForge
   ```

2. Install our extension:
   ```bash
   cd extensions
   ln -s ../../stable-diffusion-webui-forge/extensions/sd-forge-deforum
   ```

3. Check if huggingface-hub issue exists in reForge

### Phase 2: Dependency Verification
- [ ] Check reForge's `requirements_versions.txt` for huggingface-hub version
- [ ] Verify if install.py fix is still needed
- [ ] Test if diffusers from git main causes conflicts
- [ ] Document any reForge-specific adjustments needed

### Phase 3: Basic Functionality
- [ ] Start reForge with `--deforum-api` flag
- [ ] Verify extension loads without errors
- [ ] Test simple 3D animation generation
- [ ] Test Flux + Interpolation mode (if Flux works in reForge)
- [ ] Test Wan video generation

### Phase 4: Tuning System
- [ ] Run `./run-tuning-tests.sh --start-server`
- [ ] Verify metrics tests pass (10/10 unit tests)
- [ ] Run color preservation sweep test
- [ ] Verify outputs in `outputs/deforum-tuning/`

## Key Differences Found

### reForge Dependencies (requirements_versions.txt)
```
gradio==3.41.2          # vs Forge 4.40.0 (MAJOR DIFFERENCE!)
huggingface_hub==0.25.0 # vs Forge 0.26.2 (older)
diffusers==0.32.2       # vs our git main requirement
```

### Implications
1. **Gradio 3.41.2** - Old version, likely doesn't have HfFolder import issue
   - Our install.py fix may not be needed!
   - UI differences expected (Gradio 3 vs 4)

2. **huggingface_hub 0.25.0** - Older than Forge's 0.26.2
   - Has HfFolder class (pre-1.0)
   - Missing DDUFEntry (added in 0.27.0) - may cause issues with reForge features

3. **diffusers 0.32.2** - Pinned version
   - Our git main requirement will upgrade it
   - Need to check if 0.32.2 has WanImageToVideoPipeline

## Expected Issues

### Likely Compatible
- ✅ HfFolder issue likely doesn't exist (Gradio 3.41.2 is old)
- ✅ Extension is API-based, should work with both Gradio versions
- ✅ Tuning system only uses standard APIs

### Potential Issues
1. **Gradio version conflict** - Our code may assume Gradio 4 APIs
2. **diffusers upgrade** - Installing git main may break reForge's pinned 0.32.2
3. **huggingface_hub upgrade** - Our 0.36.0 requirement higher than reForge's 0.25.0
4. **UI rendering** - Extension UI built for Gradio 4, may look different in Gradio 3

## Success Criteria

✅ Extension loads without errors
✅ Basic 3D animation works
✅ Flux support (if available in reForge)
✅ Wan video generation works
✅ Tuning system tests pass
✅ No huggingface-hub conflicts

## Rollback Plan

If reForge has critical incompatibilities:
1. Document issues in this file
2. Keep dev branch on original Forge
3. Maintain reforge compatibility branch separately
4. OR: Fix issues and make extension work with both

## Notes

- reForge README recommends other forks for stability (Forge Classic, Forge Neo, ersatzForge)
- May want to test those as well if reForge has issues
- Keep original Forge compatibility - don't break it for reForge support
