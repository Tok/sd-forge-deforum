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

## Expected Issues

### Likely Compatible
- Our extension is API-based, so should work with any Forge fork
- Tuning system only uses standard diffusers/torch APIs
- Most features don't depend on Forge internals

### Potential Issues
1. **Different Gradio version** - reForge may use Gradio 5.x (check compatibility)
2. **Backend differences** - reForge removed medvram/lowvram flags (might affect memory management)
3. **Extension loading order** - Different hook timing could affect install.py fix
4. **Dependency versions** - May have different transformers/accelerate/peft versions

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
