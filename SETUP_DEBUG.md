# Setup Debugging - Z-Image CUDA Crashes

**Issue:** Z-Image crashes with `CUDA error: unspecified launch failure` at tanh operation
**Hypothesis:** Local setup issue (drivers, PyTorch, venv), not Forge bug

---

## Current Environment

**GPU:** RTX 4070 Ti SUPER (16GB VRAM)
**PyTorch:** 2.9.1+cu128
**CUDA:** Runtime version needs verification
**Driver:** Needs verification

**Crash signature:**
```
at::_ops::tanh::call(at::Tensor const&)
CUDA error: unspecified launch failure
```

---

## Debug Plan

### Step 1: Check Current NVIDIA Driver

```bash
nvidia-smi
```

**Look for:**
- Driver version (should be 550+ for CUDA 12.8)
- CUDA version shown
- GPU memory state

**Expected:** Driver 560+ recommended for PyTorch 2.9.1+cu128

### Step 2: Check PyTorch Installation

```bash
cd /home/zirteq/workspace/forge-neo
source venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}')"
```

**Look for:**
- PyTorch version matches (2.9.1+cu128)
- CUDA available = True
- CUDA version = 12.8
- cuDNN version compatibility

### Step 3: Test Basic CUDA Operations

```bash
python -c "import torch; x = torch.randn(100, 100, device='cuda'); y = torch.tanh(x); print('tanh works:', y.shape)"
```

**If this fails:** PyTorch/CUDA installation broken
**If this works:** Issue is model-specific or Forge-specific

### Step 4: Update NVIDIA Drivers (If Needed)

**Ubuntu/Debian:**
```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver (560+ recommended)
sudo ubuntu-drivers autoinstall
# OR specific version
sudo apt install nvidia-driver-560

# Reboot required
sudo reboot
```

**After reboot:**
```bash
nvidia-smi  # Verify new driver
```

### Step 5: Wipe and Rebuild venv

```bash
cd /home/zirteq/workspace/forge-neo

# Backup current venv name (just in case)
mv venv venv.old

# Create fresh venv
python3.12 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.8 (RTX 4070 Ti SUPER supports this)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify install
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"

# Install Forge requirements
pip install -r requirements.txt
```

### Step 6: Install Deforum Requirements

```bash
cd extensions/sd-forge-deforum
pip install -r requirements.txt
```

### Step 7: Test Basic Forge txt2img

**Don't test Z-Image yet - start simpler:**

```bash
cd /home/zirteq/workspace/forge-neo
python webui.py
```

**Test with SDXL first** (smaller, more reliable):
- Model: Any SDXL checkpoint
- Resolution: 512x512
- Steps: 10
- Should generate without crashes

**If SDXL works → Try Z-Image next**

### Step 8: Test Z-Image

**Only if SDXL succeeds:**
- Model: z_image_turbo_bf16.safetensors
- Modules: ae.safetensors, qwen_3_4b.safetensors
- Resolution: Start at 480p (very safe)
- Steps: 20

---

## Common Issues & Fixes

### Issue: Driver Too Old
**Symptom:** `nvidia-smi` shows driver <550
**Fix:** Update to driver 560+

### Issue: PyTorch CUDA Mismatch
**Symptom:** `torch.version.cuda` != driver CUDA version
**Fix:** Reinstall PyTorch matching driver

### Issue: Corrupted venv
**Symptom:** Random import errors, weird crashes
**Fix:** Full venv rebuild (Step 5)

### Issue: Out of VRAM
**Symptom:** Works at 480p, crashes at 720p+
**Fix:** Use `--lowvram` or `--medvram` flag

### Issue: SageAttention Problems
**Symptom:** Crashes during attention operations
**Fix:** Try without SageAttention (uninstall or disable)

---

## Verification Checklist

After completing steps above, verify:

- [ ] `nvidia-smi` shows driver 560+
- [ ] `torch.cuda.is_available()` returns True
- [ ] `torch.version.cuda` shows 12.8
- [ ] Basic tanh operation works on GPU
- [ ] SDXL txt2img generates successfully
- [ ] Z-Image generates at 480p
- [ ] Z-Image generates at 720p (if VRAM allows)

---

## If Still Crashing After All Steps

**Possible causes:**
1. Hardware issue (GPU memory corruption)
2. Z-Image model file corrupted (re-download)
3. Genuine Forge bug (but test on another machine first)
4. PyTorch 2.9.1 regression (try PyTorch 2.5.1)

**Next steps:**
1. Test on different machine (if available)
2. Try older PyTorch version
3. Contact Forge Neo with full environment details
4. Check CUDA compute capability compatibility

---

## Quick Commands Reference

```bash
# Check driver
nvidia-smi

# Check PyTorch/CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

# Test basic CUDA
python -c "import torch; print(torch.tanh(torch.randn(10, 10, device='cuda')))"

# Rebuild venv
rm -rf venv && python3.12 -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Start Forge
python webui.py
```

---

**Start with Step 1 and work through sequentially. Report findings at each step.**
