# Manual Testing Procedure for Large Animations (30k+ Frames)

## Purpose
Verify that schedule truncation and progress indicators work correctly for large frame counts.

## Prerequisites
- Forge WebUI running
- Deforum extension loaded
- Settings → Deforum → Max Schedule Display Frames = 1000 (default)

## Test Cases

### Test 1: Small Animation (No Truncation)
**Frame Count:** 500 frames
**Expected:**
- ❌ No truncation indicator in schedules
- ❌ No "Schedule Display Truncated" message in status
- ❌ No progress bars (below 1000 frame threshold)

**Steps:**
1. Go to Deforum → Camera Path tab
2. Set max_frames = 500 in Run tab
3. Click "Generate Rotate-Around Preset"
4. Check Translation X schedule textbox
5. Verify schedule does NOT contain `[truncated at frame...]`

---

### Test 2: Boundary Animation (No Truncation)
**Frame Count:** 1000 frames
**Expected:**
- ❌ No truncation indicator (exactly at threshold, not exceeded)
- ❌ No progress bars (at threshold, not over)

**Steps:**
1. Set max_frames = 1000
2. Generate camera path
3. Verify no truncation

---

### Test 3: Just Over Boundary (Truncation)
**Frame Count:** 1001 frames
**Expected:**
- ✅ Truncation indicator: `[truncated at frame 1000, full schedule in settings.json]`
- ✅ "Schedule Display Truncated" message in status
- ✅ Progress bars appear (2 bars: "Generating camera path", "Converting to schedules")

**Steps:**
1. Set max_frames = 1001
2. Generate camera path
3. Verify truncation indicator in Translation X
4. Verify status message mentions truncation
5. Check console for progress bars (tqdm)

---

### Test 4: Large Animation (30k frames)
**Frame Count:** 30,000 frames
**Expected:**
- ✅ Truncation indicator in all 6 schedule textboxes
- ✅ Detailed truncation info in status:
  - "Total frames: 30,000"
  - "Displayed: 1,000 frames (first portion only)"
  - "Full schedules saved to settings.json"
- ✅ Two progress bars in console:
  - "Generating camera path | 30000/30000"
  - "Converting to schedules | 30000/30000"
- ✅ UI remains responsive (no browser freeze)
- ✅ Generation completes in reasonable time (~30-60 seconds)

**Steps:**
1. Set max_frames = 30000 in Run tab
2. Click "Generate Rotate-Around Preset"
3. **Watch console for progress bars**
4. **Verify UI doesn't freeze** (should respond to mouse clicks)
5. Check Translation X schedule length:
   - Should be ~15KB (truncated), NOT ~450KB (full)
6. Verify status message contains truncation info
7. Check that visualization is skipped (>5000 frame threshold)

---

### Test 5: Extreme Animation (100k frames) - Optional
**Frame Count:** 100,000 frames
**Expected:**
- ✅ Same as Test 4, but generation takes 2-3 minutes
- ✅ Progress bars show useful feedback during generation
- ✅ No memory issues
- ✅ Schedule strings remain reasonable length (<20KB per schedule)

**Note:** This test is optional and mainly verifies that the system doesn't break with extreme inputs.

---

## Verification Checklist

After running tests, verify:

- [ ] Small animations (<1000) have NO truncation
- [ ] Large animations (>1000) HAVE truncation
- [ ] Progress bars appear for >1000 frames
- [ ] Progress bars DO NOT appear for ≤1000 frames
- [ ] UI remains responsive during generation
- [ ] Browser doesn't freeze when populating schedules
- [ ] Status message explains truncation when it occurs
- [ ] Schedule textboxes show truncated preview (not full schedules)
- [ ] Truncation threshold can be adjusted in Settings

---

## Adjusting Truncation Threshold

To change where truncation kicks in:
1. Go to Settings → Deforum
2. Find "Max Schedule Display Frames" slider
3. Adjust range: 100-5000 frames (default: 1000)
4. Click "Apply settings"
5. Restart WebUI for changes to take effect

**Use Cases:**
- **Low VRAM / Slow Browser:** Lower to 500 frames
- **Fast System / Need More Preview:** Raise to 2000-3000 frames
- **Never Truncate:** Set to 5000 (not recommended for 30k+ animations)

---

## Troubleshooting

**Progress bars don't appear:**
- Check Settings → Deforum → Console Theme (should be set to "slopcore" or "classic")
- Verify frame count > 1000
- Check console for tqdm output

**UI still freezes:**
- Verify truncation threshold is set appropriately (default 1000)
- Check schedule textbox content length (<50KB is safe)
- Try lowering truncation threshold to 500 in settings

**Schedule seems wrong:**
- Full schedules are still generated correctly
- Only the UI display is truncated
- Check settings.json after saving to see full schedules

---

## Performance Expectations

| Frames | Generation Time | Schedule Length | Progress Bars | Truncation |
|--------|----------------|-----------------|---------------|------------|
| 500    | <1 second      | 8KB (full)      | No            | No         |
| 1000   | ~2 seconds     | 15KB (full)     | No            | No         |
| 1001   | ~2 seconds     | 15KB (trunc)    | Yes           | Yes        |
| 5000   | ~10 seconds    | 15KB (trunc)    | Yes           | Yes        |
| 30000  | ~60 seconds    | 15KB (trunc)    | Yes           | Yes        |
| 100000 | ~3 minutes     | 20KB (trunc)    | Yes           | Yes        |

*Times measured on modern CPU (i7/Ryzen 5+). May vary by system.*
