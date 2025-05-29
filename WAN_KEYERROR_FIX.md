# WAN KeyError Fix

## Problem Summary

After applying the FFmpeg video stitching fixes, a new error occurred during WAN video generation:

```
KeyError: 'frames'
```

This error was happening in the WAN integration code when trying to access clip data.

## Root Cause

The issue was a **data structure mismatch** between the UI elements and the WAN integration:

- **UI Elements** (`ui_elements.py`) creates clips with the key `'num_frames'`
- **WAN Integration** (`wan_simple_integration.py`) was trying to access `clip['frames']`

## Error Location

**File**: `scripts/deforum_helpers/wan_simple_integration.py`  
**Function**: `generate_video_with_i2v_chaining()`  
**Line**: Approximately 1123

```python
# This was causing the KeyError:
print(f"   🎞️ Frames: {clip['frames']}")  # ❌ 'frames' key doesn't exist

# The actual key in the data structure is:
print(f"   🎞️ Frames: {clip['num_frames']}")  # ✅ Correct key
```

## Fix Applied

### 1. Fixed Key References

**Before**:
```python
print(f"   🎞️ Frames: {clip['frames']}")
wan_frames = self._calculate_wan_frames(clip['frames'])
discard_info = self._calculate_frame_discarding(clip['frames'], wan_frames)
print(f"🎯 Wan will generate {wan_frames} frames, targeting {clip['frames']} final frames")
```

**After**:
```python
print(f"   🎞️ Frames: {clip['num_frames']}")
wan_frames = self._calculate_wan_frames(clip['num_frames'])
discard_info = self._calculate_frame_discarding(clip['num_frames'], wan_frames)
print(f"🎯 Wan will generate {wan_frames} frames, targeting {clip['num_frames']} final frames")
```

### 2. Fixed Error Message Reference

**Before**:
```python
print(f"🗑️ Will discard {discard_info['discard_count']} frames using {discard_info['method']}")
```

**After**:
```python
print(f"🗑️ Will discard {discard_info['discard_count']} frames from the middle to preserve start/end frames")
```

The `discard_info` dictionary doesn't contain a `'method'` key, so this was also causing potential issues.

## Data Structure Reference

For future reference, the clip data structure from `ui_elements.py` contains:

```python
{
    'prompt': str,           # The prompt text for this clip
    'start_frame': int,      # Starting frame number
    'end_frame': int,        # Ending frame number  
    'num_frames': int        # Number of frames in this clip
}
```

## Expected Results

After this fix:

1. ✅ **No More KeyError**: The WAN integration correctly accesses clip data
2. ✅ **Proper Frame Calculation**: Frame counts are correctly passed to WAN processing
3. ✅ **Correct Logging**: Status messages display the right frame information
4. ✅ **Continued Processing**: The generation pipeline can proceed to the actual WAN model loading

## Files Modified

1. `scripts/deforum_helpers/wan_simple_integration.py` - Fixed key references and error message
2. `WAN_KEYERROR_FIX.md` - This documentation (created)

## Next Steps

With both the FFmpeg fixes and this KeyError fix applied, the WAN video generation should now:

1. ✅ Correctly parse clip data from the UI
2. ✅ Generate frames with proper naming convention
3. ✅ Successfully stitch frames into video using FFmpeg
4. ✅ Add audio to the final video

The pipeline should now work end-to-end without the previous errors. 