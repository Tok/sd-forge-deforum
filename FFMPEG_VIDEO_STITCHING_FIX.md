# FFmpeg Video Stitching Fix

## Problem Summary

The WAN video generation was completing successfully, but FFmpeg was failing to stitch the frames into a video file. The error message showed:

```
Error opening input file C:\Users\Zirteq\Documents\workspace\webui-forge\webui\outputs\img2img-images\Deforum_20250529192907_\20250529192907.mp4.
Error opening input files: No such file or directory
```

## Root Causes Identified

### 1. Frame Naming Pattern Mismatch

**Issue**: The WAN integration was saving frames with simple numeric names:
- `000000000.png`, `000000001.png`, etc.

**Expected**: Deforum expects frames with timestamp prefix:
- `20250529192907_000000000.png`, `20250529192907_000000001.png`, etc.

**Impact**: FFmpeg couldn't find the frames because it was looking for the wrong pattern.

### 2. Malformed FFmpeg Command

**Issue**: The FFmpeg command had redundant and incorrect parameters:
```bash
# Old (broken) command structure:
ffmpeg -y -r 8 -start_number 0 -i "pattern.png" -frames:v 394 -c:v libx264 -vf fps=8 -pix_fmt yuv420p -crf 17 -preset veryslow -pattern_type sequence -vcodec png output.mp4
```

**Problems**:
- Redundant `-vcodec png` parameter conflicting with `-c:v libx264`
- Poor error handling when FFmpeg fails

## Fixes Implemented

### 1. Fixed Frame Naming in WAN Integration

**File**: `scripts/deforum_helpers/wan_simple_integration.py`

**Changes**:
- Modified `generate_video_with_i2v_chaining()` to extract timestring from output directory
- Updated frame naming to use Deforum's convention: `{timestring}_{frame_idx:09d}.png`
- Added fallback timestring generation if none found

**Before**:
```python
dst_filename = f"{total_frame_idx:09d}.png"  # 000000000.png
```

**After**:
```python
dst_filename = f"{timestring}_{total_frame_idx:09d}.png"  # 20250529192907_000000000.png
```

### 2. Fixed FFmpeg Command Structure

**File**: `scripts/deforum_helpers/video_audio_utilities.py`

**Changes**:
- Removed redundant `-vcodec` parameter
- Improved error handling with return code checking
- Added stderr/stdout logging for debugging

**Before**:
```python
cmd = [ffmpeg_location, '-y', '-r', str(float(fps)), '-start_number', str(stitch_from_frame), 
       '-i', imgs_path, '-frames:v', str(stitch_to_frame), '-c:v', 'libx264', 
       '-vf', f'fps={float(fps)}', '-pix_fmt', 'yuv420p', '-crf', str(crf), 
       '-preset', preset, '-pattern_type', 'sequence']
cmd.append('-vcodec')  # ❌ Redundant and conflicting
cmd.append('png' if imgs_path[0].find('.png') != -1 else 'libx264')
cmd.append(outmp4_path)
```

**After**:
```python
cmd = [ffmpeg_location, '-y', '-r', str(float(fps)), '-start_number', str(stitch_from_frame),
       '-i', imgs_path, '-frames:v', str(stitch_to_frame), '-c:v', 'libx264',
       '-vf', f'fps={float(fps)}', '-pix_fmt', 'yuv420p', '-crf', str(crf),
       '-preset', preset, '-pattern_type', 'sequence', outmp4_path]

# Added proper error checking
if process.returncode != 0:
    print(f"FFmpeg stderr: {stderr}")
    print(f"FFmpeg stdout: {stdout}")
    raise RuntimeError(f"FFmpeg failed with return code {process.returncode}: {stderr}")
```

## Technical Details

### Frame Pattern Matching

The fix ensures that:
1. WAN generates frames with names like: `20250529192907_000000000.png`
2. FFmpeg receives pattern: `20250529192907_%09d.png`
3. Pattern matching works correctly for sequential frame processing

### Timestring Extraction Logic

```python
# Extract timestring from output directory name
timestring = os.path.basename(output_dir).split('_')[-1]
if not timestring or len(timestring) != 14:
    # Fallback: search for 14-digit timestamp in directory name
    dir_parts = os.path.basename(output_dir).split('_')
    for part in dir_parts:
        if len(part) == 14 and part.isdigit():
            timestring = part
            break
    else:
        # Last resort: create new timestamp
        timestring = datetime.now().strftime("%Y%m%d%H%M%S")
```

## Expected Results

After these fixes:

1. **Frame Generation**: WAN will save frames with correct naming pattern
2. **FFmpeg Processing**: Video stitching will work without "No such file or directory" errors
3. **Audio Integration**: Audio will be properly added to the generated video
4. **Complete Pipeline**: Full WAN → Frames → Video → Audio workflow will succeed

## Testing

The fixes address the core issues seen in the error log:
- ✅ Frames are now named correctly for FFmpeg pattern matching
- ✅ FFmpeg command is properly structured without conflicting parameters
- ✅ Better error reporting for debugging future issues

## Files Modified

1. `scripts/deforum_helpers/wan_simple_integration.py` - Frame naming fix
2. `scripts/deforum_helpers/video_audio_utilities.py` - FFmpeg command fix
3. `test_ffmpeg_fix.py` - Test script for verification (created)
4. `FFMPEG_VIDEO_STITCHING_FIX.md` - This documentation (created)

The fixes are backward compatible and don't affect other Deforum functionality. 