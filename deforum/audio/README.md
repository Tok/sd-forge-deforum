# Audio Synchronization Module

Audio event detection and keyframe generation for syncing Deforum animations to music.

## Features

- **Event Detection**: Multiple detection methods (onsets, beats, bass energy)
- **Signal Processing**: Lowpass filter and distortion for isolating bass kicks
- **Keyframe Generation**: Convert audio events to animation keyframes with spacing constraints
- **Visualization**: Waveform and event marker plots (coming soon)

## Modules

### `analysis.py`
Audio event detection using librosa:
- `detect_onsets()` - Detect transients (kicks, snares, attacks)
- `detect_beats()` - Detect rhythmic pulse and tempo
- `extract_bass_energy()` - Low-frequency energy envelope
- `detect_events()` - Unified detection interface

### `processing.py`
Signal processing for better event detection:
- `apply_lowpass_filter()` - Butterworth lowpass (isolate bass 20-250Hz)
- `apply_bandpass_filter()` - Isolate specific frequency ranges
- `apply_distortion()` - Emphasize transients (tanh/hard/arctan)
- `process_audio_for_detection()` - Complete processing pipeline

### `keyframe_generation.py`
Convert events to animation keyframes:
- `filter_events_by_intensity()` - Filter weak events
- `cluster_nearby_events()` - Enforce minimum spacing
- `generate_keyframes_from_events()` - Main keyframe generation
- `keyframes_to_schedule_string()` - Export to Deforum format

## Usage Example

```python
from deforum.audio import (
    load_audio_file,
    process_audio_for_detection,
    detect_events,
    generate_keyframes_from_events
)

# Load audio
audio, sr = load_audio_file("song.mp3", sample_rate=22050)

# Process for bass kick detection
processed = process_audio_for_detection(
    audio,
    sample_rate=sr,
    frequency_band='bass',
    lowpass_cutoff=200,
    distortion_gain=1.2
)

# Detect events
event_times, intensities = detect_events(
    processed,
    sr,
    method='onset',
    sensitivity=0.7
)

# Generate keyframes
keyframes = generate_keyframes_from_events(
    event_times,
    intensities,
    fps=24,
    max_frames=240,
    min_spacing_frames=12,
    intensity_threshold=0.5
)

print(f"Generated {len(keyframes)} keyframes from audio")
for kf in keyframes[:5]:
    print(f"  Frame {kf['frame']}: intensity={kf['intensity']:.2f}")
```

## Dependencies

- **librosa** - Audio analysis and event detection
- **scipy** - Signal processing filters
- **soundfile** - Audio file I/O
- **plotly** - Interactive visualizations (for UI)

Install with:
```bash
pip install librosa soundfile scipy plotly
```

## Roadmap

**Phase 1: Core Analysis** ✅
- Audio loading
- Event detection (onset/beat/bass)
- Signal processing (lowpass/distortion)

**Phase 2: Keyframe Generation** ✅
- Event filtering and clustering
- Frame number conversion
- Deforum schedule export

**Phase 3: Visualization** (In Progress)
- Waveform plots
- Event marker overlay
- Interactive controls

**Phase 4: UI Integration** (Pending)
- Gradio Audio Sync tab
- Settings controls
- Preview and apply

## Technical Details

### Bass Kick Isolation

Bass kicks typically occur in the 60-250Hz range. To isolate them:

1. **Lowpass Filter**: Remove frequencies above 250Hz
2. **Highpass Filter**: Remove rumble below 20Hz
3. **Distortion**: Soft clipping emphasizes transients
4. **Onset Detection**: Find sharp energy increases

```python
# Optimal settings for bass kick detection
bass_audio = process_audio_for_detection(
    audio,
    frequency_band='bass',
    lowpass_cutoff=200,  # Isolate bass frequencies
    distortion_gain=1.5,  # Emphasize transients
)
```

### Keyframe Spacing

Minimum spacing prevents keyframe spam:
- **12 frames @ 24fps** = 0.5 seconds
- **18 frames @ 30fps** = 0.6 seconds
- **30 frames @ 60fps** = 0.5 seconds

When events occur closer than minimum spacing, only the strongest is kept.

## Integration with Parseq

Audio Sync is **disabled** when Parseq is active, as Parseq has its own audio reactivity features. The Audio Sync tab will show a warning in this case.
