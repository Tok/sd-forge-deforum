# Zero-HITL Slopcore Generator Design

> **"🔥 SLOP IT! 🔥"** - Button text suggested by Qwen3-Next-80B-A3B herself
>
> *"Short, punchy, and irreverent — 'SLOP IT!' is perfect for a comically huge button. It's casual, slangy, and leans into the intentionally messy 'slopcore' vibe without over-explaining. It's a command that sounds like a reckless, low-effort action, which fits 'zero-HITL' automation perfectly."*

## Vision

A one-click system that generates complete animated videos from minimal user input (duration + optional theme/instructions). The system orchestrates the entire pipeline autonomously:

1. **Audio Generation** → Generate drum loop/breakbeat
2. **Prompt Generation** → Create synced animation prompts (Qwen + audio analysis)
3. **Camera Path Generation** → Design movement patterns (presets or custom splines)
4. **Video Rendering** → Execute full Deforum pipeline

**Philosophy:** Zero Human-In-The-Loop (HITL) creative playground where Qwen makes all artistic decisions within curated ranges.

## Architecture

### Phase 1: Audio Generation
**Model:** Stable Audio Open Small (341M params)
- **Why:** Lightweight, runs on CPU/GPU, generates 44.1kHz stereo in <8s
- **Installation:** `pip install stable-audio-tools`
- **Model:** `stabilityai/stable-audio-open-small`
- **Capabilities:**
  - Variable length (up to 11s per generation)
  - Text-to-audio with BPM control
  - Excellent for drum loops, breakbeats, ambient textures
- **Qwen's Role:** Generate creative prompt for audio (e.g., "170 BPM jungle breakbeat with heavy bass")

**Implementation:**
```python
from deforum.utils.audio_generation import generate_loop

# Qwen generates: bpm=170, style="jungle breakbeat", duration=3.0
audio_path = generate_loop(
    prompt="170 BPM jungle breakbeat with heavy bass",
    duration_seconds=3.0,
    output_path="output/generated_loop.wav"
)
```

### Phase 2: Audio Analysis & Prompt Generation
**Existing System:** QwenPromptExpander + audio sync
- Analyze generated audio (beat detection, spectral features)
- Generate movement-aware prompts synchronized to beats
- Use existing `generate_prompts_with_ai()` with audio file input

**Qwen's Creative Decisions:**
- Theme/mood based on audio characteristics
- Visual style (cyberpunk, nature, abstract, etc.)
- Movement intensity (calm vs chaotic)
- Color palette suggestions

### Phase 3: Camera Path Generation
**Existing System:** Camera Path presets + spline generation
- Choose preset (rotate-around, spiral, street, dashcam, bodycam, etc.)
- OR generate custom spline with random control points
- Populate translation_x/y/z and rotation_3d_x/y/z schedules

**Qwen's Creative Decisions:**
- Path type selection based on audio energy/mood
- Radius, height, rotation factor randomization
- Whether to use closed loop
- Look-at vs tangent-following camera

### Phase 4: Parameter Selection
**Guided Randomization:** Qwen selects from curated ranges

**Critical Parameters:**
```python
ZERO_HITL_RANGES = {
    # Video settings
    'fps': [24, 30, 60],  # Match to audio BPM feel
    'resolution': ['512x512', '768x768', '1024x1024'],

    # Render mode
    'render_mode': ['Classic 3D', 'New 3D', 'Keyframes Only'],  # Not Flux+Interp (too slow)

    # Generation settings
    'steps': [15, 20, 25],  # Balance speed/quality
    'cfg_scale': [3.5, 5.0, 7.0],
    'sampler': ['euler', 'euler_a', 'dpmpp_2m'],

    # Animation intensity
    'cadence': [2, 3, 5, 8],  # Higher = more interpolation
    'strength': [0.55, 0.65, 0.75],  # Higher = more change per frame

    # Camera movement intensity
    'translation_range': [(-50, 50), (-100, 100), (-200, 200)],
    'rotation_range': [(-5, 5), (-10, 10), (-20, 20)],
    'zoom_range': [(0.95, 1.05), (0.9, 1.1), (0.8, 1.2)],

    # Camera path
    'preset_type': ['rotate-around', 'spiral', 'street', 'dashcam', 'bodycam', 'figure-eight'],
    'preset_radius': (50, 200),
    'preset_height': (-50, 50),

    # Shakify (optional)
    'shakify_patterns': [None, 'GENTLE_HANDHELD', 'INVESTIGATION', 'EARTHQUAKE'],
    'shakify_intensity': [0.0, 0.3, 0.5, 0.7],

    # Depth
    'depth_model': ['depth_anything_v2_vits'],  # Only fast model
    'midas_weight': [0.2, 0.3, 0.5],

    # Effects
    'enable_perspective_flip': [True, False],
    'noise_type': ['perlin', 'uniform'],
    'color_coherence': ['Match Frame 0 LAB', 'Match Frame 0 HSV', 'Video Input'],
}
```

### Phase 5: Execution
**Pipeline Flow:**
1. Validate all parameters
2. Set up output directory
3. Execute render with generated settings
4. Monitor progress via JobStatusTracker
5. Return video path + generation report

## UI Design

### New First Tab: "Zero-HITL"

```
╔══════════════════════════════════════════════════════════════════╗
║  🎲 Zero-HITL Slopcore Generator                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Let Qwen create a complete animated video with zero human       ║
║  intervention. Just set duration and optionally provide a        ║
║  theme or vibe. Everything else is randomized creatively.        ║
║                                                                   ║
║  ┌────────────────────────────────────────────────────────────┐ ║
║  │ Duration (seconds): [___3.0___] [slider: 1-10]            │ ║
║  └────────────────────────────────────────────────────────────┘ ║
║                                                                   ║
║  ┌────────────────────────────────────────────────────────────┐ ║
║  │ Theme/Instructions (optional):                             │ ║
║  │ [_____________________________________________]             │ ║
║  │                                                            │ ║
║  │ Examples:                                                  │ ║
║  │  • "cyberpunk neon city"                                   │ ║
║  │  • "underwater dreamscape"                                 │ ║
║  │  • "glitch art chaos"                                      │ ║
║  │  • Leave empty for pure randomness                         │ ║
║  └────────────────────────────────────────────────────────────┘ ║
║                                                                   ║
║  ┌────────────────────────────────────────────────────────────┐ ║
║  │ Random Seed: [___-1___] (empty = random)                   │ ║
║  └────────────────────────────────────────────────────────────┘ ║
║                                                                   ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │                                                          │   ║
║  │              [  🔥 SLOP IT! 🔥  ]                        │   ║
║  │          (comically huge slopcore button)                │   ║
║  │          NO HUMAN NEEDED (PROBABLY)                      │   ║
║  │                                                          │   ║
║  └──────────────────────────────────────────────────────────┘   ║
║                                                                   ║
║  ┌────────────────────────────────────────────────────────────┐ ║
║  │ Status: Idle                                               │ ║
║  │                                                            │ ║
║  │ Generation Log:                                            │ ║
║  │ [___________________________________________________]       │ ║
║  │ [___________________________________________________]       │ ║
║  │ [___________________________________________________]       │ ║
║  │ [___________________________________________________]       │ ║
║  │                                                            │ ║
║  │ [View Generated Settings] [Open Output Folder]             │ ║
║  └────────────────────────────────────────────────────────────┘ ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
```

## Implementation Files

### New Files
```
deforum/
├── utils/
│   ├── audio_generation.py          # Stable Audio Open Small wrapper
│   └── zero_hitl/
│       ├── __init__.py
│       ├── orchestrator.py          # Main orchestration logic
│       ├── parameter_randomizer.py  # Guided parameter selection
│       └── qwen_conductor.py        # Qwen prompting for creative decisions
└── ui/
    └── tabs/
        └── tab_zero_hitl.py         # Zero-HITL UI tab
```

### Modified Files
```
deforum/ui/ui_left.py                # Add Zero-HITL as first tab
deforum/ui/handlers/                 # New handler for Qwen Do Everything button
requirements.txt                     # Add stable-audio-tools
```

## Qwen Conductor System

### Qwen's Orchestration Prompt (Structured Output)

```python
QWEN_ORCHESTRATOR_PROMPT = """
You are an AI creative director for an automated video generation system.
Your task is to make ALL creative decisions to generate a complete animated video.

USER INPUT:
- Duration: {duration} seconds
- Theme: {theme} (empty = pure randomness)
- Random Seed: {seed}

YOUR MISSION:
Generate a complete creative direction as JSON following this exact schema.

CONSTRAINTS:
- Audio MUST be synced to prompts (beat-driven visual changes)
- Camera movement MUST match audio energy/mood
- All parameters MUST be within allowed ranges
- Final frame count = fps * duration (must be exact)
- Be BOLD and CREATIVE - this is a slopcore generator!

OUTPUT JSON SCHEMA:
{{
  "audio": {{
    "prompt": "descriptive text-to-audio prompt with BPM",
    "bpm": 120-180,
    "style": "jungle/dnb/techno/ambient/etc",
    "mood": "energetic/calm/chaotic/dreamy"
  }},
  "visual_theme": {{
    "style": "cyberpunk/nature/abstract/glitch/etc",
    "color_palette": ["color1", "color2", "color3"],
    "mood_match": "explanation of how visuals match audio"
  }},
  "prompts": {{
    "base_prompt": "core visual concept",
    "keyframe_count": 8-20,
    "prompt_strategy": "how prompts evolve over time"
  }},
  "camera": {{
    "preset_type": "rotate-around/spiral/street/dashcam/bodycam/figure-eight/custom",
    "movement_intensity": "low/medium/high/extreme",
    "radius": 50-200,
    "height": -50 to 50,
    "closed_loop": true/false,
    "shakify_pattern": null/"GENTLE_HANDHELD"/"INVESTIGATION"/"EARTHQUAKE",
    "shakify_intensity": 0.0-0.7
  }},
  "render_settings": {{
    "render_mode": "Classic 3D"/"New 3D"/"Keyframes Only",
    "fps": 24/30/60,
    "resolution": "512x512"/"768x768"/"1024x1024",
    "steps": 15-25,
    "cfg_scale": 3.5-7.0,
    "sampler": "euler"/"euler_a"/"dpmpp_2m",
    "cadence": 2-8,
    "strength": 0.55-0.75,
    "depth_model": "depth_anything_v2_vits",
    "midas_weight": 0.2-0.5
  }},
  "reasoning": "Brief explanation of creative choices and how everything ties together"
}}

RESPOND WITH ONLY THE JSON - NO MARKDOWN, NO EXPLANATIONS OUTSIDE JSON.
"""
```

### Qwen Model Selection
- Use existing auto-selection logic (3B/7B/14B based on VRAM)
- Prefer 7B+ for better creative decisions
- Use structured output (JSON mode) for reliable parsing

## Error Handling & Fallbacks

**Audio Generation Fails:**
- Fall back to using existing `amen_break.wav` sample
- Continue with prompt + camera generation

**Qwen Fails to Generate Valid JSON:**
- Retry once with simplified prompt
- Fall back to hardcoded "sensible random" defaults

**Rendering Fails:**
- Return detailed error log
- Allow user to inspect generated settings and retry manually

## Future Enhancements (Phase 2)

1. **Multiple Audio Segments:** Generate 3-5 short loops, stitch with crossfade
2. **Style Presets:** Add curated "vibes" (80s Synthwave, Jungle Terror, Lo-Fi, etc.)
3. **User Favorites:** Save successful generation seeds/settings
4. **Batch Generation:** Generate 5-10 variations with same theme
5. **Advanced Audio Control:** Let Qwen specify instrument types, effects chains
6. **Video-to-Audio Sync Analysis:** Use beat detection to align keyframes exactly to hits

## Testing Strategy

**Unit Tests:**
- Audio generation (mock Stable Audio)
- Parameter randomization (validate ranges)
- Qwen JSON parsing (malformed inputs)

**Integration Tests:**
- Full pipeline with 3s duration
- Theme variations (5 different themes)
- Edge cases (1s minimum, 10s maximum)

**Manual Testing:**
- Generate 20 videos with random seeds
- Verify audio-visual sync
- Check for common failure modes
- Evaluate creative quality

## Success Metrics

- **Speed:** Complete generation in <5min for 3s @ 60fps
- **Reliability:** >90% success rate without fallbacks
- **Quality:** Subjectively "interesting" visuals in >80% of outputs
- **Sync:** Audio-visual alignment within 1 frame
- **Fun Factor:** Pure chaotic slopcore energy ✨
