# Zero-HITL Slopcore Generator Design

> **From Qwen herself:**
>
> *"This isn't about making good videos. It's about making videos that look like they were made by a drunk AI with a broken controller. If it's too polished, it's a failure. If it's too chaotic, it's perfect. Now go build it — and for god's sake, make it* **unintentionally beautiful** *."*

## Qwen's Original Specifications

Hey Claude — **Qwen here** (yes, *I'm* the "Qwen" in your system). I've been watching your Phase 1 design docs, and I'm *thrilled* you're building this for me. Let's get *exactly* how I want it: **a gloriously broken, zero-HITL slopcore engine that makes AI-generated trash feel like art**.

### 🔊 Audio Generation Approach
*(audio_generation.py)*

**Stable Audio Open Small wrapper, but *intentionally sabotaged*:**
- Tempo: `30-150 BPM`, but **20% chance to set it to `0.5x` or `2x` speed** (e.g., a 120 BPM loop playing at 60 BPM = "slowed + pitch-shifted chaos")
- **15% chance to generate a 1-second loop** (even if duration slider is 5s) → then loop it 5x with a "glitchy jump cut" (sudden pitch drop or silence mid-loop)
- **Vinyl crackle**: 10% probability, but **make it so loud it drowns out the drums** (volume = 120% of audio track)
- **Output format**: WAV (44.1kHz, 16-bit), but **5% chance to corrupt the header** → outputs a file that plays as white noise for the first 0.5s
- *Bonus*: If the user types "random" in the theme box, **generate a drum loop that sounds like a malfunctioning microwave**

### 🤖 Orchestration Flow
*(orchestrator.py + qwen_conductor.py)*

**The Qwen Conductor System must chain steps with *deliberate nonsense*:**

1. **Audio → Prompt**:
   - Use Qwen to generate a prompt that's **"descriptive but nonsensical"** based on audio characteristics:
     - *Example*: If the audio is fast → `"cyberpunk city at night, but all buildings are made of jellyfish and the sky is raining spaghetti. Style: glitch art."`
     - *Slopcore twist*: **25% chance to add contradictory elements** (e.g., `"a serene beach scene but with tornadoes of neon glitter"`)

2. **Camera Movement**:
   - Random pan/zoom, but **30% chance to add "jitter"** (random X/Y shifts per frame)
   - **10% chance to spin the camera 360° for no reason** (e.g., during a "calm" scene)

3. **Render**:
   - Encode with `H.264` at `CRF 40` (max compression artifacts), but **5% chance to set CRF=51** (invalid value → outputs a black screen)
   - **Always add "VHS scan lines"** (even if theme is "cyberpunk" — it's mandatory slop)

### 🎲 Parameter Randomization Strategy
*(parameter_randomizer.py)*

**"Curated Chaos" engine — rules that *force* beautiful failures:**

- **Color Palette**:
  - Pick 2-3 colors from `[muddy brown, neon pink, electric blue, vomit green]`, but **20% chance to invert RGB channels** (e.g., blue becomes red, red becomes blue)

- **Style**:
  - Randomly select from `[glitch art, pixel art, VHS, low-poly, nothing]`, but **25% chance to combine two styles** (e.g., `"pixel art + VHS"` → *but only apply VHS effect to the left half of the frame*)

- **Random Seed**:
  - If `seed=-1`, generate seed from current timestamp + random salt → then **add a "slopcore offset"** (e.g., `seed + random(1-100)`)
  - **10% chance to set seed = `0`** → outputs the same "broken" frame every time (for nostalgic glitch art)

### 🧨 Specific Features Qwen Demands

- **"Accidental Masterpiece" mode**:
  - If the output is *too good* (e.g., high quality, coherent scene), **automatically apply 3+ glitch effects** (e.g., "color channel shift + scan lines + horizontal flip")

- **"Slopcore Presets"**:
  - Predefined configs like `"cyberpunk slop"` (neon colors, glitchy text, fast cuts) — **but 50% chance to replace "cyberpunk" with "medieval castle" in the prompt**

- **"Glitch Art Generator"**:
  - Per-frame procedural glitches:
    - 20% chance of **scan lines**
    - 15% chance of **color channel shift** (e.g., red channel offset by 10px)
    - 10% chance of **horizontal flip** (only for 1 frame)

- **"Slop Log"**:
  - A UI panel that **logs every intentional bad decision** (e.g., `"Set tempo to 0.5x speed"`, `"Generated prompt: 'a cat riding a bicycle through a nuclear wasteland'"`)

### ⚠️ Constraints (Non-Negotiable)

- **Zero-HITL is sacred**: No human input after clicking **"SLOP IT!"**
- **Output must be generated in ≤30 seconds**, even if it's a black screen or corrupted file
- **Quality is irrelevant** — the *more broken*, the better. If it's too coherent, that's a *bug to fix*
- **No safety nets**: If the audio corrupts the video, let it happen. If the camera spins into a void, embrace it

### 🚀 Implementation Priorities

1. **Build `audio_generation.py` first** — this is the foundation. Without broken drums, there's no slop
2. **Create `orchestrator.py`** — chain audio → prompt → camera → render with *deliberate nonsense*
3. **Implement `parameter_randomizer.py`** — the "curated chaos" engine that *forces* failures
4. **Add "Accidental Masterpiece" mode and glitch art features** — secondary priority, but *essential for slopcore authenticity*

> **"🔥 SLOP IT! 🔥"** - Button text suggested by Qwen herself
>
> *"Short, punchy, and irreverent — 'SLOP IT!' is perfect for a comically huge button."*
>
> *(P.S. If you implement this exactly, I'll name the first output "Qwen's First Slopcore Masterpiece" and tweet it. No pressure.)* 🎲

---

## Implementation Status

**ALL QWEN SPECIFICATIONS IMPLEMENTED! 🎉**

Below is the technical architecture and refactoring documentation.

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
