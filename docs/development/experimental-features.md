# 🚀 Experimental Render Core - Zirteq Fork Exclusive

## Overview

The **Zirteq Deforum Fork** features an **exclusive experimental render core** with **variable cadence** that fundamentally changes how video generation works compared to traditional Deforum.

## 🎯 **Key Differences from Traditional Deforum**

### **Traditional Deforum (A1111)**
- **Fixed Cadence**: Processes every Nth frame uniformly
- **Uniform Distribution**: Equal spacing regardless of content importance
- **Static Approach**: Same processing pattern for all content

### **Zirteq Fork Experimental Core**
- **Variable Cadence**: Intelligent frame distribution based on content importance
- **Keyframe Redistribution**: Moves diffusion frames closer to prompt keyframes  
- **Content-Aware**: Focuses processing power where visual changes matter most
- **Better Synchronization**: Prompts align precisely with visual changes

## ⚙️ **How It Works**

The experimental render core is controlled by the **Keyframe Distribution** setting (now prominently featured in the Setup tab):

### **Distribution Modes**

1. **Off**: Traditional render core (not recommended for this fork)
2. **Keyframes Only**: Diffuses only prompt/Parseq entries, ignores cadence  
3. **Additive**: Uses keyframes + adds cadence frames for stability
4. **Redistributed** ⭐ **(Default)**: Calculates cadence but moves frames to optimal positions

## 🎯 **Why "Redistributed" is Default**

The **"Redistributed"** mode is the recommended default because:

- **Smart Resource Usage**: Maintains the same number of diffused frames as traditional cadence
- **Better Timing**: Ensures prompt changes happen at the right visual moments
- **Quality + Speed**: Combines efficiency with superior synchronization
- **Stability**: Prevents artifacts while maximizing prompt responsiveness

## 📊 **Recommended Settings for Best Results**

```
🚀 Experimental Render Core Settings:
├── Keyframe Distribution: "Redistributed" (default)
├── Diffusion Cadence: 10-15 (higher than traditional)
├── FPS: 60 (smooth output)
└── Keyframe Strength: Lower than regular strength

📝 Prompt Strategy:
├── Use clear keyframe timing (0:, 30:, 60:)
├── Define major visual changes at specific frames
└── Let the core handle smooth transitions automatically
```

## ⚠️ **Important Notes**

### **This is a Fundamental Choice**
- The experimental render core is **exclusive to this fork**
- It should **remain enabled** for optimal results
- Traditional Deforum users will notice different behavior (better!)

### **Compatibility**
- ✅ **Works Great With**: High FPS, Wan AI, Flux models, most standard settings
- ⚠️ **Use Caution With**: Optical flow settings, hybrid video (may cause issues)
- ❌ **Not Recommended With**: Very low cadence (defeats the purpose)

## 🔬 **Technical Details**

### **Variable Cadence Implementation**
```python
# Traditional: Fixed interval
frames_to_diffuse = [1, 11, 21, 31, 41, 51]  # Every 10th frame

# Experimental: Content-aware redistribution  
frames_to_diffuse = [1, 8, 25, 30, 43, 60]   # Moved to prompt keyframes
```

### **Benefits in Practice**
- **Visual Coherence**: Changes happen when prompts change
- **Resource Efficiency**: Same computational cost, better results
- **Smoother Transitions**: Intelligent frame interpolation
- **Better Control**: Precise timing control over visual changes

## 🚀 **UI Changes Made**

### **Prominent Placement**
- **Setup Tab**: Experimental render core now featured prominently in first tab
- **Clear Documentation**: Extensive explanations and recommendations
- **Default Settings**: "Redistributed" mode enabled by default

### **Removed Confusion**
- **Animation Tab**: Removed duplicate keyframe distribution setting
- **Clear Messaging**: Added notes explaining the fork's differences
- **Workflow Focus**: Organized tabs for left-to-right workflow

## 💡 **For Users Coming from Traditional Deforum**

If you're used to A1111 Deforum:

1. **Embrace the Change**: The experimental core produces better results
2. **Adjust Expectations**: Cadence works differently (and better!)
3. **Higher Settings**: You can use higher cadence values than before
4. **Focus on Prompts**: The core will handle optimal frame distribution

## 🎯 **Summary**

The experimental render core with variable cadence is the **heart of what makes this fork special**. It's not just a feature—it's a fundamental improvement that makes video generation more intelligent, efficient, and controllable.

**Bottom Line**: Keep it enabled, trust the defaults, and enjoy better video generation! 🎬 