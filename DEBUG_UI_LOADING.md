# Debug UI Loading

## 🔍 **Debug Steps Added**

I've added debug output to the UI creation to help identify what's happening:

### **Changes Made:**
1. **Debug Print**: Added `print("🔧 DEBUG: Creating Wan inference steps slider with minimum=5")`
2. **Unique Label**: Changed label to `"Inference Steps (Fixed Min=5)"` 
3. **Unique ID**: Added `elem_id="wan_inference_steps_fixed"`

### **What to Look For After WebUI Restart:**

#### **1. Console Output**
Look for this message in the WebUI console/terminal:
```
🔧 DEBUG: Creating Wan inference steps slider with minimum=5
```

#### **2. UI Label**
In the Wan tab, the slider should now show:
```
Inference Steps (Fixed Min=5)
```
Instead of just "Inference Steps"

#### **3. Element ID**
The slider should have the HTML element ID: `wan_inference_steps_fixed`

## 🎯 **Diagnostic Results**

### **If you see the debug message and new label:**
✅ Our code IS being executed
❌ Something else is overriding the minimum value
→ **Next step**: Check for JavaScript/CSS overrides or Gradio bugs

### **If you DON'T see the debug message or new label:**
❌ Our code is NOT being executed
❌ There's another UI file being loaded instead
→ **Next step**: Find the actual UI file being used

### **If you see the new label but minimum is still 20:**
✅ Our code IS being executed
❌ Gradio is ignoring our minimum parameter
→ **Next step**: Try a different approach (JavaScript override)

## 🔧 **Next Actions Based on Results**

Please restart WebUI and report:
1. **Console message**: Do you see the debug print?
2. **Label change**: Does the slider show "(Fixed Min=5)"?
3. **Minimum value**: Is it still 20 or now 5?

This will help us identify exactly where the problem is occurring. 