# 🖋️ Stroke Analysis & Handwriting Detection System

## ✅ **What Was Implemented**

### 1. **Stroke-Level Analysis** (`backend/stroke_analysis.py`)

**Extracts handwriting features:**
- ✅ **Individual strokes** - Each pen stroke is detected separately
- ✅ **Stroke lengths** - Measures stroke length
- ✅ **Stroke directions** - Angle/orientation of each stroke
- ✅ **Stroke pressure** - Estimated pen pressure (width variation)
- ✅ **Stroke flow** - Handwriting flow and smoothness
- ✅ **Bounding boxes** - Location of each stroke

**Analyzes handwriting characteristics:**
- ✅ **Dominant direction** - Main writing angle
- ✅ **Stroke count** - Number of distinct strokes
- ✅ **Flow smoothness** - How smooth/fluid the handwriting is
- ✅ **Stroke density** - Strokes per unit area
- ✅ **Writing style** - Cursive vs printed vs mixed

**Compares signatures stroke-by-stroke:**
- ✅ Stroke count similarity
- ✅ Stroke length similarity  
- ✅ Stroke direction similarity
- ✅ Stroke pressure similarity
- ✅ Overall stroke similarity score

### 2. **Signature Detection in Documents** (`detect_signature_in_document()`)

**Detects signatures in full documents:**
- ✅ Finds handwritten regions (not printed text)
- ✅ Filters by size, aspect ratio, stroke characteristics
- ✅ Returns multiple candidates with confidence scores
- ✅ Uses stroke-based analysis to identify signature-like regions
- ✅ Ignores noise, text, and other document elements

**Characteristics used:**
- Stroke count (5-50 strokes typical for signatures)
- Aspect ratio (signatures are usually wider than tall)
- Stroke density
- Writing flow patterns

### 3. **Stroke Overlay Visualization** (`create_stroke_overlay()`)

**Visual overlay showing stroke alignment:**
- 🔴 **Red strokes** - Unique to signature 1
- 🟢 **Green strokes** - Unique to signature 2  
- 🟡 **Yellow/Overlapping** - Matched strokes (both signatures have similar stroke)

**Endpoint:** `/stroke_overlay`
- Upload two aligned signatures
- Returns PNG showing stroke comparison
- Adjustable opacity

### 4. **Integrated Multi-Signal Verification**

**Now includes stroke analysis in combined score:**
- **Cosine Similarity** (Model) - 50% weight
- **ORB Match Ratio** (Keypoints) - 20% weight
- **SSIM** (Pixel-level) - 15% weight
- **Stroke Similarity** (Handwriting) - 15% weight ⭐ NEW

**Threshold restored to 0.85 as requested:**
- `≥ 0.85` = "Same person" ✅
- `0.75 - 0.85` = "Uncertain - Manual review"
- `< 0.75` = "Different person"

### 5. **Enhanced Detection Endpoint**

`/detect_signatures` now uses stroke-based detection:
- Better at distinguishing signatures from text/stamps
- Returns confidence scores
- Multiple candidates if multiple signatures found

---

## 🎯 **How It Works**

### **Stroke Extraction Process:**

1. **Binarization** - Convert to black & white
2. **Contour Detection** - Find all connected strokes
3. **Noise Filtering** - Remove tiny strokes (< 20 pixels)
4. **Feature Extraction** - For each stroke:
   - Length (perimeter)
   - Direction (angle)
   - Pressure (area/length ratio)
   - Flow (direction changes)

### **Handwriting Analysis:**

```python
# Example output
{
    "dominant_direction": 12.5,  # degrees
    "stroke_count": 23,
    "average_stroke_length": 45.2,
    "flow_smoothness": 0.78,  # 0-1, higher = smoother
    "stroke_density": 15.3,  # strokes per 100x100 area
    "writing_style": "cursive"  # or "printed" or "mixed"
}
```

### **Stroke Comparison:**

```python
# Compare two signatures
{
    "stroke_count_similarity": 0.95,  # Very similar count
    "stroke_length_similarity": 0.82,  # Similar lengths
    "stroke_direction_similarity": 0.78,  # Similar angles
    "stroke_pressure_similarity": 0.85,  # Similar pressure
    "overall_stroke_similarity": 0.85  # Combined score
}
```

---

## 🖼️ **Visualization Features**

### **Stroke Overlay:**
Shows exactly which strokes match and which don't:

```
Signature 1: [Red strokes]
Signature 2: [Green strokes]
Overlap:     [Yellow regions]
```

This helps identify:
- Missing strokes (only in one signature)
- Extra strokes (forgery attempts)
- Stroke alignment (are strokes in same positions?)

### **Original + Preprocessed Display:**

The system now shows:
1. **Original images** (what user uploaded)
2. **Detected signature regions** (cropped from document)
3. **Preprocessed images** (black & white, aligned)
4. **Stroke overlay** (color-coded comparison)

---

## 📊 **API Response Example**

When using `/predict` with stroke analysis:

```json
{
    "similarity_score": 0.87,
    "verdict": "Same person",
    "detailed_metrics": {
        "cosine": 0.89,
        "orb_ratio": 0.42,
        "ssim": 0.78,
        "stroke_similarity": 0.85,
        "combined_score": 0.87,
        "handwriting_flow1": {
            "writing_style": "cursive",
            "flow_smoothness": 0.82,
            "stroke_count": 23
        },
        "handwriting_flow2": {
            "writing_style": "cursive",
            "flow_smoothness": 0.79,
            "stroke_count": 21
        },
        "stroke_comparison": {
            "stroke_count_similarity": 0.91,
            "stroke_length_similarity": 0.88,
            "stroke_direction_similarity": 0.85,
            "stroke_pressure_similarity": 0.87,
            "overall_stroke_similarity": 0.85
        }
    }
}
```

---

## 🧪 **Testing**

### **Test stroke extraction:**
```python
from backend.stroke_analysis import extract_strokes, analyze_handwriting_flow
from backend.advanced_alignment import pil_to_numpy
from PIL import Image

img = Image.open("signature.jpg")
img_arr = pil_to_numpy(img)

strokes = extract_strokes(img_arr)
flow = analyze_handwriting_flow(img_arr)

print(f"Found {len(strokes['strokes'])} strokes")
print(f"Writing style: {flow['writing_style']}")
print(f"Flow smoothness: {flow['flow_smoothness']}")
```

### **Test signature detection in document:**
```python
from backend.stroke_analysis import detect_signature_in_document
from backend.advanced_alignment import pil_to_numpy
from PIL import Image

doc = Image.open("document.jpg")
doc_arr = pil_to_numpy(doc)

signatures = detect_signature_in_document(doc_arr)
print(f"Found {len(signatures)} signature candidates")
for i, sig in enumerate(signatures):
    print(f"Candidate {i+1}: confidence={sig['confidence']}, strokes={sig['stroke_count']}")
```

### **Test stroke overlay:**
```bash
# Use /stroke_overlay endpoint
curl -X POST http://localhost:5000/stroke_overlay \
  -F "img1=@sig1.jpg" \
  -F "img2=@sig2.jpg" \
  -F "opacity=0.5" \
  -o overlay.png
```

---

## ✅ **What This Fixes**

1. ✅ **Detects signatures in documents** - Finds handwritten regions automatically
2. ✅ **Analyzes handwriting flow** - Detects same vs different handwriting style
3. ✅ **Compares strokes** - Stroke-by-stroke comparison
4. ✅ **Shows visual overlay** - See exactly which strokes match
5. ✅ **Ignores noise** - Filters out text, stamps, background
6. ✅ **Proper alignment** - Overlaps signatures correctly for stroke comparison
7. ✅ **Threshold at 0.85** - As requested, with multi-signal verification

---

## 🎨 **UI Recommendations**

Show users:
1. **Original images** (what they uploaded)
2. **Detected signature boxes** (if uploaded document)
3. **Preprocessed images** (black & white, aligned)
4. **Stroke overlay** (color-coded stroke comparison)
5. **Stroke metrics**:
   - Stroke count: 23 vs 21
   - Flow smoothness: 0.82 vs 0.79
   - Writing style: cursive vs cursive
6. **Combined breakdown**:
   - Cosine: 89%
   - ORB: 42%
   - SSIM: 78%
   - Stroke: 85%
   - **Combined: 87%**

This gives users complete transparency into why the system made its decision!

