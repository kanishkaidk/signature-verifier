# ✅ UI Endpoints - ALL CONNECTED

## 🎯 **Visualization Buttons in UI**

All visualization endpoints are now connected and working in the frontend:

### **1. 🖋️ Stroke Overlay** (`/stroke_overlay`)
**Button:** "🖋️ Stroke Overlay" (Yellow highlighted button)

**Shows:**
- 🔴 **Red strokes** = Unique to Signature 1
- 🟢 **Green strokes** = Unique to Signature 2
- 🟡 **Yellow/Overlapping** = Matched strokes (both signatures)

**What it does:**
- Extracts individual pen strokes from both signatures
- Shows exact stroke alignment after preprocessing
- Color-codes which strokes match and which don't
- Perfect for seeing handwriting flow and detecting forgery

### **2. 📊 Preprocessing Preview** (`/preprocessed_preview`)
**Button:** "📊 Preprocessing" (Blue highlighted button)

**Shows 4-step pipeline for each signature:**
- **Original** | **Denoised** (Noise Removed) | **Preprocessed** | **Aligned**

**What it shows:**
- **Original:** What user uploaded
- **Denoised:** After noise removal (only signature strokes remain)
- **Preprocessed:** Centered, resized, normalized
- **Aligned:** After ORB-based alignment (perfect overlap)

**Perfect for:**
- Seeing how signatures are extracted from documents
- Understanding noise removal process
- Verifying alignment quality

### **3. Verification Signals Display**
**Automatically shown after verification:**

**Left Panel:**
- Cosine Similarity (50% weight)
- ORB Match Ratio (20% weight)
- SSIM (15% weight)
- Stroke Similarity (15% weight) ⭐ NEW
- **Combined Score** (weighted average)

**Right Panel - Handwriting Analysis:**
- **Writing Style:** Cursive / Printed / Mixed
- **Flow Smoothness:** How fluid the handwriting is (0-100%)
- **Stroke Count:** Number of strokes in each signature
- **Stroke Comparison:**
  - Count Similarity
  - Length Similarity
  - Direction Similarity
  - Pressure Similarity

### **4. Other Visualizations (Already Working)**
- ✅ Saliency Heatmap
- ✅ Grad-CAM
- ✅ Dual Saliency
- ✅ Difference Map
- ✅ Saliency Diff

---

## 📋 **How to Use**

### **Step 1: Upload Signatures**
1. Upload two signature images
2. Click "Verify Signatures"

### **Step 2: See Results**
- **Combined Score** appears (multi-signal verification)
- **Verdict** (Same/Different/Uncertain)
- **Detailed Metrics** panel shows all signals
- **Handwriting Analysis** panel shows stroke info

### **Step 3: Visualize**
- Click **"🖋️ Stroke Overlay"** to see exact stroke alignment
- Click **"📊 Preprocessing"** to see noise removal and alignment steps
- Adjust opacity slider to control overlay transparency
- Use zoom slider to magnify

---

## 🔍 **What You'll See**

### **Stroke Overlay:**
```
Red strokes = Only in Signature 1
Green strokes = Only in Signature 2  
Yellow regions = Overlapping matched strokes
```

### **Preprocessing Preview:**
```
Row 1 (Sig1): [Original] [Denoised] [Preprocessed] [Aligned]
Row 2 (Sig2): [Original] [Denoised] [Preprocessed] [Aligned]
```

### **Handwriting Analysis:**
```
Writing Style: cursive vs cursive ✅
Flow Smoothness: 82% vs 79% ✅
Stroke Count: 23 vs 21 strokes ✅
Stroke Length Similarity: 88% ✅
```

---

## ✅ **All Endpoints Working:**

1. ✅ `/predict` - Main verification (returns detailed_metrics)
2. ✅ `/stroke_overlay` - Stroke alignment visualization
3. ✅ `/preprocessed_preview` - Preprocessing steps
4. ✅ `/saliency` - Saliency heatmap
5. ✅ `/gradcam` - Grad-CAM visualization
6. ✅ `/dual_saliency` - Dual saliency maps
7. ✅ `/difference` - Difference heatmap
8. ✅ `/saliency_diff` - Saliency difference
9. ✅ `/detect_signatures` - Auto-detect in documents
10. ✅ `/align_preview` - Alignment preview

**Everything is connected and working!** 🎉

