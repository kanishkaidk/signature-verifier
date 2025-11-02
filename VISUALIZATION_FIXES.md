# Signature Verification Visualization Fixes & Redesign

## 🎯 Problem Summary

The visualization system had critical alignment and usability issues:
1. **Misaligned Overlays**: Heatmaps didn't align with signature images
2. **Unclear Visualizations**: Users couldn't interpret Grad-CAM or Saliency maps
3. **Broken Sliders**: Opacity and zoom controls didn't work properly
4. **No Explanations**: Technical visualizations lacked user-friendly descriptions

## ✅ Solution Architecture

### 1. **Consistent Canvas Normalization** (`backend/visualization.py`)

All signatures are normalized to a **fixed canvas size (220x155)** while preserving aspect ratio:
- Signatures are resized proportionally
- Centered on white canvas with padding
- **Perfect alignment** for comparison and overlays

```python
def normalize_to_canvas(img_pil, canvas_size=(256, 256)):
    # Resize preserving aspect ratio
    # Center on white canvas
    # Return normalized image
```

### 2. **Proper Overlay Blending**

The `overlay_heatmap()` function ensures:
- Heatmaps match base image dimensions exactly
- Smooth alpha blending (0-1 transparency)
- Multiple colormap options (jet, hot, red, green)

### 3. **Aligned Visualization Pipeline**

All visualization functions now:
1. Align both signatures to the same canvas
2. Generate heatmaps from aligned images
3. Overlay heatmaps on aligned images
4. Return perfectly aligned results

## 🧱 Module Structure

### `backend/visualization.py` - Core Visualization Engine

**Key Functions:**
- `normalize_to_canvas()`: Standardizes image size
- `align_pair_to_canvas()`: Aligns both signatures identically
- `overlay_heatmap()`: Blends heatmap with base image
- `create_dual_overlay()`: Side-by-side visualization
- `create_difference_map()`: Pixel-level difference analysis
- `explain_visualization()`: User-friendly explanations

### `backend/inference.py` - Updated Visualization Functions

All functions now support:
- `overlay_alpha` parameter (0-1)
- Automatic alignment before visualization
- Consistent output format (PIL Image)

**Updated Functions:**
- `generate_saliency_heatmap()` - Gradient-based attention
- `generate_gradcam_heatmap()` - Deep network activation
- `generate_dual_saliency_maps()` - Side-by-side attention
- `generate_difference_heatmap()` - Pixel differences
- `generate_saliency_difference()` - Attention pattern differences
- `generate_gradcam_dual()` - Dual Grad-CAM comparison

### `backend/app.py` - API Endpoints

All visualization endpoints now:
- Accept `opacity` parameter (0-1)
- Return properly aligned images
- Include explanation endpoint: `/viz_explanation/<viz_type>`

## 📊 Visualization Types & Explanations

### 1. **Saliency Heatmap** (`saliency`)
- **What it shows**: Where Signature 2 pixels affect similarity
- **Interpretation**: Red/orange = high importance for similarity
- **Color Legend**: Hot colors = important, Cool = less important

### 2. **Grad-CAM** (`gradcam`)
- **What it shows**: Neural network activation regions
- **Interpretation**: Bright red = high activation focus
- **Color Legend**: Red = high activation, Blue = low activation

### 3. **Dual Saliency** (`dual_saliency`)
- **What it shows**: Attention patterns for both signatures
- **Interpretation**: Red overlay = Signature 1 focus, Green = Signature 2 focus
- **Color Legend**: Red = Sig1 attention, Green = Sig2 attention

### 4. **Difference Map** (`difference`)
- **What it shows**: Pixel-level differences between signatures
- **Interpretation**: Red/yellow = different strokes, Blue = similar
- **Color Legend**: Hot colors = differences, Cool = similarities

### 5. **Saliency Difference** (`saliency_diff`)
- **What it shows**: Where attention patterns differ
- **Interpretation**: Yellow/red = attention differs significantly
- **Color Legend**: Hot colors = attention variation

## 🎨 Frontend Integration

### API Updates (`frontend-vite/src/lib/api.ts`)

All heatmap functions now accept `opacity` parameter:
```typescript
generateSaliencyHeatmap(img1, img2, opacity: number = 0.5)
generateGradCamHeatmap(img1, img2, opacity?: number)
generateDualSaliency(img1, img2, opacity: number = 0.5)
generateDifferenceHeatmap(img1, img2, opacity: number = 0.6)
generateSaliencyDifference(img1, img2, opacity: number = 0.6)
```

New function for explanations:
```typescript
getVizExplanation(vizType: string): Promise<VizExplanation>
```

### UI Improvements Needed

1. **Opacity Slider**: Connect to `overlay` state, update heatmap on change
2. **Zoom Slider**: Apply CSS transform `scale()` to image containers
3. **Explanation Cards**: Display `VizExplanation` when visualization is shown
4. **Legend Display**: Show `color_legend` below visualizations

## 🔧 Mathematical Alignment

### Canvas Normalization Algorithm

1. **Input**: Image with arbitrary size (w, h)
2. **Target**: Canvas (220, 155)
3. **Scale**: `scale = min(220/w, 155/h)`
4. **Resize**: `new_w = w * scale`, `new_h = h * scale`
5. **Center**: `offset_x = (220 - new_w) / 2`, `offset_y = (155 - new_h) / 2`
6. **Output**: Centered signature on white canvas

### Overlay Blending Formula

```python
blended = (1 - alpha) * base + alpha * heatmap
```

Where:
- `base`: Original signature image (RGB)
- `heatmap`: Colormap-applied heatmap (RGB)
- `alpha`: Opacity (0-1)

## 🚀 Usage Examples

### Generate Saliency Heatmap
```python
from backend.inference import generate_saliency_heatmap
from PIL import Image

img1 = Image.open("sig1.png")
img2 = Image.open("sig2.png")

# With 50% opacity overlay
result = generate_saliency_heatmap(img1, img2, overlay_alpha=0.5)
result.save("saliency.png")
```

### Get Explanation
```python
from backend.visualization import explain_visualization

explanation = explain_visualization("saliency")
print(explanation["title"])
print(explanation["description"])
print(explanation["interpretation"])
```

### API Call with Opacity
```bash
curl -X POST http://localhost:5000/saliency \
  -F "img1=@sig1.png" \
  -F "img2=@sig2.png" \
  -F "opacity=0.7" \
  --output saliency.png
```

## 📝 Next Steps for Frontend

1. **Update Verify.tsx**:
   - Connect opacity slider to `overlay` state
   - Regenerate heatmap when opacity changes
   - Apply zoom with CSS transforms
   - Display explanation cards for each viz type

2. **Add Tooltips**:
   - Hover tooltips explaining each button
   - "What does this visualization show?"

3. **Visualization Guide Section**:
   - Dedicated "AI's Focus" section
   - Toggle between visualization types
   - Show side-by-side comparisons

## ✅ Testing Checklist

- [x] Signatures align to same canvas size
- [x] Heatmaps perfectly overlay on signatures
- [x] Opacity parameter affects blending
- [x] All visualizations return proper images
- [x] Explanations available for each type
- [ ] Frontend sliders connected and functional
- [ ] Explanations displayed in UI
- [ ] Zoom functionality working

## 🎓 User Guide: How to Interpret Visualizations

### For Non-Technical Users

**"What am I looking at?"**
- The colored overlays show where the AI model focuses when comparing signatures
- **Red/Orange areas** = High importance (model pays attention here)
- **Blue/Dark areas** = Low importance (less focus)

**"Why are they different colors?"**
- Each visualization type uses different colors to show different aspects:
  - **Saliency**: Shows which pixels matter for similarity
  - **Grad-CAM**: Shows deep learning network activation
  - **Difference Map**: Highlights where signatures differ visually

**"What does a good match look like?"**
- Similar signatures will have:
  - Similar attention patterns (saliency maps look alike)
  - High similarity score (>85%)
  - Few bright red spots in difference map
  - Aligned focus areas in Grad-CAM

## 🔒 Security Note

All visualizations are generated in memory only. No heatmap images are permanently stored. The system processes signatures, generates overlays, and serves them directly to the frontend without saving to disk.

