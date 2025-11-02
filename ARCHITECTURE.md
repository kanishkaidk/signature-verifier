# SignGuard Architecture

## 🏗️ Modular Pipeline Architecture

### Components

1. **Signature Detector** (`backend/signature_detector.py`)
   - Detects and localizes signatures in document images
   - Uses contour detection and morphological operations
   - SHA-256 hashing for integrity verification
   - Automatic cropping of signature regions

2. **Preprocessing Module** (`backend/preprocessing.py`)
   - Alignment: Centering, cropping, deskewing
   - Normalization: Consistent sizing and orientation
   - Binarization: Adaptive thresholding for clean signatures

3. **Verification Engine** (`backend/inference.py`)
   - Siamese Network inference
   - Multiple visualization types:
     - Saliency Maps
     - Grad-CAM (single & dual)
     - Dual Saliency Maps
     - Difference Heatmaps
     - Saliency Difference Maps

4. **Pipeline Controller** (`backend/pipeline.py`)
   - Orchestrates: Detection → Preprocessing → Verification → Visualization
   - Configurable: Enable/disable detection, alignment
   - Modular design for easy extension

5. **Security Layer** (`backend/app.py`)
   - SHA-256 image hashing
   - File validation (type, size)
   - Rate limiting (20 req/min per IP)
   - UUID-based identifiers (no PII)
   - No permanent image storage

## 🔄 Processing Flow

```
User Upload
    ↓
[Security] Hash & Validate
    ↓
[Detection] Extract Signatures (optional)
    ↓
[Preprocessing] Align & Normalize
    ↓
[Verification] Siamese Network
    ↓
[Visualization] Grad-CAM/Saliency
    ↓
[Reporting] PDF Generation
```

## 🎯 Visualization Fixes

### Issues Fixed:
- ✅ **Grad-CAM Alignment**: Now uses `cv2.normalize()` for proper contrast
- ✅ **Saliency Maps**: Normalized with `cv2.NORM_MINMAX` for better visibility
- ✅ **Dual Visualizations**: Side-by-side with color coding (red/green)
- ✅ **Difference Maps**: Proper normalization and colormaps
- ✅ **Overlay Alignment**: Heatmaps match preprocessed image dimensions

### Normalization:
All heatmaps now use `cv2.normalize()` with `cv2.NORM_MINMAX` for consistent contrast and visibility.

## 🔒 Security Features

1. **Image Hashing**: SHA-256 hash computed for every upload
2. **Integrity Verification**: Hashes stored in metadata
3. **No PII**: UUIDs instead of filenames
4. **Memory-Only**: Images processed in RAM, never saved to disk
5. **Rate Limiting**: Prevents abuse
6. **File Validation**: Type and size checks

## 📡 API Endpoints

### Core
- `POST /predict` - Verify signatures
- `POST /batch_predict` - Batch verification
- `GET /metrics` - Model performance metrics
- `GET /history` - Verification history
- `GET /health` - Server status

### Visualizations
- `POST /saliency` - Single saliency map
- `POST /gradcam` - Grad-CAM (single/dual)
- `POST /dual_saliency` - Dual saliency maps
- `POST /difference` - Feature difference heatmap
- `POST /saliency_diff` - Saliency pattern differences

### Utilities
- `POST /detect_signatures` - Detect signature regions
- `POST /align_preview` - Preview aligned signatures
- `POST /report` - Generate PDF report
- `GET /disclaimer` - Security/privacy info

## 🚀 Usage

### Basic Verification
```python
from backend.pipeline import get_pipeline

pipeline = get_pipeline()
result = pipeline.process(img1_bytes, img2_bytes, auto_detect_signatures=True)
```

### With Detection
```python
# Auto-detect and crop signatures from documents
result = pipeline.process(document1_bytes, document2_bytes, auto_detect_signatures=True)
```

### Generate Visualization
```python
# Generate specific visualization type
heatmap = pipeline.generate_visualization(img1, img2, viz_type="gradcam")
```

## 🔧 Configuration

Pipeline can be configured:
```python
pipeline = VerificationPipeline(
    enable_detection=True,  # Auto-detect signatures
    enable_alignment=True   # Auto-align before verification
)
```

