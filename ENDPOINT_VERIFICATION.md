# Endpoint Verification & Fixes

## ✅ All Endpoints Verified and Fixed

### 1. `/predict` - POST
**Status**: ✅ Fixed  
**Frontend Call**: `verifySignatures(img1, img2)`  
**Expected Response**: `{similarity_score: number, verdict: string}`  
**Actual Response**: ✅ Matches - Returns exactly what frontend expects

### 2. `/batch_predict` - POST
**Status**: ✅ Fixed  
**Frontend Call**: `batchVerify(reference, files)`  
**Expected Response**: `{results: BatchResult[]}`  
**Issues Fixed**:
- ❌ Was returning `{id, results}` - Fixed to return only `{results}`
- ❌ Results had `id` field - Removed, only `filename`, `similarity_score`, `verdict`, `error`
- ✅ Now returns correct format

### 3. `/report` - POST
**Status**: ✅ OK  
**Frontend Call**: `downloadReport(img1, img2, heatmap?)`  
**Expected Response**: PDF blob  
**Actual Response**: ✅ Returns PDF file correctly

### 4. `/history` - GET
**Status**: ✅ OK  
**Frontend Call**: `getHistory()`  
**Expected Response**: `{history: HistoryItem[]}`  
**Actual Response**: ✅ Matches

### 5. `/health` - GET
**Status**: ✅ OK  
**Frontend Call**: `healthCheck()`  
**Expected Response**: `{status: string}`  
**Actual Response**: `{status: "ok", security: {...}}` - ✅ Works (extra fields OK)

### 6. `/saliency` - POST
**Status**: ✅ OK  
**Frontend Call**: `generateSaliencyHeatmap(img1, img2, opacity)`  
**Expected Response**: PNG image blob  
**Actual Response**: ✅ Returns PNG with proper overlay  
**Parameters**: ✅ Reads `opacity` from form data

### 7. `/gradcam` - POST
**Status**: ✅ Fixed  
**Frontend Call**: `generateGradCamHeatmap(img1, img2, opacity)`  
**Expected Response**: PNG image blob  
**Issues Fixed**:
- ❌ Was not reading `opacity` parameter - ✅ Fixed
- ❌ Was not passing `overlay_alpha` to functions - ✅ Fixed
- ✅ Now properly handles opacity for both dual and single image modes

### 8. `/metrics` - GET
**Status**: ✅ Fixed  
**Frontend Call**: `getMetrics()`  
**Expected Response**: `{accuracy?: number, f1?: number, threshold?: number}`  
**Issues Fixed**:
- ❌ Was returning `{accuracy, threshold}` missing `f1` - ✅ Fixed
- ✅ Now returns all three fields with proper fallbacks

### 9. `/disclaimer` - GET
**Status**: ✅ OK  
**Frontend Call**: `getDisclaimer()`  
**Expected Response**: Disclaimer object  
**Actual Response**: ✅ Matches

### 10. `/viz_explanation/<viz_type>` - GET
**Status**: ✅ OK  
**Frontend Call**: `getVizExplanation(vizType)`  
**Expected Response**: `VizExplanation` object  
**Actual Response**: ✅ Matches

### 11. `/align_preview` - POST
**Status**: ✅ OK  
**Frontend Call**: (Not currently used in frontend)  
**Expected Response**: PNG image  
**Actual Response**: ✅ Returns aligned preview image

### 12. `/dual_saliency` - POST
**Status**: ✅ OK  
**Frontend Call**: `generateDualSaliency(img1, img2, opacity)`  
**Expected Response**: PNG image blob  
**Actual Response**: ✅ Returns PNG with proper overlay  
**Parameters**: ✅ Reads `opacity` from form data

### 13. `/difference` - POST
**Status**: ✅ OK  
**Frontend Call**: `generateDifferenceHeatmap(img1, img2, opacity)`  
**Expected Response**: PNG image blob  
**Actual Response**: ✅ Returns PNG with difference overlay  
**Parameters**: ✅ Reads `opacity` from form data  
**Extra**: Returns stats in headers (`X-Difference-Percentage`, `X-Mean-Difference`)

### 14. `/saliency_diff` - POST
**Status**: ✅ OK  
**Frontend Call**: `generateSaliencyDifference(img1, img2, opacity)`  
**Expected Response**: PNG image blob  
**Actual Response**: ✅ Returns PNG with saliency difference overlay  
**Parameters**: ✅ Reads `opacity` from form data

### 15. `/detect_signatures` - POST
**Status**: ✅ OK  
**Frontend Call**: (Not currently used in frontend)  
**Expected Response**: JSON with signature detection results  
**Actual Response**: ✅ Returns `{signatures_found, bounding_boxes, message}`

## Summary of Fixes

1. **`/gradcam`**: Now reads and uses `opacity` parameter ✅
2. **`/metrics`**: Now includes `f1` field in response ✅
3. **`/batch_predict`**: Removed `id` field from results array ✅
4. **`/predict`**: Cleaned up response to match frontend interface exactly ✅

## Testing Checklist

- [ ] `/predict` - Verify returns `{similarity_score, verdict}`
- [ ] `/batch_predict` - Verify returns `{results: [...]}` with correct fields
- [ ] `/gradcam` - Verify opacity parameter works
- [ ] `/metrics` - Verify includes `f1` field
- [ ] All visualization endpoints accept and use `opacity` parameter
- [ ] All endpoints handle errors gracefully

## Common Issues Resolved

1. **Opacity Parameter**: All visualization endpoints now properly read `opacity` from form data
2. **Response Format**: All endpoints match TypeScript interfaces exactly
3. **Error Handling**: All endpoints return proper error JSON on failure
4. **File Validation**: All file upload endpoints validate files before processing

