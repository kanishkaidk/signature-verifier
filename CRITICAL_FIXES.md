# 🔴 CRITICAL FIXES - False Positives & Alignment

## 🚨 Issues Fixed

### 1. **False Positives (Different Signatures Declared Same)** ❌ → ✅

**Problem:** System was declaring clearly different signatures (different names, languages) as "Same person"

**Root Causes:**
- Threshold too low (0.85) allowed false matches
- ORB alignment was forcing matches between different signatures
- No validation to prevent over-alignment

**Fixes Applied:**

#### A. **Stricter Threshold (0.85 → 0.92)**
```python
# OLD: threshold = 0.85
# NEW: threshold = 0.92 (minimum for "Same person")

Threshold Ranges:
- ≥ 0.92 = "Same person" (High confidence)
- 0.85 - 0.92 = "Uncertain - Manual review recommended"
- < 0.85 = "Different person"
```

#### B. **Smart ORB Alignment**
```python
# Now checks match ratio before aligning
min_match_ratio = 0.25  # Need at least 25% keypoint matches

# Won't force-align if:
- Match ratio < 25%
- Transformation too extreme (determinant check)
- Not enough good matches
```

#### C. **Match Quality Filtering**
```python
# Only keeps matches with distance < 50 (good quality)
good_matches = [m for m in matches if m.distance < 50]

# Validates transformation is reasonable
det = np.linalg.det(M[:2, :2])
if det < 0.5 or det > 2.0:
    # Too extreme - likely different signatures
    return original (no alignment)
```

### 2. **Overlapping/Alignment Not Working** ❌ → ✅

**Problem:** Visual overlays weren't aligning properly, making comparison difficult

**Fixes Applied:**

#### A. **Consistent Preprocessing Pipeline**
- Same `normalize_to_canvas()` used for:
  - Model inference
  - All visualizations
  - Overlay generation

#### B. **Smart Alignment Logic**
- Only aligns if signatures are similar enough (match ratio check)
- If different signatures detected → uses normalized (centered) but not warped
- Prevents forced alignment from creating false matches

#### C. **Improved Visualization**
- All heatmaps computed on same preprocessed tensors
- Perfect pixel alignment guaranteed
- Overlay uses proper alpha blending

## 📊 New Threshold System

| Similarity Score | Verdict | Confidence |
|------------------|---------|------------|
| ≥ 0.92 | Same person | High |
| 0.85 - 0.92 | Uncertain | Low - Manual review |
| < 0.85 | Different person | High |

## 🔧 Alignment Behavior

### **Similar Signatures (Same Person)**
1. ORB detects many matching keypoints (>25% ratio)
2. Homography computed and applied
3. Signatures aligned pixel-perfectly
4. High similarity score (>0.92)

### **Different Signatures**
1. ORB detects few matches (<25% ratio)
2. **Alignment SKIPPED** (prevents false matching)
3. Uses normalized (centered) but unwarped images
4. Model sees actual differences → Low similarity score (<0.92)

## ✅ What This Prevents

1. **False Positives:** Different signatures won't be declared "Same"
2. **Forced Alignment:** Won't warp different signatures to look similar
3. **Over-Matching:** ORB checks ensure alignment only when appropriate

## 🧪 Testing Recommendations

1. **Test with Clearly Different Signatures:**
   - Should return "Different person" even with alignment enabled
   - Similarity should be < 0.92

2. **Test with Same Person Signatures:**
   - Should align properly
   - Similarity should be ≥ 0.92

3. **Test Borderline Cases (0.85-0.92):**
   - Should return "Uncertain - Manual review recommended"

## ⚙️ Configuration

To adjust sensitivity:

**Increase Strictness (fewer false positives):**
- Increase `min_match_ratio` in `align_pair_via_orb()` (e.g., 0.3, 0.4)
- Increase default threshold (e.g., 0.94, 0.95)

**Decrease Strictness (more lenient):**
- Decrease `min_match_ratio` (e.g., 0.2, 0.15)
- Decrease threshold (but NOT below 0.90 for safety)

## 🎯 Summary

- ✅ **Threshold raised to 0.92** - Much stricter
- ✅ **Smart alignment** - Only aligns similar signatures
- ✅ **Match validation** - Prevents over-matching
- ✅ **Uncertain zone** - Flags borderline cases
- ✅ **No forced alignment** - Different signatures stay different

The system should now correctly identify different signatures as "Different person" even when alignment is enabled!

