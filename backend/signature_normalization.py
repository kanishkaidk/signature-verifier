"""
5-Stage Signature Normalization Pipeline
Ensures perfect alignment for accurate verification.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional
from backend.advanced_alignment import pil_to_numpy, numpy_to_pil


# ============================================================================
# STAGE 1: SIGNATURE DETECTION
# ============================================================================

def detect_signatures_robust(img_rgb: np.ndarray) -> List[Dict]:
    """
    IMPROVED signature detection using histogram equalization + adaptive thresholding.
    Based on robust OpenCV contour-based approach that handles:
    - Variable lighting and scan contrast
    - Text, boxes, stamps nearby
    - Faint signatures or angled scans
    
    Returns:
        List of dicts with keys: 'bbox' (x, y, w, h), 'confidence', 'image_crop'
    """
    # Handle both RGB and grayscale
    if len(img_rgb.shape) == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.copy()
    
    h, w = gray.shape
    
    # IMPROVEMENT 1: Increase contrast using histogram equalization
    # This helps with faint signatures and variable lighting
    gray_eq = cv2.equalizeHist(gray)
    
    # IMPROVEMENT 2: Adaptive threshold with better parameters
    # Block size 25 (instead of 35) for better local adaptation
    # C value 15 for better noise handling
    binary = cv2.adaptiveThreshold(
        gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 15  # Changed from 35, 15
    )
    
    # IMPROVEMENT 3: Morphological operations to clean up
    # Small kernel (3x3) to remove tiny specks without losing signature strokes
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove small noise
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)  # Connect signature strokes
    
    # Find all contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = w_box * h_box  # Use bounding box area (more reliable than contourArea for signatures)
        aspect_ratio = w_box / (h_box + 1e-6)
        
        # IMPROVEMENT 4: Better filtering based on area and aspect ratio
        # Signature-like: wide (aspect > 1.5), reasonable size (4000-150000 pixels)
        # This range handles small signatures and large ones
        if (4000 < area < 150000 and  # Reasonable signature size range
            1.5 < aspect_ratio < 8.0):  # Wide, not too tall
            
            # Extract signature region (with padding)
            padding = 15
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + w_box + padding)
            y2 = min(h, y + h_box + padding)
            
            # Ensure we're cropping from original RGB image
            if len(img_rgb.shape) == 3:
                crop = img_rgb[y1:y2, x1:x2].copy()
            else:
                # Grayscale - convert to RGB for consistency
                crop_gray = img_rgb[y1:y2, x1:x2].copy()
                crop = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)
            
            # IMPROVED confidence scoring with location awareness
            # Signatures can be anywhere, but we prefer certain locations
            area_score = min(1.0, area / 50000.0)  # Normalize to max expected signature area
            aspect_score = 1.0 if 2.0 < aspect_ratio < 6.0 else 0.7
            
            # Location-based scoring (signatures often at bottom-right, but can be anywhere)
            # Don't penalize top/middle locations - just give bonus to bottom-right
            h, w = gray.shape
            y_center = y + h_box // 2
            x_center = x + w_box // 2
            
            # Bottom-right bonus (but not required)
            bottom_bonus = 0.2 if y_center > h * 0.7 else 0.0  # Bottom 30% of image
            right_bonus = 0.1 if x_center > w * 0.6 else 0.0  # Right 40% of image
            location_score = 0.5 + bottom_bonus + right_bonus  # Base 0.5, max 0.8
            
            # Stroke density check - signatures have dense horizontal strokes
            crop_binary = clean[y1:y2, x1:x2] if y1 < clean.shape[0] and x1 < clean.shape[1] else clean
            if crop_binary.size > 0:
                stroke_density = np.sum(crop_binary > 0) / crop_binary.size
                density_score = min(1.0, stroke_density * 10)  # Normalize density
            else:
                density_score = 0.5
            
            # Combined confidence
            confidence = (area_score * 0.3 + aspect_score * 0.2 + location_score * 0.3 + density_score * 0.2)
            
            candidates.append({
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'confidence': float(confidence),
                'image_crop': crop,
                'area': int(area)
            })
    
    # If no candidates found, return whole image as fallback
    if not candidates:
        print("⚠️ No signature region detected - using whole image")
        # Return whole image as a single candidate
        if len(img_rgb.shape) == 3:
            crop = img_rgb.copy()
        else:
            crop = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        
        candidates.append({
            'bbox': (0, 0, w, h),
            'confidence': 0.3,  # Low confidence since detection failed
            'image_crop': crop,
            'area': w * h
        })
    
    # Sort by confidence (highest first), then by area (largest first)
    candidates.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
    
    return candidates


# ============================================================================
# STAGE 2: NOISE REMOVAL
# ============================================================================

def clean_signature(img_rgb: np.ndarray) -> np.ndarray:
    """
    IMPROVED noise reduction using adaptive thresholding with MEAN_C.
    
    Handles:
    - Variable lighting and scan contrast
    - Text/background leakage
    - Small noise blobs
    - Preserves signature stroke integrity
    
    Returns:
        Clean signature: pure black strokes on pure white background (RGB format)
    """
    if len(img_rgb.shape) == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.copy()
    
    # IMPROVEMENT 1: Gaussian blur with smaller kernel (3x3) for less smoothing
    # Preserves fine signature details while reducing noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # FIELD-TESTED: Adaptive threshold with GAUSSIAN_C (more robust than MEAN_C)
    # Block size 35 (field-tested value), C=10 (threshold adjustment)
    # THRESH_BINARY_INV: dark pixels (signature) become white, background becomes black
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # FIELD-TESTED: GAUSSIAN_C for signatures
        cv2.THRESH_BINARY_INV, 35, 10  # FIELD-TESTED: blockSize=35
    )
    
    # IMPROVEMENT 3: Morphological opening to remove small noise blobs
    # Small kernel (2x2) to remove tiny specks without affecting signature strokes
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # IMPROVEMENT 4: Optional dilation to slightly thicken strokes
    # Compensates for any thinning from opening, helps preserve stroke continuity
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    
    # Remove very small connected components (additional noise removal)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    
    # Keep only components above threshold (signature strokes)
    min_area = 20
    mask = np.zeros_like(cleaned)
    
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == i] = 255
    
    cleaned = mask
    
    # Convert to pure white background with black signature strokes
    # Final format: pure black (0,0,0) strokes on pure white (255,255,255) background
    result = np.ones((cleaned.shape[0], cleaned.shape[1], 3), dtype=np.uint8) * 255  # White background
    result[cleaned > 0] = [0, 0, 0]  # Black strokes
    
    return result


# ============================================================================
# STAGE 3: BASELINE & ENDPOINT ALIGNMENT
# ============================================================================

def detect_baseline(binary_img: np.ndarray) -> float:
    """
    FIELD-TESTED baseline detection - finds the "medium line" (lowest ink pixel).
    
    Uses the bottom-most stroke pixel (max Y coordinate of non-zero pixels).
    This is the canonical method used in forensic handwriting analysis.
    
    Returns:
        Y-coordinate of baseline (pixels from top)
    """
    # Handle both grayscale (1 channel) and RGB (3 channels)
    if len(binary_img.shape) == 3:
        # 3-channel image - convert to grayscale
        if binary_img.shape[2] == 3:
            binary = cv2.cvtColor(binary_img, cv2.COLOR_RGB2GRAY)
        else:
            binary = binary_img[:, :, 0]  # Take first channel if unexpected shape
    else:
        # Already grayscale
        binary = binary_img.copy()
    
    # Ensure it's a 2D array
    if len(binary.shape) > 2:
        binary = binary[:, :, 0]
    
    h, w = binary.shape
    
    # FIELD-TESTED METHOD: Find lowest ink pixel (bottom-most stroke)
    # For cleaned signatures: black strokes (value < 128) or white strokes on black (value > 128)
    # We check for non-background pixels
    
    # Method: Find all non-background pixels
    # Background is typically 0 (black) or 255 (white) depending on format
    # Strokes are the opposite
    
    # Check if image is inverted (black background, white strokes)
    mean_val = np.mean(binary)
    if mean_val < 127:
        # Dark image - probably black background with white strokes
        # Find white pixels (strokes)
        ys, xs = np.where(binary > 128)
    else:
        # Light image - probably white background with black strokes
        # Find black pixels (strokes)
        ys, xs = np.where(binary < 128)
    
    # If no strokes found, return default baseline (near bottom)
    if len(ys) == 0:
        return float(h - 10)  # 10px from bottom
    
    # Baseline is the lowest (maximum Y) stroke pixel
    baseline_y = float(np.max(ys))
    
    return baseline_y


def align_baseline(binary_img: np.ndarray, target_baseline: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Align signature to a target baseline position.
    
    Args:
        binary_img: Signature image (RGB or grayscale)
        target_baseline: Target Y-coordinate for baseline (None = keep current)
    
    Returns:
        (aligned_image, shift_amount) - same format as input
    """
    # Detect current baseline using the full image
    current_baseline = detect_baseline(binary_img)
    
    # Get dimensions
    if len(binary_img.shape) == 3:
        h, w, c = binary_img.shape
        is_rgb = True
    else:
        h, w = binary_img.shape
        is_rgb = False
    
    if target_baseline is None:
        target_baseline = float(h - 20)  # Default: 20px from bottom
    
    # Calculate shift needed (positive shift = move down, negative = move up)
    shift = target_baseline - current_baseline
    
    # If shift is very small, don't bother aligning
    if abs(shift) < 2:
        return binary_img, 0.0
    
    # Apply translation - keep original format
    # Translation matrix: [1, 0, tx], [0, 1, ty]
    # ty = shift (move vertically)
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    
    # Set border value based on image type (white background)
    if is_rgb:
        border_value = (255, 255, 255)  # White for RGB
    else:
        border_value = 255  # White for grayscale
    
    aligned = cv2.warpAffine(
        binary_img,  # Use original image
        M, (w, h),
        borderValue=border_value,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    
    return aligned, float(shift)


def align_pair_baseline(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    IMPROVED baseline alignment - aligns both signatures to the same baseline.
    
    CRITICAL: This ensures both signatures sit on the same "writing line".
    Without this, signatures appear misaligned even if they're the same person.
    
    Uses improved baseline detection that finds actual signature strokes.
    Validates baselines and uses proportional alignment if needed.
    
    Args:
        img1, img2: Signature images (RGB or grayscale) - should be cropped signatures, not full documents
    
    Returns:
        (aligned_img1, aligned_img2) - both with same baseline Y coordinate
    """
    # Get image dimensions
    h1, w1 = img1.shape[:2] if len(img1.shape) == 2 else img1.shape[:2]
    h2, w2 = img2.shape[:2] if len(img2.shape) == 2 else img2.shape[:2]
    
    # Detect baselines (Y coordinate of bottom-most signature stroke)
    baseline1 = detect_baseline(img1)
    baseline2 = detect_baseline(img2)
    
    # CRITICAL: Validate baselines are reasonable
    # If baseline is too close to image edges, it might be wrong (detected document edge, not signature)
    if baseline1 > h1 * 0.95 or baseline1 < h1 * 0.05:
        print(f"⚠️ Baseline1 ({baseline1:.1f}) seems wrong for image height {h1}, using 85%")
        baseline1 = h1 * 0.85  # Use 85% from top as fallback
    
    if baseline2 > h2 * 0.95 or baseline2 < h2 * 0.05:
        print(f"⚠️ Baseline2 ({baseline2:.1f}) seems wrong for image height {h2}, using 85%")
        baseline2 = h2 * 0.85  # Use 85% from top as fallback
    
    # Use proportional alignment (same % of image height) instead of absolute pixels
    # This works better when images are different sizes
    ratio1 = baseline1 / h1
    ratio2 = baseline2 / h2
    target_ratio = max(ratio1, ratio2)  # Use lower baseline (higher ratio)
    
    # Convert back to absolute pixels for each image
    target_baseline1 = h1 * target_ratio
    target_baseline2 = h2 * target_ratio
    
    # Align both to their respective target baselines
    img1_aligned, shift1 = align_baseline(img1, target_baseline1)
    img2_aligned, shift2 = align_baseline(img2, target_baseline2)
    
    # Verify alignment worked
    final_baseline1 = detect_baseline(img1_aligned)
    final_baseline2 = detect_baseline(img2_aligned)
    
    # Calculate final ratio to check alignment quality
    h1_final = img1_aligned.shape[0]
    h2_final = img2_aligned.shape[0]
    final_ratio1 = final_baseline1 / h1_final
    final_ratio2 = final_baseline2 / h2_final
    
    # Debug output
    print(f"📐 Baseline alignment: Before - sig1={baseline1:.1f} ({ratio1:.3f}), sig2={baseline2:.1f} ({ratio2:.3f}) | "
          f"Target ratio={target_ratio:.3f} | Shifts: sig1={shift1:.1f}, sig2={shift2:.1f} | "
          f"After - sig1={final_baseline1:.1f} ({final_ratio1:.3f}), sig2={final_baseline2:.1f} ({final_ratio2:.3f})")
    
    # Check alignment quality by comparing ratios (should be very close)
    ratio_diff = abs(final_ratio1 - final_ratio2)
    
    if ratio_diff > 0.05:  # More than 5% difference in ratio
        print(f"⚠️ Alignment ratio gap too large ({ratio_diff:.3f}), trying direct pixel alignment")
        # Fallback: align to same absolute pixel position (if images are similar size)
        if abs(h1_final - h2_final) < 10:  # Images are similar height
            target_pixel = max(final_baseline1, final_baseline2)
            img1_aligned, _ = align_baseline(img1, target_pixel)
            img2_aligned, _ = align_baseline(img2, target_pixel)
            
            final_baseline1 = detect_baseline(img1_aligned)
            final_baseline2 = detect_baseline(img2_aligned)
            print(f"🔄 Direct pixel alignment: target={target_pixel:.1f}, after: sig1={final_baseline1:.1f}, sig2={final_baseline2:.1f}, diff={abs(final_baseline1 - final_baseline2):.1f}px")
    
    return img1_aligned, img2_aligned


# ============================================================================
# STAGE 4: SIZE NORMALIZATION & PADDING
# ============================================================================

def center_and_pad(img: np.ndarray, size: int = 256) -> np.ndarray:
    """
    FIELD-TESTED center and pad - trims all empty borders and re-centers on square canvas.
    
    This is critical for ORB matching - both signatures must be:
    - Same canvas size (square)
    - Centered identically
    - Scaled to fit within canvas
    
    Args:
        img: Signature image (grayscale, black strokes on white background)
        size: Target square canvas size (default 256 for model input compatibility)
    
    Returns:
        Centered signature on square canvas (grayscale)
    """
    # Handle RGB images
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Find all non-zero (stroke) pixels
    ys, xs = np.where(gray > 0)
    
    # If no strokes found, return empty canvas
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    
    # Get bounding box
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    
    # Crop to signature region
    crop = gray[y1:y2+1, x1:x2+1]
    h, w = crop.shape[:2]
    
    # Scale to fit within square canvas (preserve aspect ratio)
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    h2, w2 = resized.shape[:2]
    
    # Create square canvas (white background)
    canvas = np.ones((size, size), dtype=np.uint8) * 255
    
    # Center the resized signature
    yoff = (size - h2) // 2
    xoff = (size - w2) // 2
    canvas[yoff:yoff+h2, xoff:xoff+w2] = resized
    
    # Invert to get black strokes on white background (if needed)
    # Check if we need inversion based on mean pixel value
    if np.mean(canvas) > 127:
        # Mostly white - assume we want black strokes
        # Find strokes and ensure they're black
        stroke_mask = canvas < 128
        canvas[stroke_mask] = 0  # Black strokes
        canvas[~stroke_mask] = 255  # White background
    
    return canvas


def resize_and_pad(
    img: np.ndarray,
    size: Tuple[int, int] = (256, 256),
    preserve_aspect: bool = True
) -> np.ndarray:
    """
    Resize signature to target size while preserving aspect ratio,
    then center it in a fixed canvas with padding.
    
    Uses center_and_pad internally for better ORB matching.
    
    Args:
        img: Signature image (RGB or grayscale)
        size: Target (width, height) - if square, use center_and_pad
        preserve_aspect: If True, maintain aspect ratio and pad
    
    Returns:
        Normalized image of exact size
    """
    target_w, target_h = size
    
    # If square canvas, use field-tested center_and_pad
    if target_w == target_h and preserve_aspect:
        # Use square size (use larger dimension)
        square_size = max(target_w, target_h)
        centered = center_and_pad(img, size=square_size)
        # Resize to exact target if needed
        if square_size != target_w:
            centered = cv2.resize(centered, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return centered
    
    # Non-square or direct resize - use original method
    if len(img.shape) == 3:
        h, w, c = img.shape
        is_color = True
    else:
        h, w = img.shape
        is_color = False
    
    if preserve_aspect:
        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calculate padding to center
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        # Add padding (white background)
        if is_color:
            pad_value = (255, 255, 255)
        else:
            pad_value = 255
        
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=pad_value
        )
    else:
        # Direct resize (may distort aspect ratio)
        img_padded = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
    return img_padded


# ============================================================================
# STAGE 5: BRIGHTNESS & STROKE NORMALIZATION
# ============================================================================

def match_brightness(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Match brightness and contrast of img2 to img1.
    
    Uses mean and standard deviation normalization.
    
    Args:
        img1: Reference image (RGB or grayscale)
        img2: Image to normalize (RGB or grayscale)
    
    Returns:
        Normalized img2 matching img1's brightness/contrast (same format as input)
    """
    # Check if images are RGB
    is_rgb = len(img1.shape) == 3 and img1.shape[2] == 3
    
    # Convert to grayscale for statistics (handles both RGB and grayscale safely)
    if is_rgb:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        # Already grayscale, but ensure 2D
        gray1 = img1.copy()
        gray2 = img2.copy()
        if len(gray1.shape) > 2:
            gray1 = gray1[:, :, 0]
        if len(gray2.shape) > 2:
            gray2 = gray2[:, :, 0]
    
    # Ensure 2D arrays
    if len(gray1.shape) > 2:
        gray1 = gray1[:, :, 0]
    if len(gray2.shape) > 2:
        gray2 = gray2[:, :, 0]
    
    # Compute mean and std
    mean1, std1 = cv2.meanStdDev(gray1)
    mean2, std2 = cv2.meanStdDev(gray2)
    
    # Normalize img2 to match img1
    if std2[0] > 1e-5:
        normalized_gray = ((gray2.astype(np.float32) - mean2[0]) * (std1[0] / std2[0])) + mean1[0]
        normalized_gray = np.clip(normalized_gray, 0, 255).astype(np.uint8)
    else:
        normalized_gray = gray2.copy()
    
    # Convert back to RGB if original was color
    if is_rgb:
        normalized = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2RGB)
    else:
        normalized = normalized_gray
    
    return normalized


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def normalize_signature_pair(
    img1: Image.Image,
    img2: Image.Image,
    target_size: Tuple[int, int] = (220, 155),  # Match model input size
    enable_baseline_align: bool = True,
    enable_brightness_match: bool = True,
    auto_detect_signatures: bool = True  # NEW: Auto-detect signatures in documents
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Complete 5-stage normalization pipeline for signature pair.
    
    CRITICAL: This ensures perfect alignment, noise removal, and consistent preprocessing.
    
    STAGE 1 (NEW): Automatic signature detection - crops signatures from documents if needed
    STAGE 2: Noise Removal + Background Normalization
    STAGE 3: Baseline Alignment
    STAGE 4: Size Normalization
    STAGE 5: Brightness Matching
    
    Both signatures will have:
    - Same size (target_size)
    - Same baseline position
    - Same brightness/contrast
    - Pure white background (background color differences removed)
    - Noise removed (text, stamps, artifacts)
    
    Returns:
        (normalized_img1, normalized_img2, processing_info)
    """
    # Convert to numpy RGB
    img1_arr = pil_to_numpy(img1.convert('RGB'))
    img2_arr = pil_to_numpy(img2.convert('RGB'))
    
    processing_info = {}
    
    # ====================================================================
    # STAGE 1: SIGNATURE DETECTION (CRITICAL!)
    # Detect and crop signatures from documents BEFORE processing
    # This ensures we're working with actual signatures, not full documents
    # ====================================================================
    print("=" * 60)
    print("🚀 NORMALIZATION PIPELINE STARTED")
    print("=" * 60)
    
    # CHECKPOINT: Verify input images
    assert img1 is not None and img2 is not None, "❌ CHECKPOINT FAILED: PIL images are None"
    print(f"✅ CHECKPOINT: PIL images valid - sig1 size: {img1.size}, sig2 size: {img2.size}")
    
    # Convert to numpy RGB
    img1_arr = pil_to_numpy(img1.convert('RGB'))
    img2_arr = pil_to_numpy(img2.convert('RGB'))
    
    # CHECKPOINT: Verify numpy arrays
    assert img1_arr is not None and img2_arr is not None, "❌ CHECKPOINT FAILED: Numpy arrays are None"
    assert len(img1_arr.shape) >= 2 and len(img2_arr.shape) >= 2, "❌ CHECKPOINT FAILED: Invalid array dimensions"
    print(f"✅ CHECKPOINT: Numpy arrays valid - sig1 shape: {img1_arr.shape}, sig2 shape: {img2_arr.shape}")
    
    if auto_detect_signatures:
        print("🔍 Detecting signatures in images...")
        try:
            # Try to detect signatures
            candidates1 = detect_signatures_robust(img1_arr)
            candidates2 = detect_signatures_robust(img2_arr)
            
            # Use the best candidate (highest confidence, largest area)
            # CRITICAL: Check list is not empty AND has valid confidence
            if candidates1 and len(candidates1) > 0 and candidates1[0].get('confidence', 0) > 0.4:
                img1_crop = candidates1[0]['image_crop']
                print(f"   ✅ Signature 1 detected (confidence={candidates1[0]['confidence']:.2f}, area={candidates1[0]['area']})")
                img1_arr = img1_crop
                processing_info['signature1_detected'] = True
                processing_info['signature1_bbox'] = candidates1[0]['bbox']
            else:
                print(f"   ⚠️ Signature 1 detection failed - using whole image")
                processing_info['signature1_detected'] = False
            
            if candidates2 and len(candidates2) > 0 and candidates2[0].get('confidence', 0) > 0.4:
                img2_crop = candidates2[0]['image_crop']
                print(f"   ✅ Signature 2 detected (confidence={candidates2[0]['confidence']:.2f}, area={candidates2[0]['area']})")
                img2_arr = img2_crop
                processing_info['signature2_detected'] = True
                processing_info['signature2_bbox'] = candidates2[0]['bbox']
            else:
                print(f"   ⚠️ Signature 2 detection failed - using whole image")
                processing_info['signature2_detected'] = False
        except Exception as e:
            import traceback
            print(f"⚠️ Signature detection error (continuing with whole images): {str(e)}")
            traceback.print_exc()
            processing_info['signature1_detected'] = False
            processing_info['signature2_detected'] = False
            # Continue with original images if detection fails
    
    # ====================================================================
    # STAGE 2: Noise Removal + Background Normalization
    # Removes text, stamps, lines, AND ensures pure white background
    # This fixes issues where same signature has different background colors
    # ====================================================================
    print("🧹 Cleaning signatures (noise removal)...")
    
    # CHECKPOINT: Verify input images before cleaning
    assert img1_arr is not None and img2_arr is not None, "❌ CHECKPOINT FAILED: Input images are None"
    assert img1_arr.size > 0 and img2_arr.size > 0, "❌ CHECKPOINT FAILED: Input images are empty"
    print(f"✅ CHECKPOINT: Input images valid - sig1 shape: {img1_arr.shape}, sig2 shape: {img2_arr.shape}")
    
    img1_clean = clean_signature(img1_arr)
    img2_clean = clean_signature(img2_arr)
    
    # CHECKPOINT: Verify cleaning worked
    assert img1_clean is not None and img2_clean is not None, "❌ CHECKPOINT FAILED: Cleaned images are None"
    assert img1_clean.size > 0 and img2_clean.size > 0, "❌ CHECKPOINT FAILED: Cleaned images are empty"
    print(f"✅ CHECKPOINT: Cleaning complete - sig1 shape: {img1_clean.shape}, sig2 shape: {img2_clean.shape}")
    
    processing_info['noise_removed'] = True
    processing_info['background_normalized'] = True  # Always pure white background
    
    # ====================================================================
    # STAGE 3: Baseline Alignment (CRITICAL - MUST WORK!)
    # Aligns both signatures to the same baseline (writing line)
    # CRITICAL: This must happen BEFORE size normalization to preserve alignment
    # If alignment fails, signatures cannot be compared accurately
    # ====================================================================
    if enable_baseline_align:
        print("🔧 Aligning baselines...")
        try:
            baseline_before1 = detect_baseline(img1_clean)
            baseline_before2 = detect_baseline(img2_clean)
            print(f"   Before: sig1 baseline={baseline_before1:.1f}, sig2 baseline={baseline_before2:.1f}")
            
            # Validate baselines are reasonable (not at image edges - indicates detection failed)
            if len(img1_clean.shape) == 2:
                h1, w1 = img1_clean.shape[:2]
            else:
                h1, w1 = img1_clean.shape[:2]
            
            if len(img2_clean.shape) == 2:
                h2, w2 = img2_clean.shape[:2]
            else:
                h2, w2 = img2_clean.shape[:2]
            
            baseline1_valid = 0.1 * h1 < baseline_before1 < 0.95 * h1
            baseline2_valid = 0.1 * h2 < baseline_before2 < 0.95 * h2
            
            if not baseline1_valid:
                print(f"⚠️ WARNING: Signature 1 baseline ({baseline_before1:.1f}) seems invalid for image height {h1}")
            if not baseline2_valid:
                print(f"⚠️ WARNING: Signature 2 baseline ({baseline_before2:.1f}) seems invalid for image height {h2}")
            
            # CRITICAL: For same signatures with different alignment, baselines MUST align perfectly
            # Use proportional alignment to handle different image sizes
            img1_baseline, img2_baseline = align_pair_baseline(img1_clean, img2_clean)
            
            baseline_after1 = detect_baseline(img1_baseline)
            baseline_after2 = detect_baseline(img2_baseline)
            baseline_diff_px = abs(baseline_after1 - baseline_after2)
            
            processing_info['baseline_aligned'] = True
            processing_info['baseline1'] = float(baseline_after1)
            processing_info['baseline2'] = float(baseline_after2)
            processing_info['baseline_diff'] = float(baseline_diff_px)
            
            # Calculate alignment quality as ratio (better for different image sizes)
            h1_final, h2_final = img1_baseline.shape[0], img2_baseline.shape[0]
            ratio1 = baseline_after1 / h1_final
            ratio2 = baseline_after2 / h2_final
            ratio_diff = abs(ratio1 - ratio2)
            processing_info['baseline_ratio_diff'] = float(ratio_diff)
            
            print(f"   After: sig1 baseline={baseline_after1:.1f} ({ratio1:.3f}), sig2 baseline={baseline_after2:.1f} ({ratio2:.3f}), diff={baseline_diff_px:.1f}px (ratio diff={ratio_diff:.3f})")
            
            # CRITICAL: If alignment is poor, log severe warning
            if baseline_diff_px > 10 or ratio_diff > 0.05:
                print(f"❌ SEVERE WARNING: Baseline alignment quality is POOR!")
                print(f"   - Pixel diff: {baseline_diff_px:.1f}px (should be < 5px)")
                print(f"   - Ratio diff: {ratio_diff:.3f} (should be < 0.02)")
                print(f"   - Signatures may not be properly aligned - comparison may be INACCURATE")
                processing_info['baseline_alignment_warning'] = True
            else:
                processing_info['baseline_alignment_warning'] = False
                print(f"✅ Baseline alignment quality: GOOD (diff={baseline_diff_px:.1f}px, ratio diff={ratio_diff:.3f})")
                
        except Exception as e:
            import traceback
            print(f"❌ BASELINE ALIGNMENT FAILED: {str(e)}")
            traceback.print_exc()
            # Use unaligned images but mark alignment as failed
            img1_baseline = img1_clean
            img2_baseline = img2_clean
            processing_info['baseline_aligned'] = False
            processing_info['baseline_alignment_warning'] = True
            processing_info['baseline_error'] = str(e)
    else:
        img1_baseline = img1_clean
        img2_baseline = img2_clean
        processing_info['baseline_aligned'] = False
        processing_info['baseline_alignment_warning'] = False
    
    # ====================================================================
    # STAGE 4: Size Normalization & Padding (FIELD-TESTED)
    # Trims empty borders and re-centers on square canvas
    # CRITICAL: Both signatures must be same size and centered identically
    # ====================================================================
    print(f"📏 Resizing and centering to {target_size}...")
    
    # CHECKPOINT: Verify images are valid before resizing
    assert img1_baseline is not None and img2_baseline is not None, "❌ CHECKPOINT FAILED: Baseline images are None"
    assert img1_baseline.size > 0 and img2_baseline.size > 0, "❌ CHECKPOINT FAILED: Baseline images are empty"
    print(f"✅ CHECKPOINT: Baseline images valid - sig1 shape: {img1_baseline.shape}, sig2 shape: {img2_baseline.shape}")
    
    # FIELD-TESTED: Use square canvas for better ORB matching
    # If target is square (or close), use center_and_pad
    target_w, target_h = target_size
    if target_w == target_h or abs(target_w - target_h) < 10:
        # Square canvas - use field-tested center_and_pad
        square_size = max(target_w, target_h)
        img1_normalized = center_and_pad(img1_baseline, size=square_size)
        img2_normalized = center_and_pad(img2_baseline, size=square_size)
        
        # CHECKPOINT: Verify center_and_pad worked
        assert img1_normalized.shape == img2_normalized.shape, f"❌ CHECKPOINT FAILED: Different sizes after center_and_pad - sig1: {img1_normalized.shape}, sig2: {img2_normalized.shape}"
        print(f"✅ CHECKPOINT: center_and_pad complete - both signatures: {img1_normalized.shape}")
        
        # Resize to exact target if needed
        if square_size != target_w or square_size != target_h:
            img1_normalized = cv2.resize(img1_normalized, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img2_normalized = cv2.resize(img2_normalized, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        # Non-square - use standard resize_and_pad
        # Get baseline positions BEFORE resize (as ratios)
        if enable_baseline_align:
            baseline1_before = detect_baseline(img1_baseline)
            baseline2_before = detect_baseline(img2_baseline)
            h1_before, h2_before = img1_baseline.shape[0], img2_baseline.shape[0]
            baseline_ratio1 = baseline1_before / h1_before
            baseline_ratio2 = baseline2_before / h2_before
            target_baseline_ratio = (baseline_ratio1 + baseline_ratio2) / 2.0
            print(f"   Baseline ratios before resize: sig1={baseline_ratio1:.3f}, sig2={baseline_ratio2:.3f}, target={target_baseline_ratio:.3f}")
        
        img1_normalized = resize_and_pad(img1_baseline, size=target_size, preserve_aspect=True)
        img2_normalized = resize_and_pad(img2_baseline, size=target_size, preserve_aspect=True)
        
        # CHECKPOINT: Verify resize_and_pad worked
        assert img1_normalized.shape == img2_normalized.shape, f"❌ CHECKPOINT FAILED: Different sizes after resize_and_pad - sig1: {img1_normalized.shape}, sig2: {img2_normalized.shape}"
        print(f"✅ CHECKPOINT: resize_and_pad complete - both signatures: {img1_normalized.shape}")
    
    # CRITICAL: After resize, baselines should be at the SAME ratio
    # Force-align them to ensure equality
    if enable_baseline_align:
        h1_after, h2_after = img1_normalized.shape[0], img2_normalized.shape[0]
        
        # Target baseline positions (same ratio for both)
        target_baseline1 = h1_after * target_baseline_ratio
        target_baseline2 = h2_after * target_baseline_ratio
        
        # Force-align both to same target baseline
        img1_normalized, _ = align_baseline(img1_normalized, target_baseline1)
        img2_normalized, _ = align_baseline(img2_normalized, target_baseline2)
        
        # Final verification - baselines MUST be equal
        final_baseline1 = detect_baseline(img1_normalized)
        final_baseline2 = detect_baseline(img2_normalized)
        baseline_diff_px = abs(final_baseline1 - final_baseline2)
        final_ratio1 = final_baseline1 / h1_after
        final_ratio2 = final_baseline2 / h2_after
        baseline_diff_ratio = abs(final_ratio1 - final_ratio2)
        
        print(f"   Baselines after resize & realignment: sig1={final_baseline1:.1f} ({final_ratio1:.3f}), sig2={final_baseline2:.1f} ({final_ratio2:.3f})")
        print(f"   Baseline diff: {baseline_diff_px:.1f}px (ratio diff: {baseline_diff_ratio:.4f})")
        
        # CRITICAL: Verify baselines are EQUAL (within 1px tolerance for exact alignment)
        max_attempts = 5
        attempt = 0
        while baseline_diff_px > 1 and attempt < max_attempts:
            attempt += 1
            print(f"   ⚠️ Attempt {attempt}/{max_attempts}: Baselines not equal (diff={baseline_diff_px:.1f}px), force-aligning...")
            avg_baseline = (final_baseline1 + final_baseline2) / 2.0
            img1_normalized, _ = align_baseline(img1_normalized, avg_baseline)
            img2_normalized, _ = align_baseline(img2_normalized, avg_baseline)
            
            # Re-check
            final_baseline1 = detect_baseline(img1_normalized)
            final_baseline2 = detect_baseline(img2_normalized)
            baseline_diff_px = abs(final_baseline1 - final_baseline2)
            final_ratio1 = final_baseline1 / h1_after
            final_ratio2 = final_baseline2 / h2_after
            
        if baseline_diff_px <= 1:
            print(f"   ✅ Baselines are EQUAL (diff={baseline_diff_px:.1f}px) after {attempt} attempt(s)")
        else:
            print(f"   ⚠️ WARNING: Baseline alignment incomplete (diff={baseline_diff_px:.1f}px) after {max_attempts} attempts")
    
    processing_info['size_normalized'] = True
    processing_info['target_size'] = target_size
    
    # ====================================================================
    # STAGE 5: Brightness Matching
    # Matches brightness and contrast between signatures
    # Ensures fair comparison even with different pens/scanning conditions
    # ====================================================================
    if enable_brightness_match:
        img2_final = match_brightness(img1_normalized, img2_normalized)
        img1_final = img1_normalized.copy()
        processing_info['brightness_matched'] = True
    else:
        img1_final = img1_normalized
        img2_final = img2_normalized
        processing_info['brightness_matched'] = False
    
    return img1_final, img2_final, processing_info


# ============================================================================
# OVERLAY VISUALIZATION WITH BASELINE MARKERS
# ============================================================================

def overlay_signatures_with_baseline(
    sig1: np.ndarray,
    sig2: np.ndarray,
    alpha: float = 0.5,
    show_baseline: bool = True
) -> np.ndarray:
    """
    Create overlay visualization with baseline markers.
    
    Args:
        sig1, sig2: Normalized signature images (RGB or grayscale)
        alpha: Blend factor (0-1)
        show_baseline: If True, draw baseline markers
    
    Returns:
        RGB overlay image with baseline indicators
    """
    # Ensure same size
    if sig1.shape[:2] != sig2.shape[:2]:
        h, w = sig1.shape[:2]
        sig2 = cv2.resize(sig2, (w, h), interpolation=cv2.INTER_AREA)
    
    # Normalize both to 3-channel BGR for consistent blending
    # Helper function to safely convert any image format to BGR
    def to_bgr(img):
        """Convert any image format to BGR (3 channels)"""
        if len(img.shape) == 2:
            # Grayscale (H, W) - convert to BGR
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                # Already 3 channels - check if RGB or BGR
                # Assume RGB and convert to BGR
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[2] == 1:
                # Single channel in 3D array (H, W, 1) - squeeze to 2D then convert
                img_2d = img[:, :, 0]
                return cv2.cvtColor(img_2d, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                # RGBA - convert to RGB first, then BGR
                rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                # Unexpected channel count - take first channel
                img_2d = img[:, :, 0]
                return cv2.cvtColor(img_2d, cv2.COLOR_GRAY2BGR)
        else:
            # Unexpected shape - try to reshape to 2D
            raise ValueError(f"Unexpected image shape: {img.shape}")
    
    # Convert both to BGR
    try:
        color1 = to_bgr(sig1)
    except Exception as e:
        # Fallback: force to grayscale then BGR
        if len(sig1.shape) == 2:
            sig1_gray = sig1
        elif len(sig1.shape) == 3:
            sig1_gray = cv2.cvtColor(sig1, cv2.COLOR_RGB2GRAY) if sig1.shape[2] == 3 else sig1[:, :, 0]
        else:
            sig1_gray = np.mean(sig1, axis=2).astype(np.uint8) if len(sig1.shape) == 3 else sig1.reshape(sig1.shape[:2])
        color1 = cv2.cvtColor(sig1_gray, cv2.COLOR_GRAY2BGR)
    
    try:
        color2 = to_bgr(sig2)
    except Exception as e:
        # Fallback: force to grayscale then BGR
        if len(sig2.shape) == 2:
            sig2_gray = sig2
        elif len(sig2.shape) == 3:
            sig2_gray = cv2.cvtColor(sig2, cv2.COLOR_RGB2GRAY) if sig2.shape[2] == 3 else sig2[:, :, 0]
        else:
            sig2_gray = np.mean(sig2, axis=2).astype(np.uint8) if len(sig2.shape) == 3 else sig2.reshape(sig2.shape[:2])
        color2 = cv2.cvtColor(sig2_gray, cv2.COLOR_GRAY2BGR)
    
    # Ensure both are same shape for blending
    if color1.shape != color2.shape:
        h, w = color1.shape[:2]
        color2 = cv2.resize(color2, (w, h), interpolation=cv2.INTER_AREA)
    
    # Create overlay
    overlay = cv2.addWeighted(color1, alpha, color2, 1 - alpha, 0)
    
    # Draw baseline markers if requested
    if show_baseline:
        baseline1 = detect_baseline(sig1)
        baseline2 = detect_baseline(sig2)
        
        h, w = overlay.shape[:2]
        
        # Draw baseline lines
        cv2.line(overlay, (0, int(baseline1)), (w, int(baseline1)), (0, 255, 0), 2)  # Green for sig1
        cv2.line(overlay, (0, int(baseline2)), (w, int(baseline2)), (255, 0, 0), 2)  # Blue for sig2
        
        # Label
        cv2.putText(overlay, "Baseline 1", (5, int(baseline1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(overlay, "Baseline 2", (5, int(baseline2) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Convert back to RGB
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return overlay_rgb

