"""
Advanced Signature Alignment using ORB Keypoints + RANSAC
Ensures pixel-perfect alignment for comparison and visualization.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List


def detect_signatures_in_image(
    rgb_image: np.ndarray, 
    min_area: int = 1500
) -> List[Tuple[int, int, int, int]]:
    """
    Detect candidate signature bounding boxes using threshold -> contours.
    Returns list of (x, y, w, h) boxes sorted by position.
    
    Args:
        rgb_image: RGB image array (H, W, 3)
        min_area: Minimum area threshold for signature regions
    
    Returns:
        List of bounding boxes [(x, y, w, h), ...]
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Adaptive threshold to highlight ink
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological closing to join strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    h_img, w_img = gray.shape
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Heuristics: signature width < image width and area > min_area
        if area > min_area and w < 0.95 * w_img:
            boxes.append((x, y, w, h))
    
    # Sort left-to-right, top-to-bottom
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    
    if len(boxes) == 0:
        # Fallback: whole image
        boxes = [(0, 0, w_img, h_img)]
    
    return boxes


def denoise_signature(
    crop_rgb: np.ndarray, 
    min_component_area: int = 30
) -> np.ndarray:
    """
    IMPROVED noise reduction using adaptive thresholding with MEAN_C.
    Handles variable lighting and scan contrast better than fixed threshold.
    
    Args:
        crop_rgb: RGB signature crop
        min_component_area: Minimum area to keep a connected component
    
    Returns:
        Denoised RGB image (black strokes on white background)
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    
    # IMPROVEMENT: Use adaptive threshold instead of OTSU for variable lighting
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 41, 10
    )
    
    # Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove noise
    bw = cv2.dilate(bw, kernel, iterations=1)  # Slightly thicken strokes
    
    # Remove tiny components
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    sizes = stats[:, -1]
    
    bw_clean = np.zeros_like(bw)
    for i in range(1, nb_components):
        if sizes[i] >= min_component_area:
            bw_clean[output == i] = 255
    
    # Reconstruct RGB with kept strokes (black on white)
    mask = bw_clean > 0
    out = np.ones_like(crop_rgb) * 255  # White background
    out[mask] = [0, 0, 0]  # Black strokes
    
    return out


def align_pair_safe(
    imgA: np.ndarray,
    imgB: np.ndarray,
    use_similarity_only: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """
    SAFER alignment using similarity/affine transform instead of projective homography.
    Homography can artificially align unrelated shapes - this prevents that.
    
    Args:
        imgA: Reference image (RGB)
        imgB: Image to warp (RGB)
        use_similarity_only: If True, use similarity transform (scale+rotate+translate only)
                           If False, use affine (allows skew but not projective)
    
    Returns:
        (warped_imgB, transform_matrix, warp_type)
        warp_type: "similarity", "affine", "none", or "homography" (fallback)
    """
    a = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
    
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)
    
    if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
        return imgB, None, "none"
    
    # Use knnMatch for better filtering (Lowe's ratio test)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    if len(good) < 6:
        return imgB, None, "none"
    
    ptsA = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    h, w = imgA.shape[:2]
    M = None
    warp_type = "none"
    
    if use_similarity_only:
        # Try similarity transform first (preserves angles, no skew)
        try:
            # estimateAffinePartial2D gives similarity (4 DOF: tx, ty, rotation, scale)
            M, inliers = cv2.estimateAffinePartial2D(
                ptsB, ptsA, 
                method=cv2.RANSAC, 
                ransacReprojThreshold=5.0,
                confidence=0.99,
                maxIters=2000
            )
            if M is not None:
                warp_type = "similarity"
        except:
            pass
    
    if M is None:
        # Fallback to full affine (6 DOF: allows skew but not projective)
        try:
            M, inliers = cv2.estimateAffine2D(
                ptsB, ptsA,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                confidence=0.99,
                maxIters=2000
            )
            if M is not None:
                warp_type = "affine"
        except:
            pass
    
    if M is None:
        return imgB, None, "none"
    
    # Validate transformation is reasonable
    # For similarity: check scale is reasonable (0.5 to 2.0)
    if warp_type == "similarity":
        scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        if scale < 0.5 or scale > 2.0:
            return imgB, None, "none"
    
    # Warp using affine (2x3 matrix)
    warped = cv2.warpAffine(
        imgB, M, (w, h),
        borderValue=(255, 255, 255),
        flags=cv2.INTER_LINEAR
    )
    
    return warped, M, warp_type


def align_pair_via_orb(
    imgA: np.ndarray, 
    imgB: np.ndarray,
    min_match_ratio: float = 0.25,
    use_safe_alignment: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Find transform from imgB -> imgA coordinate frame using ORB matches.
    
    Args:
        imgA: Reference image (RGB)
        imgB: Image to warp (RGB)
        min_match_ratio: Minimum ratio of good matches to total keypoints (0.25 = 25%)
        use_safe_alignment: If True, use similarity/affine (safer). If False, use homography (legacy)
    
    Returns:
        (warped_imgB, transform_matrix)
    """
    if use_safe_alignment:
        # Use safer similarity/affine alignment
        warped, M, warp_type = align_pair_safe(imgA, imgB, use_similarity_only=True)
        return warped, M
    else:
        # Legacy homography method (kept for compatibility but not recommended)
        return _align_pair_homography_legacy(imgA, imgB, min_match_ratio)


def _align_pair_homography_legacy(
    imgA: np.ndarray,
    imgB: np.ndarray,
    min_match_ratio: float = 0.25
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    LEGACY homography-based alignment. Use align_pair_safe() instead.
    Kept for backward compatibility but marked as unsafe.
    """
    a = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
    
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return imgB, None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 10:
        return imgB, None
    
    total_keypoints = min(len(kp1), len(kp2))
    match_ratio = len(matches) / total_keypoints
    
    if match_ratio < min_match_ratio:
        return imgB, None
    
    good_matches = [m for m in matches if m.distance < 50]
    if len(good_matches) < 8:
        return imgB, None
    
    ptsA = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    try:
        M, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 3.0)
    except:
        M = None
    
    h, w = imgA.shape[:2]
    
    if M is None:
        return imgB, None
    
    det = np.linalg.det(M[:2, :2])
    if det < 0.5 or det > 2.0:
        return imgB, None
    
    warped = cv2.warpPerspective(
        imgB, M, (w, h),
        borderValue=(255, 255, 255),
        flags=cv2.INTER_LINEAR
    )
    
    return warped, M


def preprocess_for_model(
    img_rgb: np.ndarray, 
    target_size: Tuple[int, int] = (220, 155),
    pad: int = 16
) -> np.ndarray:
    """
    Preprocess signature: center, pad to square, resize to target size.
    This is the SAME preprocessing used for model inference and visualization.
    
    Args:
        img_rgb: RGB signature image
        target_size: (width, height) target size
        pad: Padding pixels around signature
    
    Returns:
        Preprocessed RGB image (target_size)
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Bounding box of ink
    ys, xs = np.where(bw > 0)
    
    if len(xs) == 0 or len(ys) == 0:
        # Empty, return white canvas
        return 255 * np.ones((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    
    crop = img_rgb[y0:y1+1, x0:x1+1]
    h, w, _ = crop.shape
    
    # Pad to square
    size = max(h, w) + pad
    canvas = 255 * np.ones((size, size, 3), dtype=np.uint8)
    
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = crop
    
    # Resize to target
    target_w, target_h = target_size
    resized = cv2.resize(
        canvas, (target_w, target_h), 
        interpolation=cv2.INTER_AREA
    )
    
    return resized


def visualize_overlay(
    base_rgb: np.ndarray,
    over_rgb: np.ndarray,
    alpha: float = 0.5,
    zoom: float = 1.0,
    rotate_deg: float = 0
) -> np.ndarray:
    """
    Create overlay visualization with optional zoom/rotate on overlay.
    
    Args:
        base_rgb: Base signature image
        over_rgb: Overlay signature image
        alpha: Blend opacity (0-1)
        zoom: Zoom factor for overlay
        rotate_deg: Rotation angle in degrees
    
    Returns:
        Blended RGB image
    """
    h, w = base_rgb.shape[:2]
    
    # Apply zoom and rotate to overlay
    if zoom != 1.0 or rotate_deg != 0:
        M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), rotate_deg, zoom)
        over_t = cv2.warpAffine(
            over_rgb, M_rot, (w, h), 
            borderValue=(255, 255, 255),
            flags=cv2.INTER_LINEAR
        )
    else:
        over_t = over_rgb.copy()
    
    # Blend
    blended = cv2.addWeighted(
        base_rgb.astype(np.float32),
        1 - alpha,
        over_t.astype(np.float32),
        alpha,
        0
    )
    
    return np.clip(blended, 0, 255).astype(np.uint8)


def pil_to_numpy(img_pil: Image.Image) -> np.ndarray:
    """Convert PIL Image to RGB numpy array."""
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    return np.array(img_pil)


def numpy_to_pil(img_arr: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    return Image.fromarray(img_arr)

