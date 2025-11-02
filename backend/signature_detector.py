"""
Signature Detection and Localization Module
Detects signatures in document images and crops them for processing.
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import hashlib


def hash_image(image_bytes: bytes) -> str:
    """
    Compute SHA-256 hash of image for integrity checking.
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(image_bytes).hexdigest()


def detect_signatures_contour(img_pil: Image.Image, min_area: int = 4000) -> List[Tuple[int, int, int, int]]:
    """
    IMPROVED signature detection using histogram equalization + robust filtering.
    Based on the improved detection algorithm.
    
    Args:
        img_pil: PIL Image
        min_area: Minimum contour area (default 4000 to match improved detector)
    
    Returns:
        List of bounding boxes [(x, y, w, h), ...]
    """
    # Convert to OpenCV format
    if img_pil.mode == 'RGBA':
        img_pil = img_pil.convert('RGB')
    img_array = np.array(img_pil)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # IMPROVEMENT 1: Histogram equalization to increase contrast
    gray_eq = cv2.equalizeHist(gray)
    
    # IMPROVEMENT 2: Adaptive threshold with better parameters
    binary = cv2.adaptiveThreshold(
        gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 15  # Better block size and C value
    )
    
    # IMPROVEMENT 3: Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove small noise
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)  # Connect strokes
    
    # Find contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    signatures = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h  # Use bounding box area (more reliable)
        
        # IMPROVEMENT 4: Better filtering based on area and aspect ratio
        if area < min_area or area > 150000:  # Skip too small or too large
            continue
        
        aspect_ratio = w / (h + 1e-6)
        
        # Signature-like: wide (aspect > 1.5), reasonable size
        if aspect_ratio < 1.5 or aspect_ratio > 8.0:
            continue
        
        # Add padding
        padding = 15
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_cv.shape[1] - x, w + 2 * padding)
        h = min(img_cv.shape[0] - y, h + 2 * padding)
        
        signatures.append((x, y, w, h))
    
    # Sort by area (largest first)
    signatures.sort(key=lambda box: box[2] * box[3], reverse=True)
    
    return signatures


def crop_signature(img_pil: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop signature from image using bounding box.
    
    Args:
        img_pil: Original PIL Image
        bbox: Bounding box (x, y, w, h)
    
    Returns:
        Cropped PIL Image
    """
    x, y, w, h = bbox
    return img_pil.crop((x, y, x + w, y + h))


def detect_and_extract_signatures(img_pil: Image.Image, select_largest: bool = True) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
    """
    Detect and extract all signatures from an image.
    
    Args:
        img_pil: PIL Image containing signature(s)
        select_largest: If True and multiple signatures found, return only the largest
    
    Returns:
        Tuple of (list of cropped signature images, list of bounding boxes)
    """
    bboxes = detect_signatures_contour(img_pil)
    
    if not bboxes:
        # If no signature detected, return the whole image
        return [img_pil], [(0, 0, img_pil.width, img_pil.height)]
    
    if select_largest and len(bboxes) > 1:
        # Return only the largest signature
        bboxes = [bboxes[0]]
    
    cropped_signatures = [crop_signature(img_pil, bbox) for bbox in bboxes]
    
    return cropped_signatures, bboxes


def process_document_with_detection(img_pil: Image.Image) -> Image.Image:
    """
    High-level function: detect and extract signature from document.
    If multiple signatures found, returns the largest.
    
    Args:
        img_pil: Document image (may contain text, multiple signatures, etc.)
    
    Returns:
        Cropped signature image
    """
    signatures, _ = detect_and_extract_signatures(img_pil, select_largest=True)
    return signatures[0] if signatures else img_pil

