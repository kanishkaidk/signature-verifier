"""
Create EXACT overlap visualization of two signatures.
Shows perfect alignment with visual blending.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple
from backend.advanced_alignment import pil_to_numpy, numpy_to_pil


def create_exact_signature_overlay(
    img1: Image.Image,
    img2: Image.Image,
    alpha: float = 0.5
) -> Image.Image:
    """
    Create EXACT OVERLAP of two aligned signatures.
    
    This shows:
    - Both signatures overlaid on top of each other
    - Perfect pixel-level alignment
    - Visual blend showing where they match
    
    Args:
        img1: First signature (PIL)
        img2: Second signature (PIL, should be aligned)
        alpha: Blend factor (0-1, 0.5 = equal blend)
    
    Returns:
        PIL Image showing exact overlap
    """
    # Convert to numpy arrays
    img1_arr = pil_to_numpy(img1.convert('RGB'))
    img2_arr = pil_to_numpy(img2.convert('RGB'))
    
    h1, w1 = img1_arr.shape[:2]
    h2, w2 = img2_arr.shape[:2]
    
    # Ensure same size
    if (h1, w1) != (h2, w2):
        img2_arr = cv2.resize(img2_arr, (w1, h1), interpolation=cv2.INTER_AREA)
    
    # Create exact overlay (50/50 blend shows true overlap)
    overlay = cv2.addWeighted(img1_arr, 0.5, img2_arr, 0.5, 0)
    
    return numpy_to_pil(overlay)


def create_overlay_with_colors(
    img1: Image.Image,
    img2: Image.Image,
    show_unique: bool = True
) -> Image.Image:
    """
    Create overlap visualization with color coding:
    - Red tint: Signature 1 unique areas
    - Green tint: Signature 2 unique areas
    - Yellow: Overlapping/matched areas
    
    Args:
        img1: First signature (PIL)
        img2: Second signature (PIL, aligned)
        show_unique: If True, color-code unique vs overlapping regions
    
    Returns:
        PIL Image with color-coded overlap
    """
    img1_arr = pil_to_numpy(img1.convert('RGB'))
    img2_arr = pil_to_numpy(img2.convert('RGB'))
    
    h1, w1 = img1_arr.shape[:2]
    h2, w2 = img2_arr.shape[:2]
    
    if (h1, w1) != (h2, w2):
        img2_arr = cv2.resize(img2_arr, (w1, h1), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale for masks
    gray1 = cv2.cvtColor(img1_arr, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_arr, cv2.COLOR_RGB2GRAY)
    
    # Create binary masks (signature pixels)
    _, mask1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find overlaps
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    unique1 = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))
    unique2 = cv2.bitwise_and(mask2, cv2.bitwise_not(mask1))
    
    # Create color-coded overlay
    overlay = img1_arr.copy().astype(np.float32)
    
    if show_unique:
        # Red tint for Signature 1 unique
        overlay[unique1 > 0] = overlay[unique1 > 0] * 0.7 + np.array([255, 100, 100]) * 0.3
        
        # Green tint for Signature 2 unique
        overlay[unique2 > 0] = img2_arr[unique2 > 0].astype(np.float32) * 0.7 + np.array([100, 255, 100]) * 0.3
        
        # Yellow for overlaps
        overlay[overlap_mask > 0] = (
            overlay[overlap_mask > 0] * 0.4 + 
            img2_arr[overlap_mask > 0].astype(np.float32) * 0.3 + 
            np.array([255, 255, 100]) * 0.3
        )
    else:
        # Simple 50/50 blend
        overlay = cv2.addWeighted(img1_arr, 0.5, img2_arr, 0.5, 0).astype(np.float32)
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return numpy_to_pil(overlay)

