"""
Advanced Visualization Module for Signature Verification
Properly aligned heatmaps, overlays, and explainable visualizations.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from backend.preprocessing import preprocess_signature, align_signatures


# Standard canvas size for all visualizations (matches model input size)
CANVAS_SIZE = (220, 155)  # (width, height) - matches model training size


def normalize_to_canvas(img_pil: Image.Image, canvas_size: Tuple[int, int] = CANVAS_SIZE) -> Image.Image:
    """
    Normalize signature to fixed canvas size while preserving aspect ratio.
    Pads with white background and centers the signature.
    
    Args:
        img_pil: PIL Image
        canvas_size: Target (width, height)
    
    Returns:
        Normalized PIL Image (RGB)
    """
    # Convert to RGB
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    
    img_array = np.array(img_pil)
    h, w = img_array.shape[:2]
    target_w, target_h = canvas_size
    
    # Calculate scaling to fit within canvas while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Create white canvas
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    
    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return Image.fromarray(canvas)


def align_pair_to_canvas(
    img1_pil: Image.Image, 
    img2_pil: Image.Image, 
    canvas_size: Tuple[int, int] = CANVAS_SIZE
) -> Tuple[Image.Image, Image.Image]:
    """
    Align both signatures to the same canvas size.
    Ensures perfect alignment for comparison.
    
    Returns:
        Tuple of (normalized_img1, normalized_img2)
    """
    img1_normalized = normalize_to_canvas(img1_pil, canvas_size)
    img2_normalized = normalize_to_canvas(img2_pil, canvas_size)
    return img1_normalized, img2_normalized


def create_exact_overlay(
    img1: Image.Image,
    img2: Image.Image,
    alpha: float = 0.5
) -> Image.Image:
    """
    Create EXACT OVERLAP of two signatures.
    Shows both images perfectly aligned and overlaid on top of each other.
    
    Args:
        img1: First signature (PIL)
        img2: Second signature (PIL, should be aligned)
        alpha: Blend transparency (0-1)
    
    Returns:
        PIL Image showing exact overlap
    """
    from backend.advanced_alignment import pil_to_numpy, numpy_to_pil
    
    img1_arr = pil_to_numpy(img1.convert('RGB'))
    img2_arr = pil_to_numpy(img2.convert('RGB'))
    
    h1, w1 = img1_arr.shape[:2]
    h2, w2 = img2_arr.shape[:2]
    
    # Ensure same size
    if (h1, w1) != (h2, w2):
        img2_arr = cv2.resize(img2_arr, (w1, h1), interpolation=cv2.INTER_AREA)
    
    # Create exact overlay (50/50 blend)
    overlay = cv2.addWeighted(img1_arr, 0.5, img2_arr, 0.5, 0)
    
    return numpy_to_pil(overlay)


def overlay_heatmap(
    base_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> Image.Image:
    """
    Overlay heatmap on base image with proper blending.
    
    Args:
        base_img: Base signature image (PIL)
        heatmap: Heatmap array (same size as base_img, normalized 0-1 or 0-255)
        alpha: Transparency of overlay (0-1)
        colormap: Colormap name ('jet', 'hot', 'cool', 'red', 'green')
    
    Returns:
        Blended image (PIL RGB)
    """
    base_array = np.array(base_img.convert('RGB'))
    h, w = base_array.shape[:2]
    
    # Ensure heatmap is same size
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize heatmap to 0-255 if needed
    if heatmap.max() <= 1.0:
        heatmap = (heatmap * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
    
    # Apply colormap
    if colormap == 'jet':
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    elif colormap == 'hot':
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    elif colormap == 'cool':
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_COOL)
    elif colormap == 'red':
        # Red overlay
        heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap_colored[:, :, 0] = heatmap  # Red channel
    elif colormap == 'green':
        # Green overlay
        heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap_colored[:, :, 1] = heatmap  # Green channel
    else:
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    blended = cv2.addWeighted(base_array, 1 - alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(blended)


def create_dual_overlay(
    base_img1: Image.Image,
    base_img2: Image.Image,
    heatmap1: np.ndarray,
    heatmap2: np.ndarray,
    alpha: float = 0.5
) -> Image.Image:
    """
    Create side-by-side visualization with two heatmaps.
    Red overlay for img1, Green overlay for img2.
    
    Args:
        base_img1: First signature (normalized)
        base_img2: Second signature (normalized)
        heatmap1: Heatmap for first signature
        heatmap2: Heatmap for second signature
        alpha: Overlay transparency
    
    Returns:
        Combined side-by-side image
    """
    overlay1 = overlay_heatmap(base_img1, heatmap1, alpha, colormap='red')
    overlay2 = overlay_heatmap(base_img2, heatmap2, alpha, colormap='green')
    
    # Combine side by side
    w, h = CANVAS_SIZE
    combined = Image.new('RGB', (w * 2 + 20, h), (255, 255, 255))
    combined.paste(overlay1, (0, 0))
    combined.paste(overlay2, (w + 20, 0))
    
    return combined


def create_difference_map(
    img1: Image.Image,
    img2: Image.Image,
    show_unique_regions: bool = True
) -> Tuple[Image.Image, dict]:
    """
    Create difference map showing where signatures differ.
    
    Args:
        img1: First signature (normalized)
        img2: Second signature (normalized)
        show_unique_regions: If True, highlight unique regions
    
    Returns:
        (difference_image, stats_dict)
    """
    # Convert to grayscale for comparison
    img1_gray = np.array(img1.convert('L'))
    img2_gray = np.array(img2.convert('L'))
    
    # Compute absolute difference
    diff = np.abs(img1_gray.astype(np.float32) - img2_gray.astype(np.float32))
    diff = diff.astype(np.uint8)
    
    # Normalize for visualization
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Create colored difference map
    # Blue = similar regions, Red/Yellow = different regions
    diff_colored = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay on one of the signatures
    base_array = np.array(img1.convert('RGB'))
    diff_overlay = overlay_heatmap(
        Image.fromarray(base_array),
        diff_norm,
        alpha=0.6,
        colormap='hot'
    )
    
    # Calculate statistics
    diff_percentage = (diff > 30).sum() / diff.size * 100  # Threshold for "different"
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    stats = {
        "difference_percentage": round(diff_percentage, 2),
        "max_difference": int(max_diff),
        "mean_difference": round(float(mean_diff), 2)
    }
    
    return diff_overlay, stats


def explain_visualization(viz_type: str) -> dict:
    """
    Return user-friendly explanation for each visualization type.
    
    Returns:
        Dictionary with title, description, and interpretation guide
    """
    explanations = {
        "saliency": {
            "title": "Saliency Map - Model Attention",
            "description": "Shows which parts of Signature 2 the AI considers most important when comparing to Signature 1.",
            "interpretation": "Red/orange areas = high importance for similarity. Darker areas = less influence.",
            "color_legend": "Hot colors (red/yellow) = important regions, Cool colors (blue) = less important"
        },
        "gradcam": {
            "title": "Grad-CAM - Deep Learning Focus",
            "description": "Reveals which regions of the signature activate the neural network most strongly.",
            "interpretation": "Bright red areas show where the AI model focuses its attention for verification.",
            "color_legend": "Red = high activation, Blue = low activation"
        },
        "dual_saliency": {
            "title": "Dual Saliency - Both Signatures",
            "description": "Shows attention patterns for both signatures side-by-side.",
            "interpretation": "Left (red) = Signature 1 attention. Right (green) = Signature 2 attention.",
            "color_legend": "Red overlay = Signature 1 focus, Green overlay = Signature 2 focus"
        },
        "difference": {
            "title": "Difference Map - Visual Comparison",
            "description": "Highlights pixel-level differences between the two signatures.",
            "interpretation": "Hot colors (red/yellow) = significant differences. Cool colors = similar regions.",
            "color_legend": "Red/Yellow = different strokes, Blue = similar strokes"
        },
        "saliency_diff": {
            "title": "Saliency Difference - Attention Variation",
            "description": "Shows where the AI's attention patterns differ between signatures.",
            "interpretation": "Highlights regions where one signature gets more attention than the other.",
            "color_legend": "Yellow/Red = attention differs, Dark = similar attention"
        }
    }
    
    return explanations.get(viz_type, {
        "title": "Visualization",
        "description": "AI model explanation",
        "interpretation": "See how the model analyzes the signatures",
        "color_legend": ""
    })


def create_legend_image(
    title: str,
    color_legend: str,
    canvas_size: Tuple[int, int] = CANVAS_SIZE
) -> Image.Image:
    """
    Create a text legend image for visualizations.
    """
    img = Image.new('RGB', canvas_size, (255, 255, 255))
    # This is a placeholder - actual text rendering would use PIL ImageDraw
    return img

