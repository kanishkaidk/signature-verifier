"""
Stroke-level signature analysis for handwriting and forgery detection.
Analyzes strokes, flow, pressure patterns, and structural similarity.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List, Optional
from backend.advanced_alignment import pil_to_numpy, numpy_to_pil


def extract_strokes(img_rgb: np.ndarray) -> Dict:
    """
    Extract stroke-level features from signature.
    
    Returns:
        dict with:
            - strokes: List of stroke contours
            - stroke_lengths: List of stroke lengths
            - stroke_directions: List of stroke directions (angles)
            - stroke_pressure: Estimated pressure (stroke width)
            - stroke_flow: Flow direction changes
            - bounding_boxes: Bounding boxes for each stroke
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Binarize (signature is dark on light background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours (each stroke/connected component)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out noise (very small strokes)
    min_stroke_area = 20  # pixels
    strokes = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_stroke_area]
    
    stroke_data = {
        'strokes': [],
        'stroke_lengths': [],
        'stroke_directions': [],
        'stroke_pressure': [],
        'stroke_flow': [],
        'bounding_boxes': []
    }
    
    for stroke in strokes:
        # Stroke length (perimeter)
        length = cv2.arcLength(stroke, False)
        stroke_data['stroke_lengths'].append(length)
        stroke_data['strokes'].append(stroke)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(stroke)
        stroke_data['bounding_boxes'].append((x, y, w, h))
        
        # Estimate stroke direction (angle of bounding box)
        if w > h:
            angle = 0  # Horizontal
        else:
            angle = 90  # Vertical
        
        # More precise: use minAreaRect for better angle
        rect = cv2.minAreaRect(stroke)
        angle = rect[2]
        stroke_data['stroke_directions'].append(angle)
        
        # Estimate pressure (stroke width)
        # Approximate as ratio of area to length
        area = cv2.contourArea(stroke)
        pressure = area / max(length, 1) if length > 0 else 0
        stroke_data['stroke_pressure'].append(pressure)
        
        # Stroke flow: analyze direction changes along the stroke
        if len(stroke) > 2:
            points = stroke.reshape(-1, 2)
            directions = []
            for i in range(1, len(points)):
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                directions.append(angle)
            
            # Flow consistency (lower std = smoother stroke)
            flow_consistency = np.std(directions) if len(directions) > 0 else 180
            stroke_data['stroke_flow'].append(flow_consistency)
        else:
            stroke_data['stroke_flow'].append(180)
    
    return stroke_data


def analyze_handwriting_flow(img_rgb: np.ndarray) -> Dict:
    """
    Analyze handwriting flow, style, and characteristics.
    
    Returns:
        dict with:
            - dominant_direction: Main writing direction (angle)
            - stroke_count: Number of distinct strokes
            - average_stroke_length: Average stroke length
            - flow_smoothness: How smooth the handwriting is
            - stroke_density: Strokes per unit area
            - writing_style: Cursive vs printed (estimated)
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Extract strokes
    stroke_data = extract_strokes(img_rgb)
    
    if len(stroke_data['strokes']) == 0:
        return {
            'dominant_direction': 0,
            'stroke_count': 0,
            'average_stroke_length': 0,
            'flow_smoothness': 0,
            'stroke_density': 0,
            'writing_style': 'unknown'
        }
    
    # Dominant direction (weighted by stroke length)
    if stroke_data['stroke_directions']:
        lengths = np.array(stroke_data['stroke_lengths'])
        directions = np.array(stroke_data['stroke_directions'])
        # Weight directions by stroke length
        weighted_direction = np.average(directions, weights=lengths)
        dominant_direction = weighted_direction % 180  # Normalize to 0-180
    else:
        dominant_direction = 0
    
    # Stroke statistics
    stroke_count = len(stroke_data['strokes'])
    avg_stroke_length = np.mean(stroke_data['stroke_lengths']) if stroke_data['stroke_lengths'] else 0
    
    # Flow smoothness (lower std in flow = smoother)
    flow_smoothness = 1.0 / (1.0 + np.mean(stroke_data['stroke_flow']) / 180.0)
    
    # Stroke density
    total_area = w * h
    stroke_area = sum([cv2.contourArea(s) for s in stroke_data['strokes']])
    stroke_density = stroke_count / (total_area / 10000)  # strokes per 100x100 area
    
    # Writing style estimation
    # Cursive: fewer, longer strokes, smooth flow
    # Printed: more, shorter strokes, less smooth
    if avg_stroke_length > 100 and flow_smoothness > 0.5 and stroke_count < 10:
        writing_style = 'cursive'
    elif stroke_count > 15 and avg_stroke_length < 50:
        writing_style = 'printed'
    else:
        writing_style = 'mixed'
    
    return {
        'dominant_direction': float(dominant_direction),
        'stroke_count': int(stroke_count),
        'average_stroke_length': float(avg_stroke_length),
        'flow_smoothness': float(flow_smoothness),
        'stroke_density': float(stroke_density),
        'writing_style': writing_style,
        'stroke_data': stroke_data  # Include raw stroke data for comparison
    }


def compare_strokes(stroke_data1: Dict, stroke_data2: Dict) -> Dict:
    """
    Compare stroke-level features between two signatures.
    
    Returns:
        dict with:
            - stroke_count_similarity: How similar the stroke counts are
            - stroke_length_similarity: How similar average stroke lengths are
            - stroke_direction_similarity: How similar stroke directions are
            - stroke_pressure_similarity: How similar pressure patterns are
            - stroke_overlap: Percentage of strokes that overlap
            - overall_stroke_similarity: Combined stroke similarity score
    """
    # IMPROVED Stroke count similarity
    # Use more forgiving calculation that accounts for slight variations in stroke detection
    count1 = len(stroke_data1['strokes'])
    count2 = len(stroke_data2['strokes'])
    if count1 == 0 or count2 == 0:
        count_sim = 0.0
    else:
        # More forgiving: Allow ±1 stroke difference with high similarity
        count_diff = abs(count1 - count2)
        count_ratio = min(count1, count2) / max(count1, count2)
        
        # If difference is 0: 100% similarity
        # If difference is 1 and counts are similar: ~85% similarity (stroke detection can be off by 1)
        # If difference is 2+: Use ratio-based calculation
        if count_diff == 0:
            count_sim = 1.0
        elif count_diff == 1 and count_ratio > 0.6:  # e.g., 3 vs 4, 4 vs 5
            count_sim = 0.85  # High similarity for ±1 difference
        elif count_diff == 2 and count_ratio > 0.5:  # e.g., 3 vs 5, 4 vs 6
            count_sim = 0.70  # Moderate similarity for ±2 difference
        else:
            # Use ratio-based calculation for larger differences
            count_sim = count_ratio * 0.8  # Slightly scale down ratio to be more conservative
    
    count_sim = max(0.0, min(1.0, count_sim))  # Clamp to [0, 1]
    
    # IMPROVED Stroke length similarity
    # Compare distributions, not just averages (more robust)
    lengths1 = stroke_data1['stroke_lengths'] if stroke_data1['stroke_lengths'] else []
    lengths2 = stroke_data2['stroke_lengths'] if stroke_data2['stroke_lengths'] else []
    
    if not lengths1 or not lengths2:
        length_sim = 0.0
    else:
        avg_len1 = np.mean(lengths1)
        avg_len2 = np.mean(lengths2)
        
        # Also compare median (more robust to outliers)
        median_len1 = np.median(lengths1) if len(lengths1) > 0 else avg_len1
        median_len2 = np.median(lengths2) if len(lengths2) > 0 else avg_len2
        
        # Compare both mean and median, take average of similarities
        if avg_len1 == 0 or avg_len2 == 0:
            mean_sim = 0.0
        else:
            # More forgiving ratio calculation
            mean_ratio = min(avg_len1, avg_len2) / max(avg_len1, avg_len2)
            mean_diff_ratio = abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2)
            # If difference is < 10%: Very high similarity
            # If difference is < 25%: Good similarity
            # Otherwise: Use ratio
            if mean_diff_ratio < 0.10:
                mean_sim = 0.95
            elif mean_diff_ratio < 0.25:
                mean_sim = 0.80 + (0.25 - mean_diff_ratio) * 0.75  # Scales from 0.80 to 0.95
            else:
                mean_sim = mean_ratio
        
        # Median similarity (same logic)
        if median_len1 == 0 or median_len2 == 0:
            median_sim = 0.0
        else:
            median_ratio = min(median_len1, median_len2) / max(median_len1, median_len2)
            median_diff_ratio = abs(median_len1 - median_len2) / max(median_len1, median_len2)
            if median_diff_ratio < 0.10:
                median_sim = 0.95
            elif median_diff_ratio < 0.25:
                median_sim = 0.80 + (0.25 - median_diff_ratio) * 0.75
            else:
                median_sim = median_ratio
        
        # Combine mean and median similarities (weighted: 60% mean, 40% median)
        length_sim = 0.6 * mean_sim + 0.4 * median_sim
    
    length_sim = max(0.0, min(1.0, length_sim))  # Clamp to [0, 1]
    
    # Stroke direction similarity
    if stroke_data1['stroke_directions'] and stroke_data2['stroke_directions']:
        dir1 = np.mean(stroke_data1['stroke_directions'])
        dir2 = np.mean(stroke_data2['stroke_directions'])
        # Normalize angles to 0-180 range
        dir1 = dir1 % 180
        dir2 = dir2 % 180
        angle_diff = min(abs(dir1 - dir2), abs(180 - abs(dir1 - dir2)))
        direction_sim = 1.0 - (angle_diff / 90.0)  # Max 90 degree difference
    else:
        direction_sim = 0.0
    
    # Stroke pressure similarity
    if stroke_data1['stroke_pressure'] and stroke_data2['stroke_pressure']:
        avg_press1 = np.mean(stroke_data1['stroke_pressure'])
        avg_press2 = np.mean(stroke_data2['stroke_pressure'])
        if avg_press1 == 0 or avg_press2 == 0:
            pressure_sim = 0.0
        else:
            pressure_sim = 1.0 - abs(avg_press1 - avg_press2) / max(avg_press1, avg_press2)
    else:
        pressure_sim = 0.0
    
    # Overall stroke similarity (weighted average)
    overall = (
        0.3 * count_sim +
        0.3 * length_sim +
        0.2 * direction_sim +
        0.2 * pressure_sim
    )
    
    return {
        'stroke_count_similarity': float(count_sim),
        'stroke_length_similarity': float(length_sim),
        'stroke_direction_similarity': float(direction_sim),
        'stroke_pressure_similarity': float(pressure_sim),
        'overall_stroke_similarity': float(overall)
    }


def create_stroke_overlay(img1_rgb: np.ndarray, img2_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create EXACT OVERLAP visualization showing signature alignment.
    Shows both signatures overlaid on top of each other with color coding:
        - Red tint: Areas in Signature 1
        - Green tint: Areas in Signature 2
        - Yellow/Orange: Overlapping areas (matched strokes)
        - White: Empty background
    
    Args:
        img1_rgb: First signature (RGB) - should be preprocessed and aligned
        img2_rgb: Second signature (RGB) - should be preprocessed and aligned to img1
        alpha: Transparency for blending (0-1)
    
    Returns:
        RGB overlay image showing exact overlap
    """
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    
    # Ensure same size
    if (h1, w1) != (h2, w2):
        img2_rgb = cv2.resize(img2_rgb, (w1, h1), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale for mask creation
    gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
    
    # Create binary masks (signature pixels vs background)
    # Threshold to find signature strokes (dark pixels)
    _, mask1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find overlapping regions (both signatures have strokes)
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    
    # Create color-coded overlay
    overlay = np.ones((h1, w1, 3), dtype=np.uint8) * 255  # White background
    
    # Red tint for Signature 1 (unique areas)
    unique1 = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))
    overlay[unique1 > 0] = [255, 200, 200]  # Light red
    
    # Green tint for Signature 2 (unique areas)
    unique2 = cv2.bitwise_and(mask2, cv2.bitwise_not(mask1))
    overlay[unique2 > 0] = [200, 255, 200]  # Light green
    
    # Yellow/Orange for overlapping regions (matched strokes)
    overlay[overlap_mask > 0] = [255, 255, 0]  # Bright yellow for overlap
    
    # Blend with original signatures for better visualization
    # Signature 1 (red tinted)
    img1_tinted = img1_rgb.copy()
    img1_tinted[unique1 > 0] = np.clip(img1_tinted[unique1 > 0] * 0.7 + np.array([255, 100, 100]) * 0.3, 0, 255).astype(np.uint8)
    img1_tinted[overlap_mask > 0] = np.clip(img1_tinted[overlap_mask > 0] * 0.5 + np.array([255, 255, 0]) * 0.5, 0, 255).astype(np.uint8)
    
    # Signature 2 (green tinted)
    img2_tinted = img2_rgb.copy()
    img2_tinted[unique2 > 0] = np.clip(img2_tinted[unique2 > 0] * 0.7 + np.array([100, 255, 100]) * 0.3, 0, 255).astype(np.uint8)
    img2_tinted[overlap_mask > 0] = np.clip(img2_tinted[overlap_mask > 0] * 0.5 + np.array([255, 255, 0]) * 0.5, 0, 255).astype(np.uint8)
    
    # Create final overlay: blend both signatures with color coding
    # Start with white background
    final_overlay = np.ones((h1, w1, 3), dtype=np.uint8) * 255
    
    # Blend Signature 1
    final_overlay = cv2.addWeighted(final_overlay, 1-alpha/2, img1_tinted, alpha/2, 0)
    
    # Blend Signature 2
    final_overlay = cv2.addWeighted(final_overlay, 1-alpha/2, img2_tinted, alpha/2, 0)
    
    # Enhance overlap regions
    overlap_mask_3d = cv2.merge([overlap_mask, overlap_mask, overlap_mask])
    overlap_regions = np.where(overlap_mask_3d > 0)
    if len(overlap_regions[0]) > 0:
        # Make overlapping regions more visible with yellow tint
        final_overlay[overlap_regions] = np.clip(
            final_overlay[overlap_regions] * 0.6 + np.array([255, 255, 100]) * 0.4, 
            0, 255
        ).astype(np.uint8)
    
    return final_overlay


def detect_signature_in_document(doc_img: np.ndarray) -> List[Dict]:
    """
    Detect signature regions in a full document image.
    Uses stroke-based detection: looks for handwritten regions with characteristic patterns.
    
    Returns:
        List of detected signature regions, each with:
            - bbox: (x, y, w, h) bounding box
            - confidence: Detection confidence (0-1)
            - stroke_count: Number of strokes in region
            - is_signature_like: True if region matches signature characteristics
    """
    gray = cv2.cvtColor(doc_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Adaptive threshold to find handwritten regions
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to connect strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours (potential signature regions)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for contour in contours:
        x, y, w_box, h_box = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter by size (signatures are typically medium-sized)
        min_area = 500  # Minimum signature area
        max_area = w * h * 0.3  # Max 30% of image
        
        if area < min_area or area > max_area:
            continue
        
        # Filter by aspect ratio (signatures are typically wider than tall)
        aspect_ratio = w_box / max(h_box, 1)
        if aspect_ratio < 0.3 or aspect_ratio > 5.0:
            continue
        
        # Extract region
        region = doc_img[y:y+h_box, x:x+w_box]
        
        # Analyze stroke characteristics
        stroke_data = extract_strokes(region)
        stroke_count = len(stroke_data['strokes'])
        
        # Signature characteristics:
        # - Multiple strokes (not just a single blob)
        # - Smooth flow (handwritten)
        # - Moderate stroke count (typically 5-50 strokes)
        is_signature_like = (
            stroke_count >= 5 and
            stroke_count <= 50 and
            area > min_area and
            aspect_ratio > 0.5  # Not too tall/thin
        )
        
        # Confidence based on characteristics
        confidence = 0.0
        if is_signature_like:
            # Higher confidence for moderate stroke count and good aspect ratio
            stroke_score = min(stroke_count / 20.0, 1.0)  # Peak at ~20 strokes
            aspect_score = min(aspect_ratio / 2.0, 1.0) if aspect_ratio > 1 else aspect_ratio
            confidence = (stroke_score + aspect_score) / 2.0
        
        candidates.append({
            'bbox': (x, y, w_box, h_box),
            'confidence': float(confidence),
            'stroke_count': int(stroke_count),
            'is_signature_like': is_signature_like,
            'region': region  # Cropped region for analysis
        })
    
    # Sort by confidence
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    return candidates

