"""
Multi-signal verification combining cosine similarity, ORB matches, and SSIM.
This prevents false positives by using multiple independent signals.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, Dict, Optional
from backend.inference import _ensure_model_loaded, _to_model_tensor


def orb_match_ratio(imgA: np.ndarray, imgB: np.ndarray) -> Tuple[float, int, list, list]:
    """
    CRITICALLY IMPROVED ORB match ratio - optimized for normalized signature images.
    
    Issues with previous version:
    - Normalized images (220x155) are too small for good keypoint detection
    - Images are too clean/binary after normalization
    - ORB needs contrast and edges to find features
    
    Solutions:
    - Upscale images before ORB (2x-4x larger for better keypoint detection)
    - Sharpen edges to create more features
    - Use adaptive thresholding to preserve details
    - More aggressive keypoint detection
    """
    # CRITICAL: ORB matching requires IDENTICAL preprocessing
    # Both images must be:
    # - Same size
    # - Same brightness
    # - Same alignment (baseline)
    # - Same canvas (centered)
    # This is ensured by the normalization pipeline BEFORE ORB
    
    # Convert to grayscale
    a = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY) if len(imgA.shape) == 3 else imgA
    b = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY) if len(imgB.shape) == 3 else imgB
    
    # CRITICAL: Validate images are not empty
    if a.size == 0 or b.size == 0:
        print(f"   ❌ ORB: Empty images detected - a.shape={a.shape}, b.shape={b.shape}")
        return 0.0, 0, [], []
    
    h, w = a.shape
    if h == 0 or w == 0:
        print(f"   ❌ ORB: Invalid image dimensions - {w}x{h}")
        return 0.0, 0, [], []
    
    # CRITICAL: Images should already be preprocessed by normalization pipeline
    # Verify they are same size (required for ORB)
    if a.shape != b.shape:
        print(f"   ⚠️ ORB: Images different sizes - resizing to match")
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    
    # CRITICAL FIX 1: Upscale images for better keypoint detection
    # Normalized images are 220x155 which is too small for ORB
    # Upscale by 2x-4x to get better feature detection
    if h < 300 or w < 400:  # If images are small, upscale them
        scale_factor = max(2.0, 400.0 / w, 300.0 / h)  # At least 2x, or to reach 400x300
        new_w = max(400, int(w * scale_factor))  # Ensure minimum 400px width
        new_h = max(300, int(h * scale_factor))  # Ensure minimum 300px height
        a = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_CUBIC)  # Cubic for better quality
        b = cv2.resize(b, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"   📏 ORB: Upscaled images from {w}x{h} to {new_w}x{new_h} (scale={scale_factor:.2f}x)")
    
    # Ensure same size (should be same after upscale, but double-check)
    if a.shape != b.shape:
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    
    # CRITICAL FIX 2: Enhance images for better feature detection
    # Normalized signatures might be too clean/binary - add contrast and sharpen
    
    # Step 1: CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Higher clipLimit for more contrast
    a_enhanced = clahe.apply(a)
    b_enhanced = clahe.apply(b)
    
    # Step 2: Sharpen edges to create more features
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    a_sharp = cv2.filter2D(a_enhanced, -1, kernel_sharpen * 0.3)  # Light sharpening
    b_sharp = cv2.filter2D(b_enhanced, -1, kernel_sharpen * 0.3)
    a_enhanced = cv2.addWeighted(a_enhanced, 0.7, a_sharp, 0.3, 0)
    b_enhanced = cv2.addWeighted(b_enhanced, 0.7, b_sharp, 0.3, 0)
    
    # Step 3: Adaptive threshold to ensure good edges (if image is too uniform)
    # This helps if signatures are too light/clean after normalization
    mean_a = np.mean(a_enhanced)
    mean_b = np.mean(b_enhanced)
    if mean_a > 200 or mean_b > 200:  # Very light/clean images
        # Apply adaptive threshold to create edges
        a_thresh = cv2.adaptiveThreshold(a_enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 5)
        b_thresh = cv2.adaptiveThreshold(b_enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 5)
        # Blend original and thresholded (keep some gradient info)
        a_enhanced = cv2.addWeighted(a_enhanced, 0.6, a_thresh, 0.4, 0)
        b_enhanced = cv2.addWeighted(b_enhanced, 0.6, b_thresh, 0.4, 0)
    
    # CRITICAL FIX 3: Optimized ORB parameters for signature images
    # Lower fastThreshold to detect more features in clean signatures
    # Larger patchSize for better descriptor matching
    orb = cv2.ORB_create(
        nfeatures=3000,  # Even more features
        scaleFactor=1.2,
        nlevels=10,  # More levels for multi-scale detection
        edgeThreshold=21,  # Smaller edge threshold (was 31)
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=41,  # Larger patch for better matching (was 31)
        fastThreshold=15  # Lower threshold to detect more features in clean images (was 20)
    )
    
    kp1, des1 = orb.detectAndCompute(a_enhanced, None)
    kp2, des2 = orb.detectAndCompute(b_enhanced, None)
    
    print(f"   🔑 ORB: Detected {len(kp1)} keypoints in img1, {len(kp2)} in img2")

    if des1 is None or des2 is None:
        print(f"   ⚠️ ORB: Descriptors are None - img1={len(kp1)} kp, img2={len(kp2)} kp")
        print(f"   🔍 Debug: Image shapes - a_enhanced={a_enhanced.shape}, b_enhanced={b_enhanced.shape}")
        print(f"   🔍 Debug: Image stats - a mean={np.mean(a_enhanced):.1f}, b mean={np.mean(b_enhanced):.1f}")
        return 0.0, 0, kp1, kp2
    
    if len(kp1) == 0 or len(kp2) == 0:
        print(f"   ⚠️ ORB: No keypoints detected - img1={len(kp1)}, img2={len(kp2)}")
        print(f"   🔍 Debug: Try lowering fastThreshold or increasing image size")
        return 0.0, 0, kp1, kp2

    # CRITICAL FIX 4: Better matching with homography filtering (geometric validation)
    # Strategy 1: FLANN with Lowe's ratio test + Homography filtering
    try:
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=12,
                           key_size=12,
                           multi_probe_level=2)
        search_params = dict(checks=100)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test (0.75 is better than 0.8 for signatures)
        ratio_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                    ratio_matches.append(m)
        
        print(f"   🎯 ORB (FLANN): Found {len(ratio_matches)} matches after ratio test")
        
        # STEP 5: Homography filtering to reject geometric outliers
        # This validates that matches are spatially consistent
        if len(ratio_matches) >= 10:  # Need at least 10 matches for homography
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in ratio_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in ratio_matches]).reshape(-1, 1, 2)
            
            # Find homography with RANSAC to filter outliers
            H, mask = cv2.findHomography(src_pts, dst_pts, 
                                        cv2.RANSAC, 
                                        ransacReprojThreshold=5.0)  # 5px tolerance
            
            if mask is not None:
                # Count inliers (matches that fit the geometric transformation)
                inliers = np.sum(mask)
                inlier_ratio = inliers / len(ratio_matches)
                
                print(f"   🔍 Homography filtering: {inliers}/{len(ratio_matches)} inliers ({inlier_ratio*100:.1f}%)")
                
                # Use inliers as final matches (geometrically consistent)
                good_matches = [ratio_matches[i] for i in range(len(ratio_matches)) if mask[i]]
                num_matches = len(good_matches)
                
                # If homography is good, boost the match ratio
                if inlier_ratio > 0.5:  # >50% inliers = good geometric consistency
                    print(f"   ✅ Good geometric consistency - using inliers")
                else:
                    print(f"   ⚠️ Low geometric consistency - many outliers")
            else:
                # Homography failed - use ratio matches as-is
                good_matches = ratio_matches
                num_matches = len(good_matches)
                print(f"   ⚠️ Homography computation failed - using ratio matches")
        else:
            # Not enough matches for homography - use ratio matches
            good_matches = ratio_matches
            num_matches = len(good_matches)
            print(f"   ⚠️ Not enough matches for homography (need 10, got {len(ratio_matches)})")
        
    except Exception as e:
        print(f"   ⚠️ ORB: FLANN failed ({e}), using brute force")
        # Strategy 2: Brute force with distance filtering
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Filter by distance
        if len(matches) > 0:
            distances = [m.distance for m in matches]
            percentile_75 = np.percentile(distances, 75)
            good_matches = [m for m in matches if m.distance < percentile_75 * 1.2]
            num_matches = len(good_matches)
            
            # Try homography on brute force matches too
            if num_matches >= 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = np.sum(mask)
                    good_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
                    num_matches = len(good_matches)
                    print(f"   🔍 Homography (BF): {inliers}/{len(good_matches)} inliers")
        else:
            num_matches = 0
            good_matches = []
        
        print(f"   🎯 ORB (Brute Force): Found {num_matches} matches")
    
    # Match ratio: normalize by minimum keypoints
    total_keypoints = min(len(kp1), len(kp2))
    match_ratio = num_matches / max(1, total_keypoints)
    
    print(f"   📊 ORB: Match ratio = {num_matches}/{total_keypoints} = {match_ratio:.3f} ({match_ratio*100:.1f}%)")
    
    return float(match_ratio), int(num_matches), kp1, kp2


def compute_ssim_score(imgA: np.ndarray, imgB: np.ndarray) -> float:
    """
    Compute Structural Similarity Index between two images.
    
    CRITICAL: Images should already be preprocessed and aligned.
    Upscale for better SSIM calculation (similar to ORB fix).
    """
    a = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY) if len(imgA.shape) == 3 else imgA
    b = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY) if len(imgB.shape) == 3 else imgB
    
    # Ensure same size
    if a.shape != b.shape:
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    
    # CRITICAL: Upscale for better SSIM (small images = less detail for comparison)
    # Normalized images are 220x155 - upscale to at least 440x310 for better SSIM
    h, w = a.shape
    if h < 300 or w < 400:
        scale_factor = max(2.0, 400.0 / w, 300.0 / h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        a = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(b, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Compute SSIM with appropriate window size
    # Window size should be odd and less than image dimensions
    win_size = min(7, min(a.shape[0], a.shape[1]) // 2)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3
    
    score = ssim(a, b, win_size=win_size, data_range=255)
    return float(score)


def combined_verification(
    img1_array: np.ndarray,
    img2_array: np.ndarray,
    img1_pil: Optional[Image.Image] = None,
    img2_pil: Optional[Image.Image] = None,
    model=None,
    include_stroke_analysis: bool = True
) -> Dict:
    """
    Multi-signal verification with comprehensive checkpoints.
    
    CRITICAL: All images must be preprocessed (normalized) before calling this function.
    """
    print("=" * 60)
    print("🔍 MULTI-SIGNAL VERIFICATION STARTED")
    print("=" * 60)
    
    # CHECKPOINT: Verify input arrays
    assert img1_array is not None and img2_array is not None, "❌ CHECKPOINT FAILED: Input arrays are None"
    assert img1_array.size > 0 and img2_array.size > 0, "❌ CHECKPOINT FAILED: Input arrays are empty"
    assert len(img1_array.shape) >= 2 and len(img2_array.shape) >= 2, "❌ CHECKPOINT FAILED: Invalid array dimensions"
    print(f"✅ CHECKPOINT: Input arrays valid - sig1 shape: {img1_array.shape}, sig2 shape: {img2_array.shape}")
    
    # CHECKPOINT: Verify arrays are same size (required for fair comparison)
    if img1_array.shape[:2] != img2_array.shape[:2]:
        print(f"⚠️ CHECKPOINT WARNING: Different sizes - sig1: {img1_array.shape}, sig2: {img2_array.shape}")
        print(f"   Resizing to match...")
        h, w = min(img1_array.shape[0], img2_array.shape[0]), min(img1_array.shape[1], img2_array.shape[1])
        img1_array = cv2.resize(img1_array, (w, h), interpolation=cv2.INTER_AREA)
        img2_array = cv2.resize(img2_array, (w, h), interpolation=cv2.INTER_AREA)
        print(f"✅ CHECKPOINT: Resized to match - both: {img1_array.shape}")
    
    if model is None:
        model = _ensure_model_loaded()
    
    # 1. Cosine similarity from model embeddings (deep learning - most reliable)
    print("📊 Computing cosine similarity (model embeddings)...")
    
    if img1_pil is None:
        from backend.advanced_alignment import numpy_to_pil
        img1_pil = numpy_to_pil(img1_array)
    if img2_pil is None:
        from backend.advanced_alignment import numpy_to_pil
        img2_pil = numpy_to_pil(img2_array)
    
    # CHECKPOINT: Verify PIL images before model inference
    assert img1_pil is not None and img2_pil is not None, "❌ CHECKPOINT FAILED: PIL images are None"
    assert img1_pil.size[0] > 0 and img1_pil.size[1] > 0, "❌ CHECKPOINT FAILED: PIL images have zero size"
    print(f"✅ CHECKPOINT: PIL images valid - sig1 size: {img1_pil.size}, sig2 size: {img2_pil.size}")
    
    img1_tensor = _to_model_tensor(img1_pil)
    img2_tensor = _to_model_tensor(img2_pil)
    
    with torch.no_grad():
        emb1 = model.forward_once(img1_tensor)
        emb2 = model.forward_once(img2_tensor)
        cosine = F.cosine_similarity(emb1, emb2).item()
    
    # CHECKPOINT: Verify cosine similarity is valid
    assert 0.0 <= cosine <= 1.0, f"❌ CHECKPOINT FAILED: Invalid cosine similarity: {cosine}"
    print(f"✅ CHECKPOINT: Cosine similarity computed: {cosine:.4f}")
    
    # 2. ORB REMOVED - Always showing very low values (3-7%) even for same signatures
    # ORB is unreliable for thin stroke patterns like signatures
    # Better to rely on Cosine (deep embeddings) + SSIM (structural similarity) + Handwriting analysis
    orb_ratio = 0.0
    num_matches = 0
    kp1, kp2 = [], []
    orb_contribution = 0.0
    print("ℹ️ ORB matching skipped (unreliable for signatures)")
    
    # 3. SSIM
    print("📊 Computing SSIM (structural similarity)...")
    
    # CHECKPOINT: Verify arrays before SSIM
    assert img1_array.shape == img2_array.shape, f"❌ CHECKPOINT FAILED: Size mismatch before SSIM - sig1: {img1_array.shape}, sig2: {img2_array.shape}"
    print(f"✅ CHECKPOINT: Arrays ready for SSIM - shape: {img1_array.shape}")
    
    ssim_score = compute_ssim_score(img1_array, img2_array)
    
    # CHECKPOINT: Verify SSIM result
    assert 0.0 <= ssim_score <= 1.0, f"❌ CHECKPOINT FAILED: Invalid SSIM score: {ssim_score}"
    print(f"✅ CHECKPOINT: SSIM computed: {ssim_score:.4f}")
    
    # 4. Stroke analysis (if enabled)
    stroke_similarity = 0.0
    handwriting_flow1 = {}
    handwriting_flow2 = {}
    stroke_comparison = {}
    
    if include_stroke_analysis:
        try:
            from backend.stroke_analysis import (
                analyze_handwriting_flow, compare_strokes, extract_strokes
            )
            
            # Analyze handwriting flow and strokes
            handwriting_flow1 = analyze_handwriting_flow(img1_array)
            handwriting_flow2 = analyze_handwriting_flow(img2_array)
            
            # Compare strokes
            stroke_data1 = handwriting_flow1.get('stroke_data', {})
            stroke_data2 = handwriting_flow2.get('stroke_data', {})
            
            if stroke_data1 and stroke_data2:
                stroke_comparison = compare_strokes(stroke_data1, stroke_data2)
                stroke_similarity = stroke_comparison.get('overall_stroke_similarity', 0.0)
        except Exception as e:
            print(f"Stroke analysis error: {e}")
            stroke_similarity = 0.0
    
    # ====================================================================
    # CRITICAL: HANDWRITING ANALYSIS IS PRIMARY (60% weight)
    # User requested: 60% handwriting, 40% other signals
    # ALL ANALYSIS MUST USE PREPROCESSED IMAGES (img1_array, img2_array)
    # ====================================================================
    
    # Calculate comprehensive handwriting similarity score
    handwriting_score = 0.0
    handwriting_details = {}
    
    print(f"🔍 Handwriting analysis check:")
    print(f"   - include_stroke_analysis: {include_stroke_analysis}")
    print(f"   - handwriting_flow1 exists: {bool(handwriting_flow1)}")
    print(f"   - handwriting_flow2 exists: {bool(handwriting_flow2)}")
    print(f"   - stroke_comparison exists: {bool(stroke_comparison)}")
    
    if include_stroke_analysis and handwriting_flow1 and handwriting_flow2 and stroke_comparison:
        # Combine multiple handwriting metrics:
        # 1. Stroke similarity (from stroke_comparison)
        stroke_sim = stroke_comparison.get('overall_stroke_similarity', 0.0)
        
        # 2. Flow smoothness similarity - IF SAME = 100%, CLOSE = HIGH SCORE
        # User request: "IF FLOW% IS SAME = 100%.. SO 66 AND 65 SHOULD BE 99.9%"
        flow1_smooth = handwriting_flow1.get('flow_smoothness', 0.0)
        flow2_smooth = handwriting_flow2.get('flow_smoothness', 0.0)
        flow_diff = abs(flow1_smooth - flow2_smooth)
        
        # CRITICAL FIX: Calculate similarity based on closeness (not average)
        # If diff = 0%: 100% similarity
        # If diff = 1% (66% vs 65%): 99.9% similarity
        # Formula: similarity = 100% - (diff * 0.1), with minimum 0%
        # For 1% diff: 1.0 - (0.01 * 0.1) = 1.0 - 0.001 = 0.999 = 99.9%
        flow_similarity = max(0.0, 1.0 - (flow_diff * 0.1))
        
        # Ensure it's in valid range
        flow_similarity = max(0.0, min(1.0, flow_similarity))
        
        # 3. Writing style match (1.0 if same, 0.7 if different but both exist)
        style1 = handwriting_flow1.get('writing_style', 'unknown')
        style2 = handwriting_flow2.get('writing_style', 'unknown')
        style_match = 1.0 if style1 == style2 else (0.7 if style1 != 'unknown' and style2 != 'unknown' else 0.5)
        
        # 4. Length, direction, pressure similarities (stroke count removed)
        length_sim = stroke_comparison.get('stroke_length_similarity', 0.0)
        direction_sim = stroke_comparison.get('stroke_direction_similarity', 0.0)
        pressure_sim = stroke_comparison.get('stroke_pressure_similarity', 0.0)
        
        # Stroke Comparison component (30% of total, increased from 25%)
        # Less weight on length similarity as requested
        stroke_comparison_score = (
            0.15 * length_sim +        # Less weight (15% of stroke comp = 4.5% of total)
            0.30 * direction_sim +     # More weight (30% of stroke comp = 9% of total)
            0.30 * pressure_sim +      # More weight (30% of stroke comp = 9% of total)
            0.25 * stroke_sim          # Overall stroke similarity (25% of stroke comp = 7.5% of total)
        )
        
        # Stroke comparison = 30% of total score (will be scaled later)
        
        # USER REQUEST: Handwriting = 10% total weight
        # SLIGHTLY consider stroke count (10% of handwriting = 1% of total)
        # Flow: 45%, Style: 45%, Count: 10% (within handwriting's 10%)
        
        # Get stroke count similarity (slight weight)
        count_sim = stroke_comparison.get('stroke_count_similarity', 0.0)
        
        handwriting_score = (
            0.45 * flow_similarity +    # 45% of handwriting (4.5% of total)
            0.45 * style_match +        # 45% of handwriting (4.5% of total)
            0.10 * count_sim            # 10% of handwriting (1% of total - SLIGHT)
        )
        
        # Handwriting = 10% of total score (will be scaled later)
        
        # Ensure handwriting score is in valid range
        handwriting_score = max(0.0, min(1.0, handwriting_score))
        
        handwriting_details = {
            "flow_similarity": flow_similarity,
            "style_match": style_match,
            "count_similarity": count_sim  # Added back with slight weight
        }
        
        print(f"📊 Handwriting score (10% of total - slight stroke count):")
        print(f"   - Flow sim: {flow_similarity:.4f} × 0.45 = {0.45 * flow_similarity:.4f} (45% of handwriting)")
        print(f"   - Style match: {style_match:.4f} × 0.45 = {0.45 * style_match:.4f} (45% of handwriting)")
        print(f"   - Count sim: {count_sim:.4f} × 0.10 = {0.10 * count_sim:.4f} (10% of handwriting - SLIGHT)")
        print(f"   → Handwriting component: {handwriting_score:.4f} (10% of total)")
        
        # DEBUG: Show flow difference for user visibility
        flow1_debug = handwriting_flow1.get('flow_smoothness', 0.0)
        flow2_debug = handwriting_flow2.get('flow_smoothness', 0.0)
        print(f"   🔍 Flow: Sig1={flow1_debug:.1f}%, Sig2={flow2_debug:.1f}%, Diff={abs(flow1_debug-flow2_debug):.2f}%, Similarity={flow_similarity:.1f}%")
        
        print(f"📊 Stroke Comparison score (30% of total - less weight on length):")
        print(f"   - Stroke sim: {stroke_sim:.4f} × 0.25 = {0.25 * stroke_sim:.4f} (25% of stroke comp)")
        print(f"   - Length sim: {length_sim:.4f} × 0.15 = {0.15 * length_sim:.4f} (15% of stroke comp - LESS WEIGHT)")
        print(f"   - Direction sim: {direction_sim:.4f} × 0.30 = {0.30 * direction_sim:.4f} (30% of stroke comp)")
        print(f"   - Pressure sim: {pressure_sim:.4f} × 0.30 = {0.30 * pressure_sim:.4f} (30% of stroke comp)")
        print(f"   → Stroke comparison component: {stroke_comparison_score:.4f} (30% of total)")
    elif include_stroke_analysis and stroke_similarity > 0:
        # Fallback: use overall stroke similarity only
        handwriting_score = stroke_similarity
        stroke_comparison_score = stroke_similarity  # Use stroke similarity as fallback
    else:
        # No handwriting analysis available
        handwriting_score = 0.0
        stroke_comparison_score = 0.0
    
    # Ensure stroke_comparison_score exists (in case stroke_analysis failed)
    if 'stroke_comparison_score' not in locals():
        stroke_comparison_score = 0.0
    
    # Ensure all scores are in valid range
    handwriting_score = max(0.0, min(1.0, handwriting_score))
    stroke_comparison_score = max(0.0, min(1.0, stroke_comparison_score))
    
    # Other signals (40% total weight)
    # Improved weights based on signature-specific reliability:
    # - SSIM: 15% (more reliable for signatures than ORB)
    # - Cosine: 20% (still primary, but reduced slightly)
    # - ORB: 5% (less reliable for thin strokes - dynamic weight)
    # 
    # Rationale: SSIM measures structural similarity which is better for signatures
    # ORB is weak for thin stroke patterns, so it gets lower weight
    # ORB weight reduces further if match ratio is very low (<5%)
    
    # USER REQUEST: Updated weight distribution
    # - Cosine: 30% (reduced from 40%)
    # - SSIM: 30% (increased from 25%)
    # - Handwriting: 10% (flow + style + slight stroke count)
    # - Stroke Comparison: 30% (increased from 25%)
    # Total: 100%
    
    cosine_weight = 0.30  # 30% of total
    ssim_weight = 0.30     # 30% of total
    handwriting_weight = 0.10  # 10% of total
    stroke_comparison_weight = 0.30  # 30% of total
    
    # Calculate weighted components
    cosine_component = cosine_weight * cosine
    ssim_component = ssim_weight * ssim_score
    handwriting_component = handwriting_weight * max(0.0, min(1.0, handwriting_score))
    stroke_comparison_component = stroke_comparison_weight * max(0.0, min(1.0, stroke_comparison_score))
    
    print(f"📊 Final score components:")
    print(f"   - Cosine: {cosine:.4f} × {cosine_weight:.2f} = {cosine_component:.4f} (30%)")
    print(f"   - SSIM: {ssim_score:.4f} × {ssim_weight:.2f} = {ssim_component:.4f} (30%)")
    print(f"   - Handwriting: {handwriting_score:.4f} × {handwriting_weight:.2f} = {handwriting_component:.4f} (10%)")
    print(f"   - Stroke Comparison: {stroke_comparison_score:.4f} × {stroke_comparison_weight:.2f} = {stroke_comparison_component:.4f} (30%)")
    
    # FINAL COMBINED SCORE: All components added together
    combined = (cosine_component + 
                ssim_component + 
                handwriting_component + 
                stroke_comparison_component)
    
    # CRITICAL: Final score should be in [0, 1] range
    combined = max(0.0, min(1.0, combined))
    
    # CHECKPOINT: Verify combined score is valid
    assert 0.0 <= combined <= 1.0, f"❌ CHECKPOINT FAILED: Invalid combined score: {combined}"
    print(f"✅ CHECKPOINT: Combined score computed: {combined:.4f}")
    
    print(f"🎯 FINAL COMBINED SCORE:")
    print(f"   → FINAL SCORE: {combined:.4f} ({combined * 100:.1f}%)")
    print(f"")
    print(f"📐 WEIGHT DISTRIBUTION:")
    print(f"   - Cosine similarity: 30%")
    print(f"   - SSIM: 30%")
    print(f"   - Handwriting (10%):")
    print(f"     - Flow similarity: 4.5% (45% of handwriting)")
    print(f"     - Style match: 4.5% (45% of handwriting)")
    print(f"     - Count similarity: 1% (10% of handwriting - SLIGHT)")
    print(f"   - Stroke Comparison (30%):")
    print(f"     - Overall stroke similarity: 7.5% (25% of stroke comp)")
    print(f"     - Length similarity: 4.5% (15% of stroke comp - LESS)")
    print(f"     - Direction similarity: 9% (30% of stroke comp)")
    print(f"     - Pressure similarity: 9% (30% of stroke comp)")
    print(f"   Total: 100.0%")
    
    # Additional validation: if handwriting styles differ significantly, small penalty
    if include_stroke_analysis and handwriting_flow1 and handwriting_flow2:
        style1 = handwriting_flow1.get('writing_style', 'unknown')
        style2 = handwriting_flow2.get('writing_style', 'unknown')
        if style1 != style2 and style1 != 'unknown' and style2 != 'unknown':
            # Different styles - small penalty (handwriting already accounts for this, but extra safety)
            combined *= 0.95  # Reduce by 5%
    
    # IMPROVED THRESHOLDS - More strict to reduce false positives/negatives
    # CRITICAL: Check for language/style mismatches first
    
    # Language/Character set detection: If stroke patterns are VERY different, likely different language
    style1 = handwriting_flow1.get('writing_style', 'unknown') if include_stroke_analysis and handwriting_flow1 else 'unknown'
    style2 = handwriting_flow2.get('writing_style', 'unknown') if include_stroke_analysis and handwriting_flow2 else 'unknown'
    
    # Check stroke count difference - different languages often have different stroke counts
    count1 = handwriting_flow1.get('stroke_count', 0) if include_stroke_analysis and handwriting_flow1 else 0
    count2 = handwriting_flow2.get('stroke_count', 0) if include_stroke_analysis and handwriting_flow2 else 0
    stroke_count_diff = abs(count1 - count2)
    stroke_count_ratio = min(count1, count2) / max(count1, count2) if max(count1, count2) > 0 else 0.0
    
    # CRITICAL: Large stroke count difference (e.g., 3 vs 6) suggests different language/character set
    language_mismatch = False
    if stroke_count_diff > 3 and stroke_count_ratio < 0.5:  # More than 3 stroke diff AND ratio < 50%
        language_mismatch = True
        print(f"⚠️ LANGUAGE MISMATCH DETECTED: Stroke count diff={stroke_count_diff} (sig1: {count1}, sig2: {count2})")
    
    # Flow difference check - different languages have different writing flow
    flow1 = handwriting_flow1.get('flow_smoothness', 0.0) if include_stroke_analysis and handwriting_flow1 else 0.0
    flow2 = handwriting_flow2.get('flow_smoothness', 0.0) if include_stroke_analysis and handwriting_flow2 else 0.0
    flow_diff_abs = abs(flow1 - flow2)
    
    if flow_diff_abs > 0.30:  # More than 30% flow difference
        language_mismatch = True
        print(f"⚠️ FLOW MISMATCH DETECTED: Flow diff={flow_diff_abs:.3f} (sig1: {flow1:.2f}, sig2: {flow2:.2f})")
    
    # If language mismatch detected, force "Different person" unless scores are EXTREMELY high
    if language_mismatch and combined < 0.90:
        verdict = "Different person"
        confidence = "high"
        requires_review = False
        print(f"🚫 VERDICT: Different person (language/style mismatch detected)")
    elif handwriting_score >= 0.85 and combined >= 0.80:
        # Strong handwriting match + good overall = Same person
        verdict = "Same person"
        confidence = "high"
        requires_review = False
        print(f"✅ VERDICT: Same person (strong handwriting match)")
    elif combined >= 0.85:
        # Very high combined score = Same person
        verdict = "Same person"
        confidence = "high"
        requires_review = False
        print(f"✅ VERDICT: Same person (very high combined score)")
    elif handwriting_score < 0.60 and combined < 0.70:
        # Weak handwriting + low combined = Different person
        verdict = "Different person"
        confidence = "high"
        requires_review = False
        print(f"❌ VERDICT: Different person (weak handwriting + low score)")
    elif combined < 0.70:
        # Low combined = Different person
        verdict = "Different person"
        confidence = "high"
        requires_review = False
        print(f"❌ VERDICT: Different person (low combined score)")
    elif 0.70 <= combined < 0.80:
        # Borderline - require manual review
        verdict = "Uncertain - Manual review required"
        confidence = "medium"
        requires_review = True
        print(f"⚠️ VERDICT: Uncertain (borderline score, requires review)")
    else:
        # Fallback
        verdict = "Different person"
        confidence = "medium"
        requires_review = True
        print(f"⚠️ VERDICT: Different person (fallback)")
    
    # Additional safety flags (ORB removed - no longer checking ORB matches)
    safety_flags = []
    
    if 0.75 <= combined < 0.85:
        safety_flags.append("uncertain_score")
    
    # Check handwriting flow consistency
    if include_stroke_analysis and handwriting_flow1 and handwriting_flow2:
        flow_diff = abs(handwriting_flow1.get('flow_smoothness', 0) - 
                       handwriting_flow2.get('flow_smoothness', 0))
        if flow_diff > 0.3:
            safety_flags.append("handwriting_flow_mismatch")
            requires_review = True
        
        style1 = handwriting_flow1.get('writing_style', 'unknown')
        style2 = handwriting_flow2.get('writing_style', 'unknown')
        if style1 != style2 and style1 != 'unknown' and style2 != 'unknown':
            safety_flags.append("writing_style_mismatch")
            requires_review = True
    
    result = {
        "cosine": float(cosine),
        "cosine_weight": float(cosine_weight),  # Actual weight used (12.25% of total)
        "orb_ratio": float(orb_ratio),  # Always 0.0 (ORB removed)
        "orb_weight": 0.0,  # Always 0.0 (ORB removed)
        "orb_matches": int(num_matches),  # Always 0 (ORB removed)
        "orb_keypoints1": len(kp1),  # Always 0 (ORB removed)
        "orb_keypoints2": len(kp2),  # Always 0 (ORB removed)
        "ssim": float(ssim_score),
        "ssim_weight": float(ssim_weight),  # Actual weight used (12.25% of total)
        "combined_score": float(combined),
        "verdict": verdict,
        "confidence": confidence,
        "requires_review": requires_review,
        "safety_flags": safety_flags
    }
    
    # Add handwriting analysis results if available
    if include_stroke_analysis and handwriting_flow1:
        result["stroke_similarity"] = float(stroke_similarity)
        result["handwriting_score"] = float(handwriting_score)  # NEW: Comprehensive handwriting score (60% weight)
        result["handwriting_flow1"] = {
            "writing_style": handwriting_flow1.get('writing_style', 'unknown'),
            "flow_smoothness": handwriting_flow1.get('flow_smoothness', 0),
            "stroke_count": handwriting_flow1.get('stroke_count', 0)
        }
        result["handwriting_flow2"] = {
            "writing_style": handwriting_flow2.get('writing_style', 'unknown'),
            "flow_smoothness": handwriting_flow2.get('flow_smoothness', 0),
            "stroke_count": handwriting_flow2.get('stroke_count', 0)
        }
        if stroke_comparison:
            result["stroke_comparison"] = stroke_comparison
        if handwriting_details:
            result["handwriting_details"] = handwriting_details  # NEW: Detailed breakdown
    
    return result


# Add missing torch import
import torch

