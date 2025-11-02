"""
Diagnostic script to identify why false positives occur.
Run this on problematic signature pairs to see all metrics.
"""
import cv2
import numpy as np
import torch
from PIL import Image
import io
try:
    from skimage.metrics import structural_similarity as ssim  # type: ignore
except ImportError:
    # Fallback if scikit-image not installed
    print("Warning: scikit-image not installed. Install with: pip install scikit-image")
    ssim = None

from backend.advanced_alignment import (
    pil_to_numpy, numpy_to_pil, denoise_signature,
    preprocess_for_model, align_pair_via_orb
)
from backend.inference import get_similarity_score, _ensure_model_loaded, _to_model_tensor
import torch.nn.functional as F


def orb_match_ratio(imgA, imgB):
    """Compute ORB match ratio between two images."""
    a = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
    
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)
    
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0, [], kp1, kp2
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    match_ratio = len(matches) / max(1, min(len(kp1), len(kp2)))
    return match_ratio, matches, kp1, kp2


def compute_ssim(imgA, imgB):
    """Compute Structural Similarity Index."""
    a = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
    
    # Resize to same size for comparison
    size = (256, 256)
    a = cv2.resize(a, size)
    b = cv2.resize(b, size)
    
    score = ssim(a, b)
    return score


def combined_verification(p1, p2, model=None):
    """
    Multi-signal verification combining cosine, ORB ratio, and SSIM.
    
    Returns:
        dict with cosine, orb_ratio, ssim, combined_score, matches count
    """
    # Cosine similarity from model embeddings
    if model is None:
        model = _ensure_model_loaded()
    
    img1_tensor = _to_model_tensor(numpy_to_pil(p1))
    img2_tensor = _to_model_tensor(numpy_to_pil(p2))
    
    with torch.no_grad():
        emb1 = model.forward_once(img1_tensor)
        emb2 = model.forward_once(img2_tensor)
        cosine = F.cosine_similarity(emb1, emb2).item()
    
    # ORB match ratio
    orb_ratio, matches, kp1, kp2 = orb_match_ratio(p1, p2)
    
    # Penalize if not enough matches
    if len(matches) < 8:
        orb_ratio = orb_ratio * 0.5
    
    # SSIM
    ssim_score = compute_ssim(p1, p2)
    
    # Combined score (weighted)
    combined = 0.6 * cosine + 0.25 * orb_ratio + 0.15 * ssim_score
    
    return {
        "cosine": cosine,
        "orb_ratio": orb_ratio,
        "ssim": ssim_score,
        "combined": combined,
        "matches": len(matches),
        "keypoints1": len(kp1),
        "keypoints2": len(kp2)
    }


def diagnostics(sig1_path=None, sig1_pil=None, sig2_path=None, sig2_pil=None):
    """
    Run full diagnostics on two signatures.
    
    Args:
        sig1_path: Path to first signature image (optional if sig1_pil provided)
        sig1_pil: PIL Image of first signature (optional if sig1_path provided)
        sig2_path: Path to second signature image (optional if sig2_pil provided)
        sig2_pil: PIL Image of second signature (optional if sig2_path provided)
    
    Returns:
        dict with all diagnostic metrics
    """
    # Load images
    if sig1_pil is None:
        s1 = cv2.imread(sig1_path)
        s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2RGB)
        sig1_pil = numpy_to_pil(s1)
    else:
        s1 = pil_to_numpy(sig1_pil)
    
    if sig2_pil is None:
        s2 = cv2.imread(sig2_path)
        s2 = cv2.cvtColor(s2, cv2.COLOR_BGR2RGB)
        sig2_pil = numpy_to_pil(s2)
    else:
        s2 = pil_to_numpy(sig2_pil)
    
    # Denoise
    s1_clean = denoise_signature(s1)
    s2_clean = denoise_signature(s2)
    
    # Preprocess to model size
    p1 = preprocess_for_model(s1_clean, target_size=(220, 155))
    p2 = preprocess_for_model(s2_clean, target_size=(220, 155))
    
    # Get model similarity (original method)
    score, verdict = get_similarity_score(sig1_pil, sig2_pil, threshold=0.92)
    
    # Combined verification
    combined_metrics = combined_verification(p1, p2)
    
    # ORB alignment check
    warped2, M = align_pair_via_orb(p1, p2, min_match_ratio=0.25)
    alignment_success = M is not None
    
    # Determine warp type
    warp_type = "None"
    if M is not None:
        if M.shape == (3, 3):
            warp_type = "Homography (Projective)"  # Potentially unsafe
        elif M.shape == (2, 3):
            warp_type = "Affine"
        else:
            warp_type = "Unknown"
    
    # Create overlay visualization
    overlay = cv2.addWeighted(p1, 0.5, warped2, 0.5, 0)
    
    result = {
        "model_similarity": score,
        "model_verdict": verdict,
        "cosine": combined_metrics["cosine"],
        "orb_ratio": combined_metrics["orb_ratio"],
        "orb_matches": combined_metrics["matches"],
        "orb_keypoints1": combined_metrics["keypoints1"],
        "orb_keypoints2": combined_metrics["keypoints2"],
        "ssim": combined_metrics["ssim"],
        "combined_score": combined_metrics["combined"],
        "alignment_success": alignment_success,
        "warp_type": warp_type,
        "overlay_image": overlay,
        "preprocessed1": p1,
        "preprocessed2": p2,
        "warped2": warped2
    }
    
    # Print summary
    print("\n" + "="*60)
    print("DIAGNOSTIC RESULTS")
    print("="*60)
    print(f"Model Cosine Similarity: {score:.4f}")
    print(f"Model Verdict: {verdict}")
    print(f"ORB Match Ratio: {combined_metrics['orb_ratio']:.4f} ({combined_metrics['matches']} matches)")
    print(f"ORB Keypoints: {combined_metrics['keypoints1']} vs {combined_metrics['keypoints2']}")
    print(f"SSIM Score: {combined_metrics['ssim']:.4f}")
    print(f"Combined Score: {combined_metrics['combined']:.4f}")
    print(f"Alignment Applied: {alignment_success}")
    print(f"Warp Type: {warp_type}")
    print("="*60)
    
    # Safety checks
    flags = []
    if combined_metrics['combined'] < 0.75:
        flags.append("❌ LOW COMBINED SCORE - Should be Different")
    elif 0.75 <= combined_metrics['combined'] < 0.88:
        flags.append("⚠️  UNCERTAIN ZONE - Requires manual review")
    
    if combined_metrics['matches'] < 8:
        flags.append("⚠️  INSUFFICIENT ORB MATCHES - Low keypoint support")
    
    if warp_type == "Homography (Projective)":
        flags.append("⚠️  PROJECTIVE WARP DETECTED - Potentially unsafe alignment")
    
    if flags:
        print("\n⚠️  SAFETY FLAGS:")
        for flag in flags:
            print(f"  {flag}")
    
    return result


# For testing with file paths
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        sig1_path, sig2_path = sys.argv[1], sys.argv[2]
        result = diagnostics(sig1_path=sig1_path, sig2_path=sig2_path)
        
        # Save overlay for visual inspection
        cv2.imwrite("diag_overlay.png", cv2.cvtColor(result["overlay_image"], cv2.COLOR_RGB2BGR))
        print(f"\n✅ Saved diagnostic overlay to: diag_overlay.png")
    else:
        print("Usage: python -m backend.diagnostics <sig1_path> <sig2_path>")

