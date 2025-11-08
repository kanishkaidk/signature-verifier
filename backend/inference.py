import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from backend.model.model_utils import SiameseNetwork
from backend.preprocessing import align_signatures
from backend.visualization import (
    overlay_heatmap,
    create_dual_overlay,
    create_difference_map,
    explain_visualization,
    normalize_to_canvas,
    align_pair_to_canvas,
    CANVAS_SIZE
)
from typing import Tuple
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reuse the same preprocessing as training: grayscale, 155x220, normalized
# Match ResNet18 (pretrained) expected 3-channel normalization
_transform = transforms.Compose([
    transforms.Resize((155, 220)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

_model_singleton = None

def _ensure_model_loaded():
    global _model_singleton
    if _model_singleton is None:
        model = SiameseNetwork().to(device)
        # Load the existing checkpoint in the repo
        checkpoint_path = "backend/model/siamese_model.pth"
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _model_singleton = model
    return _model_singleton

def _to_model_tensor(img_pil: Image.Image):
    # Convert to RGB to match ResNet18 3-channel input
    img = img_pil.convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(device)
    return tensor

def get_similarity_score(
    img1_pil: Image.Image, 
    img2_pil: Image.Image, 
    threshold: float = 0.92,  # Increased default threshold to reduce false positives
    enable_alignment: bool = True,
    use_advanced_alignment: bool = False,
    use_multi_signal: bool = True
):
    """
    Get similarity score between two signature images.
    
    Args:
        img1_pil: First signature image (PIL)
        img2_pil: Second signature image (PIL)
        threshold: Similarity threshold for verdict (legacy, ignored if use_multi_signal=True)
        enable_alignment: Enable preprocessing alignment (centering, deskewing, etc.)
        use_advanced_alignment: Use ORB-based alignment for better matching
        use_multi_signal: If True, use combined verification (cosine + ORB + SSIM)
    
    Returns:
        (similarity_score, verdict) tuple
        If use_multi_signal=True, similarity_score is the combined_score
    """
    model = _ensure_model_loaded()
    
    # CRITICAL: If images are already normalized, SKIP all alignment!
    # Normalized images come in already aligned, same size, same baseline
    if enable_alignment or use_advanced_alignment:
        # Use advanced alignment if requested
        if use_advanced_alignment:
            from backend.advanced_alignment import (
                pil_to_numpy, numpy_to_pil, denoise_signature,
                preprocess_for_model, align_pair_via_orb
            )
            
            # Convert to numpy
            img1_arr = pil_to_numpy(img1_pil)
            img2_arr = pil_to_numpy(img2_pil)
            
            # Denoise
            img1_clean = denoise_signature(img1_arr)
            img2_clean = denoise_signature(img2_arr)
            
            # Preprocess to model size (same preprocessing for model and viz)
            img1_pre = preprocess_for_model(img1_clean, target_size=(220, 155))
            img2_pre = preprocess_for_model(img2_clean, target_size=(220, 155))
            
            # Align using ORB + RANSAC (only if signatures are similar enough)
            # This function now checks match ratio and won't force-align different signatures
            img2_warped, transform_matrix = align_pair_via_orb(img1_pre, img2_pre, min_match_ratio=0.25)
            
            # If alignment failed (different signatures), use preprocessed but unaligned
            if transform_matrix is None:
                # Signatures are likely different - don't force alignment
                img2_warped = img2_pre
            
            # Convert back to PIL
            img1_aligned = numpy_to_pil(img1_pre)
            img2_aligned = numpy_to_pil(img2_warped)
        elif enable_alignment:
            # Original alignment method
            img1_aligned, img2_aligned = align_signatures(
                img1_pil, 
                img2_pil,
                target_size=(220, 155),  # Match model input size
                enable_deskew=True,
                enable_padding=True
            )
        else:
            # No alignment requested - use as-is
            img1_aligned, img2_aligned = img1_pil, img2_pil
    else:
        # CRITICAL: Images are already normalized - use as-is without any alignment
        # This prevents double-alignment which would destroy the normalization
        img1_aligned, img2_aligned = img1_pil, img2_pil
    
    # Use multi-signal verification if requested
    if use_multi_signal:
        from backend.multi_signal_verification import combined_verification
        from backend.advanced_alignment import pil_to_numpy
        
        # Get preprocessed arrays for multi-signal computation
        img1_arr = pil_to_numpy(img1_aligned)
        img2_arr = pil_to_numpy(img2_aligned)
        
        # Run combined verification
        multi_result = combined_verification(
            img1_arr, img2_arr,
            img1_pil=img1_aligned,
            img2_pil=img2_aligned,
            model=model
        )
        
        return multi_result["combined_score"], multi_result["verdict"]
    
    # Legacy single-signal method (cosine only)
    img1 = _to_model_tensor(img1_aligned)
    img2 = _to_model_tensor(img2_aligned)

    with torch.no_grad():
        emb1 = model.forward_once(img1)
        emb2 = model.forward_once(img2)
        sim_score = F.cosine_similarity(emb1, emb2).item()

    # Use stricter threshold to reduce false positives
    # Threshold ranges: 0.92+ = High confidence same, 0.85-0.92 = Uncertain, <0.85 = Different
    strict_threshold = max(threshold, 0.92)  # Never go below 0.92 for "Same person"
    
    if sim_score >= strict_threshold:
        verdict = "Same person"
    elif sim_score >= 0.85:
        verdict = "Uncertain - Manual review recommended"
    else:
        verdict = "Different person"
    
    return sim_score, verdict

def compare_signatures(img_path1, img_path2):
    # Helper for CLI/testing with file paths
    model = _ensure_model_loaded()
    img1 = _to_model_tensor(Image.open(img_path1))
    img2 = _to_model_tensor(Image.open(img_path2))
    with torch.no_grad():
        emb1 = model.forward_once(img1)
        emb2 = model.forward_once(img2)
        sim_score = F.cosine_similarity(emb1, emb2).item()
    return sim_score

def generate_saliency_heatmap(
    img1_pil: Image.Image, 
    img2_pil: Image.Image, 
    enable_alignment: bool = True,
    overlay_alpha: float = 0.5
) -> Image.Image:
    """
    Generate saliency heatmap with proper overlay on aligned image.
    Shows where img2 pixels affect the similarity score.
    
    Returns:
        PIL Image with heatmap overlaid on Signature 2
    """
    model = _ensure_model_loaded()
    
    # Align signatures to consistent canvas for perfect alignment
    # This ensures both images are normalized to same size without warping
    if enable_alignment:
        img1_aligned, img2_aligned = align_pair_to_canvas(
            img1_pil, img2_pil, canvas_size=(220, 155)
        )
    else:
        # Still normalize sizes even without alignment for consistent model input
        from backend.visualization import normalize_to_canvas
        img1_aligned = normalize_to_canvas(img1_pil, canvas_size=(220, 155))
        img2_aligned = normalize_to_canvas(img2_pil, canvas_size=(220, 155))
    
    img1 = _to_model_tensor(img1_aligned)
    img2 = _to_model_tensor(img2_aligned)

    img2.requires_grad_(True)
    # Normalize to unit vectors to stabilize gradient scale
    with torch.enable_grad():
        emb1 = model.forward_once(img1)
        emb2 = model.forward_once(img2)
        sim = F.cosine_similarity(emb1, emb2)
        # Backpropagate similarity to img2 inputs
        sim.backward(torch.ones_like(sim))
        grad = img2.grad.detach().squeeze(0)  # C x H x W
        sal = grad.abs().sum(dim=0)  # H x W
        sal = sal / (sal.max() + 1e-8)

    # Normalize saliency map
    sal_np = sal.cpu().numpy()
    sal_np = cv2.normalize(sal_np, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Overlay on aligned Signature 2 with proper blending
    result = overlay_heatmap(img2_aligned, sal_np, alpha=overlay_alpha, colormap='jet')
    
    return result

def generate_gradcam_heatmap(
    img_pil: Image.Image, 
    overlay_on_original: bool = True,
    overlay_alpha: float = 0.5
) -> Image.Image:
    """
    Generate Grad-CAM heatmap with proper overlay on original image.
    
    Args:
        img_pil: PIL Image (should be preprocessed/aligned)
        overlay_on_original: Overlay heatmap on the image
        overlay_alpha: Transparency of overlay (0-1)
    
    Returns:
        PIL Image with Grad-CAM overlaid
    """
    model = _ensure_model_loaded()
    model.zero_grad(set_to_none=True)
    # Access last conv block of resnet18 (same layer used for features)
    target_layer = model.backbone.layer4[-1].conv2
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations['value'] = out.detach()
    def bwd_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    handle_f = target_layer.register_forward_hook(fwd_hook)
    handle_b = target_layer.register_full_backward_hook(bwd_hook)

    try:
        x = _to_model_tensor(img_pil)
        with torch.enable_grad():
            emb = model.forward_once(x)
            # Maximize L2 norm of embedding as a generic target
            target = (emb.pow(2).sum(dim=1)).mean()
            target.backward()

        A = activations['value']  # [B,C,H,W]
        dA = gradients['value']   # [B,C,H,W]
        weights = dA.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
        cam = (weights * A).sum(dim=1, keepdim=False)  # [B,H,W]
        cam = torch.relu(cam)
        
        # Normalize properly
        cam_np = cam.squeeze(0).cpu().numpy()
        cam_np = cv2.normalize(cam_np, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        if overlay_on_original:
            # Overlay on original image
            result = overlay_heatmap(img_pil, cam_np, alpha=overlay_alpha, colormap='jet')
            return result
        else:
            # Return just the heatmap as RGBA
            cam_uint8 = (cam_np * 255).astype(np.uint8)
            h, w = cam_uint8.shape
            alpha = Image.fromarray(cam_uint8, mode='L')
            red = Image.new('L', (w, h), 255)
            rgba = Image.merge('RGBA', (red, Image.new('L',(w,h),0), Image.new('L',(w,h),0), alpha))
            return rgba
    finally:
        handle_f.remove()
        handle_b.remove()

def generate_dual_saliency_maps(
    img1_pil: Image.Image, 
    img2_pil: Image.Image, 
    enable_alignment: bool = True,
    overlay_alpha: float = 0.5
) -> Image.Image:
    """
    Generate saliency maps for both signatures side by side with proper overlays.
    Shows where each signature contributes to the similarity score.
    
    Returns:
        Side-by-side visualization with red (sig1) and green (sig2) overlays
    """
    model = _ensure_model_loaded()
    
    # Align to consistent canvas
    if enable_alignment:
        img1_aligned, img2_aligned = align_pair_to_canvas(
            img1_pil, img2_pil, canvas_size=(220, 155)
        )
    else:
        img1_aligned, img2_aligned = img1_pil, img2_pil
    
    img1 = _to_model_tensor(img1_aligned)
    img2 = _to_model_tensor(img2_aligned)
    
    # Generate saliency for img1
    img1.requires_grad_(True)
    with torch.enable_grad():
        emb1 = model.forward_once(img1)
        emb2_frozen = model.forward_once(img2).detach()
        sim1 = F.cosine_similarity(emb1, emb2_frozen)
        sim1.backward(torch.ones_like(sim1))
        grad1 = img1.grad.detach().squeeze(0)
        sal1 = grad1.abs().sum(dim=0)
        sal1 = sal1 / (sal1.max() + 1e-8)
    
    # Generate saliency for img2
    img2.requires_grad_(True)
    with torch.enable_grad():
        emb1_frozen = model.forward_once(img1).detach()
        emb2 = model.forward_once(img2)
        sim2 = F.cosine_similarity(emb1_frozen, emb2)
        sim2.backward(torch.ones_like(sim2))
        grad2 = img2.grad.detach().squeeze(0)
        sal2 = grad2.abs().sum(dim=0)
        sal2 = sal2 / (sal2.max() + 1e-8)
    
    # Normalize
    sal1_np = cv2.normalize(sal1.cpu().numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sal2_np = cv2.normalize(sal2.cpu().numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Create proper overlays using visualization module
    result = create_dual_overlay(
        img1_aligned, img2_aligned, sal1_np, sal2_np, alpha=overlay_alpha
    )
    
    return result


def generate_difference_heatmap(
    img1_pil: Image.Image, 
    img2_pil: Image.Image, 
    enable_alignment: bool = True,
    overlay_alpha: float = 0.6
) -> Tuple[Image.Image, dict]:
    """
    Compute pixel-level difference and visualize as heatmap.
    Shows where the signatures differ visually.
    
    Returns:
        (difference_overlay_image, statistics_dict)
    """
    # Align to consistent canvas
    if enable_alignment:
        img1_aligned, img2_aligned = align_pair_to_canvas(
            img1_pil, img2_pil, canvas_size=(220, 155)
        )
    else:
        img1_aligned, img2_aligned = img1_pil, img2_pil
    
    # Use visualization module's difference map function
    diff_overlay, stats = create_difference_map(img1_aligned, img2_aligned)
    
    return diff_overlay, stats


def generate_gradcam_dual(
    img1_pil: Image.Image, 
    img2_pil: Image.Image, 
    enable_alignment: bool = True,
    overlay_alpha: float = 0.5
) -> Image.Image:
    """
    Generate Grad-CAM heatmaps for both images side by side with proper overlays.
    
    Args:
        img1_pil: First signature image
        img2_pil: Second signature image
        enable_alignment: Apply preprocessing alignment before Grad-CAM
        overlay_alpha: Overlay transparency
    
    Returns:
        Side-by-side Grad-CAM visualization
    """
    # Align to consistent canvas
    if enable_alignment:
        img1_aligned, img2_aligned = align_pair_to_canvas(
            img1_pil, img2_pil, canvas_size=(220, 155)
        )
    else:
        img1_aligned, img2_aligned = img1_pil, img2_pil
    
    # Generate Grad-CAM for both
    cam1_overlay = generate_gradcam_heatmap(img1_aligned, overlay_on_original=True, overlay_alpha=overlay_alpha)
    cam2_overlay = generate_gradcam_heatmap(img2_aligned, overlay_on_original=True, overlay_alpha=overlay_alpha)
    
    # Create side-by-side combined image
    w, h = CANVAS_SIZE if enable_alignment else cam1_overlay.size
    combined = Image.new('RGB', (w * 2 + 20, h), (255, 255, 255))
    combined.paste(cam1_overlay, (0, 0))
    combined.paste(cam2_overlay, (w + 20, 0))
    return combined


def generate_saliency_difference(
    img1_pil: Image.Image, 
    img2_pil: Image.Image, 
    enable_alignment: bool = True,
    overlay_alpha: float = 0.6
) -> Image.Image:
    """
    Compute saliency maps for both images and create a difference heatmap.
    Highlights where the saliency patterns differ most.
    
    Returns:
        Overlay showing where attention patterns differ
    """
    model = _ensure_model_loaded()
    
    # Align to consistent canvas
    if enable_alignment:
        img1_aligned, img2_aligned = align_pair_to_canvas(
            img1_pil, img2_pil, canvas_size=(220, 155)
        )
    else:
        img1_aligned, img2_aligned = img1_pil, img2_pil
    
    img1 = _to_model_tensor(img1_aligned)
    img2 = _to_model_tensor(img2_aligned)
    
    # Generate saliency for img1
    img1.requires_grad_(True)
    with torch.enable_grad():
        emb1 = model.forward_once(img1)
        emb2_frozen = model.forward_once(img2).detach()
        sim1 = F.cosine_similarity(emb1, emb2_frozen)
        sim1.backward(torch.ones_like(sim1))
        grad1 = img1.grad.detach().squeeze(0)
        sal1 = grad1.abs().sum(dim=0)
        sal1 = sal1 / (sal1.max() + 1e-8)
    
    # Generate saliency for img2
    img2.requires_grad_(True)
    with torch.enable_grad():
        emb1_frozen = model.forward_once(img1).detach()
        emb2 = model.forward_once(img2)
        sim2 = F.cosine_similarity(emb1_frozen, emb2)
        sim2.backward(torch.ones_like(sim2))
        grad2 = img2.grad.detach().squeeze(0)
        sal2 = grad2.abs().sum(dim=0)
        sal2 = sal2 / (sal2.max() + 1e-8)
    
    # Compute absolute difference
    diff = torch.abs(sal1 - sal2)
    diff = diff / (diff.max() + 1e-8)
    
    # Normalize and overlay
    diff_np = cv2.normalize(diff.cpu().numpy(), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Overlay on first signature with hot colormap (yellow/red for differences)
    result = overlay_heatmap(img1_aligned, diff_np, alpha=overlay_alpha, colormap='hot')
    
    return result

if __name__ == "__main__":
    sim = compare_signatures("img1.png", "img2.png")
    print("Cosine Similarity:", sim)
    print("✅ Same person" if sim > 0.85 else "❌ Different person")
