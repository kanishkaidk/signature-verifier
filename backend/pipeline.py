"""
Modular Pipeline Controller for Signature Verification
Coordinates: Detection → Preprocessing → Verification → Visualization → Reporting
"""
from PIL import Image
import io
from typing import Dict, Any, Optional
from backend.signature_detector import (
    detect_and_extract_signatures, 
    process_document_with_detection,
    hash_image
)
from backend.preprocessing import align_signatures
from backend.inference import (
    get_similarity_score,
    generate_saliency_heatmap,
    generate_gradcam_dual,
    generate_dual_saliency_maps,
    generate_difference_heatmap,
    generate_saliency_difference
)


class VerificationPipeline:
    """
    Main pipeline controller that orchestrates the verification process.
    """
    
    def __init__(self, enable_detection: bool = True, enable_alignment: bool = True):
        self.enable_detection = enable_detection
        self.enable_alignment = enable_alignment
    
    def process(
        self,
        img1_bytes: bytes,
        img2_bytes: bytes,
        auto_detect_signatures: bool = True
    ) -> Dict[str, Any]:
        """
        Main pipeline: Detection → Alignment → Verification
        
        Args:
            img1_bytes: First image as bytes
            img2_bytes: Second image as bytes
            auto_detect_signatures: Automatically detect and crop signatures
        
        Returns:
            Dictionary with verification results and metadata
        """
        # Compute hashes for integrity
        hash1 = hash_image(img1_bytes)
        hash2 = hash_image(img2_bytes)
        
        # Load images
        img1 = Image.open(io.BytesIO(img1_bytes))
        img2 = Image.open(io.BytesIO(img2_bytes))
        
        # Step 1: Signature Detection (if enabled)
        if auto_detect_signatures and self.enable_detection:
            try:
                sigs1, bboxes1 = detect_and_extract_signatures(img1, select_largest=True)
                sigs2, bboxes2 = detect_and_extract_signatures(img2, select_largest=True)
                
                if len(sigs1) > 1 or len(sigs2) > 1:
                    # Multiple signatures detected - use largest
                    img1 = sigs1[0]
                    img2 = sigs2[0]
                elif len(sigs1) == 1 and len(sigs2) == 1:
                    img1 = sigs1[0]
                    img2 = sigs2[0]
                # else: no signature detected, use whole image
            except Exception as e:
                # Detection failed, use original images
                print(f"Signature detection failed: {e}, using original images")
        
        # Step 2: Preprocessing & Alignment
        if self.enable_alignment:
            img1_aligned, img2_aligned = align_signatures(
                img1, img2, target_size=(220, 155), enable_deskew=True, enable_padding=True
            )
        else:
            img1_aligned, img2_aligned = img1, img2
        
        # Step 3: Verification
        similarity_score, verdict = get_similarity_score(
            img1_aligned, img2_aligned, enable_alignment=False  # Already aligned
        )
        
        return {
            "similarity_score": similarity_score,
            "verdict": verdict,
            "metadata": {
                "img1_hash": hash1,
                "img2_hash": hash2,
                "signatures_detected": auto_detect_signatures and self.enable_detection,
                "alignment_applied": self.enable_alignment,
            },
            "aligned_images": {
                "img1": img1_aligned,
                "img2": img2_aligned
            }
        }
    
    def generate_visualization(
        self,
        img1: Image.Image,
        img2: Image.Image,
        viz_type: str = "saliency",
        use_aligned: bool = True
    ) -> Image.Image:
        """
        Generate visualization heatmap.
        
        Args:
            img1: First signature image
            img2: Second signature image
            viz_type: Type of visualization (saliency, gradcam, dual_saliency, difference, saliency_diff)
            use_aligned: Use aligned versions if available
        
        Returns:
            Visualization image (RGBA)
        """
        if use_aligned and self.enable_alignment:
            img1, img2 = align_signatures(img1, img2, target_size=(220, 155))
        
        if viz_type == "saliency":
            return generate_saliency_heatmap(img1, img2, enable_alignment=False)
        elif viz_type == "gradcam":
            return generate_gradcam_dual(img1, img2, enable_alignment=False)
        elif viz_type == "dual_saliency":
            return generate_dual_saliency_maps(img1, img2, enable_alignment=False)
        elif viz_type == "difference":
            return generate_difference_heatmap(img1, img2, enable_alignment=False)
        elif viz_type == "saliency_diff":
            return generate_saliency_difference(img1, img2, enable_alignment=False)
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")


# Global pipeline instance
_pipeline = VerificationPipeline(enable_detection=True, enable_alignment=True)


def get_pipeline() -> VerificationPipeline:
    """Get the global pipeline instance."""
    return _pipeline

