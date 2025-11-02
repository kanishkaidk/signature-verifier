"""
Signature preprocessing for alignment and normalization.
Improves verification accuracy by centering, cropping, and deskewing signatures.
"""
import cv2
import numpy as np
from PIL import Image
import io


def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format."""
    if img_pil.mode == 'RGBA':
        # Remove alpha channel
        img_pil = img_pil.convert('RGB')
    img_array = np.array(img_pil)
    # PIL is RGB, OpenCV is BGR
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    """Convert OpenCV format to PIL Image."""
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def preprocess_signature(
    img_pil: Image.Image,
    target_size: tuple = (220, 155),
    enable_deskew: bool = True,
    enable_padding: bool = True,
    padding_ratio: float = 0.1
) -> Image.Image:
    """
    Preprocess signature image for better alignment:
    1. Convert to grayscale and binarize
    2. Find bounding box and crop to signature region
    3. Optional: Deskew (rotate to horizontal)
    4. Resize to target dimensions
    5. Optional: Add padding for centering
    6. Convert back to RGB
    
    Args:
        img_pil: PIL Image (any format)
        target_size: (width, height) for final image
        enable_deskew: Apply deskewing to normalize rotation
        enable_padding: Add padding to center signature
        padding_ratio: Ratio of padding to add (0.1 = 10% margin)
    
    Returns:
        Preprocessed PIL Image (RGB)
    """
    try:
        # Convert to OpenCV format
        img_cv2 = pil_to_cv2(img_pil)
        original_shape = img_cv2.shape
        
        # Convert to grayscale
        if len(img_cv2.shape) == 3:
            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv2
        
        # Apply adaptive thresholding for better binarization
        # Use OTSU if image has good contrast, otherwise adaptive
        if len(np.unique(gray)) > 2:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
        
        # Find bounding box of signature
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours found, return resized original
            resized = cv2.resize(gray, target_size)
            return Image.fromarray(resized).convert('RGB')
        
        # Get bounding box from all contours
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Add small margin to bounding box
        margin = max(5, int(min(w, h) * 0.05))
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray.shape[1] - x, w + 2 * margin)
        h = min(gray.shape[0] - y, h + 2 * margin)
        
        # Crop to signature region
        cropped = binary[y:y+h, x:x+w]
        
        # Optional: Deskew (rotate to make baseline horizontal)
        if enable_deskew:
            angle = _estimate_skew_angle(cropped)
            if abs(angle) > 0.5:  # Only deskew if angle > 0.5 degrees
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                # Calculate new bounding box after rotation
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
                M[0, 2] += (nW / 2) - (w / 2)
                M[1, 2] += (nH / 2) - (h / 2)
                cropped = cv2.warpAffine(cropped, M, (nW, nH), flags=cv2.INTER_CUBIC, 
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Resize to target dimensions while maintaining aspect ratio
        h_crop, w_crop = cropped.shape
        aspect_ratio = w_crop / h_crop
        target_w, target_h = target_size
        
        if enable_padding:
            # Resize with padding to maintain aspect ratio
            if aspect_ratio > (target_w / target_h):
                # Width is limiting factor
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            else:
                # Height is limiting factor
                new_h = target_h
                new_w = int(target_h * aspect_ratio)
            
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Create padded image (centered)
            padded = np.zeros((target_h, target_w), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            final = padded
        else:
            # Direct resize without padding
            final = cv2.resize(cropped, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Convert back to PIL RGB (invert back to normal signature appearance)
        final_inverted = cv2.bitwise_not(final)
        return Image.fromarray(final_inverted).convert('RGB')
        
    except Exception as e:
        # If preprocessing fails, return resized original
        print(f"Preprocessing error: {e}, returning resized original")
        img_rgb = img_pil.convert('RGB')
        img_array = np.array(img_rgb)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        resized = cv2.resize(gray, target_size)
        return Image.fromarray(resized).convert('RGB')


def _estimate_skew_angle(binary_img: np.ndarray) -> float:
    """
    Estimate skew angle using Hough Line Transform.
    Returns angle in degrees.
    """
    try:
        # Detect lines using Hough Transform
        lines = cv2.HoughLines(binary_img, 1, np.pi/180, threshold=100)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        angles = []
        for line in lines[:20]:  # Use first 20 lines
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            # Filter out vertical lines (angle close to 90/-90)
            if abs(angle) < 85:
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Return median angle (more robust than mean)
        median_angle = np.median(angles)
        return float(median_angle)
        
    except Exception:
        return 0.0


def align_signatures(
    img1_pil: Image.Image,
    img2_pil: Image.Image,
    target_size: tuple = (220, 155),
    enable_deskew: bool = True,
    enable_padding: bool = True,
    use_canvas: bool = True
) -> tuple:
    """
    Align two signatures for better comparison.
    Ensures both are processed with the same parameters.
    
    Args:
        img1_pil: First signature image
        img2_pil: Second signature image
        target_size: Target size (width, height)
        enable_deskew: Apply deskewing
        enable_padding: Add padding
        use_canvas: If True, normalize both to same canvas size for perfect alignment
    
    Returns:
        Tuple of (aligned_img1, aligned_img2)
    """
    if use_canvas:
        # Use consistent canvas normalization for perfect alignment
        from backend.visualization import align_pair_to_canvas
        return align_pair_to_canvas(img1_pil, img2_pil, canvas_size=target_size)
    else:
        # Original preprocessing (may have slight size differences)
        img1_aligned = preprocess_signature(
            img1_pil, target_size, enable_deskew, enable_padding
        )
        img2_aligned = preprocess_signature(
            img2_pil, target_size, enable_deskew, enable_padding
        )
        return img1_aligned, img2_aligned

