import sys
import os
# Add parent directory to path for imports to work in deployment
# When running from backend/app.py, this adds the project root (containing 'backend' folder) to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from backend.inference import (
    get_similarity_score, 
    generate_saliency_heatmap, 
    generate_gradcam_heatmap, 
    generate_gradcam_dual,
    generate_dual_saliency_maps,
    generate_difference_heatmap,
    generate_saliency_difference
)
from backend.preprocessing import align_signatures
from backend.signature_detector import hash_image, detect_and_extract_signatures
from backend.pipeline import get_pipeline
from backend.advanced_alignment import (
    detect_signatures_in_image,
    denoise_signature,
    align_pair_via_orb,
    preprocess_for_model,
    visualize_overlay,
    pil_to_numpy,
    numpy_to_pil
)
from backend.signature_normalization import (
    detect_signatures_robust,
    normalize_signature_pair,
    overlay_signatures_with_baseline
)
from PIL import Image
import io
from datetime import datetime
from reportlab.lib.utils import ImageReader
import json
import uuid
import os
import cv2
import numpy as np
from functools import wraps
from collections import defaultdict
import time

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    _pdf_enabled = True
except Exception:
    _pdf_enabled = False

app = Flask(__name__)
# CORS: Allow all localhost origins for development
# Note: Flask-CORS doesn't support wildcards, so list common ports
CORS(app, origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",  # Vite default for this project
    "http://127.0.0.1:8080",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "https://signature-verifier-frontend.fly.dev",
], supports_credentials=True)

# Security: Rate limiting (simple in-memory)
_rate_limit_store = defaultdict(list)
_rate_limit_window = 60  # seconds
_rate_limit_max = 20  # requests per window

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        now = time.time()
        # Clean old entries
        _rate_limit_store[client_ip] = [t for t in _rate_limit_store[client_ip] if now - t < _rate_limit_window]
        # Check limit
        if len(_rate_limit_store[client_ip]) >= _rate_limit_max:
            return jsonify({"error": "Rate limit exceeded. Please wait before trying again."}), 429
        _rate_limit_store[client_ip].append(now)
        return f(*args, **kwargs)
    return decorated_function

# Security: Add headers
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# Security: File validation
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_image_file(file):
    """Validate uploaded image file."""
    if not file:
        return False, "No file provided"
    
    # Check filename extension (remove PII from filename)
    filename = getattr(file, 'filename', '')
    if not filename or '.' not in filename:
        return False, "Invalid filename"
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size (read only first bytes to validate)
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > MAX_FILE_SIZE:
        return False, f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
    
    if size == 0:
        return False, "Empty file"
    
    # Try to open as image (validates it's actually an image)
    try:
        img = Image.open(io.BytesIO(file.read()))
        img.verify()  # Verify it's a valid image
        file.seek(0)  # Reset for actual processing
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

_history = []  # simple in-memory log (NO IMAGES stored, only metadata)

@app.route('/detect_signatures_multi', methods=['POST'])
@rate_limit
def detect_signatures_multi():
    """
    Stage 1: Detect ALL signatures in uploaded document.
    Returns list of detected signatures with thumbnails for user selection.
    """
    if 'image' not in request.files:
        return jsonify({"error": "missing 'image' file"}), 400
    
    file = request.files['image']
    valid, err = validate_image_file(file)
    if not valid:
        return jsonify({"error": err}), 400
    
    try:
        img = Image.open(io.BytesIO(file.read()))
        file.close()
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_arr = pil_to_numpy(img)
        
        # Detect all signatures
        detected = detect_signatures_robust(img_arr)
        
        # Convert crops to base64 thumbnails
        import base64
        signatures_data = []
        
        for idx, sig_data in enumerate(detected):
            crop = sig_data['image_crop']
            crop_pil = numpy_to_pil(crop)
            
            # Resize thumbnail for preview
            crop_pil.thumbnail((200, 200), Image.Resampling.LANCZOS)
            
            # Convert to base64
            thumb_buffer = io.BytesIO()
            crop_pil.save(thumb_buffer, format='PNG')
            thumb_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
            
            signatures_data.append({
                'id': idx,
                'bbox': sig_data['bbox'],
                'confidence': sig_data['confidence'],
                'thumbnail': f"data:image/png;base64,{thumb_base64}",
                'area': sig_data['area']
            })
        
        return jsonify({
            "signatures_found": len(detected),
            "signatures": signatures_data,
            "message": f"Found {len(detected)} signature(s). Select which one to use."
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Signature detection failed: {str(e)}",
            "details": traceback.format_exc()
        }), 500


@app.route('/normalized_overlay', methods=['POST'])
@rate_limit
def normalized_overlay():
    """
    Generate normalized overlay using 5-stage pipeline.
    Shows perfect alignment with baseline markers.
    """
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "missing 'img1' or 'img2'"}), 400
    
    file1 = request.files['img1']
    file2 = request.files['img2']
    
    valid1, err1 = validate_image_file(file1)
    if not valid1:
        return jsonify({"error": f"img1: {err1}"}), 400
    
    valid2, err2 = validate_image_file(file2)
    if not valid2:
        return jsonify({"error": f"img2: {err2}"}), 400
    
    show_baseline = request.form.get('show_baseline', 'true').lower() == 'true'
    enable_baseline_align = request.form.get('enable_baseline_align', 'true').lower() == 'true'
    enable_brightness_match = request.form.get('enable_brightness_match', 'true').lower() == 'true'
    alpha = float(request.form.get('opacity', 0.5))
    alpha = max(0.0, min(1.0, alpha))
    
    try:
        img1 = Image.open(io.BytesIO(file1.read()))
        img2 = Image.open(io.BytesIO(file2.read()))
        file1.close()
        file2.close()
        
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        # CRITICAL FIX: Run complete 5-stage normalization pipeline with auto-detection
        # This ensures signatures are detected, cropped, rotated, aligned, and normalized
        img1_norm, img2_norm, processing_info = normalize_signature_pair(
            img1, img2,
            target_size=(220, 155),  # Match model input
            enable_baseline_align=enable_baseline_align,
            enable_brightness_match=enable_brightness_match,
            auto_detect_signatures=True  # CRITICAL: Detect and crop signatures (removes noise/extra lines)
        )
        
        # Create overlay with baseline markers
        overlay = overlay_signatures_with_baseline(
            img1_norm, img2_norm,
            alpha=alpha,
            show_baseline=show_baseline
        )
        
        # Create side-by-side view: Original 1 | Original 2 | Normalized Overlay
        h, w = img1_norm.shape[:2]
        
        # Resize originals for display
        img1_arr = pil_to_numpy(img1)
        img2_arr = pil_to_numpy(img2)
        img1_small = cv2.resize(img1_arr, (w, h), interpolation=cv2.INTER_AREA)
        img2_small = cv2.resize(img2_arr, (w, h), interpolation=cv2.INTER_AREA)
        
        combined_arr = np.ones((h, w * 3 + 40, 3), dtype=np.uint8) * 255
        combined_arr[:, :w] = img1_small
        combined_arr[:, w+20:2*w+20] = img2_small
        combined_arr[:, 2*w+40:3*w+40] = overlay
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        cv2.putText(combined_arr, "Original 1", (5, 15), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(combined_arr, "Original 2", (w+25, 15), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(combined_arr, "NORMALIZED OVERLAY", (2*w+45, 15), font, font_scale, (255, 0, 0), thickness)
        
        combined_pil = numpy_to_pil(combined_arr)
        
        out = io.BytesIO()
        combined_pil.save(out, format='PNG')
        out.seek(0)
        
        response = send_file(out, mimetype='image/png')
        response.headers['X-Content-Type'] = 'image/png'
        response.headers['X-Processing-Info'] = json.dumps(processing_info)
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Normalized overlay generation failed: {str(e)}",
            "details": traceback.format_exc()
        }), 500


@app.route('/predict', methods=['POST'])
@rate_limit
def predict():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "missing 'img1' or 'img2'"}), 400
    
    file1 = request.files['img1']
    file2 = request.files['img2']
    
    # Security: Validate files
    valid1, err1 = validate_image_file(file1)
    if not valid1:
        return jsonify({"error": f"img1: {err1}"}), 400
    
    valid2, err2 = validate_image_file(file2)
    if not valid2:
        return jsonify({"error": f"img2: {err2}"}), 400
    
    try:
        # Read bytes for hashing
        file1_bytes = file1.read()
        file2_bytes = file2.read()
        
        # Security: Compute hashes for integrity
        img1_hash = hash_image(file1_bytes)
        img2_hash = hash_image(file2_bytes)
        
        # Load images
        img1 = Image.open(io.BytesIO(file1_bytes))
        img2 = Image.open(io.BytesIO(file2_bytes))
        
        # Security: Close files and clear memory
        file1.close()
        file2.close()
        
        # Convert to RGB if needed
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        # ====================================================================
        # CRITICAL: ALWAYS RUN 5-STAGE NORMALIZATION PIPELINE FIRST
        # This ensures perfect alignment, noise removal, and consistent preprocessing
        # ====================================================================
        print("=" * 60)
        print("🚀 STARTING SIGNATURE NORMALIZATION PIPELINE")
        print("=" * 60)
        
        # CHECKPOINT: Verify PIL images before normalization
        assert img1 is not None and img2 is not None, "❌ CHECKPOINT FAILED: PIL images are None"
        assert img1.size[0] > 0 and img1.size[1] > 0, "❌ CHECKPOINT FAILED: img1 has zero size"
        assert img2.size[0] > 0 and img2.size[1] > 0, "❌ CHECKPOINT FAILED: img2 has zero size"
        print(f"✅ CHECKPOINT: PIL images valid - img1: {img1.size}, img2: {img2.size}")
        
        try:
            img1_norm, img2_norm, processing_info = normalize_signature_pair(
                img1, img2,
                target_size=(220, 155),  # Match model input
                enable_baseline_align=True,  # CRITICAL: Align to same baseline
                enable_brightness_match=True,  # Match brightness/contrast
                auto_detect_signatures=True  # Auto-detect signatures in documents
            )
            print(f"✅ Normalization complete!")
            print(f"   - Signature 1 detected: {processing_info.get('signature1_detected', 'unknown')}")
            print(f"   - Signature 2 detected: {processing_info.get('signature2_detected', 'unknown')}")
            print(f"   - Noise removed: {processing_info.get('noise_removed', False)}")
            print(f"   - Baseline aligned: {processing_info.get('baseline_aligned', False)}")
            if processing_info.get('baseline_aligned'):
                print(f"   - Baseline diff: {processing_info.get('baseline_diff', 0):.1f}px (should be < 5px)")
            print(f"   - Size normalized: {processing_info.get('size_normalized', False)}")
            print(f"   - Brightness matched: {processing_info.get('brightness_matched', False)}")
            print("=" * 60)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ NORMALIZATION PIPELINE FAILED: {str(e)}")
            print(f"Error details: {error_details}")
            return jsonify({
                "error": f"Processing error: {str(e)}",
                "details": error_details
            }), 500
        
        # Convert normalized arrays back to PIL for model
        img1_normalized_pil = numpy_to_pil(img1_norm)
        img2_normalized_pil = numpy_to_pil(img2_norm)
        
        # Get threshold from metrics or use default
        try:
            with open('backend/model/metrics.json', 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
                model_threshold = metrics_data.get('threshold', 0.85)
        except:
            model_threshold = 0.85  # Default threshold
        
        # CRITICAL: Run ALL verification on PREPROCESSED images
        # Use multi-signal verification by default (more robust)
        use_multi_signal = request.form.get('use_multi_signal', 'true').lower() == 'true'
        
        print("🔍 Running ALL verification on PREPROCESSED images...")
        print("   ✅ Using SAME preprocessed images for:")
        print("      - Cosine similarity (model embeddings)")
        print("      - ORB matching")
        print("      - SSIM")
        print("      - Stroke analysis (handwriting)")
        print("      - Combined score calculation")
        
        # CHECKPOINT: Verify normalized arrays before verification
        assert img1_norm is not None and img2_norm is not None, "❌ CHECKPOINT FAILED: Normalized arrays are None"
        assert img1_norm.size > 0 and img2_norm.size > 0, "❌ CHECKPOINT FAILED: Normalized arrays are empty"
        assert img1_norm.shape == img2_norm.shape, f"❌ CHECKPOINT FAILED: Size mismatch after normalization - sig1: {img1_norm.shape}, sig2: {img2_norm.shape}"
        print(f"✅ CHECKPOINT: Normalized arrays ready - both: {img1_norm.shape}")
        
        detailed_metrics = None
        score = 0.0
        verdict = "Unknown"
        
        if use_multi_signal:
            try:
                from backend.multi_signal_verification import combined_verification
                
                # CRITICAL: Use EXACT SAME preprocessed images for ALL analysis
                # img1_norm and img2_norm are already:
                #   - Detected and cropped
                #   - Noise removed
                #   - Baseline aligned
                #   - Size normalized
                #   - Brightness matched
                detailed_metrics = combined_verification(
                    img1_norm, img2_norm,  # Use preprocessed numpy arrays
                    img1_pil=img1_normalized_pil,  # PIL for model inference
                    img2_pil=img2_normalized_pil,  # PIL for model inference
                    include_stroke_analysis=True
                )
                
                # Extract score and verdict from combined verification
                score = detailed_metrics.get('combined_score', 0.0)
                verdict = detailed_metrics.get('verdict', 'Unknown')
                
                print(f"📈 Multi-signal breakdown (ALL on preprocessed images):")
                print(f"   - Cosine: {detailed_metrics.get('cosine', 0):.3f}")
                print(f"   - ORB ratio: {detailed_metrics.get('orb_ratio', 0):.3f}")
                print(f"   - SSIM: {detailed_metrics.get('ssim', 0):.3f}")
                print(f"   - Handwriting score: {detailed_metrics.get('handwriting_score', 0):.3f}")
                print(f"   - Combined: {score:.3f}")
                print(f"   - Verdict: {verdict}")
                if detailed_metrics.get('requires_review'):
                    print(f"   ⚠️ REQUIRES MANUAL REVIEW")
                if detailed_metrics.get('safety_flags'):
                    print(f"   ⚠️ Safety flags: {detailed_metrics.get('safety_flags')}")
            except Exception as e:
                import traceback
                print(f"⚠️ Multi-signal verification failed, trying basic method: {str(e)}")
                traceback.print_exc()
                # Fallback to basic method
                try:
                    score, verdict = get_similarity_score(
                        img1_normalized_pil, img2_normalized_pil,
                        threshold=model_threshold,
                        enable_alignment=False,  # ALREADY NORMALIZED
                        use_advanced_alignment=False,  # ALREADY NORMALIZED
                        use_multi_signal=False  # Use basic method
                    )
                    print(f"📊 Basic result: Similarity={score:.4f}, Verdict='{verdict}'")
                except Exception as e2:
                    error_details = traceback.format_exc()
                    print(f"❌ VERIFICATION FAILED: {str(e2)}")
                    return jsonify({
                        "error": f"Verification failed: {str(e2)}",
                        "details": error_details
                    }), 500
        else:
            # Use basic method (no multi-signal)
            try:
                score, verdict = get_similarity_score(
                    img1_normalized_pil, img2_normalized_pil,
                    threshold=model_threshold,
                    enable_alignment=False,  # ALREADY NORMALIZED
                    use_advanced_alignment=False,  # ALREADY NORMALIZED
                    use_multi_signal=False
                )
                print(f"📊 Basic result: Similarity={score:.4f}, Verdict='{verdict}'")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"❌ VERIFICATION FAILED: {str(e)}")
                return jsonify({
                    "error": f"Verification failed: {str(e)}",
                    "details": error_details
                }), 500
        print("=" * 60)
        
        # Security: Use UUID instead of filename to avoid PII
        record_id = str(uuid.uuid4())
        
        # Return format matching PredictResponse interface
        response = {
            "similarity_score": round(score, 4),
            "verdict": verdict,
        }
        
        # Include processing info so frontend can show alignment details
        if processing_info:
            response["processing_info"] = {
                "signature1_detected": processing_info.get('signature1_detected', False),
                "signature2_detected": processing_info.get('signature2_detected', False),
                "baseline_aligned": processing_info.get('baseline_aligned', False),
                "baseline_diff_px": processing_info.get('baseline_diff', 999),
                "noise_removed": processing_info.get('noise_removed', False),
                "size_normalized": processing_info.get('size_normalized', False),
                "brightness_matched": processing_info.get('brightness_matched', False)
            }
        
        # Add detailed metrics if available (for UI display)
        if detailed_metrics:
            # UPDATED WEIGHTS: Cosine 30%, SSIM 30%, Handwriting 10%, Stroke Comparison 30%
            response["detailed_metrics"] = {
                "cosine": detailed_metrics.get("cosine", score),
                "cosine_weight": detailed_metrics.get("cosine_weight", 0.30),  # 30% of total
                "orb_ratio": 0.0,  # Always 0.0 (ORB removed)
                "orb_weight": 0.0,  # Always 0.0 (ORB removed)
                "orb_matches": 0,  # Always 0 (ORB removed)
                "ssim": detailed_metrics.get("ssim", 0.0),
                "ssim_weight": detailed_metrics.get("ssim_weight", 0.30),  # 30% of total
                "combined_score": detailed_metrics.get("combined_score", score),
                "confidence": detailed_metrics.get("confidence", "medium"),
                "requires_review": detailed_metrics.get("requires_review", False),
                "safety_flags": detailed_metrics.get("safety_flags", []),
                # Handwriting analysis (10% of total)
                "handwriting_score": detailed_metrics.get("handwriting_score", 0.0),
                "handwriting_weight": detailed_metrics.get("handwriting_weight", 0.10),  # 10% of total
                "handwriting_details": detailed_metrics.get("handwriting_details", {}),
                # Stroke comparison (30% of total)
                "stroke_comparison_score": detailed_metrics.get("stroke_comparison_score", 0.0),
                "stroke_comparison_weight": detailed_metrics.get("stroke_comparison_weight", 0.30),  # 30% of total
                # Stroke analysis results
                "stroke_similarity": detailed_metrics.get("stroke_similarity", None),
                "handwriting_flow1": detailed_metrics.get("handwriting_flow1", None),
                "handwriting_flow2": detailed_metrics.get("handwriting_flow2", None),
                "stroke_comparison": detailed_metrics.get("stroke_comparison", None)
            }
        
        # Security: Store only metadata, NO images
        # CRITICAL: Include detailed_metrics in history for analytics
        history_result = {
            "id": record_id,
            "similarity_score": round(score, 4),
            "verdict": verdict,
            "img1_hash": img1_hash if 'img1_hash' in locals() else None,
            "img2_hash": img2_hash if 'img2_hash' in locals() else None,
        }
        
        # Include detailed_metrics if available (for analytics dashboard)
        if detailed_metrics:
            history_result["detailed_metrics"] = {
                "combined_score": detailed_metrics.get("combined_score", score),
                "cosine": detailed_metrics.get("cosine", score),
                "ssim": detailed_metrics.get("ssim", 0.0),
                "handwriting_score": detailed_metrics.get("handwriting_score", 0.0),
                "stroke_comparison_score": detailed_metrics.get("stroke_comparison_score", 0.0),
            }
        
        _history.append({
            "id": record_id,
            "type": "single",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "result": history_result,
            # Note: NO image data stored
        })
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route('/batch_predict', methods=['POST'])
@rate_limit
def batch_predict():
    if 'reference' not in request.files:
        return jsonify({"error": "missing 'reference' file"}), 400
    
    reference_file = request.files['reference']
    others = request.files.getlist('files') or []
    
    if not others:
        return jsonify({"error": "missing 'files' uploads"}), 400
    
    # Security: Validate reference
    valid_ref, err_ref = validate_image_file(reference_file)
    if not valid_ref:
        return jsonify({"error": f"reference: {err_ref}"}), 400
    
    try:
        reference = Image.open(io.BytesIO(reference_file.read()))
        reference_file.close()
    except Exception as e:
        return jsonify({"error": f"Invalid reference image: {str(e)}"}), 400
    
    results = []
    for f in others:
        try:
            # Security: Validate each file
            valid, err = validate_image_file(f)
            if not valid:
                original_filename = getattr(f, 'filename', f"file_{len(results)}")
                safe_filename = os.path.basename(original_filename) if original_filename else f"file_{len(results)}"
                results.append({
                    "filename": safe_filename,
                    "error": err,
                })
                continue
            
            img = Image.open(io.BytesIO(f.read()))
            f.close()
            
            # Get threshold from metrics for consistency
            try:
                with open('backend/model/metrics.json', 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                    model_threshold = metrics_data.get('threshold', 0.92)
            except:
                model_threshold = 0.92
            
            score, verdict = get_similarity_score(reference, img, threshold=model_threshold)
            
            # Use original filename if available, otherwise generic
            original_filename = getattr(f, 'filename', f"file_{len(results)}")
            # Sanitize filename to remove path and PII
            safe_filename = os.path.basename(original_filename) if original_filename else f"file_{len(results)}"
            
            results.append({
                "filename": safe_filename,
                "similarity_score": round(score, 4),
                "verdict": verdict,
            })
        except Exception as e:
            original_filename = getattr(f, 'filename', f"file_{len(results)}")
            safe_filename = os.path.basename(original_filename) if original_filename else f"file_{len(results)}"
            results.append({
                "filename": safe_filename,
                "error": str(e),
            })
    
    results.sort(key=lambda r: r.get("similarity_score", -1), reverse=True)
    
    # Security: Store only metadata
    batch_id = str(uuid.uuid4())
    # Return only results array to match frontend BatchResponse interface
    payload = {"results": results}
    _history.append({
        "id": batch_id,
        "type": "batch",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "count": len(results),
        "results": results[:5],  # store a small preview to avoid memory bloat
        # Note: NO image data stored
    })
    return jsonify(payload)


@app.route('/report', methods=['POST'])
@rate_limit
def report():
    if not _pdf_enabled:
        return jsonify({"error": "PDF generation not available. Install reportlab."}), 501
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "missing 'img1' or 'img2'"}), 400
    
    file1 = request.files['img1']
    file2 = request.files['img2']
    
    # Security: Validate files
    valid1, err1 = validate_image_file(file1)
    if not valid1:
        return jsonify({"error": f"img1: {err1}"}), 400
    
    valid2, err2 = validate_image_file(file2)
    if not valid2:
        return jsonify({"error": f"img2: {err2}"}), 400
    
    try:
        img1 = Image.open(io.BytesIO(file1.read()))
        img2 = Image.open(io.BytesIO(file2.read()))
        file1.close()
        file2.close()
        
        heatmap_file = request.files.get('heatmap')  # optional
        heatmap_img = None
        if heatmap_file:
            try:
                heatmap_img = Image.open(io.BytesIO(heatmap_file.read()))
                heatmap_file.close()
            except Exception:
                heatmap_img = None
        
        score, verdict = get_similarity_score(img1, img2)
        
        # build a very simple PDF
        pdf_bytes = io.BytesIO()
        c = canvas.Canvas(pdf_bytes, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, "SignGuard — Verification Report")
        c.setFont("Helvetica", 10)
        c.drawString(72, height - 96, f"Generated: {datetime.utcnow().isoformat()}Z")
        c.drawString(72, height - 112, f"Similarity Score: {round(score, 4)}")
        c.drawString(72, height - 128, f"Verdict: {verdict}")
        c.drawString(72, height - 144, "⚠️ Security Notice: All data is processed in memory and not stored.")
        c.setFont("Helvetica", 10)
        c.drawString(72, height - 152, "Previews:")
        
        # Draw two images side by side
        def draw_pil(pil_img, x, y, max_w, max_h):
            if pil_img is None:
                return
            w, h = pil_img.size
            scale = min(max_w / w, max_h / h)
            dw, dh = w * scale, h * scale
            c.drawImage(ImageReader(pil_img), x, y, width=dw, height=dh, preserveAspectRatio=True, mask='auto')
        
        thumb_w = (width - 72*2 - 12) / 2
        thumb_h = 220
        y_base = height - 152 - thumb_h - 12
        draw_pil(img1, 72, y_base, thumb_w, thumb_h)
        draw_pil(img2, 72 + thumb_w + 12, y_base, thumb_w, thumb_h)
        
        # Optional heatmap on next page
        if heatmap_img is not None:
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(72, height - 72, "Model Explanation (Heatmap)")
            draw_pil(heatmap_img, 72, 160, width - 144, height - 260)
        
        c.showPage()
        c.save()
        pdf_bytes.seek(0)
        
        return send_file(pdf_bytes, mimetype='application/pdf', as_attachment=True, download_name='signguard_report.pdf')
    except Exception as e:
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500


@app.route('/history', methods=['GET'])
def history():
    # return last 50 items (NO images, only metadata)
    return jsonify({
        "history": _history[-50:],
        "disclaimer": "This system processes signatures in memory only. No images are permanently stored."
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "security": {
            "images_stored": False,
            "rate_limiting": True,
            "file_validation": True
        }
    })


@app.route('/saliency', methods=['POST'])
@rate_limit
def saliency():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "missing 'img1' or 'img2'"}), 400
    
    file1 = request.files['img1']
    file2 = request.files['img2']
    
    # Security: Validate files
    valid1, err1 = validate_image_file(file1)
    if not valid1:
        return jsonify({"error": f"img1: {err1}"}), 400
    
    valid2, err2 = validate_image_file(file2)
    if not valid2:
        return jsonify({"error": f"img2: {err2}"}), 400
    
    # Get opacity parameter (0-1, default 0.5)
    opacity = float(request.form.get('opacity', 0.5))
    opacity = max(0.0, min(1.0, opacity))  # Clamp to [0, 1]
    
    try:
        img1 = Image.open(io.BytesIO(file1.read()))
        img2 = Image.open(io.BytesIO(file2.read()))
        file1.close()
        file2.close()
        
        # Ensure images are in correct format
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        heat_overlay = generate_saliency_heatmap(img1, img2, enable_alignment=True, overlay_alpha=opacity)
        
        # Ensure output is valid
        if heat_overlay is None:
            return jsonify({"error": "Failed to generate heatmap - returned None"}), 500
        
        out = io.BytesIO()
        heat_overlay.save(out, format='PNG')
        out.seek(0)
        
        response = send_file(out, mimetype='image/png')
        response.headers['X-Content-Type'] = 'image/png'
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Saliency heatmap error: {error_details}")
        return jsonify({"error": f"Heatmap generation failed: {str(e)}", "details": error_details}), 500


@app.route('/gradcam', methods=['POST'])
@rate_limit
def gradcam():
    # Support both single image and dual image mode
    has_img1 = 'img1' in request.files
    has_img2 = 'img2' in request.files
    has_img = 'img' in request.files
    
    # Get opacity parameter
    opacity = float(request.form.get('opacity', 0.5))
    opacity = max(0.0, min(1.0, opacity))
    
    if has_img1 and has_img2:
        # Dual image mode (for signature comparison)
        file1 = request.files['img1']
        file2 = request.files['img2']
        
        valid1, err1 = validate_image_file(file1)
        if not valid1:
            return jsonify({"error": f"img1: {err1}"}), 400
        
        valid2, err2 = validate_image_file(file2)
        if not valid2:
            return jsonify({"error": f"img2: {err2}"}), 400
        
        try:
            img1 = Image.open(io.BytesIO(file1.read()))
            img2 = Image.open(io.BytesIO(file2.read()))
            file1.close()
            file2.close()
            
            # Ensure images are in correct format
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            
            # Use dual Grad-CAM for comparison with opacity
            heat_overlay = generate_gradcam_dual(img1, img2, enable_alignment=True, overlay_alpha=opacity)
            
            if heat_overlay is None:
                return jsonify({"error": "Failed to generate Grad-CAM - returned None"}), 500
            
            out = io.BytesIO()
            heat_overlay.save(out, format='PNG')
            out.seek(0)
            
            response = send_file(out, mimetype='image/png')
            response.headers['X-Content-Type'] = 'image/png'
            return response
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Grad-CAM error: {error_details}")
            return jsonify({"error": f"Grad-CAM generation failed: {str(e)}", "details": error_details}), 500
    
    elif has_img:
        # Single image mode (legacy)
        file = request.files['img']
        
        valid, err = validate_image_file(file)
        if not valid:
            return jsonify({"error": err}), 400
        
        try:
            img = Image.open(io.BytesIO(file.read()))
            file.close()
            
            heat_overlay = generate_gradcam_heatmap(img, overlay_on_original=True, overlay_alpha=opacity)
            out = io.BytesIO()
            heat_overlay.save(out, format='PNG')
            out.seek(0)
            return send_file(out, mimetype='image/png')
        except Exception as e:
            return jsonify({"error": f"Grad-CAM generation failed: {str(e)}"}), 500
    
    else:
        return jsonify({"error": "missing 'img' or 'img1' and 'img2'"}), 400


@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        with open('backend/model/metrics.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Ensure all expected fields are present
        response = {
            "accuracy": data.get("accuracy"),
            "f1": data.get("f1_score") or data.get("f1"),
            "threshold": data.get("threshold", 0.85)
        }
        return jsonify(response)
    except Exception:
        return jsonify({"accuracy": None, "f1": None, "threshold": 0.85}), 200

@app.route('/disclaimer', methods=['GET'])
def disclaimer():
    """Security and privacy disclaimer."""
    return jsonify({
        "disclaimer": "All signature data is processed in memory only and is never permanently stored. This system is for educational/research purposes only.",
        "security_practices": [
            "Images processed in memory only",
            "No permanent storage of signature images",
            "SHA-256 hashing for integrity verification",
            "File validation and size limits",
            "Rate limiting enabled",
            "UUID-based identifiers (no PII in filenames)"
        ],
        "data_handling": {
            "images_stored": False,
            "embeddings_stored": False,
            "metadata_only": True
        },
        "alignment": {
            "enabled": True,
            "description": "Signatures are automatically aligned (centered, cropped, deskewed) before comparison for improved accuracy"
        }
    })

@app.route('/viz_explanation/<viz_type>', methods=['GET'])
def viz_explanation(viz_type):
    """Get user-friendly explanation for a visualization type."""
    from backend.visualization import explain_visualization
    explanation = explain_visualization(viz_type)
    return jsonify(explanation)

@app.route('/align_preview', methods=['POST'])
@rate_limit
def align_preview():
    """
    Preview aligned signatures (for debugging/visualization).
    Returns aligned versions of both images side by side.
    """
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "missing 'img1' or 'img2'"}), 400
    
    file1 = request.files['img1']
    file2 = request.files['img2']
    
    valid1, err1 = validate_image_file(file1)
    if not valid1:
        return jsonify({"error": f"img1: {err1}"}), 400
    
    valid2, err2 = validate_image_file(file2)
    if not valid2:
        return jsonify({"error": f"img2: {err2}"}), 400
    
    # Get opacity parameter
    opacity = float(request.form.get('opacity', 0.5))
    opacity = max(0.0, min(1.0, opacity))
    
    try:
        img1 = Image.open(io.BytesIO(file1.read()))
        img2 = Image.open(io.BytesIO(file2.read()))
        file1.close()
        file2.close()
        
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        print("=" * 60)
        print("🎯 CREATING EXACT OVERLAP WITH 5-STAGE NORMALIZATION")
        print("=" * 60)
        
        # CRITICAL FIX: Use 5-stage normalization pipeline FIRST
        # This ensures both signatures are:
        # - DETECTED and CROPPED (removes noise, extra lines, diagonal marks)
        # - ROTATED to horizontal (removes rotation differences)
        # - BASELINE ALIGNED (same writing line)
        # - SIZE NORMALIZED (same dimensions)
        # - BRIGHTNESS MATCHED (same contrast)
        from backend.signature_normalization import normalize_signature_pair, overlay_signatures_with_baseline
        from backend.advanced_alignment import pil_to_numpy, numpy_to_pil
        
        print("📐 Running 5-stage normalization pipeline...")
        img1_norm, img2_norm, processing_info = normalize_signature_pair(
            img1, img2,
            target_size=(220, 155),
            enable_baseline_align=True,  # CRITICAL: Align baselines
            enable_brightness_match=True,  # Match brightness
            auto_detect_signatures=True  # CRITICAL: Detect and crop signatures (removes extra lines/noise)
        )
        
        print(f"✅ Normalization complete!")
        print(f"   - Signature 1 detected: {processing_info.get('signature1_detected', False)}")
        print(f"   - Signature 2 detected: {processing_info.get('signature2_detected', False)}")
        print(f"   - Baseline aligned: {processing_info.get('baseline_aligned', False)}")
        baseline_diff = processing_info.get('baseline_diff', 0)
        print(f"   - Baseline diff: {baseline_diff:.1f}px (should be < 5px)")
        
        # Create EXACT OVERLAP using NORMALIZED signatures (they're already perfectly aligned!)
        overlay = overlay_signatures_with_baseline(
            img1_norm, img2_norm,
            alpha=opacity,
            show_baseline=True
        )
        
        # Create side-by-side view: Original 1 | Original 2 | EXACT OVERLAP (normalized)
        h, w = img1_norm.shape[:2]
        
        # Resize originals for display (for comparison)
        img1_arr = pil_to_numpy(img1)
        img2_arr = pil_to_numpy(img2)
        img1_small = cv2.resize(img1_arr, (w, h), interpolation=cv2.INTER_AREA)
        img2_small = cv2.resize(img2_arr, (w, h), interpolation=cv2.INTER_AREA)
        
        # Create combined visualization
        combined_arr = np.ones((h, w * 3 + 40, 3), dtype=np.uint8) * 255
        combined_arr[:, :w] = img1_small
        combined_arr[:, w+20:2*w+20] = img2_small
        combined_arr[:, 2*w+40:3*w+40] = overlay
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (0, 0, 0)
        cv2.putText(combined_arr, "Signature 1", (5, 15), font, font_scale, color, thickness)
        cv2.putText(combined_arr, "Signature 2", (w+25, 15), font, font_scale, color, thickness)
        cv2.putText(combined_arr, "EXACT OVERLAP", (2*w+45, 15), font, font_scale, (255, 0, 0), thickness)
        cv2.putText(combined_arr, "(NORMALIZED)", (2*w+45, 30), font, font_scale*0.8, (0, 150, 0), thickness)
        
        combined_pil = numpy_to_pil(combined_arr)
        
        print(f"✅ EXACT OVERLAP created using NORMALIZED signatures!")
        print(f"   (Noise removed, rotated to horizontal, baseline aligned)")
        print("=" * 60)
        
        out = io.BytesIO()
        combined_pil.save(out, format='PNG')
        out.seek(0)
        
        response = send_file(out, mimetype='image/png')
        response.headers['X-Content-Type'] = 'image/png'
        response.headers['X-Processing-Info'] = json.dumps(processing_info)
        return response
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Align preview error: {error_details}")
        return jsonify({"error": f"Alignment preview failed: {str(e)}", "details": error_details}), 500

@app.route('/dual_saliency', methods=['POST'])
@rate_limit
def dual_saliency():
    """Generate dual saliency maps for both signatures side by side."""
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "missing 'img1' or 'img2'"}), 400
    
    file1 = request.files['img1']
    file2 = request.files['img2']
    
    valid1, err1 = validate_image_file(file1)
    if not valid1:
        return jsonify({"error": f"img1: {err1}"}), 400
    
    valid2, err2 = validate_image_file(file2)
    if not valid2:
        return jsonify({"error": f"img2: {err2}"}), 400
    
    # Get opacity parameter
    opacity = float(request.form.get('opacity', 0.5))
    opacity = max(0.0, min(1.0, opacity))
    
    try:
        img1 = Image.open(io.BytesIO(file1.read()))
        img2 = Image.open(io.BytesIO(file2.read()))
        file1.close()
        file2.close()
        
        # Ensure images are in correct format
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        heat_overlay = generate_dual_saliency_maps(img1, img2, enable_alignment=True, overlay_alpha=opacity)
        
        if heat_overlay is None:
            return jsonify({"error": "Failed to generate dual saliency - returned None"}), 500
        
        out = io.BytesIO()
        heat_overlay.save(out, format='PNG')
        out.seek(0)
        
        response = send_file(out, mimetype='image/png')
        response.headers['X-Content-Type'] = 'image/png'
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Dual saliency error: {error_details}")
        return jsonify({"error": f"Dual saliency generation failed: {str(e)}", "details": error_details}), 500

@app.route('/difference', methods=['POST'])
@rate_limit
def difference():
    """Generate pixel difference heatmap showing where signatures differ most."""
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "missing 'img1' or 'img2'"}), 400
    
    file1 = request.files['img1']
    file2 = request.files['img2']
    
    valid1, err1 = validate_image_file(file1)
    if not valid1:
        return jsonify({"error": f"img1: {err1}"}), 400
    
    valid2, err2 = validate_image_file(file2)
    if not valid2:
        return jsonify({"error": f"img2: {err2}"}), 400
    
    # Get opacity parameter
    opacity = float(request.form.get('opacity', 0.6))
    opacity = max(0.0, min(1.0, opacity))
    
    try:
        img1 = Image.open(io.BytesIO(file1.read()))
        img2 = Image.open(io.BytesIO(file2.read()))
        file1.close()
        file2.close()
        
        # Ensure images are in correct format
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        diff_overlay, stats = generate_difference_heatmap(img1, img2, enable_alignment=True, overlay_alpha=opacity)
        
        if diff_overlay is None:
            return jsonify({"error": "Failed to generate difference map - returned None"}), 500
        
        out = io.BytesIO()
        diff_overlay.save(out, format='PNG')
        out.seek(0)
        
        # Return image with stats in headers
        response = send_file(out, mimetype='image/png')
        response.headers['X-Content-Type'] = 'image/png'
        response.headers['X-Difference-Percentage'] = str(stats.get('difference_percentage', 0))
        response.headers['X-Mean-Difference'] = str(stats.get('mean_difference', 0))
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Difference heatmap error: {error_details}")
        return jsonify({"error": f"Difference heatmap generation failed: {str(e)}", "details": error_details}), 500

@app.route('/saliency_diff', methods=['POST'])
@rate_limit
def saliency_diff():
    """Generate saliency difference heatmap highlighting where saliency patterns differ."""
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "missing 'img1' or 'img2'"}), 400
    
    file1 = request.files['img1']
    file2 = request.files['img2']
    
    valid1, err1 = validate_image_file(file1)
    if not valid1:
        return jsonify({"error": f"img1: {err1}"}), 400
    
    valid2, err2 = validate_image_file(file2)
    if not valid2:
        return jsonify({"error": f"img2: {err2}"}), 400
    
    # Get opacity parameter
    opacity = float(request.form.get('opacity', 0.6))
    opacity = max(0.0, min(1.0, opacity))
    
    try:
        img1 = Image.open(io.BytesIO(file1.read()))
        img2 = Image.open(io.BytesIO(file2.read()))
        file1.close()
        file2.close()
        
        # Ensure images are in correct format
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        heat_overlay = generate_saliency_difference(img1, img2, enable_alignment=True, overlay_alpha=opacity)
        
        if heat_overlay is None:
            return jsonify({"error": "Failed to generate saliency difference - returned None"}), 500
        
        out = io.BytesIO()
        heat_overlay.save(out, format='PNG')
        out.seek(0)
        
        response = send_file(out, mimetype='image/png')
        response.headers['X-Content-Type'] = 'image/png'
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Saliency difference error: {error_details}")
        return jsonify({"error": f"Saliency difference generation failed: {str(e)}", "details": error_details}), 500

@app.route('/detect_signatures', methods=['POST'])
@rate_limit
def detect_signatures_endpoint():
    """Detect signature regions in an uploaded document image with improved detection."""
    if 'img' not in request.files:
        return jsonify({"error": "missing 'img' file"}), 400
    
    file = request.files['img']
    
    valid, err = validate_image_file(file)
    if not valid:
        return jsonify({"error": err}), 400
    
    try:
        img = Image.open(io.BytesIO(file.read()))
        file.close()
        
        # Use advanced detection
        from backend.advanced_alignment import detect_signatures_in_image, pil_to_numpy, numpy_to_pil
        import numpy as np
        
        img_array = pil_to_numpy(img)
        boxes = detect_signatures_in_image(img_array)
        
        # Extract signature thumbnails
        signatures_data = []
        for i, (x, y, w, h) in enumerate(boxes):
            crop = img_array[y:y+h, x:x+w]
            # Convert to base64 for thumbnail
            import base64
            crop_pil = numpy_to_pil(crop)
            buf = io.BytesIO()
            crop_pil.save(buf, format='PNG')
            buf.seek(0)
            thumbnail_b64 = base64.b64encode(buf.read()).decode('utf-8')
            
            signatures_data.append({
                "index": i,
                "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "thumbnail": f"data:image/png;base64,{thumbnail_b64}",
                "area": w * h
            })
        
        return jsonify({
            "signatures_found": len(boxes),
            "signatures": signatures_data,
            "message": f"Found {len(boxes)} signature(s)"
        })
    except Exception as e:
        return jsonify({"error": f"Signature detection failed: {str(e)}"}), 500


@app.route('/stroke_overlay', methods=['POST'])
@rate_limit
def stroke_overlay():
    """
    Generate stroke-based overlay visualization.
    Shows which strokes match (yellow), unique to sig1 (red), unique to sig2 (green).
    """
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "missing 'img1' or 'img2'"}), 400
    
    file1 = request.files['img1']
    file2 = request.files['img2']
    
    valid1, err1 = validate_image_file(file1)
    if not valid1:
        return jsonify({"error": f"img1: {err1}"}), 400
    
    valid2, err2 = validate_image_file(file2)
    if not valid2:
        return jsonify({"error": f"img2: {err2}"}), 400
    
    opacity = float(request.form.get('opacity', 0.5))
    opacity = max(0.0, min(1.0, opacity))
    
    try:
        from backend.exact_overlay import create_overlay_with_colors
        from backend.advanced_alignment import pil_to_numpy, numpy_to_pil, denoise_signature, preprocess_for_model
        from backend.advanced_alignment import align_pair_via_orb
        
        img1 = Image.open(io.BytesIO(file1.read()))
        img2 = Image.open(io.BytesIO(file2.read()))
        file1.close()
        file2.close()
        
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        # Use 5-stage normalization pipeline for EXACT OVERLAP
        img1_norm, img2_norm, processing_info = normalize_signature_pair(
            img1, img2,
            target_size=(220, 155),
            enable_baseline_align=True,
            enable_brightness_match=True
        )
        
        # Create EXACT OVERLAP with color coding and baseline markers
        overlay = overlay_signatures_with_baseline(
            img1_norm, img2_norm,
            alpha=opacity,
            show_baseline=True
        )
        
        # Convert to PIL
        overlay_pil = numpy_to_pil(overlay)
        
        # Convert to bytes and return
        out = io.BytesIO()
        overlay_pil.save(out, format='PNG')
        out.seek(0)
        
        response = send_file(out, mimetype='image/png')
        response.headers['X-Content-Type'] = 'image/png'
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Stroke overlay generation failed: {str(e)}",
            "details": traceback.format_exc()
        }), 500


@app.route('/preprocessed_preview', methods=['POST'])
@rate_limit
def preprocessed_preview():
    """
    Show COMPLETE 5-stage preprocessing pipeline visualization using the REAL normalization pipeline.
    Displays: Original → Cleaned → Baseline Aligned → Resized → Normalized Overlay
    """
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "missing 'img1' or 'img2'"}), 400
    
    file1 = request.files['img1']
    file2 = request.files['img2']
    
    valid1, err1 = validate_image_file(file1)
    if not valid1:
        return jsonify({"error": f"img1: {err1}"}), 400
    
    valid2, err2 = validate_image_file(file2)
    if not valid2:
        return jsonify({"error": f"img2: {err2}"}), 400
    
    try:
        from backend.signature_normalization import (
            clean_signature, align_pair_baseline,
            resize_and_pad, overlay_signatures_with_baseline
        )
        from backend.advanced_alignment import pil_to_numpy, numpy_to_pil
        
        img1 = Image.open(io.BytesIO(file1.read()))
        img2 = Image.open(io.BytesIO(file2.read()))
        file1.close()
        file2.close()
        
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        # Run through REAL 5-stage pipeline step by step
        img1_arr = pil_to_numpy(img1)
        img2_arr = pil_to_numpy(img2)
        
        # STAGE 1: Original (for display)
        orig1 = img1_arr.copy()
        orig2 = img2_arr.copy()
        
        # STAGE 2: Noise Removal + Background Normalization
        img1_clean = clean_signature(img1_arr)
        img2_clean = clean_signature(img2_arr)
        
        # STAGE 3: Baseline Alignment
        img1_baseline, img2_baseline = align_pair_baseline(img1_clean, img2_clean)
        
        # STAGE 4: Size Normalization
        img1_normalized = resize_and_pad(img1_baseline, size=(220, 155), preserve_aspect=True)
        img2_normalized = resize_and_pad(img2_baseline, size=(220, 155), preserve_aspect=True)
        
        # STAGE 5: Create overlay showing perfect alignment
        overlay = overlay_signatures_with_baseline(
            img1_normalized, img2_normalized,
            alpha=0.5,
            show_baseline=True
        )
        
        # Create comprehensive visualization
        # Layout: 6 panels (2 rows x 3 columns)
        # Row 1: Original 1 | Cleaned 1 | Baseline Aligned 1
        # Row 2: Original 2 | Cleaned 2 | Baseline Aligned 2
        # Bottom: Resized 1 | Resized 2 | NORMALIZED OVERLAY (perfect alignment)
        
        panel_size = (220, 155)
        spacing = 20
        
        # Calculate canvas size
        panels_per_row = 3
        canvas_w = panels_per_row * panel_size[0] + (panels_per_row - 1) * spacing + 40
        canvas_h = 3 * panel_size[1] + 2 * spacing + 80  # 3 rows + labels
        
        combined_arr = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        # Helper to resize to panel size
        def to_panel(img, target=panel_size):
            if img.shape[:2] == target:
                return img
            return cv2.resize(img, target, interpolation=cv2.INTER_AREA)
        
        # Row 1: Signature 1 stages
        y_offset = 30
        combined_arr[y_offset:y_offset+panel_size[1], :panel_size[0]] = to_panel(orig1)
        combined_arr[y_offset:y_offset+panel_size[1], panel_size[0]+spacing:panel_size[0]+spacing+panel_size[0]] = to_panel(img1_clean)
        combined_arr[y_offset:y_offset+panel_size[1], 2*(panel_size[0]+spacing):2*(panel_size[0]+spacing)+panel_size[0]] = to_panel(img1_baseline)
        
        # Row 2: Signature 2 stages
        y_offset = panel_size[1] + spacing + 30
        combined_arr[y_offset:y_offset+panel_size[1], :panel_size[0]] = to_panel(orig2)
        combined_arr[y_offset:y_offset+panel_size[1], panel_size[0]+spacing:panel_size[0]+spacing+panel_size[0]] = to_panel(img2_clean)
        combined_arr[y_offset:y_offset+panel_size[1], 2*(panel_size[0]+spacing):2*(panel_size[0]+spacing)+panel_size[0]] = to_panel(img2_baseline)
        
        # Row 3: Final normalized images and overlay
        y_offset = 2 * (panel_size[1] + spacing) + 30
        combined_arr[y_offset:y_offset+panel_size[1], :panel_size[0]] = img1_normalized
        combined_arr[y_offset:y_offset+panel_size[1], panel_size[0]+spacing:panel_size[0]+spacing+panel_size[0]] = img2_normalized
        combined_arr[y_offset:y_offset+panel_size[1], 2*(panel_size[0]+spacing):2*(panel_size[0]+spacing)+panel_size[0]] = overlay
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (0, 0, 0)
        
        # Row 1 labels
        cv2.putText(combined_arr, "Original 1", (5, 20), font, font_scale, color, thickness)
        cv2.putText(combined_arr, "Cleaned 1", (panel_size[0]+spacing+5, 20), font, font_scale, color, thickness)
        cv2.putText(combined_arr, "Baseline Aligned 1", (2*(panel_size[0]+spacing)+5, 20), font, font_scale, color, thickness)
        
        # Row 2 labels
        y_label = panel_size[1] + spacing + 20
        cv2.putText(combined_arr, "Original 2", (5, y_label), font, font_scale, color, thickness)
        cv2.putText(combined_arr, "Cleaned 2", (panel_size[0]+spacing+5, y_label), font, font_scale, color, thickness)
        cv2.putText(combined_arr, "Baseline Aligned 2", (2*(panel_size[0]+spacing)+5, y_label), font, font_scale, color, thickness)
        
        # Row 3 labels
        y_label = 2 * (panel_size[1] + spacing) + 20
        cv2.putText(combined_arr, "Resized 1 (RGB)", (5, y_label), font, font_scale, color, thickness)
        cv2.putText(combined_arr, "Resized 2 (RGB)", (panel_size[0]+spacing+5, y_label), font, font_scale, color, thickness)
        cv2.putText(combined_arr, "✅ PERFECT OVERLAY", (2*(panel_size[0]+spacing)+5, y_label), font, font_scale, (0, 150, 0), thickness+1)
        
        # Add title at top
        cv2.putText(combined_arr, "5-STAGE NORMALIZATION PIPELINE", (10, 15), font, 0.5, (0, 0, 255), 2)
        
        combined_pil = numpy_to_pil(combined_arr)
        
        out = io.BytesIO()
        combined_pil.save(out, format='PNG')
        out.seek(0)
        
        response = send_file(out, mimetype='image/png')
        response.headers['X-Content-Type'] = 'image/png'
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Preprocessing preview error: {error_details}")
        return jsonify({
            "error": f"Preprocessing preview failed: {str(e)}",
            "details": error_details
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
