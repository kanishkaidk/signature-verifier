"""
Endpoint for creating stroke-based overlay visualizations.
Shows which strokes match, which are unique to each signature.
"""
from flask import request, jsonify, send_file
from backend.advanced_alignment import pil_to_numpy, numpy_to_pil
from backend.stroke_analysis import create_stroke_overlay
from PIL import Image
import io
import cv2

def create_stroke_overlay_endpoint(app):
    """
    Add stroke overlay endpoint to Flask app.
    Shows stroke alignment with color coding:
    - Red: Unique to signature 1
    - Green: Unique to signature 2
    - Yellow: Overlapping/matched strokes
    """
    
    @app.route('/stroke_overlay', methods=['POST'])
    def stroke_overlay():
        """
        Generate stroke-based overlay visualization.
        
        Form data:
            - img1: First signature image
            - img2: Second signature image (should be aligned)
            - opacity: Overlay opacity (0-1, default 0.5)
        
        Returns:
            PNG image showing stroke overlay
        """
        if 'img1' not in request.files or 'img2' not in request.files:
            return jsonify({"error": "missing 'img1' or 'img2'"}), 400
        
        file1 = request.files['img1']
        file2 = request.files['img2']
        
        opacity = float(request.form.get('opacity', 0.5))
        opacity = max(0.0, min(1.0, opacity))
        
        try:
            img1 = Image.open(io.BytesIO(file1.read()))
            img2 = Image.open(io.BytesIO(file2.read()))
            file1.close()
            file2.close()
            
            # Ensure RGB
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            
            # Convert to numpy
            img1_arr = pil_to_numpy(img1)
            img2_arr = pil_to_numpy(img2)
            
            # Create stroke overlay
            overlay = create_stroke_overlay(img1_arr, img2_arr, alpha=opacity)
            
            # Convert back to PIL and return
            overlay_pil = numpy_to_pil(overlay)
            out = io.BytesIO()
            overlay_pil.save(out, format='PNG')
            out.seek(0)
            
            return send_file(out, mimetype='image/png')
            
        except Exception as e:
            import traceback
            return jsonify({
                "error": f"Stroke overlay generation failed: {str(e)}",
                "details": traceback.format_exc()
            }), 500

