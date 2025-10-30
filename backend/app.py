from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from backend.inference import get_similarity_score
from PIL import Image
import io
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    _pdf_enabled = True
except Exception:
    _pdf_enabled = False

app = Flask(__name__)
CORS(app)

_history = []  # simple in-memory log of recent operations

@app.route('/predict', methods=['POST'])
def predict():
    img1 = Image.open(io.BytesIO(request.files['img1'].read()))
    img2 = Image.open(io.BytesIO(request.files['img2'].read()))

    score, verdict = get_similarity_score(img1, img2)

    record = {
        "similarity_score": round(score, 4),
        "verdict": verdict
    }
    _history.append({
        "type": "single",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "result": record,
    })
    return jsonify(record)


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'reference' not in request.files:
        return jsonify({"error": "missing 'reference' file"}), 400
    reference = Image.open(io.BytesIO(request.files['reference'].read()))
    others = request.files.getlist('files') or []
    if not others:
        return jsonify({"error": "missing 'files' uploads"}), 400

    results = []
    for f in others:
        try:
            img = Image.open(io.BytesIO(f.read()))
            score, verdict = get_similarity_score(reference, img)
            results.append({
                "filename": getattr(f, 'filename', "unknown"),
                "similarity_score": round(score, 4),
                "verdict": verdict,
            })
        except Exception as e:
            results.append({
                "filename": getattr(f, 'filename', "unknown"),
                "error": str(e),
            })

    results.sort(key=lambda r: r.get("similarity_score", -1), reverse=True)

    payload = {"results": results}
    _history.append({
        "type": "batch",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "count": len(results),
        "results": results[:5],  # store a small preview to avoid memory bloat
    })
    return jsonify(payload)


@app.route('/report', methods=['POST'])
def report():
    if not _pdf_enabled:
        return jsonify({"error": "PDF generation not available. Install reportlab."}), 501
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({"error": "missing 'img1' or 'img2'"}), 400

    img1 = Image.open(io.BytesIO(request.files['img1'].read()))
    img2 = Image.open(io.BytesIO(request.files['img2'].read()))
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
    c.showPage()
    c.save()
    pdf_bytes.seek(0)

    return send_file(pdf_bytes, mimetype='application/pdf', as_attachment=True, download_name='signguard_report.pdf')


@app.route('/history', methods=['GET'])
def history():
    # return last 50 items
    return jsonify({"history": _history[-50:]})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True)
