from flask import Flask, request, jsonify
from inference import get_similarity_score
from PIL import Image
import io

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    img1 = Image.open(io.BytesIO(request.files['img1'].read())).convert('RGB')
    img2 = Image.open(io.BytesIO(request.files['img2'].read())).convert('RGB')

    score, verdict = get_similarity_score(img1, img2)

    return jsonify({
        "similarity_score": round(score, 4),
        "verdict": verdict
    })

if __name__ == '__main__':
    app.run(debug=True)
