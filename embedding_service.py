from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import os

app = Flask(__name__)
model = None  # Lazy loading to save memory

def get_model():
    global model
    if model is None:
        # Use a smaller model to prevent memory overflow on Render Free tier
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    return model

def normalize(vec):
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.tolist()
    return (vec / norm).tolist()

@app.route('/embedding', methods=['POST'])
def get_embedding():
    data = request.json
    texts = data.get('texts', [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    model_instance = get_model()
    embeddings = model_instance.encode(texts)
    normalized_embeddings = [normalize(emb) for emb in embeddings]

    return jsonify({"embeddings": normalized_embeddings})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
