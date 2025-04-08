from flask import Flask, request, jsonify, render_template
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load SHL assessment catalog
with open("individual_test_solutions_2.json", "r", encoding="utf-8") as f:
    product_catalog = json.load(f)

# Load precomputed embeddings
with open("embeddings_2.json", "r", encoding="utf-8") as f:
    embeddings_data = json.load(f)

catalog_embeddings = np.array([v for v in embeddings_data.values()])
catalog_ids = list(embeddings_data.keys())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/recommend", methods=["POST"])
def recommend():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"recommended_assessments": []})

    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, catalog_embeddings)[0]
    top_indices = similarities.argsort()[-10:][::-1]  # Get top 10 matches

    recommendations = []
    for idx in top_indices:
        item = next((i for i in product_catalog if i["name"] == catalog_ids[idx]), None)
        if item:
            recommendations.append({
                "url": item.get("url"),
                "adaptive_support": item.get("adaptiveSupport", "No"),
                "description": item.get("description", "No description available"),
                "duration": item.get("duration", 0),
                "remote_support": item.get("remoteTesting", "No"),
                "test_type": item.get("testTypes", [])
            })

    return jsonify({"recommended_assessments": recommendations})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
