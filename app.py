from flask import Flask, request, jsonify, render_template
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("templates/product_catalog.json", "r", encoding="utf-8") as f:
    product_catalog = json.load(f)

with open("embeddings.json", "r", encoding="utf-8") as f:
    embeddings_data = json.load(f)

catalog_embeddings = np.array([v for v in embeddings_data.values()])
catalog_ids = list(embeddings_data.keys())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    query = request.json.get("query", "")
    if not query:
        return jsonify([])

    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, catalog_embeddings)[0]
    top_indices = similarities.argsort()[-5:][::-1]

    recommendations = []
    for idx in top_indices:
        item = next((i for i in product_catalog if i["name"] == catalog_ids[idx]), None)
        if item:
            recommendations.append({
                "name": item["name"],
                "url": item["url"],
                "remoteTesting": item.get("remoteTesting", "N/A"),
                "adaptiveSupport": item.get("adaptiveSupport", "N/A")
            
            })

    return jsonify(recommendations)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
