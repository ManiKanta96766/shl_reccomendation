from sentence_transformers import SentenceTransformer
import json

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load assessments
with open("public/product_catalog.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Generate and save embeddings
print(data[0])
embeddings = {a["name"]: model.encode(a["description"]).tolist() for a in data}

# Save the embeddings to a JSON file
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f)

print("✅ Embeddings saved successfully!")