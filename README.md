# SHL Assessment Recommendation System 🔍🧠

This project is a semantic search-based recommendation system for SHL assessments. It uses sentence embeddings and cosine similarity to recommend the most relevant assessments based on a user's query or job description.

## 🚀 Demo

- 🔗 **Live App**: [https://shl-reccomendation.onrender.com](https://shl-reccomendation.onrender.com)
- 📦 **API Endpoint**: `POST https://shl-reccomendation.onrender.com/recommend`
- 💻 **GitHub Repo**: [https://github.com/ManiKanta96766/shl_reccomendation](https://github.com/ManiKanta96766/shl_reccomendation)

---

## 🧠 How it works

1. **Product Catalog**: Contains test names, URLs, and metadata stored in a JSON file.
2. **Embeddings**: SHL test names and descriptions are embedded using the `all-MiniLM-L6-v2` model.
3. **Query Handling**: User inputs a job role or query via UI or API.
4. **Recommendation Logic**: Cosine similarity is calculated between the query and test embeddings.
5. **Top 5 Matches**: Most similar tests are returned as recommendations.

---

## 🛠 Tech Stack

- **Frontend**: HTML + Vanilla JavaScript
- **Backend**: Flask + Gunicorn
- **ML**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Deployment**: Render.com
- **Similarity**: Scikit-learn (cosine similarity)

---

## 📂 Folder Structure

. ├── app.py # Main Flask backend ├── embeddings.json # Precomputed test embeddings ├── generate_embeddings.py # Script to generate embeddings ├── templates/ │ └── index.html # UI for the recommender ├── static/ # (Optional) for CSS/JS assets ├── requirements.txt ├── product_catalog.json # SHL assessment data

---

## 📡 API Usage

### POST `/recommend`

**Request Body**
```json
{
  "query": "Frontend Developer with React experience"
}
[
  {
    "name": "SHL Cognitive Ability Test",
    "url": "https://example.com/shl-test-1",
    "description": "Measures general mental ability."
  },
  ...
]
# Clone the repo
git clone https://github.com/ManiKanta96766/shl_reccomendation.git
cd shl_reccomendation

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
