from pathlib import Path
import faiss
import pickle
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[2]
RAG_DIR = BASE_DIR / "rag_store"

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(str(RAG_DIR / "maintenance_docs.index"))

with open(RAG_DIR / "documents.pkl", "rb") as f:
    documents = pickle.load(f)


def retrieve_docs(query: str, top_k: int = 2):
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx, distance in zip(indices[0], distances[0]):
        doc = documents[idx]

        results.append({
            "source": doc["source"],
            "text": doc["text"],
            "distance": float(distance)
        })

    return results