from pathlib import Path
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = BASE_DIR / "data" / "docs"
RAG_DIR = BASE_DIR / "rag_store"

RAG_DIR.mkdir(parents=True, exist_ok=True)


def load_documents():
    documents = []

    for file_path in DOCS_DIR.glob("*.txt"):
        text = file_path.read_text()

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def build_vector_store():
    print("Loading documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, convert_to_numpy=True)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, str(RAG_DIR / "maintenance_docs.index"))

    with open(RAG_DIR / "documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    print("Vector store saved.")


if __name__ == "__main__":
    build_vector_store()