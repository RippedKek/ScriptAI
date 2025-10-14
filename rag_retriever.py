
import os, pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# 1️⃣ Paths & Config
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH  = os.path.join(DATA_DIR, "index.pkl")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


print("[RAG] Loading embedding model and FAISS index...")
embedder = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    data = pickle.load(f)
documents, metadata = data["documents"], data["metadata"]

print(f"[RAG] Ready. Index vectors: {index.ntotal}, Chunks loaded: {len(documents)}")


def retrieve_context(query: str, top_k: int = TOP_K):
    """Return top-k relevant chunks (text + source)."""
    q_emb = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb, dtype=np.float32), k=top_k)

    results = []
    for rank, idx in enumerate(I[0]):
        if 0 <= idx < len(documents):
            results.append({
                "rank": rank + 1,
                "text": documents[idx],
                "source": metadata[idx],
                "distance": float(D[0][rank])
            })
    return results


# if __name__ == "__main__":
#     print("\n[RAG] Running quick self-test...")
#     test_query = input("Enter a short query (e.g., 'muscle tissue anatomy'): ")
#     hits = retrieve_context(test_query, top_k=5)

#     for h in hits:
#         print(f"\n[{h['rank']}]  Source: {h['source']} (distance={h['distance']:.4f})")
#         print("-" * 90)
#         print(h["text"][:500].replace("\n", " "))
#         print("-" * 90)
#     print(f"\n[RAG] Returned {len(hits)} chunks.")
