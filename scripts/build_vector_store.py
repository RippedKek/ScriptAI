
import os
import faiss
import pickle
import fitz             
import textwrap
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

PDF_PATH = os.path.join(PROJECT_ROOT, "books", "anatomy_v2.pdf")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
META_FILE  = os.path.join(DATA_DIR, "index.pkl")
CHUNK_SIZE = 800   # characters per chunk


EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
print(f"[INFO] Loading embedding model: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)
dimension = embedder.get_sentence_embedding_dimension()
print(f"[INFO] Model loaded. Embedding dimension = {dimension}")


print(f"[INFO] Reading PDF: {PDF_PATH}")
doc = fitz.open(PDF_PATH)

documents, metadata = [], []

for page_num in range(len(doc)):
    text = doc[page_num].get_text("text")
    if not text.strip():
        continue

    # Split long page text into chunks of ~CHUNK_SIZE characters
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    for chunk in chunks:
        documents.append(chunk)
        metadata.append(f"{os.path.basename(PDF_PATH)} - page {page_num + 1}")

doc.close()
print(f"[INFO] Total chunks created: {len(documents)}")


print("[INFO] Generating embeddings...")
embeddings = embedder.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


print("[INFO] Building FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump({"documents": documents, "metadata": metadata}, f)

print(f"[SUCCESS] Saved {len(documents)} chunks into:")
print(f"   → {INDEX_FILE}")
print(f"   → {META_FILE}")


print("\n========== SAMPLE CHUNKS ==========\n")
for i in range(min(5, len(documents))):
    print(f"ID: {i} | {metadata[i]}")
    print(textwrap.fill(documents[i], width=100))
    print("-" * 100)

print(f"\n[INFO] FAISS index built with {index.ntotal} vectors.")
print("[DONE] Vector store ready for Phase 2.")
