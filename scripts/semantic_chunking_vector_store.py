import os
import faiss
import pickle
import fitz
import textwrap
from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# 1️⃣ Paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

PDF_PATH = os.path.join(PROJECT_ROOT, "books", "BD Chaurasia - Human Anatomy Volume 1.pdf")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
META_FILE  = os.path.join(DATA_DIR, "index.pkl")

# ------------------------------------------------------------
# 2️⃣ Embedding Model (LangChain-compatible)
# ------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
print(f"[INFO] Loading embedding model: {EMBED_MODEL}")

# use HuggingFaceEmbeddings instead of SentenceTransformer
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
st_model = SentenceTransformer(EMBED_MODEL)
dimension = st_model.get_sentence_embedding_dimension()

print(f"[INFO] Model loaded. Embedding dimension = {dimension}")

# ------------------------------------------------------------
# 3️⃣ PDF Extraction
# ------------------------------------------------------------
print(f"[INFO] Reading PDF: {PDF_PATH}")
doc = fitz.open(PDF_PATH)
pages = [(i + 1, p.get_text("text").strip()) for i, p in enumerate(doc) if p.get_text("text").strip()]
doc.close()
print(f"[INFO] Extracted {len(pages)} non-empty pages.")

# ------------------------------------------------------------
# 4️⃣ Semantic Chunking
# ------------------------------------------------------------
print("[STEP] Performing semantic chunking using LangChain-compatible embedding...")
chunker = SemanticChunker(
    embedding,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=0.75
)

documents, metadata = [], []
for page_num, text in tqdm(pages, desc="[CHUNKING PAGES]"):
    try:
        chunks = chunker.create_documents([text])
        for c in chunks:
            content = c.page_content if hasattr(c, "page_content") else c
            documents.append(content)
            metadata.append(f"{os.path.basename(PDF_PATH)} - page {page_num}")
    except Exception as e:
        print(f"[WARN] Skipped page {page_num} due to error: {e}")

print(f"[INFO] Raw semantic chunks created: {len(documents)}")

# ------------------------------------------------------------
# 5️⃣ Filter, Embed, and Build FAISS
# ------------------------------------------------------------
filtered_docs, filtered_meta = [], []
for d, m in zip(documents, metadata):
    txt = d.strip().lower()
    if len(txt.split()) > 15 and not txt.startswith("fig"):
        filtered_docs.append(d)
        filtered_meta.append(m)

documents, metadata = filtered_docs, filtered_meta
print(f"[INFO] ✅ Final semantic chunks: {len(documents)}")

print("[STEP] Generating embeddings for all chunks...")
embeddings = st_model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("[STEP] Building FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump({"documents": documents, "metadata": metadata}, f)

print(f"[SUCCESS] Saved {len(documents)} semantic chunks into:")
print(f"   → {INDEX_FILE}")
print(f"   → {META_FILE}")

print("\n========== SAMPLE SEMANTIC CHUNKS ==========\n")
for i in range(min(5, len(documents))):
    print(f"ID: {i} | {metadata[i]}")
    print(textwrap.fill(documents[i], width=100))
    print("-" * 100)

print(f"\n[INFO] FAISS index built with {index.ntotal} vectors.")
print("[DONE] Semantic vector store ready for use.")
