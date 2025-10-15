import io
import os
import zipfile
import tempfile
import shutil
import base64
import cv2
from typing import List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# ---- Domain imports from your project ----
from assessment_core_prometheus import evaluate_text
from gemini_assessment import grade  
from model_loader import load_model
from ocr_engine import OCREngine
from format_answer import format_answer
from figure_processor import process_figure

# ---- RAG index utilities (inline, adapted from build_vector_store.py) ----
import fitz
import pickle
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
META_FILE  = os.path.join(DATA_DIR, "index.pkl")

# ------------- FastAPI app -------------
app = FastAPI(title="OCR-Based Student Assessment API", version="1.0.0")

# CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load VLM once (Qwen2-VL) to reuse across requests
try:
    _model, _processor, _device = load_model()
    _ocr = OCREngine(_model, _processor, _device)
except Exception as e:
    # You can still use the /upload-pdf endpoint without OCR
    _ocr = None

# ---------- Models ----------
class Health(BaseModel):
    status: str = "ok"

# ---------- Utils ----------
def _b64_of_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def _classify_path(p: Path) -> str:
    """Return 'title' | 'figure' | 'answer' based on filename."""
    name = p.name.lower()
    if "title" in name:
        return "title"
    if name.startswith("figure"):
        return "figure"
    return "answer"

def _gather_images(root: str):
    # image files under extracted ZIP
    paths = []
    for r, _, files in os.walk(root):
        for fn in files:
            fp = os.path.join(r, fn)
            if _is_image(fp):
                paths.append(Path(fp))
    # stable order
    paths.sort(key=lambda x: x.name.lower())
    return paths

def _is_image(name: str) -> bool:
    name = name.lower()
    return name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"))

def _extract_zip_to_tmp(upload: UploadFile) -> str:
    try:
        raw = upload.file.read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ZIP file.")
    tmpdir = tempfile.mkdtemp(prefix="scripts_")
    try:
        zf.extractall(tmpdir)
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Failed to extract ZIP.")
    return tmpdir

def _ocr_folder_concat_text(folder: str) -> str:
    if _ocr is None:
        raise HTTPException(status_code=500, detail="OCR model not loaded on server.")
    # Gather images in a stable, human-friendly order
    all_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            if _is_image(p):
                all_paths.append(p)
    if not all_paths:
        raise HTTPException(status_code=400, detail="No images found in ZIP.")
    # Sort paths by name
    all_paths.sort()
    parts = []
    for p in all_paths:
        print(f"OCR processing: {p}")
        txt = _ocr.extract_all(p)
        parts.append(txt)
    return "\n\n".join(parts).strip()

def _build_index_from_pdf(pdf_bytes: bytes, chunk_size: int = 800) -> Dict[str, Any]:
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(pdf_bytes)
        tmp_pdf_path = tf.name

    try:
        doc = fitz.open(tmp_pdf_path)
        documents, metadata = [], []
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            if not text.strip():
                continue
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            for chunk in chunks:
                documents.append(chunk)
                metadata.append(f"{os.path.basename(tmp_pdf_path)} - page {page_num + 1}")
        doc.close()
        if not documents:
            raise HTTPException(status_code=400, detail="PDF has no extractable text.")

        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedder.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump({"documents": documents, "metadata": metadata}, f)

        return {"chunks": len(documents)}
    finally:
        try:
            os.remove(tmp_pdf_path)
        except Exception:
            pass

# Routes
@app.get("/health", response_model=Health)
def health():
    return Health()

@app.post("/api/v1/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    """
    Upload a textbook/reference PDF. Builds/overwrites the FAISS vector store
    used by RAG retrieval (rag_retriever.py expects files in ./data).
    """
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    pdf_bytes = await pdf.read()
    info = _build_index_from_pdf(pdf_bytes)
    return JSONResponse({"status": "ok", "message": "Vector store rebuilt.", **info})

@app.post("/api/v1/check-scripts")
async def check_scripts(
    zipfile_in: UploadFile = File(...),
    rubric_answer: str = Form(...)
):
    """
    Accepts a ZIP of page images and a rubric answer text (with delimiters).
    - Title page -> extract_student_info
    - Figures pages (name starts with 'figures') -> crop with process_figure, assess each crop
    - Remaining pages -> OCR (extract_all), concatenate, format_answer() to repair delimiters
    Returns JSON: { student_info, marks, figures }
    """
    tmpdir = _extract_zip_to_tmp(zipfile_in)
    try:
        if _ocr is None:
            raise HTTPException(status_code=500, detail="OCR model not loaded on server.")

        all_imgs = _gather_images(tmpdir)
        if not all_imgs:
            raise HTTPException(status_code=400, detail="No images found in ZIP.")

        # ---- 1) TITLE → student info ----
        title_pages = [p for p in all_imgs if _classify_path(p) == "title"]
        student_info = ""
        if title_pages:
            # If multiple possible title pages, take the first
            student_info = _ocr.extract_student_info(str(title_pages[0]))

        # ---- 2) FIGURES → crop + assess ----
        figure_pages = [p for p in all_imgs if _classify_path(p) == "figure"]
        figures_payload = []
        figures_output_dir = os.path.join(tmpdir, "figures_crops")
        os.makedirs(figures_output_dir, exist_ok=True)

        for fig_page in figure_pages:
            # crop figures from the page
            img = cv2.imread(str(fig_page))
            if img is None:
                continue
            base_out = os.path.join(figures_output_dir, fig_page.stem)
            crop_paths = process_figure(img, base_out)  # returns list of saved crop paths

            # assess each cropped figure with the VLM's figure prompt
            for cp in crop_paths:
                try:
                    assessment_json = _ocr.assess_figure(cp)  # returns JSON string per your ocr_engine.py
                except Exception as e:
                    assessment_json = f'{{"error": "figure assessment failed: {e}"}}'

                figures_payload.append({
                    "image_b64": _b64_of_image(cp),
                    "assessment": assessment_json
                })

        # ---- 3) ANSWERS → OCR -> format_answer -> evaluate_text ----
        answer_pages = [p for p in all_imgs if _classify_path(p) == "answer"]
        answer_pages.sort(key=lambda x: x.name.lower())

        answer_chunks = []
        for i, ap in enumerate(answer_pages, 1):
            print(f"→ OCR answer page {i}/{len(answer_pages)}: {ap}")
            answer_chunks.append(_ocr.extract_all(str(ap)))

        raw_student_text = "\n\n".join([t for t in answer_chunks if t]).strip()

        # Your format_answer module (as you wrote it) takes just the text
        # and re-inserts missing delimiters using Gemini; keep it as-is:
        student_text = format_answer(raw_student_text)

        print("OCR Completed. Final answer:")
        print(student_text)

        # Grade with the rubric (reference answers string with delimiters)
        marks = evaluate_text(student_text, rubric_answer)

        # ---- Build final response ----
        pages_count = len(all_imgs)
        return JSONResponse({
            "status": "ok",
            "pages_ocrd": pages_count,
            "student_info": student_info,
            "marks": marks,
            "figures": figures_payload
        })

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)