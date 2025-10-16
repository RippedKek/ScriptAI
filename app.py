import io
import os
import zipfile
import tempfile
import shutil
import base64
import cv2
import json
from typing import List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

# ---- Domain imports from your project ----
from assessment_core_prometheus import evaluate_text
from gemini_assessment import grade  
from model_loader import load_model
from ocr_engine import OCREngine
from format_answer import format_answer
from figure_processor import process_figure
from gemini_figure_processor import assess_figure_gemini, clean_json_output

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

# CORS 
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
    
def load_pdf_info():
    pdf_info = ""
    if os.path.exists("pdf_info.txt"):
        with open("pdf_info.txt", "r") as f:
            pdf_info = f.read().strip()
        f.close()
    return pdf_info

def _ensure_json_string(obj):
    """
    Ensure the returned object is a JSON string. If already a string, return as-is.
    If it's a dict/list, dump it to JSON. If something else, stringify.
    """
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        # fallback to str()
        return json.dumps({"raw": str(obj)}, ensure_ascii=False)

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

@app.get("/api/v1/pdf-info")
def pdf_info():
    """
    Returns info about the currently loaded PDF (if any).
    """
    pdf_name = load_pdf_info()
    if pdf_name:
        return {"status": "ok", "book": pdf_name}
    else:
        return {"status": "no_pdf", "book": ""}

@app.post("/api/v1/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    """
    Upload a textbook/reference PDF. Builds/overwrites the FAISS vector store
    used by RAG retrieval (rag_retriever.py expects files in ./data).
    """
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    pdf_name = pdf.filename.split(".")[0]
    with open("pdf_info.txt", "w") as f:
        f.write(pdf_name)
    f.close()
    pdf_bytes = await pdf.read()
    info = _build_index_from_pdf(pdf_bytes)
    return JSONResponse({"status": "ok", "message": "Vector store rebuilt.", "book": pdf_name, **info})

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

        title_pages = [p for p in all_imgs if _classify_path(p) == "title"]
        student_info = ""
        if title_pages:
            # If multiple possible title pages, take the first
            student_info = _ocr.extract_student_info(str(title_pages[0]))

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
                    # Step 1: Check if figure actually exists
                    does_figure_exist = _ocr.assess_figure(cp)
                    print(f"Figure existence check: {does_figure_exist} for {cp}")

                    # Step 2: If figure exists, run Gemini assessment
                    if does_figure_exist.strip().lower() == "yes":
                        raw_output = assess_figure_gemini(Image.open(cp), "What is a scaphoid fracture?")
                        
                        # Step 3: Clean Gemini’s JSON output
                        if isinstance(raw_output, str):
                            try:
                                assessment_json = clean_json_output(raw_output)
                            except Exception as e:
                                assessment_json = {"error": f"Invalid JSON: {e}", "raw": raw_output}
                        else:
                            assessment_json = raw_output
                        
                        figures_payload.append({
                            "image_b64": _b64_of_image(cp),
                            "assessment": _ensure_json_string(assessment_json)
                        })
                    else:
                        assessment_json = {"note": "No figure detected."}

                except Exception as e:
                    assessment_json = {"error": f"Figure assessment failed: {e}"}

        answer_pages = [p for p in all_imgs if _classify_path(p) == "answer"]
        answer_pages.sort(key=lambda x: x.name.lower())

        answer_chunks = []
        for i, ap in enumerate(answer_pages, 1):
            print(f"→ OCR answer page {i}/{len(answer_pages)}: {ap}")
            answer_chunks.append(_ocr.extract_all(str(ap)))

        raw_student_text = "\n\n".join([t for t in answer_chunks if t]).strip()

        student_text = format_answer(raw_student_text)

        print("OCR Completed. Final answer:")
        print(student_text)

        # Grade with the rubric (reference answers string with delimiters)
        pdf_name = load_pdf_info()
        marks = evaluate_text(student_text, rubric_answer, pdf_name)

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
        
@app.post("/api/v1/check-scripts-stream")
async def check_scripts_stream(
    zipfile_in: UploadFile = File(...),
    rubric_answer: str = Form(...)
):
    from fastapi.responses import StreamingResponse
    import json, cv2, os

    def _event(data: dict, event: str = "message"):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def work():
        tmpdir = _extract_zip_to_tmp(zipfile_in)
        try:
            if _ocr is None:
                yield _event({"error": "OCR model not loaded"}, "error")
                return

            all_imgs = _gather_images(tmpdir)
            if not all_imgs:
                yield _event({"error": "No images found in ZIP"}, "error")
                return

            title_pages  = [p for p in all_imgs if _classify_path(p) == "title"]
            figure_pages = [p for p in all_imgs if _classify_path(p) == "figure"]
            answer_pages = [p for p in all_imgs if _classify_path(p) == "answer"]
            answer_pages.sort(key=lambda x: x.name.lower())

            # let client know totals
            yield _event({"phase": "start", "answer_total": len(answer_pages), "figure_total": len(figure_pages)}, "meta")

            student_info = ""
            if title_pages:
                student_info = _ocr.extract_student_info(str(title_pages[0]))

            figures_payload = []
            if figure_pages:
                figures_output_dir = os.path.join(tmpdir, "figures_crops")
                os.makedirs(figures_output_dir, exist_ok=True)

                for i, fig_page in enumerate(figure_pages, 1):
                    yield _event({"phase": "figures", "processed": i, "total": len(figure_pages), "file": fig_page.name}, "progress")

                    img = cv2.imread(str(fig_page))
                    if img is None:
                        continue

                    base_out = os.path.join(figures_output_dir, fig_page.stem)
                    crop_paths = process_figure(img, base_out)  # list of saved crop image paths

                    # assess each crop
                    for cp in crop_paths:
                        try:
                            # Step 1: Check if figure actually exists
                            does_figure_exist = _ocr.assess_figure(cp)
                            print(f"Figure existence check: {does_figure_exist} for {cp}")

                            # Step 2: If figure exists, run Gemini assessment
                            if does_figure_exist.strip().lower() == "yes":
                                raw_output = assess_figure_gemini(Image.open(cp), "What is a scaphoid fracture?")
                                
                                # Step 3: Clean Gemini’s JSON output
                                if isinstance(raw_output, str):
                                    try:
                                        assessment_json = clean_json_output(raw_output)
                                    except Exception as e:
                                        assessment_json = {"error": f"Invalid JSON: {e}", "raw": raw_output}
                                else:
                                    assessment_json = raw_output
                                    
                                figures_payload.append({
                                    "image_b64": _b64_of_image(cp),
                                    "assessment": _ensure_json_string(assessment_json)
                                })
                            else:
                                assessment_json = {"note": "No figure detected."}

                        except Exception as e:
                            assessment_json = {"error": f"Figure assessment failed: {e}"}

            answer_chunks = []
            for i, ap in enumerate(answer_pages, 1):
                txt = _ocr.extract_all(str(ap))
                answer_chunks.append(txt)
                yield _event({"phase": "answers", "processed": i, "total": len(answer_pages), "file": ap.name}, "progress")

            raw_student_text = "\n\n".join([t for t in answer_chunks if t]).strip()
            student_text = format_answer(raw_student_text)
            pdf_name = load_pdf_info()
            marks = evaluate_text(student_text, rubric_answer, pdf_name)

            result = {
                "status": "ok",
                "pages_ocrd": len(all_imgs),
                "student_info": student_info,
                "marks": marks,
                "figures": figures_payload,   
            }
            yield _event(result, "result")
            yield _event({"done": True}, "end")

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(work(), headers=headers, media_type="text/event-stream")
