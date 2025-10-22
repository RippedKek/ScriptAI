# OCR-Based Student Assessment System

## Overview

The **OCR-Based Student Assessment System** automates the evaluation of handwritten exam scripts using **Vision-Language Models (VLMs)**, **Gemini-based assessment**, and **Retrieval-Augmented Generation (RAG)**.  
It performs OCR, text evaluation, and figure analysis — providing a complete AI-based grading pipeline suitable for academic institutions.

---

## Key Features

### 1. OCR Extraction (Text and Metadata)

- Uses **Qwen2-VL-2B-Instruct**, a vision-language model, to directly interpret handwritten text.
- Extracts student information (Name, ID, Course, etc.) from title pages.
- Detects question boundaries using standard markers such as:
  ```
  Answer to the question no-1a
  End of Answer-1a
  ```

### 2. Figure Segmentation and Evaluation

- Automatically detects and crops diagrams from scanned answer pages using **OpenCV**.
- Each extracted figure is evaluated using the **Qwen2-VL** and **Gemini** models.
- Outputs a structured JSON like:
  ```json
  {
    "figure_number": "1a",
    "target": "heart",
    "caption": "Labeled diagram of human heart",
    "marks": 92
  }
  ```

### 3. Textual Assessment

- Evaluates student answers using semantic similarity and keyword matching.
- Combines three factors:
  - **Semantic Similarity** (SentenceTransformer embeddings)
  - **Keyword Overlap** (TF-IDF weighting)
  - **Length Factor** (relative completeness)
- Two scoring pipelines available:
  - **Gemini 2.5 Flash (default)** — API-based evaluation.
  - **Prometheus RAG pipeline** — local grading using textbook retrieval.

### 4. Batch Processing

- Accepts single or multi-student `.zip` archives.
- Automatically categorizes:
  - **title** → student info extraction
  - **figure** → figure segmentation + assessment
  - **answer** → full OCR + grading
- Merges all text into structured `.txt` files for downstream grading.

### 5. RAG Integration (Optional)

- Builds **FAISS** vector store from textbooks or PDFs for contextual grading.
- Enables Prometheus to access relevant content during evaluation.

### 6. Backend API (FastAPI)

- Provides RESTful endpoints for OCR, grading, and vector store management.
- Includes real-time **streaming endpoints** for batch script checking.
- Can run locally or deployed as a microservice.

---

## Architecture Summary

| Component                           | Purpose                                                   |
| ----------------------------------- | --------------------------------------------------------- |
| `model_loader.py`                   | Loads Qwen2-VL model (supports GPU and quantization).     |
| `ocr_engine.py`                     | Extracts text and figures from scanned images.            |
| `figure_processor.py`               | Segments figures using OpenCV contour detection.          |
| `gemini_assessment.py`              | Grades answers with Gemini 2.5 Flash API.                 |
| `gemini_figure_processor.py`        | Grades figures via Gemini multimodal reasoning.           |
| `assessment_core.py`                | Local semantic evaluation pipeline.                       |
| `assessment_core_prometheus.py`     | RAG-enhanced Prometheus-based assessment.                 |
| `rag_retriever.py`                  | Retrieves semantically relevant content from FAISS index. |
| `build_vector_store.py`             | Builds FAISS index from textbook PDFs.                    |
| `semantic_chunking_vector_store.py` | Alternative semantic chunking index builder.              |
| `batch_processor.py`                | Handles ZIP batch processing of multiple scripts.         |
| `database.py`                       | Writes extracted student info and marks to CSV.           |
| `app.py`                            | FastAPI backend server for API usage.                     |
| `config.py`                         | Stores configuration and prompt templates.                |

---

## Directory Structure

```
project_root/
├── app.py
├── batch_processor.py
├── ocr_engine.py
├── assessment_core.py
├── assessment_core_prometheus.py
├── gemini_assessment.py
├── gemini_figure_processor.py
├── figure_processor.py
├── model_loader.py
├── database.py
├── config.py
├── rag_retriever.py
├── semantic_chunking_vector_store.py
├── build_vector_store.py
├── format_answer.py
├── utils.py
├── main.py
├── output/
│   ├── text/
│   ├── figures/
│   └── assessments/
└── results/
    ├── students.csv
    └── marks.csv
```

---

## Installation

### Requirements

- Python 3.9 or higher
- CUDA GPU (optional but recommended)
- Minimum 8 GB VRAM (4 GB possible with quantization)

### Setup

```bash
git clone <repo_url>
cd OCR-Based-Student-Assessment-System
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Check CUDA Availability

```bash
python cuda.py
```

---

## Usage

First clone the frontend from this [repo](https://github.com/RippedKek/easy-script-frontend)

### Environment Variables

You need to create a `.env` file and paste the gemini api key under the name `GEMINI_API_KEY`
We have created a second api key for figure assessment so the first one does not hit the token limit. It should be placed under the name `GEMINI_FIGURE_API_KEY`

### Single Student ZIP

```bash
python main.py /path/to/student_script.zip
```

### Multi-Student Batch

```bash
python main.py --multi parent_batch.zip
```

#### Example ZIP Structure

```
parent_batch/
├── student_1/
│   ├── title.png
│   ├── figure_1.png
│   └── page_1.png
├── student_2/
│   ├── title.png
│   ├── figure_2.png
│   └── page_1.png
```

---

## Backend API (FastAPI)

### Start Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoints

#### 1. Health Check

```
GET /health
```

Response:

```json
{ "status": "ok" }
```

#### 2. Upload PDF (Build Vector Store)

```
POST /api/v1/upload-pdf
```

Uploads a textbook PDF and rebuilds the FAISS index.

Response:

```json
{ "status": "ok", "message": "Vector store rebuilt." }
```

#### 3. Check Single Student Script

```
POST /api/v1/check-scripts
```

Accepts a `.zip` of scanned pages and rubric text, returns structured grading results.

#### 4. Streamed Script Checking (Batch Mode)

```
POST /api/v1/batch-check-scripts-stream
```

Processes multi-student ZIP archives with real-time progress events.

---

## Vector Store Building

Create a semantic knowledge base from textbooks for context-aware evaluation:

```bash
python build_vector_store.py
```

or for advanced semantic chunking:

```bash
python semantic_chunking_vector_store.py
```

---

## Assessment Pipelines

| Pipeline             | Model                 | Description                                |
| -------------------- | --------------------- | ------------------------------------------ |
| **Gemini (Default)** | gemini-2.5-flash-lite | Fast API-based grading                     |
| **Prometheus RAG**   | Local + FAISS         | Contextual grading with textbook retrieval |

---

## Limitations

- OCR accuracy varies with handwriting clarity.
- Prometheus pipeline requires GPU memory and FAISS index.
- Gemini API key required via `.env`.

---

## Authors

Developed by **Tanjeeb Meheran Rohan** and **Afra Anika**  
Department of Computer Science and Engineering  
**Islamic University of Technology (IUT)**

---
