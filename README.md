# OCR-Based Student Assessment System

## Overview

The **OCR-Based Student Assessment System** is a complete automated framework for evaluating handwritten exam scripts using **vision-language models (VLMs)** and **NLP-based semantic grading**. It performs OCR, figure analysis, and intelligent grading by comparing responses to reference answers using similarity metrics or API-based assessment. The system is modular, GPU-optimized, and scalable for institutional use.

---

## Key Features

### 1. OCR Extraction

- Uses **Qwen2-VL-2B-Instruct**, a multimodal vision-language model for reading handwritten text directly from scanned exam pages.
- Detects structured question boundaries like:
  - `Answer to the question no-1a`
  - `End of Answer-1a`
- Extracts student metadata (Name, ID, Course, etc.) from title pages.

### 2. Figure Extraction and Assessment

- Uses **OpenCV** contour detection to isolate figures.
- Each extracted figure is analyzed using the Qwen2-VL model.
- Outputs structured JSON results such as:
  ```json
  {
    "figure_number": "1a",
    "target": "heart",
    "caption": "Labeled diagram of human heart",
    "marks": 90
  }
  ```

### 3. Text Assessment

- Textual answers are compared with reference solutions using:
  - **Sentence-transformer** embeddings (`all-mpnet-base-v2`).
  - **TF-IDF keyword overlap** for topic relevance.
  - **Length factor** for proportional completeness.
- Grading pipelines:
  - **Gemini API (default):** Lightweight cloud-based evaluation.
  - **Prometheus RAG Pipeline (optional):** Retrieval-Augmented Generation model using FAISS; disabled by default due to hardware requirements.

### 4. Batch Processing

- Accepts ZIP files of scanned scripts (single or multi-student).
- Automatically identifies:
  - Title page → metadata extraction
  - Answer pages → text assessment
  - Figure pages → diagram analysis
- Merges all OCR text into structured files for grading.

### 5. RAG Integration (Optional)

- **scripts/build_vector_database.py** builds a **FAISS vector index** from textbook PDFs for context-aware grading.
- **rag_retriever.py** retrieves semantically relevant context chunks for use in Prometheus or Gemini assessments.

### 6. Database Integration

- Extracted student information and marks are appended to:
  - `results/students.csv`
  - `results/marks.csv`

---

## System Workflow

1. **Model Loading** → `model_loader.py` initializes Qwen2-VL with GPU or 4-bit quantization.
2. **OCR & Figure Extraction** → `ocr_engine.py` handles text and diagram extraction.
3. **Figure Processing** → `figure_processor.py` segments diagrams via OpenCV.
4. **Text Assessment** → `assessment_core.py` and `assessment_core_prometheus.py` compute grades.
5. **Batch Processing** → `batch_processor.py` manages single or multi-student ZIP archives.
6. **Main Entry** → `main.py` integrates the full pipeline.
7. **RAG Context Building** → `scripts/build_vector_database.py` builds the vector database for Prometheus.

---

## Directory Structure

```
project_root/
│
├── batch_processor.py         # Multi-student ZIP handler
├── ocr_engine.py              # OCR + figure evaluation
├── assessment_core.py         # Local semantic text assessment
├── assessment_core_prometheus.py # RAG-enhanced Prometheus pipeline (optional)
├── gemini_assessment.py       # Gemini API-based grading (default)
├── figure_processor.py        # OpenCV figure segmentation
├── model_loader.py            # Model loader with GPU/quantization
├── database.py                # CSV database appender
├── main.py                    # CLI entrypoint
├── config.py                  # Configurations & prompts
├── utils.py                   # Utility and formatting helpers
├── rag_retriever.py           # Context retriever using FAISS index
├── scripts/
│   └── build_vector_database.py  # Builds FAISS index from textbooks
├── requirements.txt           # Dependencies
└── output/
    ├── text/<student_id>/answers_combined.txt
    ├── figures/layout/<student_id>/
    └── figures/assessments/<student_id>/
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-enabled GPU (recommended)
- ≥8 GB VRAM for full model precision or 4 GB with quantization

### Setup

```bash
git clone <repo_url>
cd OCR-Based-Student-Assessment-System
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Verify CUDA

```bash
python cuda.py
```

---

## Usage

### Single Student Script

```bash
python main.py /path/to/student_script.zip
```

### Multi-Student Batch

```bash
python main.py --multi parent_batch.zip
```

#### Expected ZIP Layout

```
parent_batch/
├── std_id_1/
│   ├── title.png
│   ├── figure_1.png
│   ├── text_1.png
├── std_id_2/
│   ├── title.png
│   ├── figure_2.png
│   ├── text_1.png
```

---

## Output Files

| File Path                               | Description                |
| --------------------------------------- | -------------------------- |
| `output/text/{id}/answers_combined.txt` | Combined OCR text          |
| `output/figures/layout/{id}`            | Cropped figure images      |
| `output/figures/assessments/{id}`       | Figure JSON results        |
| `results/students.csv`                  | Extracted student metadata |
| `results/marks.csv`                     | Computed marks per student |

---

## Models and Prompts

### Model

- **Name:** `Qwen/Qwen2-VL-2B-Instruct`
- Supports CPU/GPU execution with 4-bit quantization.

### Prompts (from `config.py`)

- `PROMPT_STUDENT_INFO`: Extract structured student metadata.
- `PROMPT_EXTRACT_ALL`: Extracts all answers with delimiters.
- `PROMPT_FIGURE`: Evaluates figure accuracy and assigns marks.

---

## Gemini vs Prometheus Assessment

| Feature        | Gemini API (Default)           | Prometheus RAG (Optional)        |
| -------------- | ------------------------------ | -------------------------------- |
| Model Type     | Cloud-based (Gemini 2.5 Flash) | Local FAISS + Prometheus LLM     |
| Hardware       | Works on any system            | Requires high-end GPU            |
| Contextual RAG | Supported                      | Supported                        |
| Speed          | Fast (API-driven)              | Slower (LLM inference)           |
| Accuracy       | High                           | Very High                        |
| Usage          | Default active                 | Disabled unless manually enabled |

---

## Limitations

- OCR accuracy depends on handwriting clarity.
- Prometheus RAG mode requires a high-end GPU.
- Large batch processing may need extended runtime.

---

## License

This project is for **academic and research use only**.  
Commercial redistribution or resale is prohibited.

---

## Authors

Developed by **Tanjeeb Meheran Rohan** and **Afra Anika**  
Department of Computer Science and Engineering  
**Islamic University of Technology (IUT)**
