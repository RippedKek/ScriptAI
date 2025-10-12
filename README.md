# OCR-Based Student Assessment System

## Overview

The **OCR-Based Student Assessment System** is a complete automated solution for evaluating handwritten exam scripts. It uses state-of-the-art **vision-language models** and **natural language processing (NLP)** to extract text, identify and assess figures, and compute marks based on semantic similarity with reference answers. The system is designed to streamline the evaluation process for academic institutions.

---

## Key Features

### 1. OCR Extraction

- Uses **Qwen2-VL-2B-Instruct**, a multimodal vision-language model, to read handwritten text directly from scanned answer scripts.
- Extracts structured textual content, including question delimiters such as:
  - `Answer to the question no-1a`
  - `End of Answer-1a`
- Supports extraction of student metadata (name, ID, course info) from the title page.

### 2. Figure Extraction and Assessment

- Automatically detects and crops figures using OpenCV contour analysis.
- Each extracted figure is evaluated using the model based on the expected target (e.g., "heart", "plant cell").
- Produces a JSON file with:
  ```json
  {
    "figure_number": "1a",
    "target": "heart",
    "caption": "Labeled diagram of human heart",
    "marks": 90
  }
  ```

### 3. Text Assessment

- Evaluates answers using sentence-transformer models and linguistic analysis.
- Computes metrics:
  - **Semantic similarity** using `all-mpnet-base-v2` embeddings.
  - **TF-IDF keyword overlap** for domain relevance.
  - **Length factor** for proportional completeness.
- Produces detailed grading feedback per question with a performance table (via the `rich` console).

### 4. Batch Processing

- Accepts a **ZIP file** containing scanned pages for each student.
- Automatically identifies:
  - Title page (for metadata extraction)
  - Figure pages (for figure assessment)
  - Answer pages (for textual assessment)
- Merges all text into a single structured file for marking.

### 5. Database Integration

- Extracted student information is appended to a CSV database in `results/students.csv`.
- Marks are stored per student using their ID for easy retrieval and analysis.

---

## System Workflow

1. **Model Initialization**

   - The `model_loader.py` file loads the Qwen2-VL model and processor with GPU support and optional 4-bit quantization.

2. **OCR Extraction**

   - The `ocr_engine.py` handles text and figure extraction using the model.

3. **Figure Processing**

   - `figure_processor.py` isolates figures from images using contour detection and thresholding.

4. **Text Assessment**

   - `assessment_core.py` compares each extracted answer to reference solutions.

5. **Batch Execution**

   - The `batch_processor.py` orchestrates the pipeline for ZIP files containing all scanned pages of a script.

6. **Main Entry Point**
   - `main.py` integrates all components and runs the full process.

---

## Directory Structure

```
project_root/
│
├── assessment_core.py        # NLP-based text scoring and feedback generation
├── assessment_engine.py      # Assessment orchestration
├── assessment_initial.py     # Early scoring prototype
├── batch_processor.py        # ZIP batch processing and figure handling
├── config.py                 # Model and prompt configuration
├── cuda.py                   # CUDA device information and diagnostics
├── database.py               # CSV storage for OCR results and marks
├── figure_processor.py       # Figure segmentation using OpenCV
├── main.py                   # Primary entry point for running the system
├── model_loader.py           # Loads the Qwen2-VL model and processor
├── ocr_engine.py             # OCR logic using the multimodal model
├── utils.py                  # Helper functions (I/O and formatting)
├── requirements.txt          # Dependencies
└── output/                   # Generated outputs (texts, figures, JSON results)
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-enabled GPU (recommended)
- At least 8 GB VRAM for full model precision, or 4 GB with quantization

### Setup Steps

1. Clone the repository and navigate to the directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Verify CUDA setup:
   ```bash
   python cuda.py
   ```

---

## Usage

### Single Student Script

Run the pipeline on a ZIP file containing a student’s scanned pages:

```bash
python main.py /path/to/student_script.zip
```

### Expected ZIP Contents

```
title.png
page1.png
page2.png
figure1.png
figure2.png
```

### Output Files

| File                               | Description                      |
| ---------------------------------- | -------------------------------- |
| `output/text/answers_combined.txt` | Combined OCR text of all answers |
| `output/figures/layout/`           | Cropped figure images            |
| `output/figures/assessments/`      | JSON figure assessment results   |
| `results/students.csv`             | Student info (Name, ID, etc.)    |
| `results/marks.csv`                | Marks and grades per student     |

---

## Model and Prompts

### Model

- **Name:** `Qwen/Qwen2-VL-2B-Instruct`
- Supports both CPU and GPU execution.
- Optional **4-bit quantization** for lower memory usage.

### Prompts

Defined in `config.py`:

- `PROMPT_STUDENT_INFO`: Extracts structured student metadata.
- `PROMPT_EXTRACT_ALL`: Extracts full answer text with delimiters.
- `PROMPT_FIGURE`: Evaluates and marks figures.

---

## Extending the System

You can easily extend the system by:

- Adding new **reference answers** to `assessment_core.py`.
- Implementing **custom marking schemes** (e.g., rubric-based).
- Integrating **database storage** for long-term record management.
- Adding **metrics collection** using `metrics_logger.py` for performance analysis (optional).

---

## Limitations

- Accuracy depends on image clarity and handwriting legibility.
- Requires significant computational resources for large batches.
- Model inference speed may vary depending on hardware.

---

## License

This project is intended for academic and research use. Commercial use or redistribution without authorization is not permitted.

---

## Authors

Developed by **Tanjeeb Meheran Rohan** and **Afra Anika** under the Department of Computer Science and Engineering, **Islamic University of Technology (IUT)**.
