# OCR Based Student Assessment System

End-to-end pipeline to extract handwritten text from scanned exam scripts and auto-assess student answers. Built on **Qwen2-VL-2B-Instruct** for OCR-style extraction and **Sentence Transformers + spaCy + TF-IDF** for scoring.

---

## Features

- **Vision-LLM OCR**: Extracts student info and full answers using prompt-guided decoding (`ocr_engine.py`, `config.py`).
- **Auto-Assessment**: Semantic similarity, weighted keyword overlap, and length factor with readable feedback (`assessment_core.py`).
- **Batch Processing**: Run OCR across folders and save a consolidated report (`batch_processor.py`).
- **CSV Logging**: Parse key–value pairs and append to `results/students.csv` (`database.py`).
- **GPU-Aware**: Optional 4-bit quantization and CUDA checks (`model_loader.py`, `cuda.py`).

---

## Quick Start

### 1) Environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
pip install sentence-transformers spacy scikit-learn rich
python -m spacy download en_core_web_sm
```

### 3) Prepare folders

```bash
mkdir -p samples results
# Put your scanned image(s) in ./samples
```

### 4) Configure (optional)

Edit `config.py`:

```
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
USE_4BIT_QUANTIZATION = True|False
USE_GPU = True|False
```

Adjust prompts and output filenames as needed.

### 5) CUDA sanity check (optional)

```bash
python cuda.py
```

## Usage

1. For using the system, you must create and put all the scanned images of a script in a zip. Note that
   the **title** page for a script must be renamed `title.png/jpg/jpeg`. The system will automatically
   detect and extract student information from it.

2. When the zip is ready, run this in the terminall from the root folder

```bash
python main.py <path to zip>
```

3. Two CSVs will be generated, one `marks.csv` and another `students.csv` in **results** folder.
4. The extracted output from the scripts will also be saved in `output.txt`
5. Make sure you have updated `rubric.txt` with the standard answers

## Project Structure

```
.
├── main.py                   # Simple single-image runner (edit the img_path here)
├── model_loader.py           # Loads Qwen2-VL with optional 4-bit quantization
├── ocr_engine.py             # OCR logic (student info / structured extraction)
├── batch_processor.py        # Batch directory/image list processing
├── assessment_engine.py      # Wrapper to run assessments
├── assessment_core.py        # Scoring (semantic, TF-IDF, length) + feedback
├── assessment_initial.py     # Minimal demo of the scoring pipeline
├── database.py               # Append parsed fields to results/students.csv
├── utils.py                  # Path validation, file IO, printing
├── cuda.py                   # CUDA capability check
├── config.py                 # Model flags, prompts, output names
├── requirements.txt
├── samples/                  # Place input images here
└── results/                  # CSV and batch outputs
```

## Scoring Breakdown (high level)

Semantic Similarity (SBERT): all-mpnet-base-v2 / all-MiniLM-L6-v2
Weighted Keyword Overlap: TF-IDF on reference vs student answer
Length Factor: Proportionality to reference
Feedback: Highlights missing key nouns/verbs using spaCy
Grades are assigned from the final weighted score.

## Acknowledgments

Qwen2-VL-2B-Instruct (image-to-text)
sentence-transformers, spaCy, scikit-learn, rich
