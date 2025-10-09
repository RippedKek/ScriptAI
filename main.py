import sys
import re
from model_loader import load_model
from ocr_engine import OCREngine
from batch_processor import BatchProcessor
from database import append_ocr_to_csv, append_marks_to_csv
from utils import (
    validate_image_path,
    save_text_to_file,
    load_image_list_from_file,
    print_header,
    print_subheader
)
from config import (
    DEFAULT_OUTPUT_FILE,
    DEFAULT_BATCH_OUTPUT,
    DEFAULT_STRUCTURED_OUTPUT
)

from assessment_engine import run_assessment

def parse_student_id(text: str) -> str:
    # Capture alphanumeric student IDs
    m = re.search(r"\bID\s*:\s*([A-Za-z0-9\-]+)", text, re.IGNORECASE)
    return m.group(1) if m else ""

def handle_zip(ocr_engine: OCREngine, zip_path: str):
    print_header("BATCH ZIP PROCESSING")
    batch = BatchProcessor(ocr_engine)
    result = batch.process_zip(zip_path)

    # Persist raw OCR of title page
    print_subheader("STUDENT INFO (TITLE PAGE)")
    print(result["title_text"])
    append_ocr_to_csv(result["title_text"])

    # Combine all answer pages into one blob (in filename order)
    combined_answers = "\n\n".join(
        result["pages_texts"][k] for k in sorted(result["pages_texts"].keys())
    )
    save_text_to_file(combined_answers, "answers_combined.txt")

    # Run assessment to get per-question marks
    print_subheader("ASSESSMENT RESULTS")
    marks = run_assessment(combined_answers) or {}

    # Save marks.csv uniquely by student ID
    sid = parse_student_id(result["title_text"])
    if not sid:
        print("Could not parse Student ID from title page. Row will be skipped.")
    else:
        append_marks_to_csv(sid, marks)

    print("\nBatch processing complete.")

def main():
    try:
        model, processor, device = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Initialize engines
    ocr_engine = OCREngine(model, processor, device)

    # CL arg: path to .zip
    if len(sys.argv) >= 2 and sys.argv[1].lower().endswith(".zip"):
        handle_zip(ocr_engine, sys.argv[1])
    else:
        print("Please run: python main.py <path_to_zip>")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
