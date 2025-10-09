from model_loader import load_model
from ocr_engine import OCREngine
from batch_processor import BatchProcessor
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

def main():
    try:
        model, processor, device = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Initialize engines
    ocr_engine = OCREngine(model, processor, device)
    #batch_processor = BatchProcessor(ocr_engine)
    
    handle_single_image(ocr_engine)


def handle_single_image(ocr_engine):
    img_path = 'samples\ip3.png'
    
    if not validate_image_path(img_path):
        print("Invalid or non-existent image file")
        return
    
    try:
        print("\nProcessing...")
        text = ocr_engine.extract_text(img_path)
        
        print_subheader("EXTRACTED TEXT")
        print(text)
        
        filename = 'output.txt'
        if not filename:
            filename = DEFAULT_OUTPUT_FILE
        save_text_to_file(text, filename)

        print_subheader("ASSESSMENT RESULTS")
        run_assessment(text)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
