MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
USE_4BIT_QUANTIZATION = True  # Set to False if quantization causes issues
USE_GPU = True  # Set to False to force CPU usage

# Generation settings
DEFAULT_MAX_TOKENS = 2048
MAX_TOKENS_STUDENT_INFO = 256
MAX_TOKENS_STRUCTURED = 2048

# Prompts
PROMPT_EXTRACT_ALL = """Extract all the handwritten text from this image. 
Include everything you can read, preserving the structure and order."""

PROMPT_STUDENT_INFO = """Look at the top of this answer sheet and extract the student's name and ID/roll number.
Format your response as:
Name: [student name]
ID: [student id]"""

PROMPT_STRUCTURED = """Extract all text from this answer sheet including:
- Question numbers and delimiters (like "ANS TO QUE NO 1(a)", "END OF 1(a)")
- All handwritten answers
- Preserve the exact structure and order

Be thorough and extract everything readable."""

# File settings
DEFAULT_OUTPUT_FILE = "extracted_text.txt"
DEFAULT_BATCH_OUTPUT = "batch_results.txt"
DEFAULT_STRUCTURED_OUTPUT = "structured_text.txt"
