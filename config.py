MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
USE_4BIT_QUANTIZATION = True  # Set to False if quantization causes issues
USE_GPU = True  # Set to False to force CPU usage

# Generation settings
DEFAULT_MAX_TOKENS = 2048
MAX_TOKENS_STUDENT_INFO = 256
MAX_TOKENS_STRUCTURED = 2048

# Prompts
PROMPT_EXTRACT_ALL = """Extract all handwritten text from this image.
Preserve the exact structure and order, including question delimiters.
If visible, always include markers in this exact form:
"Answer to the question no-<id>" and "End of Answer-<id>" where <id> is like 1a, 1b, 2c, etc.
Do not invent markers if they are not present."""

PROMPT_STUDENT_INFO = """Extract student's informations and format it like:
Name: [student name]
ID: [student id]
Course: [course name]
Course Code: [course code]
Session: [academic session]
Semester: [semester number only i.e., 1, 2, 3, etc.]
Exam: [exam name]
Date: [exam date in format dd-mm-yyyy]
Section: [section]
Department: [department name abbreviation]
Institution: [institution name]"""

PROMPT_STRUCTURED = """Extract all text from this answer sheet including:
- Question numbers and delimiters (like "Answer to the question no-1a" and "End of Answer-1a")
- All handwritten answers
- Preserve the exact structure and order
Be thorough and extract everything readable."""

# File settings
DEFAULT_OUTPUT_FILE = "extracted_text.txt"
DEFAULT_BATCH_OUTPUT = "batch_results.txt"
DEFAULT_STRUCTURED_OUTPUT = "structured_text.txt"
