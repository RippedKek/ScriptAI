MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
USE_4BIT_QUANTIZATION = True  # Set to False if quantization causes issues
USE_GPU = True  # Set to False to force CPU usage

# Generation settings
DEFAULT_MAX_TOKENS = 2048
MAX_TOKENS_STUDENT_INFO = 256
MAX_TOKENS_STRUCTURED = 2048

# Prompts
PROMPT_EXTRACT_ALL = """Extract all handwritten text from this image.
Preserve the exact structure and order and keep everything unchanged.
Just clone what's written in the copy at 100 percent accuracy. 
Be thorough and extract everything readable. """

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

PROMPT_FIGURE = """
Extract the figure number for this figure located at the top left corner of the image.
The figure number is in the format "Figure of 1a", "Figure of 2b", "Figure of 3a", etc. Extract only the part after "Figure of " like "1a", "2b", "3a", etc. Also return the figure caption if visible. Target is the expected figure type.
Target will be provided as additional context at the end of this prompt.
If the caption is not visible, return a short description of the figure as caption. Based on the caption or description,
mark the figure out of 100. For marking, the following must be considered:
- If the figure is completely blank or irrelevant, mark it 0.
- If the figure has some relevant content with the target but is largely incomplete, mark it 25.
- If the figure is partially complete but missing key elements from the target, mark it 50.
- If the figure is mostly complete but has minor errors, mark it 75.
- If the figure is fully complete and correct, mark it 90.
- Add 0-10 bonus marks for exceptional quality or detail, making the maximum possible marks 100.
Return the output in the following JSON format:
{
  "figure_number": "<extracted figure number like 1a, 2b, etc.>",
  "target": <the target name provided>,
  "caption": "<extracted caption or description>",
  "marks": <marks out of 100>
}
Target: 
"""

# File settings
DEFAULT_OUTPUT_FILE = "extracted_text.txt"
DEFAULT_BATCH_OUTPUT = "batch_results.txt"
DEFAULT_STRUCTURED_OUTPUT = "structured_text.txt"
