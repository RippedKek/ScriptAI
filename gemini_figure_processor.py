import os, re, io, json
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv() 
API_KEY = os.getenv("GEMINI_FIGURE_API_KEY")
if not API_KEY:
    raise ValueError(" GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

def clean_json_output(text: str) -> dict:
    """
    Removes Markdown-style code fences (```json ... ``` or ``` ... ```),
    then parses the remaining JSON string.
    """
    # Remove triple backticks and optional 'json' or language tag
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    
    # Parse JSON safely
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON output: {e}\nCleaned text:\n{cleaned}")

def build_prompt(question):
    return f"""
    The following image is a figure extracted from a student's answer sheet on the question "{question}". The figure may contain a figure number, caption, and handwritten content. Your task is to analyze the figure and provide a structured assessment based on the following criteria: 
    1. Figure Number: Identify the figure number if present (e.g., "1a", "2b"). If not present, state "Not provided".
    2. Caption: Extract the caption or provide a brief description of the figure content.
    3. Completeness: Evaluate if the figure is complete and includes all necessary components as per standard academic figures.
    4. Accuracy: Assess the accuracy of the content in relation to the expected figure type
    5. Presentation: Comment on the clarity and neatness of the figure.
    6. Marks: Assign marks out of 100 based on the above criteria, with 0 for completely incorrect or irrelevant figures, and up to 100 for fully correct and well-presented figures. Provide a breakdown of marks if possible.
    Give response in the following JSON format:
    {{
      "figure_number": "<extracted figure number (like 1a, 1b) or 'Not provided'>",
      "target": "<the expected figure type>",
      "caption": "<extracted caption or description>",
      "marks": <marks out of 100>
    }}
    Do not include any additional text outside the JSON format.
    """.strip()

def assess_figure_gemini(image, question):
    prompt = build_prompt(question)
    print("Assessing figure with Gemini...")

    # Convert PIL image to bytes
    img_bytes_io = io.BytesIO()
    image.save(img_bytes_io, format="PNG")
    image_bytes = img_bytes_io.getvalue()

    model = genai.GenerativeModel(MODEL_NAME)

    # Proper multimodal request (image + text)
    response = model.generate_content([
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": image_bytes}},
    ])

    print("Figure assessing complete.")
    text = response.text.strip()
    text = clean_json_output(text)
    return text