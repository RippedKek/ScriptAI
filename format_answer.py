import os, re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() 
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(" GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

def build_prompt(answer):
    return f"""
    The following is a student's answer sheet for an exam. Each answer is delimited by markers like "Answer to the question no-1a" and "End of Answer-1a". The delimites must be in the exact same format (Answer to the question no-<id> and End of Answer-<id>) where <id> is like 1a, 1b, 2c, etc. The answer might have some missing delimiters like missing starting or ending markers. If you encounter such cases, you must add the missing markers in the correct format at the appropriate places to ensure that each answer is properly delimited. For example, if an answer is missing the starting marker, you should add "Answer to the question no-<id>" at the beginning of that answer. Similarly, if an answer is missing the ending marker, you should add "End of Answer-<id>" at the end of that answer. The goal is to ensure that each answer is clearly marked and separated from others. Do not touch the content of the answers, just add the missing markers where necessary. Be thorough and ensure that every answer is properly delimited. Do not write anything else other than the answers and delimiters. The answer is as follows: {answer}
    """.strip()

def format_answer(answer):
    prompt = build_prompt(answer)
    
    print("Generating formatted answers with Gemini...")

    model = genai.GenerativeModel(MODEL_NAME)

    response = model.generate_content(prompt)
    
    print("Formatting complete.")

    text = response.text.strip()

    return text