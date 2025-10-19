
import os, re
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv() 
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(" GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

embedder = SentenceTransformer(EMBED_MODEL)


def build_prompt(question, reference, student_answer, context):
    return f"""
    You are a teacher grading a student's written answer.

    Question: {question}
    Reference Answer: {reference}
    Relevant Knowledge: {context}
    Student Answer: {student_answer}

    Give a concise evaluation in the format:
    Score: <0–100>
    Feedback: <one short sentence>
    """.strip()


def grade(question, reference, student_answer, context=""):
    prompt = build_prompt(question, reference, student_answer, context)

    model = genai.GenerativeModel(MODEL_NAME)
    # for m in genai.list_models():
    #     if "generateContent" in m.supported_generation_methods:
    #         print(m.name)

    response = model.generate_content(prompt)

    text = response.text.strip()

    # Extract score and feedback
    score_match = re.search(r"score[:\-]?\s*(\d{1,3})", text, re.I)
    feedback_match = re.search(r"feedback[:\-]?\s*(.*)", text, re.I)

    score = int(score_match.group(1)) if score_match else None
    feedback = feedback_match.group(1).strip() if feedback_match else text

    # fallback if no numeric score
    if score is None:
        ref_emb = embedder.encode(reference, convert_to_tensor=True, normalize_embeddings=True)
        ans_emb = embedder.encode(student_answer, convert_to_tensor=True, normalize_embeddings=True)
        sem = float(util.cos_sim(ref_emb, ans_emb))
        score = int(sem * 100)
        feedback += f" (Estimated via similarity: {score})"

    return score, feedback


# if __name__ == "__main__":
#     q = "What is photosynthesis?"
#     ref = "Photosynthesis converts light, CO₂, and water into glucose and oxygen."
#     ans = "Plants use sunlight to make food from CO₂ and water."
#     ctx = "Chlorophyll absorbs light in chloroplasts."

#     s, fb = grade(q, ref, ans, ctx)
#     print(f"Score: {s}\nFeedback: {fb}")
