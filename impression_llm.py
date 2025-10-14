import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util


PROMPT_MODEL = "prometheus-eval/prometheus-7b-v2.0"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"  # for fallback similarity

print(f"[Prometheus] Loading model: {PROMPT_MODEL}")
tok = AutoTokenizer.from_pretrained(PROMPT_MODEL)


device = "cuda" if torch.cuda.is_available() else "cpu"

impression_model = AutoModelForCausalLM.from_pretrained(
    PROMPT_MODEL,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
)
embedder = SentenceTransformer(EMBED_MODEL)
print(f"[Prometheus] Model ready on {device.upper()}.")


def build_prompt(question, reference, student_answer, context):
    return f"""
You are a teacher grading a student's written answer.

Question: {question}
Reference Answer: {reference}
Relevant Knowledge: {context}
Student Answer: {student_answer}

Evaluate the correctness and completeness of the student's answer.
Provide:
Score: <0–100>
Feedback: <one short sentence>
""".strip()


def grade(question, reference, student_answer, context=""):
    """Return numeric score and feedback string."""
    prompt = build_prompt(question, reference, student_answer, context)

    inputs = tok(prompt, return_tensors="pt").to(device)
    output = impression_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tok.eos_token_id
    )
    response = tok.decode(output[0], skip_special_tokens=True)
    response = re.sub(r"\s+", " ", response).strip()

    # Extract score & feedback
    score_match = re.search(r"score[:\-]?\s*(\d{1,3})", response, re.I)
    feedback_match = re.search(r"feedback[:\-]?\s*(.*)", response, re.I)

    score = int(score_match.group(1)) if score_match else None
    feedback = feedback_match.group(1).strip() if feedback_match else response

    # Fallback: semantic similarity if model didn't output numeric score
    if score is None:
        ref_emb = embedder.encode(reference, convert_to_tensor=True, normalize_embeddings=True)
        ans_emb = embedder.encode(student_answer, convert_to_tensor=True, normalize_embeddings=True)
        sem = float(util.cos_sim(ref_emb, ans_emb))
        score = int(sem * 100)
        feedback += f" (Estimated via similarity: {score})"

    return score, feedback


if __name__ == "__main__":
    print("\n[Prometheus] 🔍 Running quick self-test...")
    q = "What is photosynthesis?"
    ref = "Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen."
    ans = "Plants use sunlight to make food from carbon dioxide and water."
    ctx = "Chlorophyll absorbs light in chloroplasts during photosynthesis."

    score, fb = grade(q, ref, ans, ctx)
    print(f"\nScore: {score}\nFeedback: {fb}")
