import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

PROMPT_MODEL = "prometheus-eval/prometheus-7b-v2.0"
# EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

_tok = None
_model = None
_embedder = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_models():
    """Lazy-load models only when first needed."""
    global _tok, _model, _embedder
    if _tok is not None and _model is not None and _embedder is not None:
        return  # already loaded

    print(f"[Prometheus] Initializing {PROMPT_MODEL} on {_device.upper()}...")
    _tok = AutoTokenizer.from_pretrained(PROMPT_MODEL)
    _model = AutoModelForCausalLM.from_pretrained(
        PROMPT_MODEL,
        device_map="auto" if _device == "cuda" else None,
        torch_dtype=torch.bfloat16 if _device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    _embedder = SentenceTransformer(EMBED_MODEL)
    print("[Prometheus] Ready.")

def _build_prompt(question, reference, student_answer, context):
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
    _load_models()  # <-- ensures model loaded on-demand

    inputs = _tok(_build_prompt(question, reference, student_answer, context),
                  return_tensors="pt").to(_device)
    output = _model.generate(
        **inputs,
        max_new_tokens=170,
        temperature=0.2,
        top_p=0.8,
        do_sample=True,
        pad_token_id=_tok.eos_token_id
    )
    response = _tok.decode(output[0], skip_special_tokens=True)
    response = re.sub(r"\s+", " ", response).strip()

    score_match = re.search(r"score[:\-]?\s*(\d{1,3})", response, re.I)
    feedback_match = re.search(r"feedback[:\-]?\s*(.*)", response, re.I)

    score = int(score_match.group(1)) if score_match else None
    feedback = feedback_match.group(1).strip() if feedback_match else response

    if score is None:
        ref_emb = _embedder.encode(reference, convert_to_tensor=True, normalize_embeddings=True)
        ans_emb = _embedder.encode(student_answer, convert_to_tensor=True, normalize_embeddings=True)
        sem = float(util.cos_sim(ref_emb, ans_emb))
        score = int(sem * 100)
        feedback += f" (Estimated via similarity: {score})"

    return score, feedback
