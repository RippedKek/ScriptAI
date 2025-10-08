
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy

# ------------------------------------------------------------
# 1️⃣ Model Setup
# ------------------------------------------------------------
print("Loading models...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

# ------------------------------------------------------------
# 2️⃣ Helper Functions
# ------------------------------------------------------------

def semantic_similarity(ref, ans):
    """Cosine similarity between embeddings."""
    ref_emb = model.encode(ref, convert_to_tensor=True)
    ans_emb = model.encode(ans, convert_to_tensor=True)
    return float(util.cos_sim(ref_emb, ans_emb))


def keyword_overlap(ref, ans):
    """Noun/verb keyword overlap."""
    ref_tokens = {t.lemma_.lower() for t in nlp(ref) if t.pos_ in ["NOUN", "VERB"]}
    ans_tokens = {t.lemma_.lower() for t in nlp(ans) if t.pos_ in ["NOUN", "VERB"]}
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & ans_tokens) / len(ref_tokens)


def length_factor(ref, ans):
    """Length ratio (student/reference)."""
    r, a = len(ref.split()), len(ans.split())
    ratio = min(a / r, 1.5)
    return min(ratio / 1.5, 1.0)


def final_score(ref, ans):
    """Weighted aggregate scoring."""
    sem = semantic_similarity(ref, ans)
    key = keyword_overlap(ref, ans)
    length = length_factor(ref, ans)
    score = 100 * (0.6 * sem + 0.3 * key + 0.1 * length)
    return round(score, 2), {"semantic": sem, "keyword": key, "length": length}

# ------------------------------------------------------------
# 3️⃣ Reference Answer and Test Cases
# ------------------------------------------------------------
reference = (
    "Photosynthesis is the process by which green plants use sunlight "
    "to make food from carbon dioxide and water."
)

student_cases = {
    "exact": "Photosynthesis is the process by which green plants use sunlight to make food from carbon dioxide and water.",
    "shorter": "Photosynthesis is the process by which plants make food using sunlight.",
    "longer": "Photosynthesis is the process where green plants make their own food using sunlight, water, and carbon dioxide, releasing oxygen as a byproduct which supports life on Earth.",
    "keywords_only": "Plants make food from sunlight and water.",
    "wrong_no_keywords": "Animals use oxygen to breathe and live.",
    "partial_keywords": "Photosynthesis uses sunlight and carbon dioxide but produces food without mentioning oxygen."
}

# ------------------------------------------------------------
# 4️⃣ Evaluation
# ------------------------------------------------------------
print("\n=========== STUDENT ANSWER ASSESSMENT ===========")
for case, ans in student_cases.items():
    score, metrics = final_score(reference, ans)
    print(f"\nCase: {case.upper()}")
    print(f"→ Score: {score:.2f}")
    print(f"   Semantic Similarity: {metrics['semantic']:.3f}")
    print(f"   Keyword Overlap:     {metrics['keyword']:.3f}")
    print(f"   Length Factor:       {metrics['length']:.3f}")
    if score > 90:
        grade = "A (Excellent)"
    elif score > 75:
        grade = "B (Good)"
    elif score > 60:
        grade = "C (Partial)"
    else:
        grade = "D (Poor)"
    print(f"   → Grade: {grade}")

print("\n✅ Assessment Complete.")
