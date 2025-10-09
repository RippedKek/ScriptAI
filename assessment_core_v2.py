"""
====================================================================
 ADVANCED SEMANTIC ANSWER ASSESSMENT SYSTEM (v2-final)
====================================================================
✅ Handles short-but-correct answers fairly
✅ Rewards long, detailed, correct answers (no penalty if coherent)
✅ Penalizes irrelevant, repetitive, or contradictory text
✅ Adds elaboration bonus for relevant extra ideas
✅ Uses improved coherence + semantic density logic
✅ Supports 'End of answer 1a/1b/2c...' format for OCR compatibility
====================================================================
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from rich.console import Console
from rich.table import Table

console = Console()

# ------------------------------------------------------------
# 1️⃣ Model Setup
# ------------------------------------------------------------
console.print("[bold cyan]Loading assessment models (v3-final)...[/bold cyan]")

# Semantic model for QA-style meaning comparison
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# SpaCy transformer model (fallback to small if transformer not available)
try:
    nlp = spacy.load("en_core_web_trf")
except:
    nlp = spacy.load("en_core_web_sm")

# NLI model for contradiction detection
try:
    nli = pipeline("text-classification", model="microsoft/deberta-large-mnli")
    NLI_AVAILABLE = True
except:
    NLI_AVAILABLE = False
    console.print("[yellow]⚠️ DeBERTa-MNLI not available — skipping contradiction check.[/yellow]")

console.print("[green]✅ Models loaded successfully.[/green]")


# ------------------------------------------------------------
# 2️⃣ Core Scoring Functions
# ------------------------------------------------------------
def semantic_score(ref, ans):
    """Compute cosine similarity between embeddings (normalized)."""
    ref_emb = model.encode(ref, normalize_embeddings=True)
    ans_emb = model.encode(ans, normalize_embeddings=True)
    sim = float(util.cos_sim(ref_emb, ans_emb))
    return np.clip(sim, 0.0, 1.0)


def keyword_coverage(ref, ans):
    """TF-IDF + lemma overlap for key idea coverage."""
    vect = TfidfVectorizer(stop_words="english")
    vect.fit([ref, ans])
    ref_vec, ans_vec = vect.transform([ref, ans])
    tfidf_overlap = ref_vec.multiply(ans_vec).sum() / (ref_vec.sum() + 1e-6)

    ref_lemmas = {t.lemma_.lower() for t in nlp(ref) if t.pos_ in ["NOUN", "VERB", "ADJ"]}
    ans_lemmas = {t.lemma_.lower() for t in nlp(ans) if t.pos_ in ["NOUN", "VERB", "ADJ"]}
    lemma_overlap = len(ref_lemmas & ans_lemmas) / max(1, len(ref_lemmas))
    return 0.7 * tfidf_overlap + 0.3 * lemma_overlap


def content_completeness(ref, ans):
    """Conceptual coverage of reference content."""
    ref_tokens = {t.lemma_.lower() for t in nlp(ref) if t.is_alpha}
    ans_tokens = {t.lemma_.lower() for t in nlp(ans) if t.is_alpha}
    if not ref_tokens:
        return 0
    coverage = len(ans_tokens & ref_tokens) / len(ref_tokens)
    if coverage > 0.85:
        return 1.0
    elif coverage > 0.65:
        return 0.8
    elif coverage > 0.45:
        return 0.6
    else:
        return 0.4


def contradiction_penalty(ref, ans):
    """Detect logical contradiction with NLI."""
    if not NLI_AVAILABLE:
        return 0.0
    result = nli(f"{ref} [SEP] {ans}")[0]["label"]
    return -0.3 if result == "CONTRADICTION" else 0.0


def information_density(ref, ans):
    """
    Measures how much *unique semantic content* appears in the answer.
    Long, coherent, detailed answers are rewarded; repetition penalized.
    """
    sentences = [s.text.strip() for s in nlp(ans).sents]
    if not sentences:
        return 0.5

    sims = []
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences[:i]):
            sims.append(semantic_score(s1, s2))
    redundancy = np.mean(sims) if sims else 0.0
    density = 1 - min(redundancy, 0.6)

    length_boost = min(len(sentences) / 5, 1.0)
    return np.clip(0.6 * density + 0.4 * length_boost, 0, 1.0)


def coherence_check(ref, ans):
    """Measure logical flow and topic relevance across sentences."""
    sentences = [s.text.strip() for s in nlp(ans).sents]
    if len(sentences) <= 1:
        return 1.0

    pair_sims = [semantic_score(sentences[i], sentences[i+1])
                 for i in range(len(sentences)-1)]
    mean_flow = np.mean(pair_sims)
    ref_sims = [semantic_score(ref, s) for s in sentences]
    topic_alignment = np.mean(ref_sims)
    return np.clip(0.6 * topic_alignment + 0.4 * mean_flow, 0, 1.0)


def elaboration_bonus(ref, ans):
    """
    Reward additional relevant ideas not in reference.
    Up to +5% bonus if new, related key terms appear.
    """
    ref_lemmas = {t.lemma_.lower() for t in nlp(ref) if t.pos_ in ["NOUN", "VERB"]}
    ans_lemmas = {t.lemma_.lower() for t in nlp(ans) if t.pos_ in ["NOUN", "VERB"]}
    extra = ans_lemmas - ref_lemmas
    relevant = [w for w in extra if semantic_score(ref, w) > 0.4]
    return min(len(relevant) / 10, 0.05)  # up to +5%


# ------------------------------------------------------------
# 3️⃣ Combined Final Score
# ------------------------------------------------------------
def robust_final_score(ref, ans):
    sem = semantic_score(ref, ans)
    key = keyword_coverage(ref, ans)
    comp = content_completeness(ref, ans)
    dens = information_density(ref, ans)
    coh = coherence_check(ref, ans)
    contra = contradiction_penalty(ref, ans)
    bonus = elaboration_bonus(ref, ans)

    raw = (0.55 * sem +
           0.25 * key +
           0.10 * comp +
           0.05 * dens +
           0.05 * coh +
           contra +
           bonus)

    # Additional fairness: reward long, coherent, accurate answers
    if len(ans.split()) > len(ref.split()) * 1.2 and coh > 0.8 and key > 0.8:
        raw += 0.05

    return round(100 * np.clip(raw, 0, 1), 2), {
        "semantic": sem, "keyword": key, "completeness": comp,
        "density": dens, "coherence": coh,
        "contradiction": contra, "bonus": bonus
    }


# ------------------------------------------------------------
# 4️⃣ Feedback Generator
# ------------------------------------------------------------
def feedback(ref, ans, metrics):
    sem, key, comp = metrics["semantic"], metrics["keyword"], metrics["completeness"]
    ref_lemmas = {t.lemma_.lower() for t in nlp(ref) if t.pos_ in ["NOUN", "VERB"]}
    ans_lemmas = {t.lemma_.lower() for t in nlp(ans) if t.pos_ in ["NOUN", "VERB"]}
    missing = ref_lemmas - ans_lemmas

    if sem > 0.9 and key > 0.8:
        msg = "✅ Excellent understanding with detailed coverage."
    elif sem > 0.75 and key > 0.6:
        msg = "👍 Good understanding. Add terms like: " + ", ".join(list(missing)[:3])
    elif sem > 0.5:
        msg = "⚠️ Partial correctness — some key ideas missing."
    else:
        msg = "❌ Concept unclear or off-topic."
    return msg


# ------------------------------------------------------------
# 5️⃣ Multi-Answer Extraction (Updated for 'End of answer 1a/1b/...')
# ------------------------------------------------------------
def extract_answers(text):
    """
    Extract answers between:
      'Answer to the question no-1a' ... 'End of answer 1a'
    Supports flexible OCR variations such as:
      - 'Answer to the question no 1a'
      - 'End of answer 1a / 2b / 3c'
      - 'End ans 1a' or 'End answer no 1a'
    """
    pattern = re.compile(
        r"answer\s*to\s*the\s*question\s*no[-\s]*([0-9]+[a-z])\)?\s*(.*?)\s*(?=end\s*(?:of\s*)?(?:ans(?:wer)?(?:\s*no)?\s*)?\1|$)",
        re.IGNORECASE | re.DOTALL,
    )
    return {m[0].lower(): m[1].strip() for m in pattern.findall(text)}


# ------------------------------------------------------------
# 6️⃣ Hardcoded Reference Answers
# ------------------------------------------------------------
REFERENCE_TEXT = """
Answer to the question no-1a
Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen. It occurs in chloroplasts using chlorophyll pigments that capture solar energy.
End of answer 1a

Answer to the question no-1b
Chlorophyll is the green pigment found in chloroplasts that absorbs light energy during photosynthesis. It captures mostly blue and red wavelengths while reflecting green light, giving plants their color.
End of answer 1b
"""


# ------------------------------------------------------------
# 7️⃣ Evaluation Function
# ------------------------------------------------------------
def evaluate_text(student_text):
    ref_answers = extract_answers(REFERENCE_TEXT)
    stu_answers = extract_answers(student_text)

    if not ref_answers:
        console.print("[red]❌ No reference answers found.[/red]")
        return
    if not stu_answers:
        console.print("[red]❌ No student answers found.[/red]")
        return

    console.print(f"[bold yellow]\n=========== ADVANCED MULTI-ANSWER ASSESSMENT (v3-final) ===========[/bold yellow]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Question", style="cyan", width=10)
    table.add_column("Score", justify="center")
    table.add_column("Semantic", justify="center")
    table.add_column("Keyword", justify="center")
    table.add_column("Completeness", justify="center")
    table.add_column("Grade", justify="center")
    table.add_column("Feedback", width=65)

    for qid, ref_ans in ref_answers.items():
        stu_ans = stu_answers.get(qid, "")
        if not stu_ans:
            table.add_row(qid.upper(), "-", "-", "-", "-", "N/A", "❌ No answer submitted.")
            continue

        score, metrics = robust_final_score(ref_ans, stu_ans)

        if score > 90:
            grade = "A+ (Excellent)"
        elif score > 75:
            grade = "A (Good)"
        elif score > 60:
            grade = "B (Average)"
        elif score > 45:
            grade = "C (Weak)"
        else:
            grade = "D (Poor)"

        fb = feedback(ref_ans, stu_ans, metrics)
        table.add_row(
            qid.upper(),
            f"{score:.2f}",
            f"{metrics['semantic']:.2f}",
            f"{metrics['keyword']:.2f}",
            f"{metrics['completeness']:.2f}",
            grade,
            fb,
        )

    console.print(table)
    console.print("[green]\n✅ Assessment Complete.[/green]")


# ------------------------------------------------------------
# 8️⃣ Local Test Block
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        with open("student_sample.txt", "r", encoding="utf-8") as s:
            student_text = s.read()
        evaluate_text(student_text)
    except FileNotFoundError:
        console.print("[red]❌ student_sample.txt not found. Provide sample OCR text for testing.[/red]")
