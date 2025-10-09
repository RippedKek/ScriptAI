import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from rich.console import Console
from rich.table import Table

console = Console()

console.print("[bold cyan]Loading assessment models (advanced-integrated)...[/bold cyan]")

# Sentence embeddings
try:
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
except Exception:
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# spaCy
try:
    nlp = spacy.load("en_core_web_trf")
except Exception:
    nlp = spacy.load("en_core_web_sm")

# Optional NLI for contradiction
try:
    nli = pipeline("text-classification", model="microsoft/deberta-large-mnli")
    NLI_AVAILABLE = True
except Exception:
    NLI_AVAILABLE = False
    console.print("[yellow]DeBERTa-MNLI not available — skipping contradiction check.[/yellow]")

console.print("[green]Models ready.[/green]")

def _encode_norm(text):
    "Encode with L2 normalization (compat across sentence-transformers versions)."
    try:
        return model.encode(text, normalize_embeddings=True)
    except TypeError:
        vec = model.encode(text)
        # normalize
        if hasattr(vec, "shape"):
            norm = np.linalg.norm(vec, ord=2, axis=-1, keepdims=True) + 1e-12
            return vec / norm
        return vec

def semantic_score(ref, ans):
    ref_emb = _encode_norm(ref)
    ans_emb = _encode_norm(ans)
    sim = float(util.cos_sim(ref_emb, ans_emb))
    return float(np.clip(sim, 0.0, 1.0))

def keyword_coverage(ref, ans):
    vect = TfidfVectorizer(stop_words="english")
    vect.fit([ref, ans])
    ref_vec, ans_vec = vect.transform([ref, ans])
    tfidf_overlap = ref_vec.multiply(ans_vec).sum() / (ref_vec.sum() + 1e-6)

    ref_lemmas = {t.lemma_.lower() for t in nlp(ref) if t.pos_ in ["NOUN", "VERB", "ADJ"]}
    ans_lemmas = {t.lemma_.lower() for t in nlp(ans) if t.pos_ in ["NOUN", "VERB", "ADJ"]}
    lemma_overlap = len(ref_lemmas & ans_lemmas) / max(1, len(ref_lemmas))
    return 0.7 * tfidf_overlap + 0.3 * lemma_overlap

def content_completeness(ref, ans):
    ref_tokens = {t.lemma_.lower() for t in nlp(ref) if t.is_alpha}
    ans_tokens = {t.lemma_.lower() for t in nlp(ans) if t.is_alpha}
    if not ref_tokens:
        return 0.0
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
    if not NLI_AVAILABLE:
        return 0.0
    try:
        result = nli(f"{ref} [SEP] {ans}")[0]["label"]
        return -0.3 if result == "CONTRADICTION" else 0.0
    except Exception:
        return 0.0

def information_density(ref, ans):
    sents = [s.text.strip() for s in nlp(ans).sents]
    if not sents:
        return 0.5
    sims = []
    for i, s1 in enumerate(sents):
        for j, s2 in enumerate(sents[:i]):
            sims.append(semantic_score(s1, s2))
    redundancy = float(np.mean(sims)) if sims else 0.0
    density = 1 - min(redundancy, 0.6)
    length_boost = min(len(sents) / 5, 1.0)
    return float(np.clip(0.6 * density + 0.4 * length_boost, 0, 1.0))

def coherence_check(ref, ans):
    sents = [s.text.strip() for s in nlp(ans).sents]
    if len(sents) <= 1:
        return 1.0
    pair_sims = [semantic_score(sents[i], sents[i+1]) for i in range(len(sents)-1)]
    mean_flow = float(np.mean(pair_sims)) if pair_sims else 1.0
    ref_sims = [semantic_score(ref, s) for s in sents]
    topic_alignment = float(np.mean(ref_sims)) if ref_sims else 1.0
    return float(np.clip(0.6 * topic_alignment + 0.4 * mean_flow, 0, 1.0))

def elaboration_bonus(ref, ans):
    ref_lemmas = {t.lemma_.lower() for t in nlp(ref) if t.pos_ in ["NOUN", "VERB"]}
    ans_lemmas = {t.lemma_.lower() for t in nlp(ans) if t.pos_ in ["NOUN", "VERB"]}
    extra = ans_lemmas - ref_lemmas
    # treat each extra lemma as a tiny candidate concept
    relevant = [w for w in extra if semantic_score(ref, w) > 0.4]
    return min(len(relevant) / 10.0, 0.05)  # up to +5%

def robust_final_score(ref, ans):
    sem = semantic_score(ref, ans)
    key = keyword_coverage(ref, ans)
    comp = content_completeness(ref, ans)
    dens = information_density(ref, ans)
    coh = coherence_check(ref, ans)
    contra = contradiction_penalty(ref, ans)
    bonus = elaboration_bonus(ref, ans)

    raw = (0.75 * sem +
           0.15 * key +
           0.05 * comp +
           0.025 * dens +
           0.025 * coh +
           contra +
           bonus)

    if len(ans.split()) > len(ref.split()) * 1.2 and coh > 0.8 and key > 0.8:
        raw += 0.05

    score = round(100 * float(np.clip(raw, 0, 1)), 2)
    return score, {
        "semantic": sem, "keyword": key, "completeness": comp,
        "density": dens, "coherence": coh,
        "contradiction": contra, "bonus": bonus
    }

def extract_answers(text):
    """
    Extract answers in the strict form:
        'Answer to the question no-<id>' ... 'End of Answer-<id>'
    where <id> looks like 1a, 1b, 2c, etc.
    The same <id> must close the block.
    """
    pattern = re.compile(
        r"(?:answer\s*to\s*the\s*question\s*no)\s*[-]??\s*"
        r"([0-9]+[a-z])\)??\s*"          # <id> like 1a (optional trailing ')')
        r"(.*?)\s*"
        r"(?:end\s*of\s*answer)\s*-\s*\1\b",  # require 'End of Answer-<same id>'
        re.IGNORECASE | re.DOTALL,
    )
    results = {}
    for m in pattern.finditer(text):
        qid = m.group(1).lower()
        ans = m.group(2).strip()
        results[qid] = ans
    return results

with open('rubric.txt', 'r', encoding='utf-8') as file:
    REFERENCE_TEXT = file.read()

def evaluate_text(student_text):
    """
    Evaluate OCR-extracted student answers against REFERENCE_TEXT.
    Returns:
        marks: dict mapping question id (e.g., '1a') -> score (0-100)
    """
    ref_answers = extract_answers(REFERENCE_TEXT)
    stu_answers = extract_answers(student_text)

    if not ref_answers:
        console.print("[red]No valid reference answers detected.[/red]")
        return {}
    if not stu_answers:
        console.print("[red]No valid student answers detected in OCR text.[/red]")
        # still return zeros for known reference questions
        return {qid: 0.0 for qid in ref_answers.keys()}

    console.print(f"[bold yellow]\n=========== ADVANCED MULTI-ANSWER ASSESSMENT ===========[/bold yellow]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Question", style="cyan", width=10)
    table.add_column("Score", justify="center")
    table.add_column("Semantic", justify="center")
    table.add_column("Keyword", justify="center")
    table.add_column("Completeness", justify="center")
    table.add_column("Grade", justify="center")
    table.add_column("Feedback", width=65)

    marks = {}

    def _feedback(ref, ans, metrics):
        sem, key, comp = metrics["semantic"], metrics["keyword"], metrics["completeness"]
        ref_lemmas = {t.lemma_.lower() for t in nlp(ref) if t.pos_ in ["NOUN", "VERB"]}
        ans_lemmas = {t.lemma_.lower() for t in nlp(ans) if t.pos_ in ["NOUN", "VERB"]}
        missing = ref_lemmas - ans_lemmas

        if sem > 0.9 and key > 0.8:
            msg = "Excellent understanding with detailed coverage."
        elif sem > 0.75 and key > 0.6:
            msg = "Good understanding. Add terms like: " + ", ".join(list(missing)[:3])
        elif sem > 0.5:
            msg = "Partial correctness — some key ideas missing."
        else:
            msg = "Concept unclear or off-topic."
        return msg

    for qid, ref_ans in ref_answers.items():
        stu_ans = stu_answers.get(qid, "")
        if not stu_ans:
            table.add_row(qid.upper(), "0.00", "-", "-", "-", "N/A", "No answer submitted.")
            marks[qid] = 0.0
            continue

        score, metrics = robust_final_score(ref_ans, stu_ans)
        if score > 90:
            grade = "A+"
        elif score > 75:
            grade = "A"
        elif score > 60:
            grade = "B"
        elif score > 45:
            grade = "C"
        else:
            grade = "D"

        fb = _feedback(ref_ans, stu_ans, metrics)
        table.add_row(
            qid.upper(),
            f"{score:.2f}",
            f"{metrics['semantic']:.2f}",
            f"{metrics['keyword']:.2f}",
            f"{metrics['completeness']:.2f}",
            grade,
            fb,
        )
        marks[qid] = round(score, 2)

    console.print(table)
    console.print("[green]\nAssessment Complete.[/green]")
    return marks