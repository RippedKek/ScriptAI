import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from rich.console import Console
from rich.table import Table

console = Console()

# ------------------------------------------------------------
# 1️⃣ Model Setup
# ------------------------------------------------------------
console.print("[bold cyan]Loading assessment models...[/bold cyan]")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
nlp = spacy.load("en_core_web_sm")
console.print("[green]✅ Assessment models loaded successfully.[/green]")


# ------------------------------------------------------------
# 2️⃣ Core Scoring Functions
# ------------------------------------------------------------
def semantic_similarity(ref, ans):
    """Cosine similarity between semantic embeddings of reference and answer."""
    ref_emb = model.encode(ref, convert_to_tensor=True, show_progress_bar=False)
    ans_emb = model.encode(ans, convert_to_tensor=True, show_progress_bar=False)
    return float(util.cos_sim(ref_emb, ans_emb))


def weighted_keyword_overlap(ref, ans):
    """TF-IDF weighted keyword overlap — emphasizes important domain terms."""
    vect = TfidfVectorizer(stop_words="english")
    vect.fit([ref, ans])
    ref_vec, ans_vec = vect.transform([ref, ans])
    common = ref_vec.multiply(ans_vec)
    return common.sum() / ref_vec.sum() if ref_vec.sum() != 0 else 0.0


def length_factor(ref, ans):
    """Checks proportionality of length between reference and student answer."""
    r, a = len(ref.split()), len(ans.split())
    if r == 0:
        return 0
    ratio = min(a / r, 1.5)
    return min(ratio / 1.5, 1.0)


def feedback(ref, ans, sem, key):
    """Generate readable feedback for the teacher/student."""
    ref_tokens = {t.lemma_.lower() for t in nlp(ref) if t.pos_ in ["NOUN", "VERB"]}
    ans_tokens = {t.lemma_.lower() for t in nlp(ans) if t.pos_ in ["NOUN", "VERB"]}
    missing = ref_tokens - ans_tokens

    if sem < 0.5:
        msg = "❌ Far from reference — needs major conceptual improvement."
    elif key < 0.5:
        msg = "⚠️ Missing key ideas: " + ", ".join(list(missing)[:5])
    elif sem > 0.8 and key > 0.8:
        msg = "✅ Well written and semantically close to the reference."
    else:
        msg = "👍 Mostly correct; elaborate on details for better clarity."
    return msg


def final_score(ref, ans):
    """Compute weighted final score with semantic, keyword, and length balance."""
    sem = semantic_similarity(ref, ans)
    key = weighted_keyword_overlap(ref, ans)
    length = length_factor(ref, ans)
    score = 100 * (0.8 * sem + 0.15 * key + 0.05 * length)
    return round(score, 2), {"semantic": sem, "keyword": key, "length": length}


# ------------------------------------------------------------
# 3️⃣ Multi-Answer Extractor
# ------------------------------------------------------------
def extract_answers(text):
    """
    Extract answers between delimiters:
        'Answer to the question no-1a' ... 'End of answer'
    Works with variations like 1a, 1b, 2a, etc.
    """
    pattern = re.compile(
        r"answer\s*to\s*the\s*question\s*no[\s\-./]*([0-9]+[a-z]?)\)?\s*(.*?)\s*(?=end\s*of\s*answer|$)",
        re.IGNORECASE | re.DOTALL,
    )

    return {m[0].lower(): m[1].strip() for m in pattern.findall(text)}


# ------------------------------------------------------------
# 4️⃣ Hardcoded Reference Answers
# ------------------------------------------------------------
REFERENCE_TEXT = """
Answer to the question no 1a
IPv4 (Internet Protocol version 4) is a communication protocol used to identify and locate devices on a network and route traffic across the internet. It uses a 32-bit address format (e.g., 192.168.1.1) to uniquely identify each device.
IPv4 operates on a packet-switched network, where data is divided into packets and sent independently.
Since it supports about 4.3 billion unique addresses, IPv6 was later introduced to overcome this limitation.
End of answer

Answer to the question no 1b
IPv6 (Internet Protocol version 6) is the latest version of the Internet Protocol designed to replace IPv4.
It uses a 128-bit address format (e.g., 2001:0db8:85a3::8a2e:0370:7334), allowing for a vastly larger number of unique IP addresses.
IPv6 provides improved features such as faster routing, built-in security (IPsec), and simplified address configuration.
It was developed to handle the rapid growth of internet-connected devices and overcome IPv4’s address exhaustion.
End of answer
"""


# ------------------------------------------------------------
# 5️⃣ Evaluation Function (called from main.py)
# ------------------------------------------------------------
def evaluate_text(student_text):
    """
    Evaluate OCR-extracted student answers (passed from main.py)
    against hardcoded reference answers.
    """
    ref_answers = extract_answers(REFERENCE_TEXT)
    stu_answers = extract_answers(student_text)

    if not ref_answers:
        console.print("[red]No valid reference answers detected.[/red]")
        return
    if not stu_answers:
        console.print("[red]No valid student answers detected in OCR text.[/red]")
        return

    console.print(f"[bold yellow]\n=========== MULTI-ANSWER ASSESSMENT ===========[/bold yellow]")
    console.print(f"[cyan]Detected Reference Questions:[/cyan] {list(ref_answers.keys())}")
    console.print(f"[cyan]Detected Student Questions:[/cyan] {list(stu_answers.keys())}\n")

    # Build the results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Question", style="cyan", width=10)
    table.add_column("Score", justify="center")
    table.add_column("Semantic", justify="center")
    table.add_column("Keyword", justify="center")
    table.add_column("Length", justify="center")
    table.add_column("Grade", justify="center")
    table.add_column("Feedback", width=60)

    for qid, ref_ans in ref_answers.items():
        stu_ans = stu_answers.get(qid, "")
        if not stu_ans:
            table.add_row(qid.upper(), "-", "-", "-", "-", "N/A", "No answer submitted.")
            continue

        score, metrics = final_score(ref_ans, stu_ans)

        if score > 85:
            grade = "A (Excellent)"
        elif score > 70:
            grade = "B (Good)"
        elif score > 55:
            grade = "C (Partial)"
        else:
            grade = "D (Poor)"

        fb = feedback(ref_ans, stu_ans, metrics["semantic"], metrics["keyword"])
        table.add_row(
            qid.upper(),
            f"{score:.2f}",
            f"{metrics['semantic']:.2f}",
            f"{metrics['keyword']:.2f}",
            f"{metrics['length']:.2f}",
            grade,
            fb,
        )

    console.print(table)
    console.print("[green]\n✅ Assessment Complete.[/green]")


# ------------------------------------------------------------
# 6️⃣ Local Test Block (Optional)
# ------------------------------------------------------------
if __name__ == "__main__":
    """
    Optional test: Run this file directly with a sample OCR output in 'student_sample.txt'
    """
    try:
        with open("student_sample.txt", "r", encoding="utf-8") as s:
            student_text = s.read()
        evaluate_text(student_text)
    except FileNotFoundError:
        console.print("[red]student_sample.txt not found. Provide sample OCR text for testing.[/red]")
