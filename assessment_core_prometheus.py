import re
from rich.console import Console
from rich.table import Table
from rag_retriever import retrieve_context
from impression_llm import grade #for assessment from prometheus
#from gemini_assessment import grade #for assessment from gemini
import spacy

console = Console()
nlp = spacy.load("en_core_web_sm")


def extract_answers(text):
    """
    Detect patterns like:
      'Answer to the question no-1a' ... 'End of Answer-1a'
    and return dict { '1a': 'answer text' }.
    """
    pattern = re.compile(
        r"(?:answer\s*to\s*the\s*question\s*no)\s*-?\s*([0-9]+[a-z])\)?\s*(.*?)\s*(?:end\s*of\s*answer)\s*-\s*\1\b",
        re.IGNORECASE | re.DOTALL,
    )
    results = {}
    for m in pattern.finditer(text):
        qid = m.group(1).lower()
        ans = m.group(2).strip()
        results[qid] = ans
    return results


def evaluate_text(student_text, reference_text):
    """
    Compare student OCR text against teacher reference answers.
    Returns dict: {question_id: {"score": int, "feedback": str, "sources": str}}
    """

    ref_answers = extract_answers(reference_text)
    stu_answers = extract_answers(student_text)

    if not ref_answers:
        console.print("[red]No reference answers found![/red]")
        return {}
    if not stu_answers:
        console.print("[red]No student answers detected.[/red]")
        return {
            qid: {"score": 0, "feedback": "No answer submitted.", "sources": ""}
            for qid in ref_answers
        }

    console.print(f"[bold yellow]\n========== PROMETHEUS + RAG ASSESSMENT ==========[/bold yellow]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Question", style="cyan", width=8)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Feedback", width=60)
    table.add_column("Sources (Pages)", width=40)

    marks = {}

    for qid, ref in ref_answers.items():
        ans = stu_answers.get(qid, "")
        if not ans:
            table.add_row(qid.upper(), "0", "No answer submitted.", "-")
            marks[qid] = {"score": 0, "feedback": "No answer submitted.", "sources": ""}
            continue

        #  Retrieve textbook context for this answer
        context_hits = retrieve_context(ans, top_k=3)
        context_text = "\n".join([hit["text"] for hit in context_hits])

        #  Collect textbook source info (e.g., "anatomy_v2.pdf - page 12")
        sources = [hit["source"] for hit in context_hits]
        # Option 1: full source info
        source_pages = "; ".join(sources)
        # Option 2 (optional): only page numbers 
        # source_pages = "; ".join([s.split("page")[-1].strip() for s in sources])

        #  Grading using Prometheus model
        score, feedback = grade(
            question=f"Question {qid}",
            reference=ref,
            student_answer=ans,
            context=context_text
        )

        # Add to console table
        table.add_row(qid.upper(), str(score), feedback[:60], source_pages)

        #  Store extended data 
        marks[qid] = {
            "score": score,
            "feedback": feedback,
            "sources": source_pages
        }

    console.print(table)
    console.print("[green]Assessment complete.[/green]")
    return marks
