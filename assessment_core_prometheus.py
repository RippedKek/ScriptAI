import re
import textwrap
from rich.console import Console
from rich.table import Table
from rag_retriever import retrieve_context
# from impression_llm import grade  # for assessment from prometheus
from gemini_assessment import grade  # for assessment from gemini
import spacy

console = Console()
nlp = spacy.load("en_core_web_sm")

# Optional toggle to print the full prompt sent to the model
DEBUG_PROMPT = False


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


def evaluate_text(student_text, reference_text, source_name):
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

    console.print(f"[bold yellow]\n========== GEMINI ASSESSMENT ==========[/bold yellow]")
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

        #  Reference-based Retrieval: using only teacher's correct answer
        context_hits = retrieve_context(ref, top_k=4)

        #  DEBUG PRINT — showing retrieved chunks and metadata
        # console.print(f"\n[bold cyan]→ Retrieved Context for Question {qid.upper()}:[/bold cyan]")
        # for i, hit in enumerate(context_hits, start=1):
        #     console.print(f"[yellow]Chunk {i}[/yellow]  |  [green]{hit['source']}[/green]")
        #     console.print(textwrap.fill(hit["text"].strip(), width=100))
        #     console.print("-" * 120)

        # Combining all retrieved textbook text chunks
        context_text = "\n".join([hit["text"] for hit in context_hits])

        # Collecting source info (e.g., "book.pdf - page 12")
        sources = [hit["source"] for hit in context_hits]

        # If a source_name override is provided, replace only the filename
        # portion (e.g., 'X.pdf') while preserving any suffix like ' - page 12'
        new_sources = []
        for src in sources:
            # Split on the first ' - ' to separate filename from the rest
            if " - " in src:
                _filename, suffix = src.split(" - ", 1)
                new_sources.append(f"{source_name} - {suffix}")
            else:
                # If no suffix, just use the provided source_name
                new_sources.append(source_name)
        sources = new_sources

        # Join final sources into a single string for display
        source_pages = "; ".join(sources)
        print(source_pages)

        
        if DEBUG_PROMPT:
            console.print(f"\n[bold blue]→ Full Prompt Sent to Model ({qid.upper()}):[/bold blue]")
            console.print(textwrap.fill(
                f"Question: Question {qid}\nReference: {ref}\nContext: {context_text}\nStudent Answer: {ans}",
                width=100
            ))
            console.print("-" * 120)

        #  Grading using Prometheus/Gemini
        score, feedback = grade(
            question=f"Question {qid}",
            reference=ref,
            student_answer=ans,
            context=context_text
        )

        
        table.add_row(qid.upper(), str(score), feedback[:60], source_pages)

        # results
        marks[qid] = {
            "score": score,
            "feedback": feedback,
            "sources": source_pages
        }

    console.print(table)
    console.print("[green]Assessment complete.[/green]")
    return marks
