from rich.console import Console
from assessment_core_prometheus import evaluate_text

console = Console()

def run_assessment(extracted_text: str):
    """
    Runs auto-assessment pipeline on OCR text.
    The text should contain:
      'Answer to the question no-1a' ... 'End of Answer-1a'
    Returns:
      dict of per-question marks {qid: {"score": int, "feedback": str}}
    """
    try:
        console.print("[bold cyan]Running Prometheus RAG-based assessment...[/bold cyan]")

        # Load teacher reference from rubric.txt
        with open("rubric.txt", "r", encoding="utf-8") as f:
            reference_text = f.read()

        marks = evaluate_text(extracted_text, reference_text)
        return marks

    except Exception as e:
        console.print(f"[red]Assessment failed: {e}[/red]")
        return {}
