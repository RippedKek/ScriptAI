# assessment_engine.py
from rich.console import Console
from assessment_core import evaluate_text  # optional split if you have it in a separate file

console = Console()

def run_assessment(extracted_text: str):
    """
    Runs the auto-assessment pipeline on extracted OCR text.
    The text should include:
        'Answer to the question no-1a' ... 'End of Answer-1a'
    Returns:
        dict of per-question marks.
    """
    try:
        console.print("[bold cyan]Running answer assessment on OCR text...[/bold cyan]")
        marks = evaluate_text(extracted_text)  # returns dict
        return marks
    except Exception as e:
        console.print(f"[red]Assessment failed: {e}[/red]")
        return {}
