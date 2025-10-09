# assessment_engine.py
from rich.console import Console
from assessment_core_v2 import evaluate_text  # optional split if you have it in a separate file

console = Console()

def run_assessment(extracted_text: str):
    """
    Runs the auto-assessment pipeline on extracted OCR text.
    The text should include:
        'Answer to the question no-1a' ... 'End of answer'
    """
    try:
        console.print("[bold cyan]Running answer assessment on OCR text...[/bold cyan]")
        evaluate_text(extracted_text)  # call your scoring logic
    except Exception as e:
        console.print(f"[red]Assessment failed: {e}[/red]")
