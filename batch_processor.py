import os
import re
import cv2
import json
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from figure_processor import process_figure


def _safe_extract(zipf: zipfile.ZipFile, path: str) -> None:
    """
    Guard against ZipSlip by verifying each member stays within 'path'.
    """
    def _is_within_directory(directory: str, target: str) -> bool:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        # commonpath raises on different drives on Windows, but both share root here.
        return os.path.commonpath([abs_directory, abs_target]) == abs_directory

    for member in zipf.infolist():
        target_path = os.path.join(path, member.filename)
        if not _is_within_directory(path, target_path):
            raise RuntimeError(f"Blocked ZipSlip attempt: {member.filename}")
    zipf.extractall(path)


class BatchProcessor:
    """Process multiple images (single student or many) in batch."""

    def __init__(self, ocr_engine):
        """
        Initialize batch processor

        Args:
            ocr_engine: OCREngine instance
        """
        self.ocr_engine = ocr_engine

    def process_images(
        self,
        image_paths: List[str],
        output_file: Optional[str] = None,
        verbose: bool = True,
        mode: str = "all",
    ) -> Dict[str, str]:
        """
        Process multiple images and extract text.

        Args:
            image_paths: List of image file paths
            output_file: Optional file to save results
            verbose: Print progress messages
            mode: "all" uses extract_all, "structured" uses extract_with_structure

        Returns:
            Dictionary with filenames as keys and extracted text as values
        """
        results: Dict[str, str] = {}

        for i, img_path in enumerate(image_paths, 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Processing image {i}/{len(image_paths)}: {img_path}")
                print('='*60)

            try:
                if verbose:
                    print("Extracting text...")

                if mode == "structured":
                    text = self.ocr_engine.extract_with_structure(img_path)
                else:
                    text = self.ocr_engine.extract_all(img_path)
                results[str(img_path)] = text

                if verbose:
                    print("Success")
                    preview = text[:200] + "..." if isinstance(text, str) and len(text) > 200 else text
                    print(f"\nPreview:\n{preview}\n")

            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                results[str(img_path)] = f"Error: {e}"

        # Save results to file
        if output_file:
            self._save_results(results, output_file)
            if verbose:
                print(f"\nAll results saved to {output_file}")

        return results

    def _save_results(self, results: Dict[str, str], output_file: str) -> None:
        """Save batch results to file."""
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for img_path, text in results.items():
                f.write(f"{'='*60}\n")
                f.write(f"FILE: {img_path}\n")
                f.write(f"{'='*60}\n")
                f.write(str(text))
                f.write(f"\n\n")

    def process_directory(
        self,
        directory_path: str,
        extensions: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        mode: str = "all",
        verbose: bool = True,
    ) -> Dict[str, str]:
        """
        Process all images in a directory.

        Args:
            directory_path: Path to directory containing images
            extensions: List of file extensions to process
            output_file: Optional file to save results
            mode: "all" or "structured"
            verbose: Print progress

        Returns:
            Dictionary with filenames as keys and extracted text as values
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # Get all image files (non-recursive)
        image_files: List[str] = []
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(directory_path, filename))

        if verbose:
            print(f"\nFound {len(image_files)} images in {directory_path}")

        if not image_files:
            if verbose:
                print("No images found!")
            return {}

        # Sort for stable order
        image_files = sorted(image_files)
        return self.process_images(image_files, output_file, verbose=verbose, mode=mode)

    def process_zip(
        self,
        zip_path: str,
        verbose: bool = True,
        assess_targets: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """
        Process a zip file containing scanned pages (SINGLE student).
        It is guaranteed that there will be a title image named 'title.png/jpg/jpeg'.
        Only run student info extractor on this title image.
        Run 'extract all' on every other image.

        Returns:
            dict(title_text=str, pages_texts=dict[path->text], figures=list[str])
        """
        if verbose:
            print(f"\nLoading zip: {zip_path}")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip not found: {zip_path}")

        tmpdir = tempfile.mkdtemp(prefix="exam_zip_")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                _safe_extract(z, tmpdir)

            img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            all_images = [p for p in Path(tmpdir).rglob('*') if p.suffix.lower() in img_exts]

            if not all_images:
                raise RuntimeError("No images found in zip.")

            # Identify title and figures
            title_candidates = [p for p in all_images if p.stem.lower() == "title"]
            figure_candidates = [p for p in all_images if p.stem.lower().startswith("figure")]
            if not title_candidates:
                raise RuntimeError("No 'title.*' image found in zip.")
            title_path = sorted(title_candidates)[0]
            figure_paths = sorted(figure_candidates)

            # Other pages
            other_pages = sorted([p for p in all_images if p != title_path and p not in figure_paths])

            if verbose:
                print(f"Title page: {title_path}")
                print(f"Figure pages: {len(figure_paths)}")
                print(f"Answer pages: {len(other_pages)}")

            # Extract student info
            title_text = self.ocr_engine.extract_student_info(str(title_path))

            # Extract all figures (if any) → run through layout processor
            figures: List[str] = []
            layout_dir = os.path.join('output', 'figures', 'layout')
            os.makedirs(layout_dir, exist_ok=True)

            for i, f in enumerate(figure_paths, 1):
                if verbose:
                    print(f"→ OCR figure page {i}/{len(figure_paths)}: {f}")
                image = cv2.imread(str(f))
                if image is None:
                    if verbose:
                        print(f"  Skipping (cannot read): {f}")
                    continue
                figures.extend(process_figure(image, os.path.join(layout_dir, f.stem)))

            # Assess figures (optional)
            if assess_targets is None:
                assess_targets = ["figure"] * len(figures)  # generic fallback
            if len(assess_targets) < len(figures):
                assess_targets = list(assess_targets) + ["figure"] * (len(figures) - len(assess_targets))

            assess_out_dir = os.path.join('output', 'figures', 'assessments')
            os.makedirs(assess_out_dir, exist_ok=True)

            for i, f in enumerate(figures, 1):
                if verbose:
                    print(f"→ ASSESS figure {i}/{len(figures)}: {f} (target={assess_targets[i-1]})")
                try:
                    figure_assessment = self.ocr_engine.assess_figure(str(f), target=assess_targets[i-1])
                except Exception as e:
                    if verbose:
                        print(f"  Assessment failed: {e}")
                    continue

                f_path = Path(f)

                # Parse JSON safely
                if isinstance(figure_assessment, str):
                    try:
                        data = json.loads(figure_assessment)
                    except json.JSONDecodeError:
                        data = {}
                else:
                    data = figure_assessment or {}

                # Clean "Figure of 1a" → "1a"
                figure_number_value = data.get("figure_number", "")
                match = re.search(r'Figure of\s*([0-9]+[a-z]?)', figure_number_value, re.IGNORECASE)
                data["figure_number"] = match.group(1) if match else figure_number_value or "Unknown"

                with open(os.path.join(assess_out_dir, f"{f_path.stem}_assessment.json"), 'w', encoding='utf-8') as af:
                    json.dump(data, af, indent=2, ensure_ascii=False)

            # Extract all answers
            pages_texts: Dict[str, str] = {}
            for i, p in enumerate(other_pages, 1):
                if verbose:
                    print(f"→ OCR answer page {i}/{len(other_pages)}: {p}")
                pages_texts[str(p)] = self.ocr_engine.extract_all(str(p))

            return {"title_text": title_text, "pages_texts": pages_texts, "figures": figures}

        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    # ONE student's extracted directory (used by parent-zip flow)
    def _process_student_dir(
        self,
        dir_path: str,
        verbose: bool = True,
        assess_targets: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """
        Process ONE student's extracted directory (contains title.*, figure*, and answer pages).

        Returns:
            dict(title_text=str, pages_texts=dict[path->text], figures=list[str])
        """
        p = Path(dir_path)
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        all_images = [q for q in p.rglob('*') if q.suffix.lower() in img_exts]
        if not all_images:
            raise RuntimeError(f"No images found in {dir_path}")

        # Identify files
        title_candidates = [q for q in all_images if q.stem.lower() == "title"]
        figure_candidates = [q for q in all_images if q.stem.lower().startswith("figure")]
        if not title_candidates:
            raise RuntimeError(f"No 'title.*' image found in {dir_path}")
        title_path = sorted(title_candidates)[0]
        figure_paths = sorted(figure_candidates)
        other_pages = sorted([q for q in all_images if q != title_path and q not in figure_paths])

        if verbose:
            print(f"[{p.name}] Title page: {title_path}")
            print(f"[{p.name}] Figure pages: {len(figure_paths)}")
            print(f"[{p.name}] Answer pages: {len(other_pages)}")

        # OCR title
        title_text = self.ocr_engine.extract_student_info(str(title_path))

        # Figures store per-student layout outputs
        figures: List[str] = []
        out_dir = os.path.join('output', 'figures', 'layout', p.name)
        os.makedirs(out_dir, exist_ok=True)

        for i, f in enumerate(figure_paths, 1):
            if verbose:
                print(f"[{p.name}] → OCR figure page {i}/{len(figure_paths)}: {f}")
            image = cv2.imread(str(f))
            if image is None:
                if verbose:
                    print(f"[{p.name}]   Skipping (cannot read): {f}")
                continue
            figures.extend(process_figure(image, os.path.join(out_dir, f.stem)))

        # Assess figures in the same pass if targets provided
        if assess_targets is not None:
            if len(assess_targets) < len(figures):
                assess_targets = list(assess_targets) + ["figure"] * (len(figures) - len(assess_targets))
            assess_out = os.path.join('output', 'figures', 'assessments', p.name)
            os.makedirs(assess_out, exist_ok=True)
            for i, f in enumerate(figures, 1):
                try:
                    fa = self.ocr_engine.assess_figure(str(f), target=assess_targets[i-1])
                except Exception as e:
                    if verbose:
                        print(f"[{p.name}]   Assessment failed ({f}): {e}")
                    continue
                f_path = Path(f)
                if isinstance(fa, str):
                    try:
                        data = json.loads(fa)
                    except json.JSONDecodeError:
                        data = {}
                else:
                    data = fa or {}
                figure_number_value = data.get("figure_number", "")
                match = re.search(r'Figure of\s*([0-9]+[a-z]?)', figure_number_value, re.IGNORECASE)
                data["figure_number"] = match.group(1) if match else figure_number_value or "Unknown"
                with open(os.path.join(assess_out, f"{f_path.stem}_assessment.json"), 'w', encoding='utf-8') as af:
                    json.dump(data, af, indent=2, ensure_ascii=False)

        # OCR answers
        pages_texts: Dict[str, str] = {}
        for i, pg in enumerate(other_pages, 1):
            if verbose:
                print(f"[{p.name}] → OCR answer page {i}/{len(other_pages)}: {pg}")
            pages_texts[str(pg)] = self.ocr_engine.extract_all(str(pg))

        return {"title_text": title_text, "pages_texts": pages_texts, "figures": figures}

    # Parent ZIP (many students → subfolders)
    def process_parent_zip(self, zip_path: str, verbose: bool = True, default_assess_target: str = "figure") -> Dict[str, object]:
        """
        Process a PARENT zip that contains MANY student folders.
        Each immediate subfolder name is treated as the Student ID (fallback if not parsed from title).

        Returns:
            dict[student_folder_name] -> result dict from _process_student_dir
        """
        if verbose:
            print(f"\nLoading parent zip: {zip_path}")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip not found: {zip_path}")

        tmpdir = tempfile.mkdtemp(prefix="parent_zip_")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                _safe_extract(z, tmpdir)

            # Find immediate subfolders (each is a student)
            roots = [d for d in Path(tmpdir).iterdir() if d.is_dir()]
            if not roots:
                if verbose:
                    print("No subfolders found. Falling back to single-student zip.")
                single = self._process_student_dir(tmpdir, verbose=verbose)
                return {Path(zip_path).stem: single}

            results: Dict[str, object] = {}
            for sd in sorted(roots):
                try:
                    if verbose:
                        print(f"\n=== Processing student folder: {sd.name} ===")
                    results[sd.name] = self._process_student_dir(
                        str(sd),
                        verbose=verbose,
                        assess_targets=[default_assess_target]
                    )
                except Exception as e:
                    print(f"[{sd.name}] Error: {e}")
                    results[sd.name] = {"error": str(e)}

            return results
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
