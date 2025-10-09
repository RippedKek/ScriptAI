"""
Batch processing utilities for multiple images and zip archives
"""
import os
import tempfile
import zipfile
from pathlib import Path

class BatchProcessor:
    """Process multiple images in batch"""

    def __init__(self, ocr_engine):
        """
        Initialize batch processor

        Args:
            ocr_engine: OCREngine instance
        """
        self.ocr_engine = ocr_engine

    def process_images(self, image_paths, output_file=None, verbose=True, mode="all"):
        """
        Process multiple images and extract text

        Args:
            image_paths: List of image file paths
            output_file: Optional file to save results
            verbose: Print progress messages
            mode: "all" uses extract_all, "structured" uses extract_with_structure

        Returns:
            Dictionary with filenames as keys and extracted text as values
        """
        results = {}

        for i, img_path in enumerate(image_paths, 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Processing image {i}/{len(image_paths)}: {img_path}")
                print('='*60)

            try:
                if verbose:
                    print("🔄 Extracting text...")

                if mode == "structured":
                    text = self.ocr_engine.extract_with_structure(img_path)
                else:
                    text = self.ocr_engine.extract_all(img_path)
                results[str(img_path)] = text

                if verbose:
                    print("✅ Success")
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"\n📝 Preview:\n{preview}\n")

            except Exception as e:
                if verbose:
                    print(f"❌ Error: {e}")
                results[str(img_path)] = f"Error: {e}"

        # Save results to file
        if output_file:
            self._save_results(results, output_file)
            if verbose:
                print(f"\n💾 All results saved to {output_file}")

        return results

    def _save_results(self, results, output_file):
        """Save batch results to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for img_path, text in results.items():
                f.write(f"{'='*60}\n")
                f.write(f"FILE: {img_path}\n")
                f.write(f"{'='*60}\n")
                f.write(text)
                f.write(f"\n\n")

    def process_directory(self, directory_path, extensions=None, output_file=None, mode="all"):
        """
        Process all images in a directory

        Args:
            directory_path: Path to directory containing images
            extensions: List of file extensions to process
            output_file: Optional file to save results
            mode: "all" or "structured"

        Returns:
            Dictionary with filenames as keys and extracted text as values
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # Get all image files
        image_files = []
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(directory_path, filename))

        print(f"\n📁 Found {len(image_files)} images in {directory_path}")

        if not image_files:
            print("⚠️  No images found!")
            return {}

        # Sort for stable order
        image_files = sorted(image_files)
        return self.process_images(image_files, output_file, mode=mode)

    def process_zip(self, zip_path, verbose=True):
        """
        Process a zip file containing scanned pages.
        It is guaranteed that there will be a title image named 'title.png/jpg/jpeg'.
        Only run student info extractor on this title image.
        Run 'extract all' on every other image.

        Returns:
            dict(title_text=str, pages_texts=dict[path->text])
        """
        if verbose:
            print(f"\n📦 Loading zip: {zip_path}")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip not found: {zip_path}")

        tmpdir = tempfile.mkdtemp(prefix="exam_zip_")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)

        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_images = [p for p in Path(tmpdir).rglob('*') if p.suffix.lower() in img_exts]

        if not all_images:
            raise RuntimeError("No images found in zip.")

        # Identify title image (exact file name 'title.*')
        title_candidates = [p for p in all_images if p.stem.lower() == "title"]
        if not title_candidates:
            raise RuntimeError("No 'title.*' image found in zip.")
        title_path = sorted(title_candidates)[0]

        # Other pages
        other_pages = sorted([p for p in all_images if p != title_path])

        if verbose:
            print(f"🧾 Title page: {title_path}")
            print(f"🖼️  Answer pages: {len(other_pages)}")

        # Extract student info
        title_text = self.ocr_engine.extract_student_info(str(title_path))

        # Extract all answers
        pages_texts = {}
        for i, p in enumerate(other_pages, 1):
            if verbose:
                print(f"→ OCR answer page {i}/{len(other_pages)}: {p}")
            pages_texts[str(p)] = self.ocr_engine.extract_all(str(p))

        return {"title_text": title_text, "pages_texts": pages_texts}
