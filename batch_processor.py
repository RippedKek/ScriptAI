"""
Batch processing utilities for multiple images
"""

import os


class BatchProcessor:
    """Process multiple images in batch"""
    
    def __init__(self, ocr_engine):
        """
        Initialize batch processor
        
        Args:
            ocr_engine: OCREngine instance
        """
        self.ocr_engine = ocr_engine
    
    def process_images(self, image_paths, output_file=None, verbose=True):
        """
        Process multiple images and extract text
        
        Args:
            image_paths: List of image file paths
            output_file: Optional file to save results
            verbose: Print progress messages
        
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
                
                text = self.ocr_engine.extract_text(img_path)
                results[img_path] = text
                
                if verbose:
                    print("✅ Success")
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"\n📝 Preview:\n{preview}\n")
                    
            except Exception as e:
                if verbose:
                    print(f"❌ Error: {e}")
                results[img_path] = f"Error: {e}"
        
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
    
    def process_directory(self, directory_path, extensions=None, output_file=None):
        """
        Process all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            extensions: List of file extensions to process
            output_file: Optional file to save results
        
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
        
        return self.process_images(image_files, output_file)
